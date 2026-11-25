"""Hand labels for the three judges: queue traces, pull labels, count agreement.

    uv run python -m evals.labels queue --results evals/results/latest.json
    uv run python -m evals.labels pull --results evals/results/latest.json

`queue` takes the traced trials of one judged run and adds them to LangSmith,
one annotation queue per judge. Each queue asks a person the same yes/no
question its judge was asked (`evals.judge.QUESTIONS`). The judge's own verdict
never reaches LangSmith, so the person labels blind.

`pull` reads the person's labels back and writes one row per trace and judge
to `evals/labels/labels.jsonl`, next to the question, the answer, the source
text and the judge's verdict. The rows live in the repo, so they outlast the
free plan's 14 days of trace data, and a new judge prompt can be checked
against the same labels later. It then prints each judge's agreement with the
person (eval-design.md, item 10).

Neither command calls a model. Both need `LANGSMITH_API_KEY`.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import langsmith
from loguru import logger

from evals.judge import JUDGES, JUDGES_BY_BEHAVIOR, QUESTIONS, JudgeName
from evals.tracing import LangSmithSettings

LABELS_PATH = Path("evals/labels/labels.jsonl")
DEFAULT_LIMIT = 100
"""Traces per `queue` call. About 100 labels per judge is what item 10 plans."""


def feedback_key(judge: JudgeName) -> str:
    """The LangSmith feedback key a person's label is saved under.

    Prefixed so a label can never be mistaken for an automated score with the
    judge's own name.
    """
    return f"human_{judge}"


@dataclass(frozen=True, slots=True)
class LabelTask:
    """One trace a person should label for one judge, with what the judge saw."""

    trace_id: str
    judge: JudgeName
    task_id: str
    trial: int
    outcome: str
    question: str
    answer: str
    sources: str
    judge_verdict: bool | None
    judge_reason: str


@dataclass(frozen=True, slots=True)
class HumanLabel:
    yes: bool
    comment: str
    labeled_at: str


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One line of `labels.jsonl`: a person's label next to the judge's verdict."""

    trace_id: str
    judge: str
    human: bool
    judge_verdict: bool | None
    judge_reason: str
    comment: str
    labeled_at: str
    task_id: str
    trial: int
    question: str
    answer: str
    sources: str
    run_id: str
    model: str
    prompt_version: str


class LabelStore(Protocol):
    """Where labeling happens. LangSmith in real use, a dict in tests."""

    def ensure_queue(self, judge: JudgeName) -> str:
        """The id of this judge's queue, created with its question if missing."""
        ...

    def add(self, queue_id: str, trace_ids: Sequence[str]) -> None: ...

    def labels(self, judge: JudgeName, trace_ids: Sequence[str]) -> dict[str, HumanLabel]:
        """The latest label per trace, for the traces a person has labeled."""
        ...


class LangSmithLabelStore:
    """Annotation queues and feedback in one LangSmith workspace."""

    def __init__(self, client: langsmith.Client, project: str) -> None:
        self._client = client
        self._project = project

    def ensure_queue(self, judge: JudgeName) -> str:
        key = feedback_key(judge)
        # Returns the existing config unchanged when it is the same one.
        self._client.create_feedback_config(
            key,
            feedback_config={
                "type": "categorical",
                "categories": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}],
            },
        )
        name = f"{self._project}-{judge}"
        for queue in self._client.list_annotation_queues(name=name, limit=1):
            return str(queue.id)
        queue = self._client.create_annotation_queue(
            name=name,
            description=f"Hand labels for the {judge} judge. Label without looking at the judge.",
            rubric_instructions=QUESTIONS[judge],
            rubric_items=[
                {"feedback_key": key, "description": QUESTIONS[judge], "is_required": True}
            ],
        )
        return str(queue.id)

    def add(self, queue_id: str, trace_ids: Sequence[str]) -> None:
        self._client.add_runs_to_annotation_queue(queue_id, run_ids=list(trace_ids))

    def labels(self, judge: JudgeName, trace_ids: Sequence[str]) -> dict[str, HumanLabel]:
        found: dict[str, tuple[str, HumanLabel]] = {}
        feedback = self._client.list_feedback(
            run_ids=list(trace_ids), feedback_key=[feedback_key(judge)]
        )
        for item in feedback:
            yes = _as_yes(item.score, item.value)
            if yes is None or item.run_id is None:
                continue
            stamp = item.modified_at or item.created_at
            when = stamp.isoformat() if stamp else ""
            run_id = str(item.run_id)
            # A relabel adds a second item. The newest one is the person's answer.
            if run_id not in found or when > found[run_id][0]:
                found[run_id] = (
                    when,
                    HumanLabel(yes=yes, comment=item.comment or "", labeled_at=when),
                )
        return {run_id: label for run_id, (_, label) in found.items()}


def _as_yes(score: object, value: object) -> bool | None:
    """A yes/no out of a feedback item, or `None` when it holds neither."""
    if isinstance(score, bool):
        return score
    if isinstance(score, int | float):
        return score >= 0.5
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return value.strip().lower() == "yes"
    return None


def label_tasks(results: Mapping[str, Any]) -> list[LabelTask]:
    """Every traced, graded trial of a run, once per judge its behavior reads.

    An errored trial has no answer to label. A trial with no trace id came
    from a run without `LANGSMITH_API_KEY`, so there is nothing to open.
    """
    behavior = {task["task_id"]: task.get("behavior", "") for task in results.get("tasks", [])}
    found: list[LabelTask] = []
    for trial in results.get("trials", []):
        trace_id = trial.get("trace_id")
        if not trace_id or trial.get("outcome") == "errored":
            continue
        transcript = trial.get("transcript", {})
        sources = "\n\n".join(
            f"[{item['tool']}]\n{item['text']}" for item in transcript.get("tool_results", [])
        )
        verdicts = {judge["judge"]: judge for judge in trial.get("judges", [])}
        for judge in JUDGES_BY_BEHAVIOR.get(behavior.get(trial["task_id"], ""), ()):
            verdict = verdicts.get(judge)
            found.append(
                LabelTask(
                    trace_id=trace_id,
                    judge=judge,
                    task_id=trial["task_id"],
                    trial=trial["trial"],
                    outcome=trial["outcome"],
                    question=trial.get("question", ""),
                    answer=transcript.get("answer", ""),
                    sources=sources,
                    judge_verdict=verdict["verdict"] if verdict else None,
                    judge_reason=verdict["reason"] if verdict else "",
                )
            )
    return found


def pick(tasks: Sequence[LabelTask], limit: int) -> list[LabelTask]:
    """At most `limit` traces, failed trials first, then spread across tasks.

    Sorted by trial number before task, so trial 1 of every task comes before
    trial 2 of any. A sample of 100 then covers many tasks, not 10 trials of 10.
    A trace keeps all its judges, so one opened trace is labeled once per queue.
    """
    traces: dict[str, LabelTask] = {}
    for task in tasks:
        traces.setdefault(task.trace_id, task)
    ordered = sorted(traces.values(), key=lambda t: (t.outcome != "failed", t.trial, t.task_id))
    chosen = {task.trace_id for task in ordered[:limit]}
    return [task for task in tasks if task.trace_id in chosen]


def queue(results: Mapping[str, Any], store: LabelStore, limit: int) -> dict[JudgeName, int]:
    """Add a run's traces to each judge's queue. Returns traces added per judge."""
    chosen = pick(label_tasks(results), limit)
    added: dict[JudgeName, int] = {}
    for judge in JUDGES:
        trace_ids = list(dict.fromkeys(t.trace_id for t in chosen if t.judge == judge))
        if not trace_ids:
            continue
        store.add(store.ensure_queue(judge), trace_ids)
        added[judge] = len(trace_ids)
    return added


def pull(results: Mapping[str, Any], store: LabelStore) -> list[LabelRow]:
    """The labeled traces of a run as rows. Unlabeled traces are left out."""
    tasks = label_tasks(results)
    rows: list[LabelRow] = []
    for judge in JUDGES:
        mine = [t for t in tasks if t.judge == judge]
        if not mine:
            continue
        labels = store.labels(judge, [t.trace_id for t in mine])
        for task in mine:
            label = labels.get(task.trace_id)
            if label is None:
                continue
            rows.append(
                LabelRow(
                    trace_id=task.trace_id,
                    judge=judge,
                    human=label.yes,
                    judge_verdict=task.judge_verdict,
                    judge_reason=task.judge_reason,
                    comment=label.comment,
                    labeled_at=label.labeled_at,
                    task_id=task.task_id,
                    trial=task.trial,
                    question=task.question,
                    answer=task.answer,
                    sources=task.sources,
                    run_id=str(results.get("run_id", "")),
                    model=str(results.get("model", "")),
                    prompt_version=str(results.get("prompt_version", "")),
                )
            )
    return rows


def read_rows(path: Path) -> list[LabelRow]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LabelRow(**json.loads(line)) for line in lines if line.strip()]


def merge_rows(old: Iterable[LabelRow], new: Iterable[LabelRow]) -> list[LabelRow]:
    """Old rows plus new ones, one per trace and judge. A new row wins, so a
    relabel in LangSmith replaces the stored label instead of adding a second."""
    merged = {(row.trace_id, row.judge): row for row in old}
    merged.update({(row.trace_id, row.judge): row for row in new})
    return sorted(
        merged.values(), key=lambda row: (row.judge, row.task_id, row.trial, row.trace_id)
    )


def write_rows(path: Path, rows: Sequence[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Agreement:
    """How often one judge says what the person said. The person is the truth.

    Yes is the positive class. `tpr` is the share of the person's yes labels
    the judge also called yes. `tnr` is the same for no. Two rates, not one
    accuracy, because a judge that always says yes scores high accuracy on a
    set that is mostly yes and still catches no failure.
    """

    judge: str
    labeled: int
    true_pos: int
    true_neg: int
    false_pos: int
    false_neg: int

    @property
    def compared(self) -> int:
        return self.true_pos + self.true_neg + self.false_pos + self.false_neg

    @property
    def tpr(self) -> float | None:
        positives = self.true_pos + self.false_neg
        return self.true_pos / positives if positives else None

    @property
    def tnr(self) -> float | None:
        negatives = self.true_neg + self.false_pos
        return self.true_neg / negatives if negatives else None


def agreement(rows: Iterable[LabelRow]) -> list[Agreement]:
    """One line per judge. A row whose judge call failed is labeled but not
    compared: there is no verdict to agree or disagree with."""
    counts: dict[str, list[int]] = {judge: [0, 0, 0, 0, 0] for judge in JUDGES}
    for row in rows:
        tally = counts.setdefault(row.judge, [0, 0, 0, 0, 0])
        tally[0] += 1
        if row.judge_verdict is None:
            continue
        if row.human and row.judge_verdict:
            tally[1] += 1
        elif not row.human and not row.judge_verdict:
            tally[2] += 1
        elif row.judge_verdict:
            tally[3] += 1
        else:
            tally[4] += 1
    return [Agreement(judge, *tally) for judge, tally in counts.items()]


def render_agreement(results: Sequence[Agreement], rows: Sequence[LabelRow]) -> str:
    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        "| judge | labeled | compared | TPR | TNR | judge yes, person no | judge no, person yes |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {a.judge} | {a.labeled} | {a.compared} | {rate(a.tpr)} | {rate(a.tnr)} "
            f"| {a.false_pos} | {a.false_neg} |"
            for a in results
        ),
    ]
    disagree = [r for r in rows if r.judge_verdict is not None and r.judge_verdict != r.human]
    if disagree:
        lines += ["", "Disagreements (read these to fix a judge prompt):"]
        lines += [
            f"- {r.judge} {r.task_id} trial {r.trial}: judge said "
            f"{'yes' if r.judge_verdict else 'no'}, person said {'yes' if r.human else 'no'}"
            f" (trace {r.trace_id})"
            for r in disagree
        ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.labels", description=(__doc__ or "").split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("queue", "Add a run's traces to the judges' annotation queues."),
        ("pull", "Save the labels to labels.jsonl and print agreement."),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--results",
            type=Path,
            default=Path("evals/results/latest.json"),
            help="The results file of a judged run made with LANGSMITH_API_KEY set.",
        )
        if name == "queue":
            command.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Traces to add.")
        else:
            command.add_argument("--labels", type=Path, default=LABELS_PATH, help="Where rows go.")
    commands.add_parser("agreement", help="Print agreement from labels.jsonl alone.").add_argument(
        "--labels", type=Path, default=LABELS_PATH
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, store: LabelStore | None = None) -> int:
    args = _parse_args(argv)

    if args.command == "agreement":
        rows = read_rows(args.labels)
        print(render_agreement(agreement(rows), rows))  # noqa: T201 - CLI output for a person
        return 0

    results = cast(dict[str, Any], json.loads(args.results.read_text(encoding="utf-8")))
    if not label_tasks(results):
        logger.error(
            f"{args.results} has no traced, judged trials. Run `python -m evals --mode release` "
            "with LANGSMITH_API_KEY set first."
        )
        return 2

    if store is None:
        settings = LangSmithSettings()
        client = settings.client()
        if client is None:
            logger.error("LANGSMITH_API_KEY is not set.")
            return 2
        store = LangSmithLabelStore(client, settings.langsmith_project)

    if args.command == "queue":
        added = queue(results, store, args.limit)
        for judge, count in added.items():
            print(f"{judge}: {count} traces queued")  # noqa: T201 - CLI output for a person
        return 0

    rows = merge_rows(read_rows(args.labels), pull(results, store))
    write_rows(args.labels, rows)
    print(f"{len(rows)} labels in {args.labels}\n")  # noqa: T201 - CLI output for a person
    print(render_agreement(agreement(rows), rows))  # noqa: T201 - CLI output for a person
    return 0


if __name__ == "__main__":
    sys.exit(main())
