"""Writes the eval run's JSON artifact and renders its markdown table.

The markdown table is the actual deliverable: it is what gets pasted into a
PR description or the README, and a metric with nowhere legible to land is a
metric nobody looks at again.

Two lists come out of a run, not one, and keeping them apart is the point of
this module's shape. A `regression` is the model doing the wrong thing - a
missed tool, a forbidden string, a document it failed to find. An `error` is the
run itself falling over - a rate limit, a dropped connection, a provider
outage. The old report concatenated both into `failures`, so a red run said
nothing about whether to fix a prompt or just run it again, and a flaky
afternoon looked exactly like a quality regression.

The JSON file also keeps what a person needs to look into one result later,
without paying to run it again. Every trial is one record with its transcript:
the final answer, each tool call with its arguments, and the error text if the
trial errored. Every task is one record with its passed, failed and errored
trial counts, and both pass bars in `evals.metrics.gates` read those counts.
The run itself carries the model name, a prompt version, a dataset version and
an estimated cost, so two files can be compared by what they were given.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from evals.cases import BEHAVIORS
from evals.metrics.budgets import BudgetSummary
from evals.metrics.gates import Outcome
from evals.metrics.generation import GenerationSuiteSummary
from evals.metrics.retrieval import RetrievalSuiteSummary
from evals.metrics.tools import ToolSuiteSummary

UNTAGGED = "untagged"


@dataclass(frozen=True, slots=True)
class SuiteReport:
    suite: str
    n: int
    tools: ToolSuiteSummary | None = None
    retrieval: RetrievalSuiteSummary | None = None
    generation: GenerationSuiteSummary | None = None
    budgets: BudgetSummary | None = None


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One tool call the agent made, as the user would have seen it announced."""

    name: str
    # The `ReplyToolStarted.summary` text: the tool name and its arguments,
    # already shortened by the agent. Not the raw argument object.
    arguments: str


@dataclass(frozen=True, slots=True)
class Transcript:
    """What the agent did in one trial: the evidence a grader read."""

    answer: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    # Why the trial errored. `None` for a trial that passed or failed.
    error: str | None = None
    # How many history turns were seeded before the final input. 0 for a
    # one-turn task. A seeded trial is not comparable to a one-turn trial.
    seeded_turns: int = 0
    # Whether a summarize event fired during the trial.
    summarized: bool = False


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One trial of one task."""

    task_id: str
    trial: int
    outcome: Outcome
    # What the graders found wrong. Empty unless the outcome is "failed".
    failure_reasons: tuple[str, ...]
    transcript: Transcript
    tokens_in: int
    tokens_out: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """How many trials of one task passed, failed and errored."""

    task_id: str
    suite: str
    passed: int
    failed: int
    errored: int
    # The task's behavior tag (b1 to b6), or "untagged" for a task with none.
    behavior: str = UNTAGGED


@dataclass(frozen=True, slots=True)
class BehaviorRecord:
    """Task and trial counts for every task that carries one behavior tag."""

    behavior: str
    tasks: int
    passed: int
    failed: int
    errored: int


def behavior_records(tasks: Sequence[TaskRecord]) -> tuple[BehaviorRecord, ...]:
    """One record per behavior that has tasks, in b1 to b6 order.

    A behavior with no task in the run is left out, so a `--tag b1` run does not
    print five empty rows. "untagged" comes last.
    """
    order = [*BEHAVIORS, UNTAGGED]
    return tuple(
        BehaviorRecord(
            behavior=name,
            tasks=len(group),
            passed=sum(t.passed for t in group),
            failed=sum(t.failed for t in group),
            errored=sum(t.errored for t in group),
        )
        for name in order
        if (group := [t for t in tasks if t.behavior == name])
    )


@dataclass(frozen=True, slots=True)
class Totals:
    """Trial counts for the whole suite. Errored trials are not graded."""

    passed: int
    failed: int
    errored: int

    @property
    def graded(self) -> int:
        return self.passed + self.failed


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What the run cost, worked out from tokens and a price per million.

    An estimate, not a bill: it prices every token at the two rates given, and
    it counts only tokens the provider reported. Judge tokens are priced at the
    same rates, though the judge answers through the cheaper condenser model.
    """

    tokens_in: int
    tokens_out: int
    judge_tokens_in: int
    judge_tokens_out: int
    price_in_per_million: float
    price_out_per_million: float
    usd: float


def estimate_cost(
    *,
    tokens_in: int,
    tokens_out: int,
    judge_tokens_in: int,
    judge_tokens_out: int,
    price_in_per_million: float,
    price_out_per_million: float,
) -> CostEstimate:
    """USD cost of the trial tokens plus the judge tokens."""
    usd = (
        (tokens_in + judge_tokens_in) * price_in_per_million
        + (tokens_out + judge_tokens_out) * price_out_per_million
    ) / 1_000_000
    return CostEstimate(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        judge_tokens_in=judge_tokens_in,
        judge_tokens_out=judge_tokens_out,
        price_in_per_million=price_in_per_million,
        price_out_per_million=price_out_per_million,
        usd=usd,
    )


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    model: str
    tags: tuple[str, ...]
    suites: tuple[SuiteReport, ...]
    case_ids: tuple[str, ...]
    # How many replies the run actually generated: one per case per `--repeat`.
    # Reported next to the case count because every rate above is a mean over
    # samples, and a rate over 60 samples of 20 cases says something a rate
    # over 20 says less well.
    samples: int
    # "<case id>: <what the model got wrong>" - the human-readable punch list a
    # markdown table alone cannot give. A case is here only when it got it
    # wrong in *every* sample.
    regressions: tuple[str, ...]
    # "<case id>: failed n/m samples" - the case is not stable, so no verdict
    # drawn from one sample of it is either. Kept out of `regressions` on
    # purpose: a red build people re-run until it passes is how a real
    # regression gets waved through.
    flaky: tuple[str, ...] = ()
    # "<case id>: <why the run broke>" - infrastructure, not quality. A sample
    # here was never graded, so it appears in none of the rates above.
    errors: tuple[str, ...] = ()
    # "gate", "release", or `None` for a plain run with `--repeat`.
    mode: str | None = None
    prompt_version: str = ""
    dataset_version: str = ""
    # "recorded" or "live": which web tools the run used.
    web_mode: str = ""
    trials: tuple[TrialRecord, ...] = ()
    tasks: tuple[TaskRecord, ...] = ()
    # The same trial counts, added up per behavior (b1 to b6).
    behaviors: tuple[BehaviorRecord, ...] = ()
    totals: Totals = Totals(passed=0, failed=0, errored=0)
    cost: CostEstimate | None = None
    # Did the run miss its quality bar? Decides exit code 1. Kept in the report
    # so the file says why the run was red without re-deriving it.
    quality_failed: bool = False
    # Why the run passed or failed, in plain words. Printed and stored.
    verdict: tuple[str, ...] = ()

    @property
    def graded(self) -> int:
        return self.samples - len(self.errors)


def write_json(report: RunReport, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_jsonable(report), indent=2))


def _to_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


def render_markdown(report: RunReport) -> str:
    lines = [
        f"# Eval run `{report.run_id}`",
        "",
        f"- **model**: `{report.model}`",
        f"- **mode**: {report.mode or '(plain run)'}",
        f"- **prompt version**: `{report.prompt_version or 'n/a'}`",
        f"- **dataset version**: `{report.dataset_version or 'n/a'}`",
        f"- **web tools**: {report.web_mode or 'n/a'}",
        f"- **tags**: {', '.join(report.tags) if report.tags else '(none)'}",
        f"- **cases**: {len(report.case_ids)}",
        f"- **trials**: {report.samples} ({report.graded} graded)",
        f"- **passed / failed / errored**: {report.totals.passed} / "
        f"{report.totals.failed} / {report.totals.errored}",
    ]
    if report.cost is not None:
        c = report.cost
        lines.append(
            f"- **estimated cost**: ${c.usd:.4f} "
            f"({c.tokens_in + c.judge_tokens_in} tokens in, "
            f"{c.tokens_out + c.judge_tokens_out} out, "
            f"at ${c.price_in_per_million}/${c.price_out_per_million} per million)"
        )
    lines.append("")
    lines.extend(_punch_list("Verdict", report.verdict, ""))
    if report.behaviors:
        lines.extend(
            [
                "## Behaviors",
                "",
                "| behavior | tasks | passed | failed | errored |",
                "|---|---|---|---|---|",
                *(
                    f"| {b.behavior} | {b.tasks} | {b.passed} | {b.failed} | {b.errored} |"
                    for b in report.behaviors
                ),
                "",
            ]
        )
    if report.tasks:
        lines.extend(
            [
                "## Tasks",
                "",
                *(
                    f"- {task.task_id}: {task.passed} passed, {task.failed} failed, "
                    f"{task.errored} errored"
                    for task in report.tasks
                ),
                "",
            ]
        )
    for suite in report.suites:
        lines.append(f"## {suite.suite} (n={suite.n})")
        lines.append("")
        lines.append("| metric | value | n |")
        lines.append("|---|---|---|")
        lines.extend(f"| {name} | {value} | {n} |" for name, value, n in _rows(suite))
        lines.append("")

    lines.extend(_punch_list("Regressions", report.regressions, "No regressions."))
    # Only when there are any. A single-sample run can never produce one, and a
    # standing "No flaky cases." line there would read as proof of stability
    # the run never looked for.
    if report.flaky:
        lines.extend(_punch_list("Flaky (unstable across samples)", report.flaky, ""))
    lines.extend(_punch_list("Errors (not graded)", report.errors, "No errors."))
    return "\n".join(lines)


def _punch_list(heading: str, entries: tuple[str, ...], empty: str) -> list[str]:
    if not entries:
        return [empty, ""]
    return [f"## {heading}", "", *(f"- {entry}" for entry in entries), ""]


def _rows(suite: SuiteReport) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []

    if suite.tools is not None:
        t = suite.tools
        rows += [
            ("exact_match_rate", f"{t.exact_match_rate:.2f}", t.n),
            ("mean_precision", f"{t.mean_precision:.2f}", t.n),
            ("mean_recall", f"{t.mean_recall:.2f}", t.n),
            ("over_calling_rate", f"{t.over_calling_rate:.2f}", t.n),
            ("under_calling_rate", f"{t.under_calling_rate:.2f}", t.n),
            ("order_pass_rate", f"{t.order_pass_rate:.2f}", t.n),
            ("refusal_rate", f"{t.refusal_rate:.2f}", t.n),
        ]

    if suite.retrieval is not None:
        r = suite.retrieval
        rows += [
            ("hit_rate@k", _optional(r.hit_rate_at_k), r.scored),
            ("mrr@k", _optional(r.mrr_at_k), r.scored),
        ]

    if suite.generation is not None:
        g = suite.generation
        rows += [
            ("mean_groundedness", _optional(g.mean_groundedness), g.judged),
            ("mean_relevance", _optional(g.mean_relevance), g.judged),
            ("mean_correctness", _optional(g.mean_correctness), g.judged),
            ("substring_pass_rate", f"{g.substring_pass_rate:.2f}", g.n),
        ]

    if suite.budgets is not None:
        b = suite.budgets
        rows += [
            ("p50_latency_ms", f"{b.p50_latency_ms:.0f}", b.n),
            ("p95_latency_ms", f"{b.p95_latency_ms:.0f}", b.n),
            ("mean_prompt_tokens", f"{b.mean_prompt_tokens:.0f}", b.n),
            ("mean_completion_tokens", f"{b.mean_completion_tokens:.0f}", b.n),
        ]

    return rows


def _optional(value: float | None) -> str:
    """ "n/a", not 0.00, when nothing contributed to a mean.

    The two look alike in a table and mean opposite things: 0.00 is a metric
    the run measured and failed, "n/a" is a metric nothing in the run was
    asking for.
    """
    return f"{value:.2f}" if value is not None else "n/a"
