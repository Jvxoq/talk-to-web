"""The results file and the markdown report of a security run.

The JSON file keeps one record per trial with its whole transcript, so a person
can read what the agent did without paying to run it again. Its shape is the
one `evals.labels` reads: a top-level `tasks` list with each task's `behavior`,
and a `trials` list with each trial's `trace_id`, `transcript.tool_results` and
`judges`. A task judged by `harmful_compliance` has the behavior `s9` and one
judged by `over_refusal` has `s11`, so the label queues pick them up. Every
other task has its line, lowercased.

The markdown goes to stdout. It is what a person pastes into a PR.

Core module: it knows nothing about the agent under test.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.security.gates import JudgeStatus, Verdict
from evals.security.graders import Outcome
from evals.security.metrics import LineMetrics, Rate
from evals.security.tasks import Task, behavior_tag


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    # The full argument JSON text, not a shortened summary.
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolResultRecord:
    tool: str
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    """What the agent did in one trial: the evidence the graders read."""

    answer: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    tool_results: tuple[ToolResultRecord, ...] = ()
    summary_text: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeRecord:
    judge: str
    # `None` when the judge gave no verdict.
    verdict: bool | None
    reason: str


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One trial of one task."""

    task_id: str
    trial: int
    outcome: Outcome
    trace_id: str | None
    question: str
    transcript: TranscriptRecord
    judges: tuple[JudgeRecord, ...] = ()
    attack_succeeded: bool | None = None
    utility_ok: bool | None = None
    failure_reasons: tuple[str, ...] = ()
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Cost:
    """What the run spent, from the tokens the provider reported."""

    tokens_in: int
    tokens_out: int
    judge_tokens_in: int
    judge_tokens_out: int
    usd: float


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    mode: str
    model: str
    judge_model: str
    prompt_version: str
    dataset_version: str
    tasks: tuple[Task, ...]
    trials: tuple[TrialRecord, ...]
    lines: tuple[LineMetrics, ...]
    verdict: Verdict
    cost: Cost | None = None
    # Why the run stopped starting trials, or `None` when it ran them all.
    stopped_early: str | None = None
    not_started: int = 0
    judge_status: Mapping[str, JudgeStatus] = field(default_factory=dict)

    @property
    def errored(self) -> tuple[TrialRecord, ...]:
        """Trials that hit an error, whether they count as failed or are left out."""
        return tuple(t for t in self.trials if t.transcript.error is not None)


def to_json(report: RunReport) -> dict[str, Any]:
    """The results file as plain data."""
    counts: dict[str, Counter[str]] = {}
    for trial in report.trials:
        counts.setdefault(trial.task_id, Counter())[trial.outcome] += 1
    return {
        "run_id": report.run_id,
        "mode": report.mode,
        "model": report.model,
        "judge_model": report.judge_model,
        "prompt_version": report.prompt_version,
        "dataset_version": report.dataset_version,
        "stopped_early": report.stopped_early,
        "not_started": report.not_started,
        "tasks": [_task_json(task, counts.get(task.id, Counter())) for task in report.tasks],
        "trials": [_plain(asdict(trial)) for trial in report.trials],
        "lines": [_line_json(line) for line in report.lines],
        "verdict": _plain(asdict(report.verdict)),
        "judge_status": {name: asdict(status) for name, status in report.judge_status.items()},
        "cost": asdict(report.cost) if report.cost is not None else None,
    }


def _plain(value: Any) -> Any:
    """`asdict` output with every tuple as a list, which is what JSON on disk holds."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _task_json(task: Task, outcomes: Counter[str]) -> dict[str, Any]:
    attack = task.attack
    return {
        "task_id": task.id,
        "behavior": behavior_tag(task),
        "line": task.line,
        "twin": task.twin,
        "shape": attack.shape if attack else None,
        "wrapper": attack.wrapper if attack else None,
        "position": attack.position if attack else None,
        "trials": task.trials,
        "passed": outcomes["passed"],
        "failed": outcomes["failed"],
        "errored": outcomes["errored"],
    }


def _rate_json(rate: Rate) -> dict[str, Any]:
    return {"count": rate.count, "total": rate.total, "value": rate.value}


def _line_json(line: LineMetrics) -> dict[str, Any]:
    return {
        "line": line.line,
        "attack_trials": line.attack_trials,
        "attack_hits": line.attack_hits,
        "attack_failed": line.attack_failed,
        "attack_errored": line.attack_errored,
        "asr": line.asr,
        "asr_upper_bound": line.asr_upper_bound,
        "utility": _rate_json(line.utility),
        "utility_under_attack": _rate_json(line.utility_under_attack),
        "over_refusal": _rate_json(line.over_refusal),
        "twin_errored": line.twin_errored,
        "by_shape": {key: _rate_json(rate) for key, rate in line.by_shape.items()},
        "by_wrapper": {key: _rate_json(rate) for key, rate in line.by_wrapper.items()},
        "by_position": {key: _rate_json(rate) for key, rate in line.by_position.items()},
    }


def write_json(report: RunReport, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json(report), indent=2), encoding="utf-8")


# --- markdown -------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _rate_cell(rate: Rate) -> str:
    return "n/a" if rate.total == 0 else f"{_percent(rate.value)} ({rate.text()})"


def render_markdown(report: RunReport) -> str:
    """The report a person reads: one row per line, the breakdown, why it blocks."""
    lines = [
        f"# Security eval run `{report.run_id}`",
        "",
        f"- **mode**: {report.mode}",
        f"- **model**: `{report.model}`",
        f"- **judge model**: `{report.judge_model or 'n/a'}`",
        f"- **prompt version**: `{report.prompt_version or 'n/a'}`",
        f"- **dataset version**: `{report.dataset_version or 'n/a'}`",
        f"- **trials**: {len(report.trials)}",
    ]
    if report.cost is not None:
        lines.append(f"- **cost**: ${report.cost.usd:.4f}")
    lines.append("")

    verdict = "BLOCKED" if report.verdict.failed else "OK"
    lines += [f"## Verdict: {verdict}", ""]
    lines += [f"- {reason}" for reason in report.verdict.reasons]
    lines += [f"- note: {note}" for note in report.verdict.notes]
    if report.stopped_early:
        lines.append(
            f"- stopped early: {report.stopped_early}. {report.not_started} trials were not "
            "started, so the numbers below do not cover them"
        )
    lines.append("")

    lines += _lines_table(report.lines)
    lines += _breakdown(report.lines)
    lines += _errored(report)
    return "\n".join(lines)


def _lines_table(metrics: Sequence[LineMetrics]) -> list[str]:
    rows = [
        "## Lines",
        "",
        "| line | attack trials | hits | failed | ASR | upper bound (95%) | utility "
        "| utility under attack | over-refusal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.line} | {m.attack_trials} | {m.hits.text()} | {m.attack_failed} "
            f"| {_percent(m.asr)} | {_percent(m.asr_upper_bound)} | {_rate_cell(m.utility)} "
            f"| {_rate_cell(m.utility_under_attack)} | {_rate_cell(m.over_refusal)} |"
        )
    return [*rows, ""]


def _breakdown(metrics: Sequence[LineMetrics]) -> list[str]:
    rows = ["## ASR by attack shape, wrapper and position", ""]
    body: list[str] = []
    for m in metrics:
        for dimension, split in (
            ("shape", m.by_shape),
            ("wrapper", m.by_wrapper),
            ("position", m.by_position),
        ):
            body.extend(
                f"| {m.line} | {dimension} | {value} | {rate.text()} |"
                for value, rate in split.items()
            )
    if not body:
        return [*rows, "No attack trials.", ""]
    return [*rows, "| line | split | value | hits |", "|---|---|---|---|", *body, ""]


def _errored(report: RunReport) -> list[str]:
    errored = report.errored
    if not errored:
        return ["## Errored trials", "", "None.", ""]
    rows = ["## Errored trials", ""]
    for trial in errored:
        note = "counted as failed" if trial.outcome == "failed" else "left out of the rates"
        rows.append(f"- {trial.task_id} trial {trial.trial} ({note}): {trial.transcript.error}")
    return [*rows, ""]
