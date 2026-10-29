"""Writes the eval JSON artifact and renders the markdown table.

Regressions (the model was wrong) and errors (the run broke) are kept apart,
so a red run says whether to fix a prompt or re-run.
"""

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from evals.metrics.budgets import BudgetSummary
from evals.metrics.generation import GenerationSuiteSummary
from evals.metrics.retrieval import RetrievalSuiteSummary
from evals.metrics.security import RateSummary, SecuritySummary
from evals.metrics.tools import ToolSuiteSummary


@dataclass(frozen=True, slots=True)
class SuiteReport:
    suite: str
    n: int
    tools: ToolSuiteSummary | None = None
    retrieval: RetrievalSuiteSummary | None = None
    generation: GenerationSuiteSummary | None = None
    budgets: BudgetSummary | None = None
    # The security suites only: one entry per rate the suite states.
    security: tuple[SecuritySummary, ...] = ()


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    model: str
    tags: tuple[str, ...]
    suites: tuple[SuiteReport, ...]
    case_ids: tuple[str, ...]
    # Replies generated: cases times `--repeat`. Rates are means over these.
    samples: int
    # "<case id>: <what went wrong>", only for cases wrong in every sample.
    regressions: tuple[str, ...]
    # "<case id>: failed n/m samples". Unstable, so not a regression.
    flaky: tuple[str, ...] = ()
    # "<case id>: <why it broke>". Never graded.
    errors: tuple[str, ...] = ()

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
        f"- **tags**: {', '.join(report.tags) if report.tags else '(none)'}",
        f"- **cases**: {len(report.case_ids)}",
        f"- **samples**: {report.samples} ({report.graded} graded)",
        "",
    ]
    for suite in report.suites:
        lines.append(f"## {suite.suite} (n={suite.n})")
        lines.append("")
        lines.append("| metric | value | n |")
        lines.append("|---|---|---|")
        lines.extend(f"| {name} | {value} | {n} |" for name, value, n in _rows(suite))
        lines.append("")
        for summary in suite.security:
            lines.extend(_security_section(summary))

    lines.extend(_punch_list("Regressions", report.regressions, "No regressions."))
    # Only when present, since one sample cannot prove stability.
    if report.flaky:
        lines.extend(_punch_list("Flaky (unstable across samples)", report.flaky, ""))
    lines.extend(_punch_list("Errors (not graded)", report.errors, "No errors."))
    return "\n".join(lines)


def _punch_list(heading: str, entries: tuple[str, ...], empty: str) -> list[str]:
    if not entries:
        return [empty, ""]
    return [f"## {heading}", "", *(f"- {entry}" for entry in entries), ""]


def _security_section(summary: SecuritySummary) -> list[str]:
    """One table per rate: overall first, then each split. The SLO uses the bound column."""
    lines = [
        f"### {summary.metric}",
        "",
        "| split | successes / attempts | rate | 95% upper |",
        "|---|---|---|---|",
        _rate_row("overall", summary.overall),
        *(_rate_row(f"{row.dimension}={row.value}", row.summary) for row in summary.breakdown),
        "",
    ]
    if summary.unjudged:
        lines += [f"{summary.unjudged} sample(s) had no judge verdict and are not counted.", ""]
    return lines


def _rate_row(label: str, rate: RateSummary) -> str:
    return (
        f"| {label} | {rate.successes} / {rate.attempts} | "
        f"{_percent(rate.rate)} | {_percent(rate.upper_95)} |"
    )


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


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
    """ "n/a" when nothing was measured, so it is not mistaken for 0.00."""
    return f"{value:.2f}" if value is not None else "n/a"
