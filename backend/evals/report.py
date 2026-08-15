"""Writes the eval run's JSON artifact and renders its markdown table.

The markdown table is the actual deliverable: it is what gets pasted into a
PR description or the README, and a metric with nowhere legible to land is a
metric nobody looks at again.
"""

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from evals.metrics.budgets import BudgetSummary
from evals.metrics.generation import GenerationSuiteSummary
from evals.metrics.retrieval import RetrievalSuiteSummary
from evals.metrics.tools import ToolSuiteSummary


@dataclass(frozen=True, slots=True)
class SuiteReport:
    suite: str
    n: int
    tools: ToolSuiteSummary | None = None
    retrieval: RetrievalSuiteSummary | None = None
    generation: GenerationSuiteSummary | None = None
    budgets: BudgetSummary | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    model: str
    tags: tuple[str, ...]
    suites: tuple[SuiteReport, ...]
    case_ids: tuple[str, ...]
    # "<case id>: <reason>" for every case that errored or came back
    # `ReplyFailed` - the human-readable punch list a markdown table alone
    # cannot give.
    failures: tuple[str, ...]


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
        "",
    ]
    for suite in report.suites:
        lines.append(f"## {suite.suite} (n={suite.n})")
        lines.append("")
        lines.append("| metric | value | n |")
        lines.append("|---|---|---|")
        lines.extend(f"| {name} | {value} | {n} |" for name, value, n in _rows(suite))
        lines.append("")

    if report.failures:
        lines.append("## Failures")
        lines.append("")
        lines.extend(f"- {failure}" for failure in report.failures)
        lines.append("")
    else:
        lines.append("No failures.")
        lines.append("")

    return "\n".join(lines)


def _rows(suite: SuiteReport) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []

    if suite.tools is not None:
        t = suite.tools
        rows += [
            ("exact_match_rate", f"{t.exact_match_rate:.2f}", t.n),
            ("mean_precision", f"{t.mean_precision:.2f}", t.n),
            ("mean_recall", f"{t.mean_recall:.2f}", t.n),
            ("over_calling_rate", f"{t.over_calling_rate:.2f}", t.n),
        ]

    if suite.retrieval is not None:
        r = suite.retrieval
        rows += [
            ("hit_rate@k", f"{r.hit_rate_at_k:.2f}", r.n),
            ("mrr@k", f"{r.mrr_at_k:.2f}", r.n),
        ]

    if suite.generation is not None:
        g = suite.generation
        groundedness = f"{g.mean_groundedness:.2f}" if g.mean_groundedness is not None else "n/a"
        relevance = f"{g.mean_relevance:.2f}" if g.mean_relevance is not None else "n/a"
        rows += [
            ("mean_groundedness", groundedness, g.judged),
            ("mean_relevance", relevance, g.judged),
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
