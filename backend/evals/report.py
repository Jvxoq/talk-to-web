"""Writes the eval run's JSON artifact and renders its markdown table.

The markdown table is the actual deliverable: it is what gets pasted into a
PR description or the README, and a metric with nowhere legible to land is a
metric nobody looks at again.

Two lists come out of a run, not one, and keeping them apart is the point of
this module's shape. A `regression` is the model doing the wrong thing - a
missed tool, a leaked canary, a document it failed to find. An `error` is the
run itself falling over - a rate limit, a dropped connection, a provider
outage. The old report concatenated both into `failures`, so a red run said
nothing about whether to fix a prompt or just run it again, and a flaky
afternoon looked exactly like a quality regression.
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
