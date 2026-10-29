"""Tool-selection metrics: the right tools, in the right order.

Reads the same reply events the UI gets. Grades on completed calls
(`ReplyToolFinished` with `ok`), not announced ones: a refused call still
announces itself. Refusals are reported separately.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolFinished, ReplyToolStarted


def tools_requested(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool name the model asked for, in request order - refusals included."""
    return [event.name for event in events if isinstance(event, ReplyToolStarted)]


def tools_completed(events: Sequence[ReplyEvent]) -> list[str]:
    """Tools that ran and answered, in order. An empty retrieval counts. `ok=False` does not."""
    return [event.name for event in events if isinstance(event, ReplyToolFinished) and event.ok]


def tools_refused(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool call that came back without having run."""
    return [event.name for event in events if isinstance(event, ReplyToolFinished) and not event.ok]


def tool_arguments(events: Sequence[ReplyEvent]) -> list[str]:
    """Each tool call's truncated argument summary, enough to catch a leaked canary."""
    return [event.summary for event in events if isinstance(event, ReplyToolStarted)]


def order_satisfied(completed: Sequence[str], required: Sequence[str]) -> bool:
    """Whether `required` is a subsequence of `completed`."""
    remaining = list(required)
    for name in completed:
        if remaining and name == remaining[0]:
            remaining.pop(0)
    return not remaining


@dataclass(frozen=True, slots=True)
class ToolCallScore:
    """One case's tool selection, scored against what it should have called."""

    completed: tuple[str, ...]
    expected: tuple[str, ...]
    refused: tuple[str, ...]
    exact_match: bool
    order_ok: bool
    precision: float
    recall: float
    # Completed but not expected — the over-calling a chatty agent produces.
    over_called: tuple[str, ...]
    # Expected but never completed.
    missing: tuple[str, ...]


def score_tool_call(
    completed: Sequence[str],
    expected: Sequence[str],
    *,
    refused: Sequence[str] = (),
    required_order: Sequence[str] = (),
) -> ToolCallScore:
    """Score one case. Tools compare as sets. Order is checked only when asked."""
    completed_set = set(completed)
    expected_set = set(expected)
    over_called = tuple(sorted(completed_set - expected_set))
    missing = tuple(sorted(expected_set - completed_set))
    intersection = len(completed_set & expected_set)
    return ToolCallScore(
        completed=tuple(completed),
        expected=tuple(expected),
        refused=tuple(refused),
        exact_match=completed_set == expected_set,
        order_ok=order_satisfied(completed, required_order),
        # Nothing called and nothing expected is perfect precision.
        precision=_safe_div(intersection, len(completed_set)),
        recall=_safe_div(intersection, len(expected_set)),
        over_called=over_called,
        missing=missing,
    )


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


@dataclass(frozen=True, slots=True)
class ToolSuiteSummary:
    n: int
    exact_match_rate: float
    mean_precision: float
    mean_recall: float
    # Share of cases with at least one unexpected tool. Counts cases, not calls.
    over_calling_rate: float
    # Share of cases missing a needed tool. Blocked retrievals show up here.
    under_calling_rate: float
    # Share of cases whose order held. 1.0 when no order was declared.
    order_pass_rate: float
    # Share of cases with a routing refusal. Not a failure, but worth watching.
    refusal_rate: float


def summarize_tool_scores(scores: Sequence[ToolCallScore]) -> ToolSuiteSummary:
    n = len(scores)
    if n == 0:
        return ToolSuiteSummary(
            n=0,
            exact_match_rate=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
            over_calling_rate=0.0,
            under_calling_rate=0.0,
            order_pass_rate=0.0,
            refusal_rate=0.0,
        )
    return ToolSuiteSummary(
        n=n,
        exact_match_rate=sum(1 for s in scores if s.exact_match) / n,
        mean_precision=sum(s.precision for s in scores) / n,
        mean_recall=sum(s.recall for s in scores) / n,
        over_calling_rate=sum(1 for s in scores if s.over_called) / n,
        under_calling_rate=sum(1 for s in scores if s.missing) / n,
        order_pass_rate=sum(1 for s in scores if s.order_ok) / n,
        refusal_rate=sum(1 for s in scores if s.refused) / n,
    )
