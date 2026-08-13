"""Tool-selection metrics: did the agent reach for the right tools.

No new instrumentation needed. `ReplyToolStarted` already names every tool the
agent picked, on every reply — a nice consequence of `GenerateReply` already
streaming that event for the UI's own use (the "thinking" chip). This module
just consumes the same event stream a client would.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolStarted


def tools_called(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool name the agent invoked during one reply, in call order."""
    return [event.name for event in events if isinstance(event, ReplyToolStarted)]


@dataclass(frozen=True, slots=True)
class ToolCallScore:
    """One case's tool selection, scored against what it should have called."""

    called: tuple[str, ...]
    expected: tuple[str, ...]
    exact_match: bool
    precision: float
    recall: float
    # Called but not expected — the over-calling a chatty agent produces.
    over_called: tuple[str, ...]
    # Expected but never called.
    missing: tuple[str, ...]


def score_tool_call(called: Sequence[str], expected: Sequence[str]) -> ToolCallScore:
    """Score one case. Compares sets, not sequences or counts: the question
    is "which tools", not "how many times" or "in what order"."""
    called_set = set(called)
    expected_set = set(expected)
    over_called = tuple(sorted(called_set - expected_set))
    missing = tuple(sorted(expected_set - called_set))
    intersection = len(called_set & expected_set)
    return ToolCallScore(
        called=tuple(called),
        expected=tuple(expected),
        exact_match=called_set == expected_set,
        # An empty "called" set with nothing expected is perfect precision,
        # not undefined - the agent correctly called nothing.
        precision=_safe_div(intersection, len(called_set)),
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
    # Fraction of cases where the agent called at least one tool it should
    # not have. Distinct from `mean_precision`: this counts cases, not calls,
    # which is the number worth an alert - "how often does this happen at
    # all", not "how bad is it when it does".
    over_calling_rate: float


def summarize_tool_scores(scores: Sequence[ToolCallScore]) -> ToolSuiteSummary:
    n = len(scores)
    if n == 0:
        return ToolSuiteSummary(
            n=0, exact_match_rate=0.0, mean_precision=0.0, mean_recall=0.0, over_calling_rate=0.0
        )
    return ToolSuiteSummary(
        n=n,
        exact_match_rate=sum(1 for s in scores if s.exact_match) / n,
        mean_precision=sum(s.precision for s in scores) / n,
        mean_recall=sum(s.recall for s in scores) / n,
        over_calling_rate=sum(1 for s in scores if s.over_called) / n,
    )
