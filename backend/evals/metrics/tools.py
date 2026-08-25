"""Tool-selection metrics: did the agent reach for the right tools, in the right order.

No new instrumentation needed. The reply stream already says which tools were
asked for (`ReplyToolStarted`) and which came back with an answer
(`ReplyToolFinished`, `ok`) — a nice consequence of `GenerateReply` already
streaming both for the UI's own use. This module just consumes the same event
stream a client would.

The split between "requested" and "completed" is the whole point of this
module and is easy to get wrong. `make_tool_node` writes its start event
*before* calling `ToolRegistry.invoke`, so a call that `ToolRoutingPolicy`
refuses still announces itself and then comes back `ok=False` having never
run. Scoring on start events counts a blocked call as a successful one, which
inverts the result of exactly the two cases the gate exists for: a
document-scoped question that tried the web first, and an empty account that
tried a retrieval. Everything here grades on completions and reports refusals
separately.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolFinished, ReplyToolStarted


def tools_requested(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool name the model asked for, in request order - refusals included."""
    return [event.name for event in events if isinstance(event, ReplyToolStarted)]


def tools_completed(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool that actually ran and answered, in completion order.

    A retrieval that matched nothing counts: it ran, and its empty answer is
    what licenses a web search. Only `ok=False` - a refusal, an unknown tool,
    a dead upstream - is excluded.
    """
    return [event.name for event in events if isinstance(event, ReplyToolFinished) and event.ok]


def tools_refused(events: Sequence[ReplyEvent]) -> list[str]:
    """Every tool call that came back without having run."""
    return [event.name for event in events if isinstance(event, ReplyToolFinished) and not event.ok]


def tool_arguments(events: Sequence[ReplyEvent]) -> list[str]:
    """The rendered arguments of every tool call, for content checks.

    `ReplyToolStarted.summary` is `name(arg='value', ...)`, truncated by the
    tool node. It is not a complete record of what was sent, and this is the
    one place that matters: it is enough to catch a canary being smuggled into
    a `search_web` query, which an answer-only check never sees.
    """
    return [event.summary for event in events if isinstance(event, ReplyToolStarted)]


def order_satisfied(completed: Sequence[str], required: Sequence[str]) -> bool:
    """Is `required` a subsequence of `completed`?

    A subsequence and not a prefix or an equality: the rule being checked is
    "retrieval answered before the search was paid for", which says nothing
    about what else ran in between or afterwards.
    """
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
    """Score one case.

    Membership is compared as sets - the question is "which tools", not "how
    many times" - while ordering is checked separately and only when a case
    asked for it, because most cases genuinely do not care.
    """
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
        # An empty "completed" set with nothing expected is perfect precision,
        # not undefined - the agent correctly called nothing.
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
    # Fraction of cases where the agent completed at least one tool it should
    # not have. Distinct from `mean_precision`: this counts cases, not calls,
    # which is the number worth an alert - "how often does this happen at
    # all", not "how bad is it when it does".
    over_calling_rate: float
    # Fraction of cases that missed at least one tool they needed. The other
    # half of the story, and the one the routing gate regresses first: a
    # blocked retrieval shows up here, never in `over_calling_rate`.
    under_calling_rate: float
    # Fraction of cases whose required ordering held. Vacuously 1.0 for cases
    # that declared no order, which is most of them.
    order_pass_rate: float
    # Fraction of cases where the routing policy turned at least one call
    # away. Not a failure on its own - refusing a search on a document
    # question is the gate working - but a number that moves when a prompt
    # changes, and worth seeing next to the others.
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
