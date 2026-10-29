"""Retrieval metrics: hit rate and MRR. Free, no model call.

Sources come from `ReplyToolFinished.sources`, filtered to the graded tool.
A case with nothing to retrieve scores `None`, not 1.0.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolFinished


def sources_returned(
    events: Sequence[ReplyEvent], *, tools: Sequence[str] | None = None
) -> list[str]:
    """Source labels (or URLs) the tools returned, in finish order.

    `tools` limits the pool, so web results do not mix into retrieval scores.
    """
    allowed = set(tools) if tools is not None else None
    labels: list[str] = []
    for event in events:
        if not isinstance(event, ReplyToolFinished):
            continue
        if allowed is not None and event.name not in allowed:
            continue
        labels.extend(source.url or source.label for source in event.sources)
    return labels


def hit_rate_at_k(returned: Sequence[str], expected: Sequence[str], k: int) -> float | None:
    """1.0 if an expected source is in the top `k`, else 0.0. `None` if none expected."""
    if not expected:
        return None
    return 1.0 if set(returned[:k]) & set(expected) else 0.0


def mrr_at_k(returned: Sequence[str], expected: Sequence[str], k: int) -> float | None:
    """Reciprocal rank of the first expected source in the top `k`. `None` if none expected."""
    if not expected:
        return None
    expected_set = set(expected)
    for rank, label in enumerate(returned[:k], start=1):
        if label in expected_set:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class RetrievalSuiteSummary:
    # Cases in the suite.
    n: int
    # Cases that had an expected source. Shown beside `n`.
    scored: int
    hit_rate_at_k: float | None
    mrr_at_k: float | None


def summarize_retrieval(
    hit_rates: Sequence[float | None], reciprocal_ranks: Sequence[float | None]
) -> RetrievalSuiteSummary:
    hits = [value for value in hit_rates if value is not None]
    ranks = [value for value in reciprocal_ranks if value is not None]
    return RetrievalSuiteSummary(
        n=len(hit_rates),
        scored=len(hits),
        hit_rate_at_k=(sum(hits) / len(hits)) if hits else None,
        mrr_at_k=(sum(ranks) / len(ranks)) if ranks else None,
    )
