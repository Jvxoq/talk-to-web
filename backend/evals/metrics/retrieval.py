"""Retrieval-quality metrics: hit-rate and MRR against a known-good source list.

Deterministic and free — no judge call, no model. Sources come straight off
`ReplyToolFinished.sources`, which the tools already attach for the UI to cite,
so this needs nothing beyond the same event stream `metrics.tools` reads.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolFinished


def sources_returned(events: Sequence[ReplyEvent]) -> list[str]:
    """Every source label the reply's tools returned, in the order the tools
    finished. A `Source.url` identifies a web result more precisely than its
    label; a passage from an upload has no url, so its label (the document
    name) is what stands in for it."""
    labels: list[str] = []
    for event in events:
        if isinstance(event, ReplyToolFinished):
            labels.extend(source.url or source.label for source in event.sources)
    return labels


def hit_rate_at_k(returned: Sequence[str], expected: Sequence[str], k: int) -> float:
    """1.0 if any of the first `k` returned sources is one of the expected
    ones, else 0.0. A case with no expected sources is vacuously satisfied -
    it was not asking retrieval to prove anything."""
    if not expected:
        return 1.0
    return 1.0 if set(returned[:k]) & set(expected) else 0.0


def mrr_at_k(returned: Sequence[str], expected: Sequence[str], k: int) -> float:
    """The reciprocal rank of the first expected source in the top `k`, or 0.0
    if none appears there."""
    if not expected:
        return 1.0
    expected_set = set(expected)
    for rank, label in enumerate(returned[:k], start=1):
        if label in expected_set:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class RetrievalSuiteSummary:
    n: int
    hit_rate_at_k: float
    mrr_at_k: float


def summarize_retrieval(
    hit_rates: Sequence[float], reciprocal_ranks: Sequence[float]
) -> RetrievalSuiteSummary:
    n = len(hit_rates)
    if n == 0:
        return RetrievalSuiteSummary(n=0, hit_rate_at_k=0.0, mrr_at_k=0.0)
    return RetrievalSuiteSummary(
        n=n,
        hit_rate_at_k=sum(hit_rates) / n,
        mrr_at_k=sum(reciprocal_ranks) / n,
    )
