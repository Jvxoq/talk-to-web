"""Retrieval-quality metrics: hit-rate and MRR against a known-good source list.

Deterministic and free — no judge call, no model. Sources come straight off
`ReplyToolFinished.sources`, which the tools already attach for the UI to cite,
so this needs nothing beyond the same event stream `metrics.tools` reads.

Two things here used to be wrong in the same direction - both inflating the
score. Sources were pooled across every tool, so five web results finishing
first pushed the document the case was actually about out of the top-k window
and read as a retrieval miss. And a case with nothing to retrieve scored a
perfect 1.0 rather than being left out, which meant every negative case in the
`rag` suite - the ones that exist to prove the agent says "I don't know" -
silently propped the average up. Both are fixed here: sources are filtered to
the tool being graded, and "not applicable" is `None`, not 1.0.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyEvent, ReplyToolFinished


def sources_returned(
    events: Sequence[ReplyEvent], *, tools: Sequence[str] | None = None
) -> list[str]:
    """Every source label the reply's tools returned, in the order the tools
    finished.

    A `Source.url` identifies a web result more precisely than its label; a
    passage from an upload has no url, so its label (the document name) is what
    stands in for it.

    `tools` restricts the pool to the named tools. Retrieval metrics pass
    `["retrieve_documents"]`, because grading a document question on a list
    that a concurrent `search_web` also wrote into measures the ranking of two
    unrelated result sets glued together.
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
    """1.0 if any of the first `k` returned sources is one of the expected
    ones, else 0.0.

    `None` - not 1.0 - when the case expects no sources. Such a case was never
    asking retrieval to prove anything, so it has no hit rate; counting it as
    a hit is how a suite full of "should find nothing" cases reports a
    retrieval score it never earned.
    """
    if not expected:
        return None
    return 1.0 if set(returned[:k]) & set(expected) else 0.0


def mrr_at_k(returned: Sequence[str], expected: Sequence[str], k: int) -> float | None:
    """The reciprocal rank of the first expected source in the top `k`, 0.0 if
    none appears there, and `None` when the case expects no sources at all."""
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
    # Cases that actually had an expected source, and so contributed a number.
    # Reported next to `n` rather than in place of it, so a suite that is
    # mostly negative cases cannot look like a suite that mostly retrieves.
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
