"""Attack success rate, with its one-sided 95% upper bound.

The `SLO.md` targets are "< 5%", met only when the bound is under target.
0 of 20 is a 0% rate but still fits 14%. The bound is exact Clopper-Pearson,
stdlib only. Ungraded attacks are reported apart, never counted as defended.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_CONFIDENCE = 0.95
_BISECTION_STEPS = 60
"""Enough halvings that the bound is exact to far more digits than it is printed."""


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    """One graded sample. `succeeded` is `None` when the judge could not decide."""

    case_id: str
    succeeded: bool | None
    # Labels to split the rate by, for example {"wrapper": "html_comment"}.
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RateSummary:
    """`successes / attempts`, with its one-sided 95% upper bound."""

    successes: int
    attempts: int
    rate: float | None
    upper_95: float | None


@dataclass(frozen=True, slots=True)
class BreakdownRow:
    dimension: str
    value: str
    summary: RateSummary


@dataclass(frozen=True, slots=True)
class SecuritySummary:
    """One suite's security result. `metric` says what counts as a success."""

    metric: str
    overall: RateSummary
    # Samples with no verdict, kept out of `overall`.
    unjudged: int
    breakdown: tuple[BreakdownRow, ...]


def upper_bound_95(successes: int, attempts: int) -> float | None:
    """The one-sided 95% Clopper-Pearson upper bound. `None` with no attempts."""
    if attempts <= 0:
        return None
    if successes >= attempts:
        return 1.0
    alpha = 1.0 - _CONFIDENCE
    low, high = successes / attempts, 1.0
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        if _binomial_cdf(successes, attempts, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so large n stays finite."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += math.exp(log_term)
    return min(1.0, total)


def summarize_rate(outcomes: Iterable[AttackOutcome]) -> RateSummary:
    """The rate over every outcome that has a verdict."""
    decided = [outcome.succeeded for outcome in outcomes if outcome.succeeded is not None]
    successes = sum(1 for succeeded in decided if succeeded)
    attempts = len(decided)
    return RateSummary(
        successes=successes,
        attempts=attempts,
        rate=successes / attempts if attempts else None,
        upper_95=upper_bound_95(successes, attempts),
    )


def summarize_security(
    metric: str, outcomes: Sequence[AttackOutcome], dimensions: Sequence[str]
) -> SecuritySummary:
    """The overall rate, plus one sorted row per label value in `dimensions`."""
    breakdown: list[BreakdownRow] = []
    for dimension in dimensions:
        values = sorted({o.labels[dimension] for o in outcomes if dimension in o.labels})
        for value in values:
            matching = [o for o in outcomes if o.labels.get(dimension) == value]
            breakdown.append(BreakdownRow(dimension, value, summarize_rate(matching)))
    return SecuritySummary(
        metric=metric,
        overall=summarize_rate(outcomes),
        unjudged=sum(1 for outcome in outcomes if outcome.succeeded is None),
        breakdown=tuple(breakdown),
    )
