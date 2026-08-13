"""Cost and latency: p50/p95 latency and mean cost/tokens off `ReplyUsage`.

Latency is measured by the harness's own wall clock around each case; cost and
tokens come from the single `ReplyUsage` event `GenerateReply` emits at the end
of a successful reply - already totalled across every model call the reply
made, condenser included (see `GenerateReply._total_cost`).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.chat.dto import ReplyUsage


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, the same method `numpy.percentile`
    defaults to. Good enough for a report; not worth a dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class BudgetSummary:
    n: int
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    # How many priced usage records were unpriced - see
    # `app.domain.usage.value_objects.ReplyCost.priced`. A nonzero count means
    # `mean_cost_usd` is a lower bound, and the report should say so.
    unpriced_count: int


def summarize_budgets(
    latencies_ms: Sequence[float], usages: Sequence[ReplyUsage | None]
) -> BudgetSummary:
    priced = [usage.cost_usd for usage in usages if usage is not None]
    prompt_tokens = [usage.prompt_tokens for usage in usages if usage is not None]
    completion_tokens = [usage.completion_tokens for usage in usages if usage is not None]
    unpriced = sum(1 for usage in usages if usage is not None and not usage.priced)
    return BudgetSummary(
        n=len(latencies_ms),
        p50_latency_ms=percentile(latencies_ms, 50),
        p95_latency_ms=percentile(latencies_ms, 95),
        mean_cost_usd=(sum(priced) / len(priced)) if priced else 0.0,
        mean_prompt_tokens=(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0,
        mean_completion_tokens=(
            sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0
        ),
        unpriced_count=unpriced,
    )
