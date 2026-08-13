"""What a call cost, and the prices that decide it.

Here rather than in the application layer because "what did this reply cost"
is a business question, not a plumbing one. The prices themselves arrive from
configuration - they change when a provider changes them, not when this code
changes - but the arithmetic on them, and the decision about what an unknown
model means, belong to the domain.

Stdlib only, like every other module under `app.domain`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

# Providers publish per-million-token prices, so that is the unit stored, and
# the division happens once, here.
_TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """What one model charges, in USD per million tokens.

    Input and output are separate because every provider charges differently
    for them - usually 3-4x more for output - and a single blended rate would
    quietly misprice exactly the replies worth looking at: the long ones.
    """

    input_usd_per_million: float
    output_usd_per_million: float


@dataclass(frozen=True, slots=True)
class ReplyCost:
    """What some tokens cost, and whether anyone actually knew.

    `priced` is the load-bearing field. Without it a model missing from the
    price list reports `usd=0.0`, which reads as "this was free" - and a cost
    dashboard that silently shows zero for the one model nobody configured is
    worse than no dashboard, because it is confidently wrong.
    """

    prompt_tokens: int
    completion_tokens: int
    usd: float
    priced: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "ReplyCost") -> "ReplyCost":
        """Total two costs - one reply makes many model calls.

        The sum is priced only if both halves were. An unpriced call anywhere in
        a reply means the reply's total is a lower bound, and saying so is the
        whole point of carrying the flag.
        """
        return ReplyCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            usd=self.usd + other.usd,
            priced=self.priced and other.priced,
        )


# The identity for `sum()` and for a reply that made no model call at all.
# Priced, because nothing spent is a known cost of zero, not an unknown one.
NO_COST = ReplyCost(prompt_tokens=0, completion_tokens=0, usd=0.0, priced=True)


class CostBook:
    """Model name to price, and the only thing that turns tokens into money.

    An unknown model is not an error. Models are added by changing one env var,
    and a deployment that names a new one before pricing it should get a working
    reply and an honest `priced=False`, not a 500.
    """

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def price(self, model: str, prompt_tokens: int, completion_tokens: int) -> ReplyCost:
        entry = self._prices.get(model)
        if entry is None:
            return ReplyCost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usd=0.0,
                priced=False,
            )

        usd = (
            prompt_tokens * entry.input_usd_per_million
            + completion_tokens * entry.output_usd_per_million
        ) / _TOKENS_PER_UNIT
        return ReplyCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=usd,
            priced=True,
        )

    def knows(self, model: str) -> bool:
        """Whether this model has a price on file - for a startup warning."""
        return model in self._prices
