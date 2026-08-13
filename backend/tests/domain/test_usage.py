"""Usage/cost domain rules. No fixtures, no async, no I/O."""

from app.domain.usage.value_objects import NO_COST, CostBook, ModelPrice, ReplyCost

PRICES = {
    "gpt-priced": ModelPrice(input_usd_per_million=1.0, output_usd_per_million=4.0),
}


class TestCostBookPricing:
    def test_a_known_model_prices_input_and_output_at_different_rates(self) -> None:
        book = CostBook(PRICES)

        cost = book.price("gpt-priced", prompt_tokens=1_000_000, completion_tokens=1_000_000)

        # Different rates, so the total is not just "tokens times one price" -
        # it is the sum of two independent, per-million charges.
        assert cost.usd == 1.0 + 4.0
        assert cost.priced is True

    def test_the_division_is_per_million_not_per_token(self) -> None:
        book = CostBook(PRICES)

        cost = book.price("gpt-priced", prompt_tokens=500_000, completion_tokens=250_000)

        assert cost.usd == (500_000 * 1.0 + 250_000 * 4.0) / 1_000_000

    def test_an_unknown_model_is_reported_as_unpriced_not_free(self) -> None:
        # The whole point of `priced`: a missing price list entry must not
        # silently read as "this reply cost nothing."
        book = CostBook(PRICES)

        cost = book.price("mystery-model", prompt_tokens=100, completion_tokens=50)

        assert cost.usd == 0.0
        assert cost.priced is False
        # The tokens themselves are still real and worth keeping, even though
        # nobody has told this book what they cost.
        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50

    def test_zero_tokens_costs_zero_even_for_a_priced_model(self) -> None:
        book = CostBook(PRICES)

        cost = book.price("gpt-priced", prompt_tokens=0, completion_tokens=0)

        assert cost.usd == 0.0
        assert cost.priced is True

    def test_knows_reflects_whether_a_model_has_a_price_on_file(self) -> None:
        book = CostBook(PRICES)

        assert book.knows("gpt-priced")
        assert not book.knows("mystery-model")


class TestReplyCostAddition:
    def test_adding_two_priced_costs_sums_tokens_and_usd(self) -> None:
        first = ReplyCost(prompt_tokens=10, completion_tokens=20, usd=0.5, priced=True)
        second = ReplyCost(prompt_tokens=5, completion_tokens=15, usd=0.25, priced=True)

        total = first + second

        assert total.prompt_tokens == 15
        assert total.completion_tokens == 35
        assert total.usd == 0.75
        assert total.priced is True

    def test_priced_is_false_if_either_side_was_unpriced(self) -> None:
        priced = ReplyCost(prompt_tokens=10, completion_tokens=10, usd=1.0, priced=True)
        unpriced = ReplyCost(prompt_tokens=10, completion_tokens=10, usd=0.0, priced=False)

        # Order must not matter: one unknown model anywhere in a reply's calls
        # is enough to make the whole total a lower bound.
        assert (priced + unpriced).priced is False
        assert (unpriced + priced).priced is False

    def test_total_tokens_is_prompt_plus_completion(self) -> None:
        cost = ReplyCost(prompt_tokens=7, completion_tokens=3, usd=0.0, priced=True)

        assert cost.total_tokens == 10


class TestNoCost:
    def test_no_cost_is_priced_and_zero(self) -> None:
        assert NO_COST.usd == 0.0
        assert NO_COST.total_tokens == 0
        assert NO_COST.priced is True

    def test_no_cost_is_the_identity_for_addition(self) -> None:
        cost = ReplyCost(prompt_tokens=3, completion_tokens=4, usd=0.12, priced=True)

        assert cost + NO_COST == cost
        assert NO_COST + cost == cost

    def test_no_cost_does_not_poison_a_priced_total(self) -> None:
        # An unpriced call must drag a sum down to unpriced, but "nothing was
        # spent" must not - the two are different claims about the same field.
        cost = ReplyCost(prompt_tokens=3, completion_tokens=4, usd=0.12, priced=True)

        assert (cost + NO_COST).priced is True
