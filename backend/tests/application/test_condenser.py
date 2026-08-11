"""The condenser: one cheap model, two compression jobs, one failure policy.

The point of the shared `Condenser` is that a failure to condense costs the
reply some grounding, never the reply itself - so every test here is about
either what it returns on success or that it returns `None` (and logs) rather
than raising on failure.
"""

from app.application.chat.agent.condenser import Condenser
from app.application.chat.models import ChatMessage, ModelChunk
from tests.fakes import FakeChatModel


def make_condenser(model: FakeChatModel, max_chars: int = 100) -> Condenser:
    return Condenser(
        model=model,
        model_name="condenser",
        max_chars=max_chars,
        tool_condense_prompt="condense",
        summary_prompt="summarize",
    )


class TestCondenser:
    async def test_condense_returns_none_and_logs_on_a_failing_model(self) -> None:
        condenser = make_condenser(FakeChatModel(fail_with=RuntimeError("boom")))

        result = await condenser.condense("some text", focus="q")

        assert result is None

    async def test_condense_returns_the_joined_text(self) -> None:
        condenser = make_condenser(
            FakeChatModel(turns=[[ModelChunk(text="he"), ModelChunk(text="llo")]])
        )

        result = await condenser.condense("some text", focus="q")

        assert result == "hello"

    async def test_condense_slices_the_input_to_max_chars(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="ok")]])
        condenser = make_condenser(model, max_chars=100)

        await condenser.condense("x" * 500, focus="q")

        user_turn = model.seen_messages[0][-1]
        assert "x" * 100 in user_turn.content
        assert "x" * 101 not in user_turn.content

    async def test_summarize_returns_none_on_empty_messages(self) -> None:
        condenser = make_condenser(FakeChatModel())

        assert await condenser.summarize([]) is None

    async def test_summarize_returns_none_on_a_failing_model(self) -> None:
        condenser = make_condenser(FakeChatModel(fail_with=RuntimeError("boom")))

        result = await condenser.summarize([ChatMessage(role="user", content="hi")])

        assert result is None
