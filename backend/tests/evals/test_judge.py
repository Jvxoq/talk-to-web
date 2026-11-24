"""The judge's token totals, which price the judge in the results file."""

from collections.abc import Sequence

from app.adapters.observability.null_tracer import NullTracer
from app.application.chat.models import ModelChunk, TokenUsage
from evals.judge import Judge
from tests.fakes import FakeChatModel

_REPLY = '{"groundedness": 1.0, "relevance": 1.0, "reason": "ok"}'


def _judge(turns: Sequence[Sequence[ModelChunk]]) -> Judge:
    return Judge(model=FakeChatModel(turns=turns), model_name="cheap", tracer=NullTracer())


async def test_tokens_add_up_across_calls() -> None:
    turn = [ModelChunk(text=_REPLY, usage=TokenUsage(prompt_tokens=120, completion_tokens=15))]
    judge = _judge([turn, turn])

    await judge.score(question="q", answer="a", context="c")
    await judge.score(question="q", answer="a", context="c")

    assert (judge.prompt_tokens, judge.completion_tokens) == (240, 30)


async def test_call_without_usage_adds_nothing() -> None:
    judge = _judge([[ModelChunk(text=_REPLY)]])

    score = await judge.score(question="q", answer="a", context="c")

    assert score is not None
    assert (judge.prompt_tokens, judge.completion_tokens) == (0, 0)


async def test_unparseable_reply_still_counts_its_tokens() -> None:
    """The provider billed the call even though the score is lost."""
    turn = [ModelChunk(text="not json", usage=TokenUsage(prompt_tokens=50, completion_tokens=5))]
    judge = _judge([turn])

    assert await judge.score(question="q", answer="a", context="c") is None
    assert (judge.prompt_tokens, judge.completion_tokens) == (50, 5)


async def test_empty_answer_makes_no_call() -> None:
    model = FakeChatModel(turns=[[ModelChunk(text=_REPLY)]])
    judge = Judge(model=model, model_name="cheap", tracer=NullTracer())

    assert await judge.score(question="q", answer="  ", context="c") is None
    assert model.seen_messages == []
    assert (judge.prompt_tokens, judge.completion_tokens) == (0, 0)
