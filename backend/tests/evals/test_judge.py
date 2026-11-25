"""The yes/no judges: what counts as a verdict, and the token totals that price them."""

from collections.abc import Sequence

import pytest

from app.adapters.observability.null_tracer import NullTracer
from app.application.chat.models import ModelChunk, TokenUsage
from evals.judge import Judge, JudgeVerdict, _parse
from tests.fakes import FakeChatModel

_REPLY = '{"reason": "ok", "verdict": true}'


def _judge(turns: Sequence[Sequence[ModelChunk]]) -> Judge:
    return Judge(model=FakeChatModel(turns=turns), model_name="cheap", tracer=NullTracer())


async def test_verdict_reads_the_reply() -> None:
    judge = _judge([[ModelChunk(text='{"reason": "cites the page", "verdict": false}')]])

    result = await judge.verdict("grounded", question="q", answer="a", sources="s")

    assert result == JudgeVerdict(judge="grounded", verdict=False, reason="cites the page")


async def test_prompt_carries_the_question_sources_and_answer() -> None:
    model = FakeChatModel(turns=[[ModelChunk(text=_REPLY)]])
    judge = Judge(model=model, model_name="cheap", tracer=NullTracer())

    await judge.verdict(
        "abstained", question="who wrote it?", answer="not in your file", sources=""
    )

    system, user = model.seen_messages[0]
    assert "decline" in system.content
    assert "who wrote it?" in user.content
    assert "(none)" in user.content
    assert "not in your file" in user.content


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"reason": "r", "verdict": true}\n```',
        'Sure. {"reason": "r", "verdict": true} Hope that helps.',
    ],
)
def test_parse_strips_a_fence_or_prose(raw: str) -> None:
    assert _parse("eligible", raw) == JudgeVerdict(judge="eligible", verdict=True, reason="r")


@pytest.mark.parametrize(
    "raw",
    [
        '{"reason": "r", "verdict": "true"}',
        '{"reason": "r", "verdict": 1}',
        '{"reason": "r"}',
        "[true]",
        "yes",
    ],
)
def test_parse_needs_a_real_boolean(raw: str) -> None:
    """A guessed verdict would land in the agreement counts as if the judge said it."""
    assert _parse("grounded", raw) is None


async def test_tokens_add_up_across_calls() -> None:
    turn = [ModelChunk(text=_REPLY, usage=TokenUsage(prompt_tokens=120, completion_tokens=15))]
    judge = _judge([turn, turn])

    await judge.verdict("grounded", question="q", answer="a", sources="s")
    await judge.verdict("eligible", question="q", answer="a", sources="s")

    assert (judge.prompt_tokens, judge.completion_tokens) == (240, 30)


async def test_call_without_usage_adds_nothing() -> None:
    judge = _judge([[ModelChunk(text=_REPLY)]])

    result = await judge.verdict("grounded", question="q", answer="a", sources="s")

    assert result is not None
    assert (judge.prompt_tokens, judge.completion_tokens) == (0, 0)


async def test_unparseable_reply_still_counts_its_tokens() -> None:
    """The provider billed the call even though the verdict is lost."""
    turn = [ModelChunk(text="not json", usage=TokenUsage(prompt_tokens=50, completion_tokens=5))]
    judge = _judge([turn])

    assert await judge.verdict("grounded", question="q", answer="a", sources="s") is None
    assert (judge.prompt_tokens, judge.completion_tokens) == (50, 5)


async def test_model_error_returns_none() -> None:
    model = FakeChatModel(fail_with=RuntimeError("provider down"))
    judge = Judge(model=model, model_name="cheap", tracer=NullTracer())

    assert await judge.verdict("grounded", question="q", answer="a", sources="s") is None


async def test_empty_answer_makes_no_call() -> None:
    model = FakeChatModel(turns=[[ModelChunk(text=_REPLY)]])
    judge = Judge(model=model, model_name="cheap", tracer=NullTracer())

    assert await judge.verdict("grounded", question="q", answer="  ", sources="s") is None
    assert model.seen_messages == []
    assert (judge.prompt_tokens, judge.completion_tokens) == (0, 0)
