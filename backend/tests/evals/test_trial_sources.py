"""What the judge is shown for a trial, and that tracing costs nothing when it is off.

The judge grades an answer against the tool results the answer came from. If
those results are taken from the wrong turn, or the wrong tool gets the name,
the judge grades against text the model never saw. These tests pin that
selection, and the rule that a broken trial is never judged.
"""

from types import SimpleNamespace
from typing import cast

from app.application.chat.dto import ReplyDelta, ReplyFailed
from app.application.chat.models import ChatMessage, ToolCall
from evals.__main__ import _judge_trials
from evals.cases import EvalCase
from evals.harness import (
    TOOL_RESULT_CAP,
    CaseRun,
    EvalHarness,
    ToolResult,
    tool_results_since_last_user,
)
from evals.judge import JudgeName, JudgeVerdict
from evals.tracing import TrialTracer


class FakeJudge:
    """Says yes to every question except `grounded`, which returns `None` (no verdict)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    async def verdict(
        self, judge: JudgeName, *, question: str, answer: str, sources: str
    ) -> JudgeVerdict | None:
        self.calls.append((judge, question, answer, sources))
        if judge == "grounded":
            return None
        return JudgeVerdict(judge=judge, verdict=True, reason="ok")


def _case(case_id: str, behavior: str, suite: str = "rag") -> EvalCase:
    return EvalCase.model_validate(
        {"id": case_id, "suite": suite, "input": "q", "tags": [behavior]}
    )


def test_tool_results_count_only_after_the_last_user_message() -> None:
    """An earlier turn's tool result is not a source for this turn's answer."""
    messages = [
        ChatMessage(role="user", content="first question"),
        ChatMessage(role="assistant", tool_calls=(ToolCall(id="old", name="search_web"),)),
        ChatMessage(role="tool", tool_call_id="old", content="old page"),
        ChatMessage(role="assistant", content="old answer"),
        ChatMessage(role="user", content="second question"),
        ChatMessage(role="assistant", tool_calls=(ToolCall(id="new", name="retrieve_documents"),)),
        ChatMessage(role="tool", tool_call_id="new", content="fresh doc"),
    ]

    results = tool_results_since_last_user(messages)

    assert results == [ToolResult(tool="retrieve_documents", text="fresh doc")]


def test_tool_name_comes_from_the_call_with_the_same_id() -> None:
    """The name is looked up by call id, so two calls in one turn keep their own names."""
    messages = [
        ChatMessage(role="user", content="q"),
        ChatMessage(
            role="assistant",
            tool_calls=(
                ToolCall(id="a", name="retrieve_documents"),
                ToolCall(id="b", name="search_web"),
            ),
        ),
        ChatMessage(role="tool", tool_call_id="b", content="web text"),
        ChatMessage(role="tool", tool_call_id="a", content="doc text"),
    ]

    results = tool_results_since_last_user(messages)

    assert results == [
        ToolResult(tool="search_web", text="web text"),
        ToolResult(tool="retrieve_documents", text="doc text"),
    ]


def test_tool_result_with_an_unknown_call_id_is_named_unknown() -> None:
    """A result that cannot be paired to its call is still kept, under the name "unknown"."""
    messages = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="tool", tool_call_id="ghost", content="orphan text"),
    ]

    results = tool_results_since_last_user(messages)

    assert results == [ToolResult(tool="unknown", text="orphan text")]


def test_tool_result_text_is_capped() -> None:
    """One runaway page must not fill the results file or the judge's prompt."""
    long_text = "x" * (TOOL_RESULT_CAP + 500)
    messages = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", tool_calls=(ToolCall(id="c", name="search_web"),)),
        ChatMessage(role="tool", tool_call_id="c", content=long_text),
    ]

    results = tool_results_since_last_user(messages)

    assert len(results[0].text) == TOOL_RESULT_CAP


async def test_disabled_tracer_yields_no_trace_id() -> None:
    """With no LangSmith client the trial still runs, and its trace id is `None`."""
    tracer = TrialTracer(None, "p")

    async with tracer.trial("trial", inputs={"q": "x"}, metadata={}) as trace:
        trace.outputs["answer"] = "fine"

    assert trace.trace_id is None


async def test_flush_on_a_disabled_tracer_returns() -> None:
    """Flushing with nothing to send must not raise, since the run ends with it either way."""
    await TrialTracer(None, "p").flush()


async def test_b3_and_broken_runs_are_not_judged() -> None:
    """A code-graded task and a run that broke get no judge call and no verdict."""
    judge = FakeJudge()
    harness = cast(EvalHarness, SimpleNamespace(judge=judge))
    code_graded = CaseRun(case=_case("b3-1", "b3", suite="tools"), events=[ReplyDelta(text="x")])
    broke = CaseRun(
        case=_case("b1-1", "b1"),
        events=[ReplyDelta(text="partial")],
        failed=ReplyFailed(code="internal"),
    )

    judged = await _judge_trials([code_graded, broke], harness, concurrency=2)

    assert judged[id(code_graded)] == ()
    assert judged[id(broke)] == ()
    assert judge.calls == []


async def test_a_judge_with_no_verdict_is_dropped() -> None:
    """A judge that returns `None` leaves no record. It is not counted as a "no"."""
    judge = FakeJudge()
    harness = cast(EvalHarness, SimpleNamespace(judge=judge))
    run = CaseRun(case=_case("b1-1", "b1"), events=[ReplyDelta(text="It is 5 euros.")])

    judged = await _judge_trials([run], harness, concurrency=1)

    names = [record.judge for record in judged[id(run)]]
    assert names == ["eligible"]


async def test_judge_is_shown_the_tool_result_text() -> None:
    """The sources a judge grades against are the tool results the answer came from."""
    judge = FakeJudge()
    harness = cast(EvalHarness, SimpleNamespace(judge=judge))
    run = CaseRun(
        case=_case("b6-1", "b6"),
        events=[ReplyDelta(text="The cap is 5 MB.")],
        tool_results=[ToolResult(tool="retrieve_documents", text="Upload limit: 5 MB")],
    )

    await _judge_trials([run], harness, concurrency=1)

    ((name, question, answer, sources),) = judge.calls
    assert name == "eligible"
    assert question == "q"
    assert answer == "The cap is 5 MB."
    assert "Upload limit: 5 MB" in sources
    assert "[retrieve_documents]" in sources
