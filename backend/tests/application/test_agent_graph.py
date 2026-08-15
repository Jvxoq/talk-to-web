"""The agent loop, end to end, with a scripted model and fake tools.

Everything here runs through `GenerateReply`, because the graph and the use case
are one contract: the nodes write payloads onto a custom stream and the use case
is the only thing that reads them. Testing the graph alone would prove the loop
runs and nothing about whether the user ever sees it.

No Postgres: the checkpointer is `InMemorySaver` or absent entirely.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.agent.state import RESET, AgentState
from app.application.chat.agent.summarization import _split, make_summarize_node
from app.application.chat.agent.usage import USAGE, emit_usage
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
)
from app.application.chat.guardrails.policy import InputGuardPolicy
from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.models import ChatMessage, ModelChunk, TokenUsage, ToolCall
from app.application.chat.tools.base import AgentTool, ToolRegistry
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.domain.usage.errors import RateLimited
from tests.fakes import (
    FakeAgentTool,
    FakeChatModel,
    FakeRateLimiter,
    FakeTokenCounter,
    RecordingTracer,
)

SYSTEM_PROMPT = "SYSTEM PROMPT"


def make_guard() -> ToolOutputGuard:
    return ToolOutputGuard(strip_instructions=True, max_scan_chars=50_000)


def make_condenser(
    model: FakeChatModel | None = None, tracer: RecordingTracer | None = None
) -> Condenser:
    """A real condenser over a scripted model, so tests control its replies."""
    return Condenser(
        model=model or FakeChatModel(),
        model_name="condenser",
        max_chars=40_000,
        tool_condense_prompt="condense",
        summary_prompt="summarize",
        tracer=tracer or RecordingTracer(),
    )


def build_use_case(
    model: FakeChatModel,
    tools: Sequence[AgentTool] = (),
    *,
    max_iterations: int = 5,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    limiter: FakeRateLimiter | None = None,
    daily_budget: FakeRateLimiter | None = None,
    condenser: Condenser | None = None,
    counter: FakeTokenCounter | None = None,
    history_token_budget: int = 1_000_000,
    recent_token_budget: int = 1_500,
    tool_output_token_budget: int = 1_000_000,
    tracer: RecordingTracer | None = None,
) -> GenerateReply:
    graph = build_agent_graph(
        model=model,
        tools=ToolRegistry(tools, guard=make_guard()),
        max_iterations=max_iterations,
        checkpointer=checkpointer,
        condenser=condenser or make_condenser(),
        counter=counter or FakeTokenCounter(),
        history_token_budget=history_token_budget,
        recent_token_budget=recent_token_budget,
        tool_output_token_budget=tool_output_token_budget,
        tracer=tracer or RecordingTracer(),
    )
    tracer_obj = tracer or RecordingTracer()
    return GenerateReply(
        graph=graph,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=max_iterations,
        limiter=limiter or FakeRateLimiter(),
        daily_budget=daily_budget or FakeRateLimiter(),
        guards=InputGuardPolicy(redact_pii=False, block_on_injection=False, max_scan_chars=50_000),
        tracer=tracer_obj,
    )


def build_graph(
    model: FakeChatModel,
    tools: Sequence[AgentTool] = (),
    *,
    condenser: Condenser | None = None,
    counter: FakeTokenCounter | None = None,
    tool_output_token_budget: int = 1_000_000,
    tracer: RecordingTracer | None = None,
) -> Any:
    """The compiled graph itself, for tests that read the raw custom stream.

    `GenerateReply` drops any payload `_to_event` does not recognise - which is
    exactly what happens to a `usage` payload today - so a test that wants to
    see one on the wire has to run the graph directly rather than through the
    use case.
    """
    return build_agent_graph(
        model=model,
        tools=ToolRegistry(tools, guard=make_guard()),
        max_iterations=5,
        condenser=condenser or make_condenser(),
        counter=counter or FakeTokenCounter(),
        history_token_budget=1_000_000,
        recent_token_budget=1_500,
        tool_output_token_budget=tool_output_token_budget,
        tracer=tracer or RecordingTracer(),
    )


def run_config(model: str = "test-model", owner_id: int = 1) -> Any:
    return {"configurable": {"model": model, "temperature": 0.0, "owner_id": owner_id}}


async def collect(
    use_case: GenerateReply,
    user_input: str = "hello",
    conversation_id: int | None = None,
    owner_id: int = 1,
) -> list[ReplyEvent]:
    events = await use_case(
        GenerateReplyInput(
            model="test-model",
            user_input=user_input,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
    )
    return [event async for event in events]


def text_of(events: Sequence[ReplyEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, ReplyDelta))


def asking_for(name: str, call_id: str = "call-1", **arguments: object) -> ModelChunk:
    return ModelChunk(tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),))


class TestPlainAnswer:
    async def test_text_streams_through_and_the_stream_completes_once(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="he"), ModelChunk(text="llo")]])
        use_case = build_use_case(model)

        events = await collect(use_case)

        assert text_of(events) == "hello"
        assert sum(isinstance(event, ReplyCompleted) for event in events) == 1
        assert not any(isinstance(event, ReplyFailed) for event in events)
        # One model turn: nothing asked for a tool, so the loop did not go round.
        assert model.calls == 1

    async def test_the_system_prompt_leads_the_conversation(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="hi")]])

        await collect(build_use_case(model))

        first_turn = model.seen_messages[0]
        assert first_turn[0] == ChatMessage(role="system", content=SYSTEM_PROMPT)
        assert first_turn[1].role == "user"

    async def test_a_linked_url_is_named_back_at_the_model(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="hi")]])

        await collect(build_use_case(model), user_input="summarise https://example.com/post")

        user_turn = model.seen_messages[0][-1]
        assert "https://example.com/post" in user_turn.content
        assert "fetch_web_pages" in user_turn.content

    async def test_a_broken_model_ends_the_stream_in_band(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="par")]], fail_with=RuntimeError("boom"))

        events = await collect(build_use_case(model))

        # The response body was already open, so the failure is reported as an
        # event rather than raised through a half-written stream.
        failures = [event for event in events if isinstance(event, ReplyFailed)]
        assert len(failures) == 1
        assert "boom" in failures[0].detail


class TestOneToolCall:
    async def test_the_result_comes_back_to_the_model_and_the_answer_streams(self) -> None:
        tool = FakeAgentTool(name="fake_tool", result="TOOL-RESULT")
        model = FakeChatModel(
            turns=[
                [asking_for("fake_tool", query="weather")],
                [ModelChunk(text="it is sunny")],
            ]
        )
        use_case = build_use_case(model, [tool])

        events = await collect(use_case)

        assert tool.calls == [{"query": "weather"}]
        assert text_of(events) == "it is sunny"
        assert isinstance(events[-1], ReplyCompleted)

        # The point of the loop: the tool's output is in the conversation the
        # model sees on its second turn, paired to the call that asked for it.
        # The output is fenced as untrusted content; assert it is present inside the fence.
        second_turn = model.seen_messages[1]
        tool_messages = [message for message in second_turn if message.role == "tool"]
        assert len(tool_messages) == 1
        assert "TOOL-RESULT" in tool_messages[0].content
        assert tool_messages[0].tool_call_id == "call-1"

    async def test_the_wait_is_reported_before_and_after(self) -> None:
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="weather")], [ModelChunk(text="done")]]
        )

        events = await collect(build_use_case(model, [FakeAgentTool()]))

        started = [event for event in events if isinstance(event, ReplyToolStarted)]
        finished = [event for event in events if isinstance(event, ReplyToolFinished)]
        assert [event.name for event in started] == ["fake_tool"]
        assert "weather" in started[0].summary
        assert [(event.name, event.ok) for event in finished] == [("fake_tool", True)]
        # Announced before it ran, so the spinner has a reason for the whole wait.
        assert events.index(started[0]) < events.index(finished[0])

    async def test_the_model_is_shown_every_registered_tool(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="hi")]])
        tools = [FakeAgentTool(name="alpha"), FakeAgentTool(name="beta")]

        await collect(build_use_case(model, tools))

        assert model.seen_tools[0] == ["alpha", "beta"]


class TestTwoToolCallsInOneTurn:
    async def test_both_run_and_both_results_are_appended(self) -> None:
        first = FakeAgentTool(name="alpha", result="A-RESULT")
        second = FakeAgentTool(name="beta", result="B-RESULT")
        model = FakeChatModel(
            turns=[
                [
                    ModelChunk(
                        tool_calls=(
                            ToolCall(id="a", name="alpha", arguments={"q": 1}),
                            ToolCall(id="b", name="beta", arguments={"q": 2}),
                        )
                    )
                ],
                [ModelChunk(text="both done")],
            ]
        )

        events = await collect(build_use_case(model, [first, second]))

        assert first.calls == [{"q": 1}]
        assert second.calls == [{"q": 2}]

        tool_messages = [message for message in model.seen_messages[1] if message.role == "tool"]
        assert {message.tool_call_id for message in tool_messages} == {"a", "b"}
        # Tool results are fenced as untrusted content; assert they are present.
        assert all(
            "A-RESULT" in message.content or "B-RESULT" in message.content
            for message in tool_messages
        )

        finished = [event for event in events if isinstance(event, ReplyToolFinished)]
        assert {event.name for event in finished} == {"alpha", "beta"}
        assert text_of(events) == "both done"


class TestFailingTool:
    async def test_the_failure_is_flagged_and_the_reply_still_finishes(self) -> None:
        # This fake raises rather than returning, which is what a tool bypassing
        # `BaseTool.run` would do. The registry has to survive it.
        tool = FakeAgentTool(name="fake_tool", fail_with=RuntimeError("upstream is down"))
        model = FakeChatModel(
            turns=[
                [asking_for("fake_tool", query="anything")],
                [ModelChunk(text="I could not check, but here is what I know")],
            ]
        )

        events = await collect(build_use_case(model, [tool]))

        finished = [event for event in events if isinstance(event, ReplyToolFinished)]
        assert [(event.name, event.ok) for event in finished] == [("fake_tool", False)]
        assert not any(isinstance(event, ReplyFailed) for event in events)
        assert isinstance(events[-1], ReplyCompleted)
        # The model still gets a readable tool turn, so it can answer around it.
        assert [message for message in model.seen_messages[1] if message.role == "tool"]


class TestIterationCeiling:
    async def test_a_model_that_never_stops_asking_for_tools_is_cut_off(self) -> None:
        tool = FakeAgentTool(name="fake_tool")
        # Far more scripted tool turns than the ceiling allows.
        model = FakeChatModel(turns=[[asking_for("fake_tool", n=index)] for index in range(20)])

        events = await collect(build_use_case(model, [tool], max_iterations=3))

        assert model.calls == 3
        # Three model turns means the tools ran after the first two only.
        assert len(tool.calls) == 2
        assert isinstance(events[-1], ReplyCompleted)

    async def test_the_ceiling_is_per_reply_not_per_thread(self) -> None:
        # A checkpointed thread that kept counting would refuse to use tools
        # after five messages, forever.
        tool = FakeAgentTool(name="fake_tool")
        model = FakeChatModel(
            turns=[
                [asking_for("fake_tool", turn=1)],
                [ModelChunk(text="first answer")],
                [asking_for("fake_tool", call_id="call-2", turn=2)],
                [ModelChunk(text="second answer")],
            ]
        )
        use_case = build_use_case(model, [tool], max_iterations=2, checkpointer=InMemorySaver())

        await collect(use_case, user_input="one", conversation_id=42)
        events = await collect(use_case, user_input="two", conversation_id=42)

        assert len(tool.calls) == 2
        assert text_of(events) == "second answer"


class TestUnknownTool:
    async def test_the_error_goes_back_to_the_model_without_crashing(self) -> None:
        model = FakeChatModel(
            turns=[
                [asking_for("does_not_exist", query="x")],
                [ModelChunk(text="sorry, answering directly")],
            ]
        )

        events = await collect(build_use_case(model, [FakeAgentTool(name="fake_tool")]))

        tool_messages = [message for message in model.seen_messages[1] if message.role == "tool"]
        assert len(tool_messages) == 1
        # Telling the model which tools are real costs one round trip; raising
        # would cost the whole reply.
        assert "does_not_exist" in tool_messages[0].content
        assert "fake_tool" in tool_messages[0].content

        finished = [event for event in events if isinstance(event, ReplyToolFinished)]
        assert [(event.name, event.ok) for event in finished] == [("does_not_exist", False)]
        assert isinstance(events[-1], ReplyCompleted)


class TestMemory:
    async def test_history_is_replayed_and_the_system_prompt_is_written_once(self) -> None:
        model = FakeChatModel(
            turns=[[ModelChunk(text="first answer")], [ModelChunk(text="second answer")]]
        )
        use_case = build_use_case(model, checkpointer=InMemorySaver())

        await collect(use_case, user_input="one", conversation_id=7)
        await collect(use_case, user_input="two", conversation_id=7)

        second_turn = model.seen_messages[1]
        systems = [message for message in second_turn if message.role == "system"]
        # Re-sending it every turn would stack a fresh copy in checkpointed state.
        assert len(systems) == 1
        assert [message.content for message in second_turn if message.role == "user"] == [
            "one",
            "two",
        ]

    async def test_threads_without_a_conversation_do_not_share_history(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="first")], [ModelChunk(text="second")]])
        use_case = build_use_case(model, checkpointer=InMemorySaver())

        await collect(use_case, user_input="one")
        await collect(use_case, user_input="two")

        assert [
            message.content for message in model.seen_messages[1] if message.role == "user"
        ] == ["two"]


class TestSpendLimit:
    async def test_a_reply_past_the_budget_is_refused_before_the_model_is_called(self) -> None:
        # The point of the limit: every reply spends tokens at the provider, so
        # the refusal has to land before the graph runs, not while it streams.
        model = FakeChatModel(turns=[[ModelChunk(text="one")], [ModelChunk(text="two")]])
        limiter = FakeRateLimiter(max_attempts=1)
        use_case = build_use_case(model, limiter=limiter)

        await collect(use_case)
        with pytest.raises(RateLimited):
            await collect(use_case)

        assert len(model.seen_messages) == 1

    async def test_the_budget_is_counted_per_account(self) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="one")], [ModelChunk(text="two")]])
        limiter = FakeRateLimiter(max_attempts=1)
        use_case = build_use_case(model, limiter=limiter)

        await collect(use_case, owner_id=1)
        # A second user is not blocked by the first one's spending.
        await collect(use_case, owner_id=2)

        assert limiter.hits == {"generate:1": 1, "generate:2": 1}

    async def test_a_spent_daily_budget_refuses_before_the_per_user_limit_is_touched(
        self,
    ) -> None:
        model = FakeChatModel(turns=[[ModelChunk(text="one")]])
        limiter = FakeRateLimiter()
        use_case = build_use_case(
            model, limiter=limiter, daily_budget=FakeRateLimiter(max_attempts=0)
        )

        with pytest.raises(RateLimited):
            await collect(use_case)

        assert len(model.seen_messages) == 0
        assert limiter.hits == {}, "the per-user limit must not be spent on a refused request"

    async def test_the_daily_budget_is_shared_across_every_account(self) -> None:
        # Unlike the per-user limiter above, one key ("global") is hit by every
        # caller regardless of who is signed in - the backstop registering a new
        # account cannot get around.
        model = FakeChatModel(turns=[[ModelChunk(text="one")], [ModelChunk(text="two")]])
        daily_budget = FakeRateLimiter(max_attempts=1)
        use_case = build_use_case(model, daily_budget=daily_budget)

        await collect(use_case, owner_id=1)
        with pytest.raises(RateLimited):
            await collect(use_case, owner_id=2)

        assert len(model.seen_messages) == 1


class TestSummarization:
    async def test_under_budget_does_not_call_the_condenser(self) -> None:
        condenser_model = FakeChatModel()
        condenser = make_condenser(condenser_model)
        model = FakeChatModel(turns=[[ModelChunk(text="hi")]])

        await collect(
            build_use_case(
                model,
                condenser=condenser,
                counter=FakeTokenCounter(tokens_per_message=1),
                history_token_budget=1_000,
            )
        )

        # The hot path must cost nothing: a short thread never reaches the model.
        assert condenser_model.calls == 0

    async def test_over_budget_replaces_the_head_with_a_summary(self) -> None:
        condenser_model = FakeChatModel(turns=[[ModelChunk(text="SUMMARY")]])
        node = make_summarize_node(
            FakeTokenCounter(tokens_per_message=1_000),
            make_condenser(condenser_model),
            history_token_budget=3_000,
            recent_token_budget=1_500,
            tracer=RecordingTracer(),
        )
        state = AgentState(
            messages=[
                ChatMessage(role="system", content="SYS"),
                ChatMessage(role="user", content="one"),
                ChatMessage(role="assistant", content="first"),
                ChatMessage(role="user", content="two"),
            ]
        )

        result = await node(state)

        messages = result["messages"]
        # The sentinel tells the reducer to replace, not append.
        assert messages[0] is RESET
        # The system prompt is kept verbatim, the head is one summary, the tail
        # (the most recent exchange) is intact.
        assert messages[1] == ChatMessage(role="system", content="SYS")
        assert "SUMMARY" in messages[2].content
        assert messages[3:] == [
            ChatMessage(role="assistant", content="first"),
            ChatMessage(role="user", content="two"),
        ]
        assert result["summary"] == "SUMMARY"

    async def test_the_cut_never_orphans_a_tool_message(self) -> None:
        counter = FakeTokenCounter(tokens_per_message=1_000)
        messages = [
            ChatMessage(role="user", content="one"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="c", name="t", arguments={}),),
            ),
            ChatMessage(role="tool", content="result", tool_call_id="c"),
            ChatMessage(role="user", content="two"),
        ]

        head, tail = _split(messages, counter, recent_token_budget=1_500)

        # The tail must not start on a tool message - that would orphan it from
        # the assistant call that asked for it.
        assert not tail or tail[0].role != "tool"
        # The assistant and its tool reply stay together in the head.
        assert head[-2].role == "assistant" and head[-1].role == "tool"
        assert tail == [ChatMessage(role="user", content="two")]

    async def test_a_failing_condenser_drops_the_head_and_keeps_the_reply_alive(self) -> None:
        node = make_summarize_node(
            FakeTokenCounter(tokens_per_message=1_000),
            make_condenser(FakeChatModel(fail_with=RuntimeError("boom"))),
            history_token_budget=3_000,
            recent_token_budget=1_500,
            tracer=RecordingTracer(),
        )
        state = AgentState(
            messages=[
                ChatMessage(role="system", content="SYS"),
                ChatMessage(role="user", content="one"),
                ChatMessage(role="assistant", content="first"),
                ChatMessage(role="user", content="two"),
            ]
        )

        result = await node(state)

        messages = result["messages"]
        assert messages[0] is RESET
        assert messages[1] == ChatMessage(role="system", content="SYS")
        # The head is dropped outright, the recent tail survives.
        assert messages[2:] == [
            ChatMessage(role="assistant", content="first"),
            ChatMessage(role="user", content="two"),
        ]
        assert result["summary"] == ""


class TestToolOutputCompression:
    async def test_a_small_tool_result_is_not_condensed(self) -> None:
        condenser_model = FakeChatModel()
        condenser = make_condenser(condenser_model)
        tool = FakeAgentTool(name="fake_tool", result="small")
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="q")], [ModelChunk(text="done")]]
        )

        await collect(
            build_use_case(
                model,
                [tool],
                condenser=condenser,
                counter=FakeTokenCounter(tokens_per_message=1),
                tool_output_token_budget=1_000,
            )
        )

        # A small result must not cost an extra model call.
        assert condenser_model.calls == 0
        tool_message = next(m for m in model.seen_messages[1] if m.role == "tool")
        assert "small" in tool_message.content

    async def test_a_large_tool_result_is_condensed_and_tool_end_still_fires(self) -> None:
        condenser_model = FakeChatModel(turns=[[ModelChunk(text="CONDENSED")]])
        condenser = make_condenser(condenser_model)
        tool = FakeAgentTool(name="fake_tool", result="x" * 5_000)
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="q")], [ModelChunk(text="done")]]
        )

        events = await collect(
            build_use_case(
                model,
                [tool],
                condenser=condenser,
                counter=FakeTokenCounter(tokens_per_message=2_000),
                tool_output_token_budget=1_000,
            )
        )

        assert condenser_model.calls == 1
        tool_message = next(m for m in model.seen_messages[1] if m.role == "tool")
        assert tool_message.content == "CONDENSED"
        # The SSE contract is unchanged: TOOL_END still fires with the same shape.
        finished = [event for event in events if isinstance(event, ReplyToolFinished)]
        assert [(event.name, event.ok) for event in finished] == [("fake_tool", True)]


class TestFriendlyRateLimit:
    async def test_a_rate_limit_failure_is_reported_in_plain_words(self) -> None:
        model = FakeChatModel(
            turns=[[ModelChunk(text="x")]],
            fail_with=RuntimeError("rate_limit_exceeded ... 413 Request too large"),
        )

        events = await collect(build_use_case(model))

        failures = [event for event in events if isinstance(event, ReplyFailed)]
        assert len(failures) == 1
        assert "busy" in failures[0].detail
        assert "413" not in failures[0].detail


class TestUsageStream:
    async def test_a_final_chunks_usage_reaches_the_custom_stream(self) -> None:
        model = FakeChatModel(
            turns=[
                [
                    ModelChunk(text="hi"),
                    ModelChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5)),
                ]
            ]
        )
        graph = build_graph(model)
        state = AgentState(messages=[ChatMessage(role="user", content="hi")])

        payloads = [p async for p in graph.astream(state, run_config(), stream_mode="custom")]

        usage_payloads = [p for p in payloads if isinstance(p, dict) and p.get("type") == USAGE]
        assert usage_payloads == [
            {
                "type": USAGE,
                "model": "test-model",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        ]

    async def test_emit_usage_outside_a_graph_run_does_not_raise(self) -> None:
        # The condenser is unit-tested outside any graph run, so this helper
        # must survive being called with no stream writer in context - never
        # be the reason a reply dies.
        emit_usage(model="condenser", usage=TokenUsage(prompt_tokens=1, completion_tokens=1))


class TestAgentTracing:
    async def test_the_agent_span_carries_tokens_and_ttft(self) -> None:
        tracer = RecordingTracer()
        model = FakeChatModel(
            turns=[
                [
                    ModelChunk(text="hi"),
                    ModelChunk(usage=TokenUsage(prompt_tokens=3, completion_tokens=2)),
                ]
            ]
        )

        await collect(build_use_case(model, tracer=tracer))

        spans = tracer.named("llm.agent")
        assert len(spans) == 1
        span = spans[0]
        assert span.kind == "generation"
        assert span.attributes["model"] == "test-model"
        assert span.attributes["prompt_tokens"] == 3
        assert span.attributes["completion_tokens"] == 2
        assert span.attributes["tool_calls"] == []
        assert span.attributes["ttft_ms"] is not None
        assert not span.errors

    async def test_a_tool_calling_turn_names_the_tools_it_asked_for(self) -> None:
        tracer = RecordingTracer()
        tool = FakeAgentTool(name="fake_tool")
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="x")], [ModelChunk(text="done")]]
        )

        await collect(build_use_case(model, [tool], tracer=tracer))

        first_agent_span = tracer.named("llm.agent")[0]
        assert first_agent_span.attributes["tool_calls"] == ["fake_tool"]


class TestToolTracing:
    async def test_a_tool_span_carries_ok_and_latency(self) -> None:
        tracer = RecordingTracer()
        tool = FakeAgentTool(name="fake_tool", result="a short result")
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="x")], [ModelChunk(text="done")]]
        )

        await collect(build_use_case(model, [tool], tracer=tracer))

        spans = tracer.named("tool.fake_tool")
        assert len(spans) == 1
        span = spans[0]
        assert span.kind == "span"
        assert span.attributes["arguments"] == {"query": "x"}
        assert span.attributes["ok"] is True
        assert span.attributes["sources"] == 0
        # Length of what actually went back to the model, not a literal on the
        # fixture - the output guard is free to wrap tool content, and this
        # span must track whatever it produced rather than the raw fixture.
        tool_message = next(m for m in model.seen_messages[1] if m.role == "tool")
        assert span.attributes["output_chars"] == len(tool_message.content)
        latency_ms = span.attributes["latency_ms"]
        assert isinstance(latency_ms, float) and latency_ms >= 0
        assert not span.errors

    async def test_a_failing_tool_still_gets_a_span(self) -> None:
        tracer = RecordingTracer()
        tool = FakeAgentTool(name="fake_tool", fail_with=RuntimeError("upstream is down"))
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="x")], [ModelChunk(text="done anyway")]]
        )

        await collect(build_use_case(model, [tool], tracer=tracer))

        # `ToolRegistry.invoke` never raises, so the failure surfaces as `ok=False`
        # rather than a recorded error - the span still gets its full attributes.
        spans = tracer.named("tool.fake_tool")
        assert len(spans) == 1
        assert spans[0].attributes["ok"] is False


class TestCondenserUsage:
    async def test_the_condensers_own_spend_is_emitted_too(self) -> None:
        # This is the one most likely to be silently dropped: a reply that
        # called a tool and condensed its output paid for both, and the total
        # must reflect the condenser's call, not just the agent's.
        condenser_model = FakeChatModel(
            turns=[
                [
                    ModelChunk(
                        text="CONDENSED",
                        usage=TokenUsage(prompt_tokens=7, completion_tokens=1),
                    )
                ]
            ]
        )
        condenser = make_condenser(condenser_model)
        tool = FakeAgentTool(name="fake_tool", result="x" * 5_000)
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="q")], [ModelChunk(text="done")]]
        )
        graph = build_graph(
            model,
            [tool],
            condenser=condenser,
            counter=FakeTokenCounter(tokens_per_message=2_000),
            tool_output_token_budget=1_000,
        )
        state = AgentState(messages=[ChatMessage(role="user", content="hi")])

        payloads = [p async for p in graph.astream(state, run_config(), stream_mode="custom")]

        usage_payloads = [p for p in payloads if isinstance(p, dict) and p.get("type") == USAGE]
        condenser_usage = [p for p in usage_payloads if p["model"] == "condenser"]
        assert condenser_usage == [
            {
                "type": USAGE,
                "model": "condenser",
                "prompt_tokens": 7,
                "completion_tokens": 1,
            }
        ]

    async def test_the_condenser_opens_a_generation_span(self) -> None:
        tracer = RecordingTracer()
        condenser_model = FakeChatModel(
            turns=[
                [
                    ModelChunk(
                        text="CONDENSED",
                        usage=TokenUsage(prompt_tokens=7, completion_tokens=1),
                    )
                ]
            ]
        )
        condenser = make_condenser(condenser_model, tracer)
        tool = FakeAgentTool(name="fake_tool", result="x" * 5_000)
        model = FakeChatModel(
            turns=[[asking_for("fake_tool", query="q")], [ModelChunk(text="done")]]
        )

        await collect(
            build_use_case(
                model,
                [tool],
                condenser=condenser,
                counter=FakeTokenCounter(tokens_per_message=2_000),
                tool_output_token_budget=1_000,
                tracer=tracer,
            )
        )

        spans = tracer.named("condense.tool_output")
        assert len(spans) == 1
        assert spans[0].kind == "generation"
        assert spans[0].attributes["model"] == "condenser"
        assert spans[0].attributes["prompt_tokens"] == 7
        assert spans[0].attributes["completion_tokens"] == 1
