"""The agent loop, end to end, with a scripted model and fake tools.

Everything here runs through `GenerateReply`, because the graph and the use case
are one contract: the nodes write payloads onto a custom stream and the use case
is the only thing that reads them. Testing the graph alone would prove the loop
runs and nothing about whether the user ever sees it.

No Postgres: the checkpointer is `InMemorySaver` or absent entirely.
"""

from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
)
from app.application.chat.models import ChatMessage, ModelChunk, ToolCall
from app.application.chat.tools.base import AgentTool, ToolRegistry
from app.application.chat.use_cases.generate_reply import GenerateReply
from tests.fakes import FakeAgentTool, FakeChatModel

SYSTEM_PROMPT = "SYSTEM PROMPT"


def build_use_case(
    model: FakeChatModel,
    tools: Sequence[AgentTool] = (),
    *,
    max_iterations: int = 5,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> GenerateReply:
    graph = build_agent_graph(
        model=model,
        tools=ToolRegistry(tools),
        max_iterations=max_iterations,
        checkpointer=checkpointer,
    )
    return GenerateReply(
        graph=graph,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )


async def collect(
    use_case: GenerateReply,
    user_input: str = "hello",
    conversation_id: int | None = None,
) -> list[ReplyEvent]:
    return [
        event
        async for event in use_case(
            GenerateReplyInput(
                model="test-model",
                user_input=user_input,
                conversation_id=conversation_id,
            )
        )
    ]


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
        second_turn = model.seen_messages[1]
        tool_messages = [message for message in second_turn if message.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "TOOL-RESULT"
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
        assert {message.content for message in tool_messages} == {"A-RESULT", "B-RESULT"}

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
