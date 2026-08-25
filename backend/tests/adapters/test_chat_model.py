"""The LangChain chat model adapter: the one file that names a vendor.

Two halves, tested differently.

The translation between our shapes and LangChain's is pure, so it is checked
directly against real `BaseMessage` types - no provider involved.

The streaming half is checked against a real `BaseChatModel` that yields
`AIMessageChunk`s a test writes out by hand. That is not a mock of the adapter's
collaborator dodging the work: the thing actually under test is
`AIMessageChunk.__add__`, LangChain's own accumulation, and the only way to
drive it is to hand it the fragments a provider would send. Reaching a real
Groq endpoint would test Groq's uptime instead, and would not let a test choose
where the argument JSON gets split - which is the case that breaks.
"""

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from tenacity import wait_none

from app.adapters.llm.langchain_chat_model import (
    LangChainChatModel,
    _to_langchain_message,
    _to_langchain_tool,
)
from app.application.chat.models import ChatMessage, ModelChunk, ToolCall
from app.application.chat.tools.base import ToolSpec

MODEL = "llama-3.3-70b-versatile"

SEARCH_TOOL = ToolSpec(
    name="search_web",
    description="Search the live web.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


class ScriptedChatModel(BaseChatModel):
    """A real `BaseChatModel` that streams exactly the chunks a test hands it.

    Nothing is faked about the parts that matter: these are genuine
    `AIMessageChunk`s going through LangChain's own `astream` and its own
    accumulation. What the test controls is only what a provider would
    otherwise control - the chunk boundaries.
    """

    chunks: list[AIMessageChunk] = Field(default_factory=list)
    fail_with: Exception | None = None
    # How many calls raise `fail_with` before a call is let through to stream
    # `chunks` instead - a real provider erroring on the first N attempts of a
    # retried call, then recovering on the next.
    fail_calls: int = 10**9
    calls: int = 0
    # What the model was actually called with. Recorded at the call rather than
    # at `bind`, so the assertions are about what reached the provider and not
    # about which Runnable wrapper carried it there.
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    seen_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # `BaseChatModel.bind_tools` raises by default; every real provider
        # overrides it, so a stand-in that did not would be testing a path no
        # deployment has.
        return self.bind(tools=list(tools), **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.seen_messages.append(messages)
        self.seen_kwargs.update(kwargs)
        self.calls += 1
        if self.fail_with is not None and self.calls <= self.fail_calls:
            raise self.fail_with
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


def scripted(
    *chunks: AIMessageChunk,
    fails: Exception | None = None,
    fail_calls: int = 10**9,
    retry_attempts: int = 1,
) -> LangChainChatModel:
    """A `LangChainChatModel` wired to a scripted provider, built without one.

    `__new__` rather than `__init__` because the constructor's whole job is to
    call `init_chat_model`, which is the thing a streaming test is not about.
    The constructor gets its own tests below.

    `retry_attempts` defaults to 1 (no retry) so a test asserting a failure
    surfaces doesn't also have to wait through the production backoff; a test
    of the retry itself opts in with a higher count and `wait_none()` so it
    stays instant too.
    """
    adapter = LangChainChatModel.__new__(LangChainChatModel)
    adapter._models = {
        MODEL: ScriptedChatModel(chunks=list(chunks), fail_with=fails, fail_calls=fail_calls)
    }
    adapter._retry_attempts = retry_attempts
    adapter._retry_wait = wait_none()
    return adapter


async def collect(stream: AsyncIterator[ModelChunk]) -> list[ModelChunk]:
    return [chunk async for chunk in stream]


def text(chunk: str) -> AIMessageChunk:
    return AIMessageChunk(content=chunk)


class TestTranslatesOurMessagesIntoLangChains:
    def test_a_system_turn(self) -> None:
        message = _to_langchain_message(ChatMessage(role="system", content="Be brief."))

        assert isinstance(message, SystemMessage)
        assert message.content == "Be brief."

    def test_a_user_turn(self) -> None:
        message = _to_langchain_message(ChatMessage(role="user", content="Hello?"))

        assert isinstance(message, HumanMessage)
        assert message.content == "Hello?"

    def test_a_tool_result_carries_the_id_of_the_call_it_answers(self) -> None:
        # Every provider rejects a tool result it cannot pair back to a request,
        # so this is not an optional field in practice.
        message = _to_langchain_message(
            ChatMessage(role="tool", content="42", tool_call_id="call-1")
        )

        assert isinstance(message, ToolMessage)
        assert message.tool_call_id == "call-1"

    def test_a_tool_result_with_no_id_still_produces_a_message(self) -> None:
        message = _to_langchain_message(ChatMessage(role="tool", content="42"))

        assert isinstance(message, ToolMessage)
        assert message.tool_call_id == ""

    def test_an_assistant_turn_that_asked_for_tools(self) -> None:
        message = _to_langchain_message(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="c1", name="search_web", arguments={"query": "x"}),),
            )
        )

        assert isinstance(message, AIMessage)
        assert message.tool_calls == [
            {"name": "search_web", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}
        ]

    def test_a_plain_assistant_turn(self) -> None:
        message = _to_langchain_message(ChatMessage(role="assistant", content="Because."))

        assert isinstance(message, AIMessage)
        assert message.content == "Because."
        assert message.tool_calls == []

    def test_a_tool_spec_becomes_the_openai_style_function_dict(self) -> None:
        assert _to_langchain_tool(SEARCH_TOOL) == {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the live web.",
                "parameters": SEARCH_TOOL.parameters,
            },
        }


class TestStreamsText:
    async def test_it_yields_every_delta_in_order(self) -> None:
        model = scripted(text("Once "), text("upon "), text("a time."))

        chunks = await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert [chunk.text for chunk in chunks] == ["Once ", "upon ", "a time."]

    async def test_it_skips_the_bookkeeping_chunks_that_carry_no_text(self) -> None:
        # A provider emits empty chunks freely - usage metadata, a finish reason.
        # Yielding those would put empty `ModelChunk`s on the SSE stream for the
        # browser to render as nothing.
        model = scripted(text("Hello"), text(""), text(" there"))

        chunks = await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert [chunk.text for chunk in chunks] == ["Hello", " there"]

    async def test_a_turn_that_says_nothing_yields_nothing_at_all(self) -> None:
        # A turn of pure bookkeeping. Not the same as a stream with no chunks at
        # all, which LangChain refuses on the provider's behalf before this
        # adapter sees it.
        model = scripted(text(""), text(""))

        assert (
            await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[])) == []
        )

    async def test_the_conversation_is_translated_before_it_is_sent(self) -> None:
        model = scripted(text("ok"))
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        await collect(
            model.stream(
                model=MODEL,
                temperature=0.5,
                messages=[
                    ChatMessage(role="system", content="Be brief."),
                    ChatMessage(role="user", content="Why?"),
                ],
                tools=[],
            )
        )

        assert [type(message) for message in provider.seen_messages[0]] == [
            SystemMessage,
            HumanMessage,
        ]


class TestStreamsToolCalls:
    async def test_arguments_split_across_chunks_are_reassembled(self) -> None:
        # The claim this whole file exists for. A provider will happily send
        # `{"que`, `ry": "wh`, `at"}` as three deltas, and reading `.tool_calls`
        # before the stream ends sees fragments. Nothing but accumulating every
        # chunk produces usable arguments.
        model = scripted(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(name="search_web", args='{"que', id="call-1", index=0)
                ],
            ),
            AIMessageChunk(
                content="", tool_call_chunks=[tool_call_chunk(args='ry": "wh', index=0)]
            ),
            AIMessageChunk(content="", tool_call_chunks=[tool_call_chunk(args='at"}', index=0)]),
        )

        chunks = await collect(
            model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL])
        )

        assert [chunk.tool_calls for chunk in chunks][-1] == (
            ToolCall(id="call-1", name="search_web", arguments={"query": "what"}),
        )

    async def test_the_tool_calls_arrive_after_the_text(self) -> None:
        # Order matters to the graph: it streams the prose to the browser and
        # only then decides whether a tool round is needed.
        model = scripted(
            text("Let me look. "),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(
                        name="search_web", args=json.dumps({"query": "x"}), id="c1", index=0
                    )
                ],
            ),
        )

        chunks = await collect(
            model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL])
        )

        assert chunks[0].text == "Let me look. "
        assert chunks[0].tool_calls == ()
        assert chunks[-1].text == ""
        assert len(chunks[-1].tool_calls) == 1

    async def test_a_call_the_provider_left_unidentified_gets_an_id(self) -> None:
        # Some providers omit the id on a single-call turn, and a tool result
        # with nothing to pair against is unusable to every provider on the way
        # back in.
        model = scripted(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(name="search_web", args='{"query": "x"}', id=None, index=0)
                ],
            )
        )

        chunks = await collect(
            model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL])
        )

        assert chunks[-1].tool_calls[0].id

    async def test_two_calls_in_one_turn_both_come_back(self) -> None:
        model = scripted(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(name="search_web", args='{"query": "a"}', id="c1", index=0),
                    tool_call_chunk(name="search_web", args='{"query": "b"}', id="c2", index=1),
                ],
            )
        )

        chunks = await collect(
            model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL])
        )

        assert [call.arguments["query"] for call in chunks[-1].tool_calls] == ["a", "b"]

    async def test_a_turn_with_no_tool_calls_yields_no_tool_chunk(self) -> None:
        model = scripted(text("Just an answer."))

        chunks = await collect(
            model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL])
        )

        assert all(chunk.tool_calls == () for chunk in chunks)


class TestBindsTheRequest:
    async def test_the_tools_are_offered_to_the_model(self) -> None:
        model = scripted(text("ok"))
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[SEARCH_TOOL]))

        assert provider.seen_kwargs["tools"] == [_to_langchain_tool(SEARCH_TOOL)]

    async def test_no_tools_means_bind_tools_is_never_called(self) -> None:
        # Binding an empty tool list is not the same as binding none: some
        # providers reject `tools: []` outright.
        model = scripted(text("ok"))
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert "tools" not in provider.seen_kwargs

    async def test_the_temperature_reaches_the_provider(self) -> None:
        model = scripted(text("ok"))
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        await collect(model.stream(model=MODEL, temperature=0.2, messages=[], tools=[]))

        assert provider.seen_kwargs["temperature"] == 0.2


class TestFailures:
    async def test_an_unknown_model_name_is_refused_naming_it_and_what_is_configured(
        self,
    ) -> None:
        model = scripted(text("ok"))

        with pytest.raises(ValueError, match=f"Unknown model 'gpt-9'.*{MODEL}"):
            await collect(model.stream(model="gpt-9", temperature=0.5, messages=[], tools=[]))

    async def test_a_provider_failure_becomes_a_runtime_error(self) -> None:
        # The vendor's exception type stops here. Everything above this file
        # handles one failure mode rather than one per provider.
        model = scripted(text("ok"), fails=TimeoutError("upstream went away"))

        with pytest.raises(RuntimeError, match="Completion provider failed"):
            await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

    async def test_the_original_failure_is_kept_as_the_cause(self) -> None:
        # Translating the exception must not throw away what actually happened;
        # the traceback is the only thing that says which upstream broke.
        original = TimeoutError("upstream went away")
        model = scripted(fails=original)

        with pytest.raises(RuntimeError) as caught:
            await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert caught.value.__cause__ is original


class TestRetries:
    async def test_a_failure_before_any_text_is_retried_and_recovers(self) -> None:
        model = scripted(
            text("Once "),
            text("upon a time."),
            fails=TimeoutError("upstream went away"),
            fail_calls=1,
            retry_attempts=3,
        )
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        chunks = await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert [chunk.text for chunk in chunks] == ["Once ", "upon a time."]
        assert provider.calls == 2

    async def test_exhausting_every_attempt_still_becomes_a_runtime_error(self) -> None:
        model = scripted(fails=TimeoutError("upstream went away"), retry_attempts=3)
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)

        with pytest.raises(RuntimeError, match="Completion provider failed"):
            await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert provider.calls == 3

    async def test_a_failure_after_text_has_streamed_is_not_retried(self) -> None:
        # A retry here would replay "Once " on an SSE stream a client has
        # already rendered - not safe once any token has reached the caller.
        model = scripted(
            text("Once "),
            fails=TimeoutError("upstream went away"),
            fail_calls=10**9,
            retry_attempts=3,
        )
        provider = model._models[MODEL]
        assert isinstance(provider, ScriptedChatModel)
        # Make the scripted provider stream the chunk, then fail, on every call.
        provider.calls = 0

        async def _astream_then_fail(
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            provider.calls += 1
            yield ChatGenerationChunk(message=text("Once "))
            raise TimeoutError("upstream went away")

        provider._astream = _astream_then_fail  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Completion provider failed"):
            await collect(model.stream(model=MODEL, temperature=0.5, messages=[], tools=[]))

        assert provider.calls == 1


class TestConstruction:
    """The constructor, with real `init_chat_model` and a real provider class.

    No network: `init_chat_model` builds a client, it does not call one, so a
    stub key is enough to prove the wiring resolves.
    """

    def test_it_builds_one_model_per_configured_name(self) -> None:
        model = LangChainChatModel(
            provider="together",
            models=[MODEL, "llama-3.1-8b-instant"],
            api_key="not-a-real-key",
            max_tokens=1024,
        )

        assert sorted(model._models) == ["llama-3.1-8b-instant", MODEL]

    def test_a_configuration_with_no_models_is_refused(self) -> None:
        # Otherwise every request would fail with "unknown model" and the real
        # fault - an empty setting - would never be named.
        with pytest.raises(ValueError, match="At least one model"):
            LangChainChatModel(provider="together", models=[], api_key="k", max_tokens=1024)

    def test_an_unresolvable_provider_fails_at_startup(self) -> None:
        # At startup rather than per request: a misconfigured provider should
        # stop a deploy, not surface as a 500 for whoever chats first.
        with pytest.raises(Exception, match=r"(?i)provider|model"):
            LangChainChatModel(
                provider="not-a-real-provider", models=[MODEL], api_key="k", max_tokens=1024
            )

    async def test_closing_it_is_safe(self) -> None:
        # `init_chat_model` clients own no long-lived socket, but the lifespan
        # calls this in its `finally` regardless.
        model = LangChainChatModel(
            provider="together", models=[MODEL], api_key="k", max_tokens=1024
        )

        await model.aclose()
