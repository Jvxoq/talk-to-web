"""LangChain-backed streaming chat model: the one place a vendor is named.

Satisfies `app.application.chat.ports.ChatModel`. Everything LangChain knows how
to do - provider resolution, tool binding, chunk accumulation - stops at this
file, so the agent graph above it never imports `langchain` and never learns
which company answered.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.runnables import Runnable
from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from tenacity.wait import wait_base

from app.application.chat.models import ChatMessage, ModelChunk, TokenUsage, ToolCall
from app.application.chat.tools.base import ToolSpec

_DEFAULT_RETRY_WAIT = wait_exponential_jitter(initial=1, max=8)
"""Module-level singleton so it isn't rebuilt on every call to the default."""


def _to_langchain_message(message: ChatMessage) -> BaseMessage:
    """Translate one of our turns into the shape LangChain expects."""
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "user":
        return HumanMessage(content=message.content)
    if message.role == "tool":
        # A tool result that cannot be paired back to its request is rejected by
        # every provider, so the id is not optional in practice.
        return ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "")

    tool_calls: list[LangChainToolCall] = [
        LangChainToolCall(id=call.id, name=call.name, args=dict(call.arguments))
        for call in message.tool_calls
    ]
    return AIMessage(content=message.content, tool_calls=tool_calls)


def _to_langchain_tool(spec: ToolSpec) -> dict[str, Any]:
    """Translate a `ToolSpec` into the OpenAI-style dict `bind_tools` understands."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


class LangChainChatModel:
    """
    Streams tool-calling turns from any provider `init_chat_model` can resolve.

    One `BaseChatModel` is built per configured model name at startup: building
    one per request would re-create an HTTP client on every message. The provider
    is a plain string ("groq", "openai", "anthropic", ...), so changing vendor is
    an environment change rather than a code change.
    """

    def __init__(
        self,
        *,
        provider: str,
        models: Sequence[str],
        api_key: str,
        max_tokens: int,
        retry_attempts: int = 3,
        retry_wait: wait_base = _DEFAULT_RETRY_WAIT,
    ) -> None:
        if not models:
            raise ValueError("At least one model name must be configured")

        self._models: dict[str, BaseChatModel] = {}
        for name in models:
            built = init_chat_model(
                f"{provider}:{name}",
                api_key=api_key,
                max_tokens=max_tokens,
            )
            if not isinstance(built, BaseChatModel):
                # `init_chat_model` returns a late-bound configurable wrapper when
                # asked for one. We never ask, and the rest of this class relies
                # on `bind_tools`, so refuse loudly rather than fail per request.
                raise RuntimeError(f"Provider {provider!r} did not return a usable chat model")
            self._models[name] = built

        self._retry_attempts = retry_attempts
        self._retry_wait = retry_wait

        logger.info(f"Chat models ready via {provider!r}: {', '.join(self._models)}")

    async def stream(
        self,
        *,
        model: str,
        temperature: float,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ModelChunk]:
        """Yield text deltas as they arrive, then any tool calls the turn asked for."""
        runnable = self._runnable(model=model, temperature=temperature, tools=tools)
        payload = [_to_langchain_message(message) for message in messages]

        # Tool call arguments arrive split across many chunks - a provider will
        # happily send `{"que`, `ry": "wh`, `at"}` as three deltas - so nothing
        # can be parsed until the stream ends. `AIMessageChunk.__add__` is what
        # reassembles them; reading `.tool_calls` before the end sees fragments.
        accumulated: AIMessageChunk | None = None

        # A dropped connection or a 429/5xx before the first token is a
        # provider hiccup worth one silent retry; a failure after tokens have
        # already reached the caller is not retryable - the SSE stream is
        # already in flight, and replaying it would duplicate text the client
        # rendered. `started` gates the predicate on that boundary rather than
        # on exception type, which keeps this generic across whatever provider
        # `init_chat_model` resolved.
        started = False

        def _retryable(_: BaseException) -> bool:
            return not started

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._retry_attempts),
                wait=self._retry_wait,
                retry=retry_if_exception(_retryable),
                reraise=True,
            ):
                with attempt:
                    accumulated = None
                    async for chunk in runnable.astream(payload):
                        if not isinstance(chunk, AIMessageChunk):
                            continue
                        accumulated = chunk if accumulated is None else accumulated + chunk

                        text = str(chunk.text)
                        if text:
                            started = True
                            # `model_construct` skips validation: this runs
                            # once per token, on data we just built ourselves.
                            yield ModelChunk.model_construct(text=text)
        except Exception as error:
            logger.warning(f"Chat stream failed for model {model}: {error}")
            raise RuntimeError(f"Completion provider failed: {error}") from error

        requested = self._tool_calls(accumulated)
        if requested:
            logger.debug(f"Model {model} asked for {len(requested)} tool call(s)")
            yield ModelChunk(tool_calls=requested)

        usage = self._usage(accumulated, model=model)
        if usage is not None:
            yield ModelChunk(usage=usage)

    @staticmethod
    def _usage(accumulated: AIMessageChunk | None, *, model: str) -> TokenUsage | None:
        """Read what the provider reported, if it reported anything at all.

        Absence has to stay absence: not every provider fills in
        `usage_metadata`, and a fabricated zero would look identical to "the
        model really used no tokens" downstream, in a cost report. Reading it
        is best-effort by design - a malformed dict here must not cost a reply
        that has already streamed successfully.
        """
        if accumulated is None:
            return None
        try:
            reported = accumulated.usage_metadata
            if reported is None:
                return None
            return TokenUsage(
                prompt_tokens=reported.get("input_tokens", 0),
                completion_tokens=reported.get("output_tokens", 0),
            )
        except Exception as error:
            logger.debug(f"Could not read usage for model {model}: {error}")
            return None

    def _runnable(
        self,
        *,
        model: str,
        temperature: float,
        tools: Sequence[ToolSpec],
    ) -> Runnable[Any, BaseMessage]:
        chat_model = self._models.get(model)
        if chat_model is None:
            available = ", ".join(self._models) or "none"
            raise ValueError(f"Unknown model {model!r}. Configured models: {available}.")

        # Binding an empty tool list is not the same as binding none: some
        # providers reject `tools: []` outright.
        bound: Runnable[Any, BaseMessage] = (
            chat_model.bind_tools([_to_langchain_tool(spec) for spec in tools])
            if tools
            else chat_model
        )
        return bound.bind(temperature=temperature)

    @staticmethod
    def _tool_calls(accumulated: AIMessageChunk | None) -> tuple[ToolCall, ...]:
        """Read the reassembled calls off the finished turn, in our own shape."""
        if accumulated is None:
            return ()
        return tuple(
            ToolCall(
                # Some providers omit the id on a single-call turn, and a tool
                # result with no id to pair against is unusable downstream.
                id=call.get("id") or uuid.uuid4().hex,
                name=call["name"],
                arguments=dict(call.get("args") or {}),
            )
            for call in accumulated.tool_calls
        )

    async def aclose(self) -> None:
        """Nothing to release: `init_chat_model` clients own no long-lived socket."""
        return None
