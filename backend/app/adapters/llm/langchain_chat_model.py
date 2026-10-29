"""LangChain streaming chat model, implementing `ChatModel`.

LangChain stays in this file, so the agent graph never imports it.
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
from app.application.chat.provider_errors import is_auth_failure
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
        # Providers reject a tool result without its call id.
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
    """Streams tool-calling turns from any provider `init_chat_model` supports.

    One client per model, built at startup. The provider is an env string.
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
                # We need `bind_tools`, so fail at startup, not per request.
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

        # Tool call arguments arrive in fragments. Adding chunks reassembles them.
        accumulated: AIMessageChunk | None = None

        # Retry only before the first token. After that, a retry would
        # duplicate text the client already shows.
        started = False

        def _retryable(error: BaseException) -> bool:
            # A rejected key fails the same way every time. Do not retry.
            if is_auth_failure(str(error)):
                logger.error(f"Provider rejected our credentials for model {model}")
                return False
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
                            # Skips validation: runs per token on our own data.
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
        """The provider's reported usage, or `None`. Never a made-up zero."""
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
        except Exception as error:  # noqa: BLE001
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

        # Some providers reject `tools: []`, so bind none instead.
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
                # Some providers omit the id. Results need one.
                id=call.get("id") or uuid.uuid4().hex,
                name=call["name"],
                arguments=dict(call.get("args") or {}),
            )
            for call in accumulated.tool_calls
        )

    async def aclose(self) -> None:
        """Nothing to release: `init_chat_model` clients own no long-lived socket."""
        return None
