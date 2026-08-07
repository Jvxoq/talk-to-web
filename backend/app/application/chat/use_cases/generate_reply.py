"""Stream one agent reply, turning graph events into events the API can frame."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from loguru import logger

from app.application.chat.agent.graph import AgentGraph
from app.application.chat.agent.nodes import DELTA, TOOL_END, TOOL_START
from app.application.chat.agent.state import AgentState
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
)
from app.application.chat.models import ChatMessage
from app.domain.chat.value_objects import UserMessage


class GenerateReply:
    """
    Answer the user, with context if the agent can get it and without if it cannot.

    Context is an enhancement, not a precondition: a scraper timing out or a
    vector store being down should cost the answer some grounding, never the
    answer itself. That posture now lives one layer down - every tool returns a
    readable failure instead of raising, so a dead upstream reaches the model as
    a sentence it can route around. What is left here is the outer guarantee:
    once this stream has opened, it ends with `ReplyFailed` or `ReplyCompleted`,
    never with an exception thrown through a half-written response body.

    The compiled graph is injected, never built: compiling per request would
    rebuild the whole agent - and reconnect its checkpointer - on every message.
    """

    def __init__(
        self,
        graph: AgentGraph,
        system_prompt: str,
        max_iterations: int,
    ) -> None:
        self._graph = graph
        self._system_prompt = system_prompt
        # A backstop below LangGraph's own default, in the same units the graph
        # counts in: one lap is an agent node plus a tool node, and the +2 covers
        # the final answering turn. The router is what normally stops the loop;
        # this only catches a graph miswired into a cycle the router cannot see.
        self._recursion_limit = 2 * max_iterations + 2

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        # Typed `Any` on purpose. The concrete type is langchain_core's
        # `RunnableConfig`, and importing it to say so would put LangChain in the
        # application layer - which the import contract forbids, and for good
        # reason: LangChain is what the `ChatModel` port exists to keep out.
        config: Any = {
            "configurable": {
                # A conversation is the agent's memory key. Without one, every
                # request is its own thread and nothing is remembered - which is
                # exactly what a client that never opened a conversation wants.
                "thread_id": str(data.conversation_id)
                if data.conversation_id is not None
                else uuid4().hex,
                "model": data.model,
                "temperature": data.temperature,
            },
            "recursion_limit": self._recursion_limit,
        }

        try:
            state = AgentState(
                messages=await self._new_messages(data, config),
                # Reset per request. The count is a ceiling on this reply's tool
                # laps; carried over from the checkpoint it would only ever climb.
                iterations=0,
            )

            async for payload in self._graph.astream(state, config, stream_mode="custom"):
                event = _to_event(payload)
                if event is not None:
                    yield event
        except asyncio.CancelledError:
            # The client hung up. Let it propagate: swallowing it leaves the
            # graph running for a response nobody is reading.
            raise
        except Exception as exc:
            # The HTTP response is already open, so raising here would only
            # truncate the body. The client is told in-band instead.
            logger.warning(f"Agent run failed for model {data.model}: {exc}")
            yield ReplyFailed(detail=str(exc))
            return

        yield ReplyCompleted()

    async def _new_messages(
        self,
        data: GenerateReplyInput,
        config: Any,
    ) -> list[ChatMessage]:
        """Only what this turn adds - the checkpointer already holds the rest."""
        message = UserMessage(data.user_input)
        messages: list[ChatMessage] = []

        if await self._is_new_thread(config):
            # Appended once per thread, not once per turn: the checkpointer
            # replays the whole state, so re-sending the system prompt every
            # message would stack a fresh copy of it in the history.
            messages.append(ChatMessage(role="system", content=self._system_prompt))

        messages.append(ChatMessage(role="user", content=_user_content(message)))
        return messages

    async def _is_new_thread(self, config: Any) -> bool:
        if self._graph.checkpointer is None:
            # Nothing is remembered, so every turn starts the conversation.
            return True
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception as exc:
            # A checkpointer that cannot be read must not cost the user a reply;
            # the worst case is one duplicated system message.
            logger.warning(f"Could not read agent state, treating thread as new: {exc}")
            return True
        return not snapshot.values.get("messages")


def _user_content(message: UserMessage) -> str:
    """The user's text, with the links they mentioned called out."""
    urls = message.urls()
    if not urls:
        return message.text

    # Models routinely answer from memory about a URL they were handed instead
    # of reading it. Naming the addresses back at them, on the turn that carries
    # them, is what makes `fetch_web_pages` get picked. It rides on the user
    # turn rather than the system prompt because the system prompt is written
    # once per thread, and these links belong to this message only.
    listed = ", ".join(urls)
    return f"{message.text}\n\n[The user linked: {listed}. Read them with fetch_web_pages.]"


def _to_event(payload: object) -> ReplyEvent | None:
    """Translate one custom-stream payload from the nodes into a reply event."""
    if not isinstance(payload, dict):
        return None

    kind = payload.get("type")
    if kind == DELTA:
        return ReplyDelta(text=str(payload.get("text", "")))
    if kind == TOOL_START:
        return ReplyToolStarted(
            name=str(payload.get("name", "")),
            summary=str(payload.get("summary", "")),
        )
    if kind == TOOL_END:
        return ReplyToolFinished(name=str(payload.get("name", "")), ok=bool(payload.get("ok")))

    logger.debug(f"Ignoring unknown agent stream payload: {payload!r}")
    return None
