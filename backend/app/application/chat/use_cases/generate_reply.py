"""Stream one agent reply, turning graph events into events the API can frame."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from loguru import logger

from app.application.chat.agent.graph import AgentGraph
from app.application.chat.agent.nodes import DELTA, TOOL_END, TOOL_START
from app.application.chat.agent.state import AgentState
from app.application.chat.agent.usage import USAGE
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
    ReplyUsage,
)
from app.application.chat.guardrails.policy import GuardVerdict, InputGuardPolicy
from app.application.chat.models import ChatMessage, Source
from app.application.chat.ports import RateLimiter, Tracer
from app.domain.chat.errors import UnsafeUserMessage
from app.domain.chat.value_objects import UserMessage
from app.observability.metrics import emit_reply_metrics


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
        limiter: RateLimiter,
        daily_budget: RateLimiter,
        guards: InputGuardPolicy,
        tracer: Tracer,
    ) -> None:
        self._graph = graph
        self._system_prompt = system_prompt
        self._limiter = limiter
        self._daily_budget = daily_budget
        self._guards = guards
        self._tracer = tracer
        # A backstop below LangGraph's own default, in the same units the graph
        # counts in: one lap is now an agent node, a tool node and a summarize
        # node, and the +3 covers the final answering turn. The router is what
        # normally stops the loop; this only catches a graph miswired into a
        # cycle the router cannot see.
        self._recursion_limit = 3 * max_iterations + 3

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        """Spend one of this user's requests, then hand back the stream.

        A plain coroutine returning the generator, rather than an async
        generator itself, because the two do the check at different moments. An
        async generator body runs nothing until the first `__anext__` - by which
        point the router has already handed the iterator to `StreamingResponse`,
        the 200 and the headers are on the wire, and a `RateLimited` raised then
        can only truncate a response that already claimed to succeed. Awaited
        here, the refusal happens before a single byte is sent, and the error
        handler can answer with a real 429.
        """
        # One counter shared with every upload, URL ingestion and transcription
        # session in the deployment - see `Settings.global_daily_call_budget`.
        # Checked first: a per-user budget only bounds one account, and
        # registration has no CAPTCHA to stop someone spending it many times
        # over with fresh ones.
        await self._daily_budget.hit("global")

        # Keyed on the account, not the address: this limit exists because every
        # reply spends tokens at the model provider, and the bill follows the
        # signed-in user wherever they connect from.
        await self._limiter.hit(f"generate:{data.owner_id}")

        # Inspected here, for the same reason the limits are: this is the last
        # moment a refusal can still be an HTTP status. Once `_stream` is handed
        # to `StreamingResponse` the 200 is committed, and a guardrail that
        # fired then could only truncate a response that already claimed to
        # succeed.
        #
        # It also has to run before anything downstream sees the text. Whatever
        # this returns is what reaches the model, the checkpointer and the
        # trace - so a secret the user pasted is redacted once, here, instead of
        # being archived by three systems that each thought it was somebody
        # else's problem.
        verdict = self._guards.inspect(data.user_input)
        if verdict.action == "block":
            raise UnsafeUserMessage(", ".join(verdict.categories()))
        if verdict.findings:
            # Recorded even when nothing was blocked. With blocking off by
            # default, this line is the entire dataset for deciding whether it
            # can ever be turned on.
            logger.info(
                "Guardrail {} on input for owner {}: {}",
                verdict.action,
                data.owner_id,
                ", ".join(verdict.categories()),
            )

        return self._stream(data, verdict)

    async def _stream(
        self, data: GenerateReplyInput, verdict: GuardVerdict
    ) -> AsyncIterator[ReplyEvent]:
        # Typed `Any` on purpose. The concrete type is langchain_core's
        # `RunnableConfig`, and importing it to say so would put LangChain in the
        # application layer - which the import contract forbids, and for good
        # reason: LangChain is what the `ChatModel` port exists to keep out.
        config: Any = {
            "configurable": {
                # A conversation is the agent's memory key. Without one, every
                # request is its own thread and nothing is remembered - which is
                # exactly what a client that never opened a conversation wants.
                #
                # The owner is part of the key, not decoration. A bare
                # conversation id let anyone resume anyone's thread by guessing
                # an integer - the checkpointer replays the whole history, so
                # that handed over every message in it. Namespacing makes the
                # collision impossible without adding a database round trip to
                # a use case that has none: a stranger's id simply addresses an
                # empty thread of their own.
                "thread_id": _thread_id(data.owner_id, data.conversation_id),
                "model": data.model,
                "temperature": data.temperature,
                # Read back by the tool node, which passes it to the tools that
                # search the user's own documents.
                "owner_id": data.owner_id,
            },
            "recursion_limit": self._recursion_limit,
        }

        # Every model call this reply makes, keyed by the model that made it -
        # the answering model and the condenser both spend tokens, and a reply
        # that condensed twice really did spend them.
        spend: dict[str, list[int]] = {}
        tools_used: list[str] = []
        started = time.perf_counter()
        outcome = "completed"

        try:
            async with self._tracer.span(
                "chat.reply",
                user_id=data.owner_id,
                session_id=config["configurable"]["thread_id"],
                model=data.model,
                temperature=data.temperature,
                guardrail_action=verdict.action,
                guardrail_categories=list(verdict.categories()),
            ) as span:
                try:
                    state = AgentState(
                        messages=await self._new_messages(data, verdict.text, config),
                        # Reset per request. The count is a ceiling on this reply's tool
                        # laps; carried over from the checkpoint it would only ever climb.
                        iterations=0,
                    )

                    async for payload in self._graph.astream(state, config, stream_mode="custom"):
                        # Bookkeeping, not an event. Usage is intercepted here
                        # rather than translated by `_to_event` because it is an
                        # internal protocol between the nodes and this use case -
                        # the client is told the total once, at the end, not one
                        # model call at a time.
                        if _record_usage(payload, spend):
                            continue

                        event = _to_event(payload)
                        if event is None:
                            continue
                        if isinstance(event, ReplyToolStarted):
                            tools_used.append(event.name)
                        yield event
                except asyncio.CancelledError:
                    # The client hung up. Let it propagate: swallowing it leaves the
                    # graph running for a response nobody is reading.
                    outcome = "cancelled"
                    raise
                except Exception as exc:
                    # The HTTP response is already open, so raising here would only
                    # truncate the body. The client is told in-band instead.
                    outcome = "failed"
                    span.record_error(exc)
                    logger.warning(f"Agent run failed for model {data.model}: {exc}")
                    yield ReplyFailed(detail=_friendly_error(str(exc)))
                    return

                prompt_tokens, completion_tokens = self._total_tokens(spend)
                elapsed_ms = _elapsed_ms(started)
                span.set(
                    outcome=outcome,
                    tools_used=tools_used,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=elapsed_ms,
                )

                if spend:
                    # Skipped entirely when no provider reported usage. Sending
                    # zeros would be a claim nobody made.
                    yield ReplyUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_ms=elapsed_ms,
                        model=data.model,
                    )

                yield ReplyCompleted()
        finally:
            # In a `finally` so a cancelled reply is still counted. A user who
            # closed the tab halfway through still spent the tokens and the time.
            prompt_tokens, completion_tokens = self._total_tokens(spend)
            emit_reply_metrics(
                outcome=outcome,
                model=data.model,
                owner_id=data.owner_id,
                tools=tools_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                guardrail_action=verdict.action,
                guardrail_categories=list(verdict.categories()),
                latency_ms=_elapsed_ms(started),
            )

    def _total_tokens(self, spend: dict[str, list[int]]) -> tuple[int, int]:
        """Total the tokens every model call this reply made spent.

        Totalled rather than reported per model because the question a client
        asks is "what did this reply spend", and the split across the
        answering model and the condenser is a detail for the trace.
        """
        prompt_tokens = sum(tokens[0] for tokens in spend.values())
        completion_tokens = sum(tokens[1] for tokens in spend.values())
        return prompt_tokens, completion_tokens

    async def _new_messages(
        self,
        data: GenerateReplyInput,
        user_text: str,
        config: Any,
    ) -> list[ChatMessage]:
        """Only what this turn adds - the checkpointer already holds the rest.

        `user_text` rather than `data.user_input`, because by here the guardrail
        may have rewritten it. Taking it as an argument is what makes it
        impossible to reach for the unredacted original by habit.
        """
        message = UserMessage(user_text)
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


def _thread_id(owner_id: int, conversation_id: int | None) -> str:
    """The checkpointer's key for this turn, scoped to the person asking."""
    if conversation_id is None:
        # A one-off turn remembers nothing, so it needs a key nothing else can
        # collide with rather than a key anyone could address.
        return uuid4().hex
    return f"{owner_id}:{conversation_id}"


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


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _record_usage(payload: object, spend: dict[str, list[int]]) -> bool:
    """Add one usage payload to the running total. True if that is what it was.

    Returning a flag rather than an event, because usage is not something the
    client is told mid-reply: the model calls a reply makes are an implementation
    detail, and streaming five separate token counts would invite a frontend to
    add them up itself and disagree with the one number sent at the end.

    Tolerant of a malformed payload on purpose. This runs on the reply path, and
    a bookkeeping line that cannot be parsed is worth losing - the reply is not.
    """
    if not isinstance(payload, dict) or payload.get("type") != USAGE:
        return False

    try:
        model = str(payload["model"])
        prompt_tokens = int(payload.get("prompt_tokens", 0))
        completion_tokens = int(payload.get("completion_tokens", 0))
    except (KeyError, TypeError, ValueError):
        logger.debug(f"Ignoring malformed usage payload: {payload!r}")
        return True

    running = spend.setdefault(model, [0, 0])
    running[0] += prompt_tokens
    running[1] += completion_tokens
    return True


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
        raw_sources = payload.get("sources")
        sources = (
            tuple(Source(**raw) for raw in raw_sources if isinstance(raw, dict))
            if isinstance(raw_sources, list)
            else ()
        )
        return ReplyToolFinished(
            name=str(payload.get("name", "")), ok=bool(payload.get("ok")), sources=sources
        )

    logger.debug(f"Ignoring unknown agent stream payload: {payload!r}")
    return None


def _friendly_error(detail: str) -> str:
    """Turn a provider rate-limit failure into a sentence a person can read.

    The raw detail is Groq's JSON, which is noise in front of a user. A rate
    limit is transient - the right message is "try again in a moment", not a
    dump of the provider's error body. Anything else passes through unchanged.
    """
    lowered = detail.lower()
    if "rate_limit_exceeded" in lowered or "413" in detail or "429" in detail:
        return "The assistant is busy right now - please try again in a moment."
    return detail
