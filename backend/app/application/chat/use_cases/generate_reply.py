"""Stream one agent reply, turning graph events into events the API can frame."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from loguru import logger

from app.application.chat.agent.graph import AgentGraph
from app.application.chat.agent.nodes import DELTA, TOOL_END, TOOL_START
from app.application.chat.agent.progress import SUMMARIZE
from app.application.chat.agent.state import AgentState
from app.application.chat.agent.usage import USAGE
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplySummarizing,
    ReplyToolFinished,
    ReplyToolStarted,
    ReplyUsage,
)
from app.application.chat.guardrails.policy import GuardVerdict, InputGuardPolicy
from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.models import ChatMessage, Source
from app.application.chat.ports import RateLimiter, Tracer
from app.application.chat.provider_errors import is_auth_failure
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.errors import UnsafeUserMessage
from app.domain.chat.tool_routing import is_document_scoped
from app.domain.chat.value_objects import UserMessage
from app.domain.ingestion.entities import UploadedDocument
from app.observability.metrics import emit_reply_metrics


class GenerateReply:
    """Answer the user, with context if the agent can get it and without if it cannot.

    Context is an enhancement, not a precondition. The outer guarantee is that
    once this stream has opened it ends with `ReplyFailed` or `ReplyCompleted`,
    never with an exception through a half-written body.

    The compiled graph is injected, never built: compiling per request would
    reconnect the checkpointer on every message.
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
        uow_factory: UnitOfWorkFactory,
        tool_output_guard: ToolOutputGuard,
        max_digest_documents: int,
        max_digest_summary_chars: int,
    ) -> None:
        self._graph = graph
        self._system_prompt = system_prompt
        self._limiter = limiter
        self._daily_budget = daily_budget
        self._guards = guards
        self._tracer = tracer
        self._uow_factory = uow_factory
        # A digest is written from an uploaded file, so it is external content
        # and gets the same fence every tool result gets.
        self._tool_output_guard = tool_output_guard
        self._max_digest_documents = max_digest_documents
        self._max_digest_summary_chars = max_digest_summary_chars
        # A backstop in the graph's own units: three nodes per lap, plus the
        # final answering turn. The router normally stops the loop.
        self._recursion_limit = 3 * max_iterations + 3

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        """Spend one of this user's requests, then hand back the stream.

        A plain coroutine, not an async generator: a generator body runs nothing
        until the first `__anext__`, by which point the 200 is on the wire and a
        refusal could only truncate it. Awaited here, it can still be a 429.
        """
        # One counter for the whole deployment. Checked first, because a
        # per-user budget only bounds one account and registration is open.
        await self._daily_budget.hit("global")

        # Keyed on the account, not the address: the bill follows the user.
        await self._limiter.hit(f"generate:{data.owner_id}")

        # Here for the same reason the limits are: the last moment a refusal
        # can still be an HTTP status. Also before anything downstream sees the
        # text, so a pasted secret is redacted once rather than archived by the
        # model, the checkpointer and the trace.
        verdict = self._guards.inspect(data.user_input)
        if verdict.action == "block":
            raise UnsafeUserMessage(", ".join(verdict.categories()))
        if verdict.findings:
            # Recorded even when nothing was blocked. With blocking off, this
            # line is the dataset for deciding whether it can be turned on.
            logger.info(
                "Guardrail {} on input for owner {}: {}",
                verdict.action,
                data.owner_id,
                ", ".join(verdict.categories()),
            )

        # One query for three things: the digest, the names the routing gate
        # matches on, and whether the document tool is worth offering. Outside
        # `_stream`, so a database hiccup is still an HTTP error.
        return self._stream(
            data,
            verdict,
            await self._owned_documents(data.owner_id, data.conversation_id),
        )

    def _digest_documents(self, documents: list[UploadedDocument]) -> list[UploadedDocument]:
        """The newest few - all that fit in front of the model each turn."""
        return documents[: self._max_digest_documents]

    async def _owned_documents(
        self, owner_id: int, conversation_id: int | None
    ) -> list[UploadedDocument] | None:
        """This thread's uploads, newest first, or `None` if they cannot be read.

        Scoped to the conversation, not the account: listing another chat's
        files would open the gate on documents this thread's search filters away.

        `None` rather than `[]` on failure, because the two are not the same to
        the gate. Collapsed together, one slow query would tell a user with a
        full shelf that they have none, and refuse the retrieval.
        """
        # No conversation owns no documents, and that is a fact rather than a
        # failed read: `[]`, not `None`, so the gate closes.
        if conversation_id is None:
            return []

        try:
            async with self._uow_factory() as uow:
                return await uow.documents.list_by_conversation(owner_id, conversation_id)
        except Exception as exc:
            logger.warning(
                f"Could not read documents for owner {owner_id} "
                f"in conversation {conversation_id}: {exc}"
            )
            return None

    async def _stream(
        self,
        data: GenerateReplyInput,
        verdict: GuardVerdict,
        known: list[UploadedDocument] | None,
    ) -> AsyncIterator[ReplyEvent]:
        # `None` means the read failed, not that the shelf is empty.
        documents = known or []
        # False only when a read that succeeded came back empty. Unknown leaves
        # the document tool open.
        has_documents = known is None or bool(known)
        # `Any` on purpose: naming langchain_core's `RunnableConfig` would put
        # LangChain in the application layer, which the import contract forbids.
        config: Any = {
            "configurable": {
                # The agent's memory key. No conversation means every request
                # is its own thread and nothing is remembered.
                #
                # The owner is part of the key, not decoration. A bare
                # conversation id let anyone resume anyone's thread by guessing
                # an integer, and the checkpointer replays the whole history.
                "thread_id": _thread_id(data.owner_id, data.conversation_id),
                "model": data.model,
                "temperature": data.temperature,
                # Read back by the tool node, for the document tools.
                "owner_id": data.owner_id,
                # The thread whose documents a retrieval may read. Next to
                # `owner_id` because neither is anything the model may name.
                "conversation_id": data.conversation_id,
                # Whether this turn is about files the user supplied - the
                # input to the tool routing policy.
                #
                # Decided here once, not read back out of the history: the
                # summarize node may replace that history, and a gate that
                # matches on nothing fails open. That direction is the quiet
                # one - a private question answered off the web, no log line.
                #
                # `verdict.text`, not `data.user_input`: the decision must be
                # made on the text the model will actually read.
                # Every owned document, not just the few the digest fits. That
                # cap is a token budget, and names cost no tokens.
                "document_scoped": is_document_scoped(
                    verdict.text, document_names=[document.name for document in documents]
                ),
                # Whether the document tool has anything to search. Read once
                # here; the tool node has no unit of work to look it up with.
                "has_documents": has_documents,
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
                        messages=await self._new_messages(
                            data,
                            verdict.text,
                            config,
                            self._digest_documents(documents),
                            known_empty=not has_documents,
                        ),
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

        Totalled, not per model: the split between the answering model and the
        condenser is a detail for the trace.
        """
        prompt_tokens = sum(tokens[0] for tokens in spend.values())
        completion_tokens = sum(tokens[1] for tokens in spend.values())
        return prompt_tokens, completion_tokens

    async def _new_messages(
        self,
        data: GenerateReplyInput,
        user_text: str,
        config: Any,
        documents: list[UploadedDocument],
        *,
        known_empty: bool,
    ) -> list[ChatMessage]:
        """Only what this turn adds - the checkpointer already holds the rest.

        `user_text` as an argument, not `data.user_input`: the guardrail may
        have rewritten it, and this makes the original unreachable by habit.
        """
        message = UserMessage(user_text)
        messages: list[ChatMessage] = []

        if await self._is_new_thread(config):
            # Appended once per thread, not once per turn: the checkpointer
            # replays the whole state, so re-sending the system prompt every
            # message would stack a fresh copy of it in the history.
            messages.append(ChatMessage(role="system", content=self._system_prompt))

        messages.append(
            ChatMessage(
                role="user",
                content=_user_content(message, self._documents_note(documents, known_empty)),
            )
        )
        return messages

    def _documents_note(self, documents: list[UploadedDocument], known_empty: bool) -> str:
        """What to tell the model about this user's shelf, or "" to say nothing.

        The pair to the routing gate in `ToolRoutingPolicy`. This informs the
        choice; the gate makes it binding when the model chooses wrongly.

        Both branches open with a bracketed tag the system prompt names, so the
        model checks for it rather than inferring it. The empty case is written
        out because silence is not the same message: the prompt's fallback rule
        reads an absent digest as "unknown, try anyway".
        """
        digest = self._digest(documents)
        if digest:
            return (
                "[DOCUMENTS AVAILABLE] The user has uploaded the documents listed "
                "below. This is a standing fact about this account, not a hint - "
                "check it against every question before answering or calling "
                "search_web. If the topic, entity or time period of the question "
                "matches one of these documents, call retrieve_documents first; "
                "only skip it when the question is unambiguously unrelated to all "
                "of them (small talk, math, a question about a public site the "
                "user just linked).\n" + digest
            )
        if known_empty:
            # Unfenced, and bracketed like the URL note: this is our own
            # sentence about a count from our own database, not a line written
            # from the contents of anybody's file. Nothing crossed a trust
            # boundary to get here, and fencing it would tell the model to
            # distrust the one fact on this turn it has no reason to.
            return (
                "[NO DOCUMENTS] The user has uploaded no documents. retrieve_documents "
                "has nothing to search - answer from what you already know, or use "
                "search_web."
            )
        return ""

    def _digest(self, documents: list[UploadedDocument]) -> str:
        """What the user has uploaded, as one fenced block, or "" if nothing.

        Without it the model knows a retrieval tool exists but never whether
        this person has anything in it, so their own file competes with a web
        search on equal terms.

        Fenced like tool output: a summary is written from an uploaded document,
        so an unfenced one would let a PDF issue instructions every turn. Rides
        on the user turn, because the system prompt is written once per thread
        and uploads arrive between turns.
        """
        if not documents:
            return ""

        lines: list[str] = []
        for document in documents:
            summary = " ".join(document.summary.split())
            if len(summary) > self._max_digest_summary_chars:
                summary = summary[: self._max_digest_summary_chars].rstrip() + "..."
            # A document still being indexed has no summary yet. It is listed
            # anyway - the name alone is enough for the model to know there is
            # something to search, and omitting it would make a file the user
            # just uploaded appear not to exist.
            lines.append(f"- {document.name}: {summary}" if summary else f"- {document.name}")

        fenced, _ = self._tool_output_guard.wrap(
            tool="uploaded_documents", content="\n".join(lines)
        )
        return fenced

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
        # A one-off turn remembers nothing, so it needs an unaddressable key.
        return uuid4().hex
    return f"{owner_id}:{conversation_id}"


def _user_content(message: UserMessage, documents_note: str = "") -> str:
    """The user's text, with their links and their uploaded files called out.

    The note is composed by `GenerateReply._documents_note`, not here: whether
    the user has documents, and how to say so, is a decision that needs the
    guard and the digest budget. This function only decides where it goes.
    """
    parts = [message.text]

    if documents_note:
        parts.append(documents_note)

    # Models routinely answer from memory about a URL instead of reading it.
    # Naming the addresses back is what makes `fetch_web_pages` get picked. On
    # the user turn, because these links belong to this message only.
    urls = message.urls()
    if urls:
        listed = ", ".join(urls)
        parts.append(f"[The user linked: {listed}. Read them with fetch_web_pages.]")

    return "\n\n".join(parts)


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

    if kind == SUMMARIZE:
        raw_status = str(payload.get("status", ""))
        status: Literal["start", "done"]
        if raw_status == "start":
            status = "start"
        elif raw_status == "done":
            status = "done"
        else:
            logger.debug(f"Ignoring summarization payload with no usable status: {payload!r}")
            return None
        after = payload.get("tokens_after")
        try:
            return ReplySummarizing(
                status=status,
                tokens_before=int(payload.get("tokens_before", 0)),
                # `None` stays `None`: it means "not known yet", and turning it
                # into 0 would tell the client the thread was shortened to nothing.
                tokens_after=None if after is None else int(after),
            )
        except (TypeError, ValueError):
            # Tolerant for the same reason `_record_usage` is: this runs on the
            # reply path, and a progress notice that cannot be parsed is worth
            # losing. The reply is not.
            logger.debug(f"Ignoring malformed summarization payload: {payload!r}")
            return None

    logger.debug(f"Ignoring unknown agent stream payload: {payload!r}")
    return None


def _friendly_error(detail: str) -> str:
    """Turn a provider failure into a sentence a person can read.

    The raw detail is the provider's JSON, which is noise in front of a user.
    A rate limit is transient - the right message is "try again in a moment",
    not a dump of the provider's error body. A rejected API key is worse than
    noise: the provider's message names the key and links to its dashboard, and
    the person reading it cannot fix it anyway. Anything else passes through
    unchanged.
    """
    lowered = detail.lower()
    if "rate_limit_exceeded" in lowered or "413" in detail or "429" in detail:
        return "The assistant is busy right now - please try again in a moment."
    if is_auth_failure(detail):
        return "The assistant is unavailable right now - please try again later."
    return detail
