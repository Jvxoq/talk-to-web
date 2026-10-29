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
    """Answer the user, with context when available.

    Once open, the stream always ends with `ReplyFailed` or `ReplyCompleted`,
    never an exception. The graph is injected because compiling per request
    reconnects the checkpointer.
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
        # Digests come from uploaded files, so they get the tool-output fence.
        self._tool_output_guard = tool_output_guard
        self._max_digest_documents = max_digest_documents
        self._max_digest_summary_chars = max_digest_summary_chars
        # Backstop: three nodes per lap plus the final turn. The router normally stops first.
        self._recursion_limit = 3 * max_iterations + 3

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        """Spend one request, then return the stream.

        A coroutine, not a generator, so refusals still become a 429 before the
        200 is sent.
        """
        # Global budget first: registration is open, so per-user limits are not enough.
        await self._daily_budget.hit("global")

        # Keyed on the account, because the bill follows the user.
        await self._limiter.hit(f"generate:{data.owner_id}")

        # Guard before streaming, so a refusal is still an HTTP status and a
        # pasted secret is redacted before the model, checkpointer or trace see it.
        verdict = self._guards.inspect(data.user_input)
        if verdict.action == "block":
            raise UnsafeUserMessage(", ".join(verdict.categories()))
        if verdict.findings:
            # Logged even when not blocked, to judge whether blocking can be turned on.
            logger.info(
                "Guardrail {} on input for owner {}: {}",
                verdict.action,
                data.owner_id,
                ", ".join(verdict.categories()),
            )

        # One query feeds the digest, the routing gate and `has_documents`.
        # Outside `_stream`, so a database error is still an HTTP error.
        return self._stream(
            data,
            verdict,
            await self._owned_documents(data.owner_id, data.conversation_id),
        )

    def _digest_documents(self, documents: list[UploadedDocument]) -> list[UploadedDocument]:
        """The newest few that fit in the prompt."""
        return documents[: self._max_digest_documents]

    async def _owned_documents(
        self, owner_id: int, conversation_id: int | None
    ) -> list[UploadedDocument] | None:
        """This thread's uploads, newest first, or `None` if the read failed.

        `None` is not `[]`: treating a failed read as empty would refuse
        retrieval to a user who has documents.
        """
        # No conversation means no documents. That is a fact, so `[]`.
        if conversation_id is None:
            return []

        try:
            async with self._uow_factory() as uow:
                return await uow.documents.list_by_conversation(owner_id, conversation_id)
        except Exception as exc:  # noqa: BLE001
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
        documents = known or []
        # False only when a successful read came back empty. Unknown keeps the tool open.
        has_documents = known is None or bool(known)
        # `Any`, because naming `RunnableConfig` would import LangChain here.
        config: Any = {
            "configurable": {
                # The owner is in the key so nobody can resume a thread by guessing its id.
                "thread_id": _thread_id(data.owner_id, data.conversation_id),
                "model": data.model,
                "temperature": data.temperature,
                # Owner and thread come from the request, never from the model.
                "owner_id": data.owner_id,
                "conversation_id": data.conversation_id,
                # Decided once here, not from the history, because summarizing
                # can replace the history and the gate would fail open. Uses the
                # guarded text and every document name (names cost no tokens).
                "document_scoped": is_document_scoped(
                    verdict.text, document_names=[document.name for document in documents]
                ),
                # The tool node has no unit of work, so this is read here.
                "has_documents": has_documents,
            },
            "recursion_limit": self._recursion_limit,
        }

        # Tokens per model, including the condenser's calls.
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
                        # Reset per request, or the checkpointed count only climbs.
                        iterations=0,
                    )

                    async for payload in self._graph.astream(state, config, stream_mode="custom"):
                        # Usage is summed here and sent once at the end, not streamed.
                        if _record_usage(payload, spend):
                            continue

                        event = _to_event(payload)
                        if event is None:
                            continue
                        if isinstance(event, ReplyToolStarted):
                            tools_used.append(event.name)
                        yield event
                except asyncio.CancelledError:
                    # Client hung up. Re-raise so the graph stops.
                    outcome = "cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    # The response is already open, so report the error in-band.
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
                    # Skipped when no provider reported usage, rather than sending zeros.
                    yield ReplyUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_ms=elapsed_ms,
                        model=data.model,
                    )

                yield ReplyCompleted()
        finally:
            # In `finally`, so a cancelled reply still counts its spend.
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
        """Total prompt and completion tokens across all models."""
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
        """Only this turn's messages. The checkpointer holds the rest.

        Takes the guarded `user_text`, so the raw input cannot be used by mistake.
        """
        message = UserMessage(user_text)
        messages: list[ChatMessage] = []

        if await self._is_new_thread(config):
            # Once per thread. The checkpointer replays history, so more would stack copies.
            messages.append(ChatMessage(role="system", content=self._system_prompt))

        messages.append(
            ChatMessage(
                role="user",
                content=_user_content(message, self._documents_note(documents, known_empty)),
            )
        )
        return messages

    def _documents_note(self, documents: list[UploadedDocument], known_empty: bool) -> str:
        """The note about the user's documents, or "" for nothing.

        Informs the model. `ToolRoutingPolicy` enforces. Both notes start with a
        tag the system prompt names. The empty case is stated, because silence
        reads as "unknown, try retrieval anyway".
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
            # Unfenced: our own sentence about our own row count, not file content.
            return (
                "[NO DOCUMENTS] The user has uploaded no documents. retrieve_documents "
                "has nothing to search - answer from what you already know, or use "
                "search_web."
            )
        return ""

    def _digest(self, documents: list[UploadedDocument]) -> str:
        """The user's uploads as one fenced block, or "" if none.

        Fenced because summaries come from uploaded files. Sent on the user
        turn, because uploads arrive after the system prompt is written.
        """
        if not documents:
            return ""

        lines: list[str] = []
        for document in documents:
            summary = " ".join(document.summary.split())
            if len(summary) > self._max_digest_summary_chars:
                summary = summary[: self._max_digest_summary_chars].rstrip() + "..."
            # Still indexing means no summary yet. List the name anyway.
            lines.append(f"- {document.name}: {summary}" if summary else f"- {document.name}")

        fenced, _ = self._tool_output_guard.wrap(
            tool="uploaded_documents", content="\n".join(lines)
        )
        return fenced

    async def _is_new_thread(self, config: Any) -> bool:
        if self._graph.checkpointer is None:
            # Nothing is remembered, so every turn is new.
            return True
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception as exc:  # noqa: BLE001
            # Worst case is one duplicate system message, not a lost reply.
            logger.warning(f"Could not read agent state, treating thread as new: {exc}")
            return True
        return not snapshot.values.get("messages")


def _thread_id(owner_id: int, conversation_id: int | None) -> str:
    """The checkpointer's key for this turn, scoped to the person asking."""
    if conversation_id is None:
        # A one-off turn gets a random key nobody can resume.
        return uuid4().hex
    return f"{owner_id}:{conversation_id}"


def _user_content(message: UserMessage, documents_note: str = "") -> str:
    """The user's text, plus the documents note and any linked URLs."""
    parts = [message.text]

    if documents_note:
        parts.append(documents_note)

    # Naming the links back makes the model read them instead of guessing.
    urls = message.urls()
    if urls:
        listed = ", ".join(urls)
        parts.append(f"[The user linked: {listed}. Read them with fetch_web_pages.]")

    return "\n\n".join(parts)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _record_usage(payload: object, spend: dict[str, list[int]]) -> bool:
    """Add a usage payload to the total. Returns True if it was one.

    A malformed payload is dropped, never allowed to break the reply.
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
                # `None` means "not known yet". 0 would be wrong.
                tokens_after=None if after is None else int(after),
            )
        except (TypeError, ValueError):
            # Drop a malformed notice rather than break the reply.
            logger.debug(f"Ignoring malformed summarization payload: {payload!r}")
            return None

    logger.debug(f"Ignoring unknown agent stream payload: {payload!r}")
    return None


def _friendly_error(detail: str) -> str:
    """Turn a provider failure into a readable sentence.

    Rate limits and auth failures get our own text. Anything else passes through.
    """

    lowered = detail.lower()
    if "rate_limit_exceeded" in lowered or "413" in detail or "429" in detail:
        return "The assistant is busy right now - please try again in a moment."
    if is_auth_failure(detail):
        return "The assistant is unavailable right now - please try again later."
    return detail
