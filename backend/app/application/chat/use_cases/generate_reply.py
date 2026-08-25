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
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.errors import UnsafeUserMessage
from app.domain.chat.tool_routing import is_document_scoped
from app.domain.chat.value_objects import UserMessage
from app.domain.ingestion.entities import UploadedDocument
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
        # The same fence every tool result gets. A document digest is written
        # from an uploaded file, so it is external content by the same
        # definition, and letting it reach the model unfenced would be a hole
        # straight past the defence `ToolRegistry.invoke` exists to provide.
        self._tool_output_guard = tool_output_guard
        self._max_digest_documents = max_digest_documents
        self._max_digest_summary_chars = max_digest_summary_chars
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

        # Read once, before the stream opens, and used for three things: the
        # digest the model is shown, the names the routing gate matches on, and
        # whether the document tool is worth offering at all. One query, not
        # three, and outside `_stream` so a database hiccup here is still an
        # HTTP error rather than a truncated reply.
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

        Scoped to the conversation, not to the account. The digest is what tells
        the model which files exist, and listing files attached to a different
        chat would invite it to retrieve passages this thread cannot reach -
        the gate would open on a document the search then filters away.

        Swallowed on failure for the same reason the tools are: knowing what a
        user uploaded makes a better answer, and not knowing must never cost
        them the answer itself.

        `None` rather than `[]` on failure, and the distinction is the whole
        point of the return type. An empty list used to mean both "this account
        has uploaded nothing" and "the database did not answer", which was
        harmless while the only consumer was a digest that would simply go
        unwritten. It stopped being harmless the moment an empty list started
        *closing a gate*: collapsed together, one slow query would tell a user
        with a shelf of documents that they have none, and refuse the retrieval
        that would have found them. Callers that only want to read the list can
        still treat the two the same; the gate must not.
        """
        # A turn with no conversation owns no documents, and that is a fact
        # rather than a failed read: `[]`, not `None`, so the gate closes and
        # no retrieval is attempted.
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
        # `None` means the read failed, not that the shelf is empty - see
        # `_owned_documents`. Everything that only reads the list treats the two
        # alike; the one thing that closes a gate does not.
        documents = known or []
        # False only when a read that actually succeeded came back empty. An
        # unknown answer leaves the document tool open, which is what the
        # deployment did before this gate existed.
        has_documents = known is None or bool(known)
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
                # The thread whose documents a retrieval may read. Passed for
                # the same reason as `owner_id` and next to it: both narrow the
                # search to what this request is allowed to see, and neither is
                # anything the model may name.
                "conversation_id": data.conversation_id,
                # Whether this turn is about files the user supplied - the
                # input to the tool routing policy.
                #
                # Decided here, once, rather than read back out of the
                # checkpointed history by the tool node. The question does not
                # change between laps, so nothing is gained by recomputing it,
                # and the history is not a safe place to look it up: the
                # summarization node may replace it wholesale, and a run that
                # compressed the current user turn away would leave the node
                # matching on an older question - or on nothing - and the gate
                # simply would not fire. That failure direction is "search
                # allowed", which is the quiet one: no error, no log line, just
                # a private question answered off the web. Sent alongside
                # `owner_id` because it is the same kind of value - written by
                # us from the request, never by the model.
                #
                # `verdict.text`, not `data.user_input`: the guardrail may have
                # rewritten the message, and the routing decision must be made
                # on the text the model is actually going to read.
                # Every owned document is offered here, not just the few the
                # digest has room for. The cap on the digest is a token budget,
                # and names cost no tokens - holding a search back because the
                # user named their seventh-newest file is exactly as right as
                # doing it for their first.
                "document_scoped": is_document_scoped(
                    verdict.text, document_names=[document.name for document in documents]
                ),
                # Whether the document tool has anything to search. Decided
                # here for the same reason `document_scoped` is - it is a fact
                # about the request, read once, and the tool node has no unit
                # of work to look it up with even if it wanted to.
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
        documents: list[UploadedDocument],
        *,
        known_empty: bool,
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

        messages.append(
            ChatMessage(
                role="user",
                content=_user_content(message, self._documents_note(documents, known_empty)),
            )
        )
        return messages

    def _documents_note(self, documents: list[UploadedDocument], known_empty: bool) -> str:
        """What to tell the model about this user's shelf, or "" to say nothing.

        The pair to the routing gate in `ToolRoutingPolicy`, and neither half
        does the job alone. This one informs the choice: a model that can read
        the list decides for itself whether a retrieval is worth a lap. The
        gate makes it binding, for the turns where it decides wrongly anyway.

        Both branches open with a bracketed tag - `[DOCUMENTS AVAILABLE]` or
        `[NO DOCUMENTS]` - that the system prompt tells the model by name to
        look for and treat as decisive. A model guessing from a plain sentence
        buried in the turn can miss it; a named marker it was told to expect is
        something to check for, not infer. The two-tier design (a fact the
        model is told to notice, backed by a gate that does not ask its
        opinion) is deliberate: the notice is what changes a lucky heuristic
        into a checked one, and the gate is what still holds when the model
        does not check it.

        The empty case earns its tokens for the same reason. Silence is not
        the same message as "there is nothing here" - the system prompt's
        fallback rule (for the rare turn with no notice at all, see below) is
        to try a retrieval whenever a question names an entity it does not
        know, so an absent digest would read as "unknown, try anyway" and get
        tried regardless. Saying so outright is what stops that call being
        made at all, leaving the gate to catch only what the model tries
        despite being told.
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

        This is the piece that stops the agent guessing. Without it the model
        knows only that a `retrieve_documents` tool exists, never whether this
        person has anything in it worth searching - so a question about their
        own file competes on equal footing with a web search. Naming the files
        and what each is about turns that guess into a reading of the list.

        Fenced with the same guard as tool output, and for the same reason: a
        summary is written from an uploaded document, which is external content
        whoever wrote the document controls. Unfenced, a PDF could issue
        instructions here on every single turn.

        Rides on the user turn rather than the system prompt, exactly like the
        URL note below. The system prompt is written once per thread, and this
        changes as the user uploads - a thread that began before an upload would
        never learn the file existed.
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
        # A one-off turn remembers nothing, so it needs a key nothing else can
        # collide with rather than a key anyone could address.
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

    # Models routinely answer from memory about a URL they were handed instead
    # of reading it. Naming the addresses back at them, on the turn that carries
    # them, is what makes `fetch_web_pages` get picked. It rides on the user
    # turn rather than the system prompt because the system prompt is written
    # once per thread, and these links belong to this message only.
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
    """Turn a provider rate-limit failure into a sentence a person can read.

    The raw detail is the provider's JSON, which is noise in front of a user.
    A rate limit is transient - the right message is "try again in a moment",
    not a dump of the provider's error body. Anything else passes through
    unchanged.
    """
    lowered = detail.lower()
    if "rate_limit_exceeded" in lowered or "413" in detail or "429" in detail:
        return "The assistant is busy right now - please try again in a moment."
    return detail
