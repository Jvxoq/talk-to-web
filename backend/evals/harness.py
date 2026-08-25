"""Builds a real container for one eval run, against real APIs.

Mirrors `app.composition.lifespan` rather than importing from it: importing
the real lifespan would drag in the FastAPI app object and its Postgres
engine, neither of which an eval run has any use for. This module is the
`evals/` package's own composition root - the one place it constructs a
client - built to the same shape for the same reason `app.composition` is:
so a run is exactly the collaborators production would build, not a
simplified stand-in that could hide a real regression.

No Postgres, on purpose. `GenerateReply` reads exactly one thing through a
`UnitOfWork` on the reply path - the list of documents this owner has uploaded
- so that list is served from memory here, and fixtures are written straight to
the vector store rather than through `IndexDocument` (which also writes a
bookkeeping row this harness has no database for).

That in-memory list is not a formality, and getting it wrong is how this
harness spent a while measuring the opposite of what it claimed. An earlier
version answered "no documents" to every request, on the reasoning that an
eval case uploads nothing. But `index_fixtures` *does* upload - into Qdrant,
under this run's owner - and `ToolRoutingPolicy` refuses `retrieve_documents`
outright for an account it believes is empty. So every retrieval case in every
suite was being graded on a run where the retrieval tool could not fire, and
the resulting "the model searched the web instead" failures were the harness's
doing. The shelf below is what keeps the two halves telling the same story:
whatever is indexed is what the reply is told exists.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp
from google import genai
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
from qdrant_client import AsyncQdrantClient
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from app.adapters.embedding.gemini_embedder import GeminiEmbedder
from app.adapters.llm.approximate_token_counter import ApproximateTokenCounter
from app.adapters.llm.langchain_chat_model import LangChainChatModel
from app.adapters.observability.langfuse_tracer import LangfuseTracer
from app.adapters.observability.null_tracer import NullTracer
from app.adapters.vector.knowledge_retriever import EmbeddedKnowledgeRetriever
from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.adapters.web.tavily_search import TavilyWebSearcher
from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.dto import (
    GenerateReplyInput,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyUsage,
)
from app.application.chat.guardrails.policy import InputGuardPolicy
from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.ports import Tracer
from app.application.chat.tools.base import ToolRegistry, ToolRoutingPolicy
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.domain.ingestion.entities import Document, UploadedDocument
from app.domain.ingestion.value_objects import Chunk, DocumentName
from app.settings import Settings
from evals.cases import EvalCase, Owner
from evals.judge import Judge

EVAL_OWNER_ID = 999_999_001
"""The account every fixture is indexed under, and the one almost every case
runs as. `retrieve_documents` filters by owner - see
`app.application.chat.ports.KnowledgeRetriever` - so this is what lets a run
see its own fixtures at all."""

EMPTY_OWNER_ID = 999_999_002
"""An account with nothing indexed and no document rows, for cases declaring
`owner: "empty"`.

A separate id rather than a separate run, because the two states must be
reachable from one invocation: `ToolRoutingPolicy` reads them in opposite
directions - refusing `retrieve_documents` here while *releasing* `search_web`
that it would otherwise hold back - and a suite that could only build one of
them was testing one arm of a two-armed gate."""

_OWNER_IDS: dict[Owner, int] = {"with_documents": EVAL_OWNER_ID, "empty": EMPTY_OWNER_ID}

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

_RATE_LIMIT_MARKER = "busy right now"
"""Substring of `GenerateReply`'s own friendly rate-limit message - see
`_friendly_error` in `app.application.chat.use_cases.generate_reply`. A
provider 429 never reaches this harness as a raised exception: `GenerateReply`
catches it inside the open stream and yields `ReplyFailed` with this text
instead, so a retry has to recognise the text, not an exception type."""

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5.0

_DIGEST_SUMMARY_CHARS = 400
"""How much of a fixture stands in for the condenser-written summary.

Production writes this field with the `Condenser` at index time. Here it is
the file's opening paragraph instead, and the trade is deliberate: a
model-written digest would differ between two runs of the same suite, and a
routing metric that moves because a summary was phrased differently is a
metric that cannot be compared week to week. The shape the model sees - a
fenced list of `name: summary` lines - is identical either way, and that shape
is what the routing decision reads."""

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

EVAL_CONVERSATION_ID = 1
"""The one thread every eval fixture is filed under.

Production scopes a search to the thread the file was attached to. An eval run
has one shelf and many cases, so the harness pins that scope rather than
dropping it - see `_PinnedConversationIndex`."""

_case_threads = count(start=1000)
"""Thread ids handed out one per case, so no case remembers another's turns.

Deliberately not `EVAL_CONVERSATION_ID`: memory and the document shelf are two
different scopes here, and collapsing them would let case order change an
answer."""


def owner_id_for(owner: Owner) -> int:
    """The account id a case's declared owner runs as."""
    return _OWNER_IDS[owner]


class AlwaysAllowRateLimiter:
    """A `RateLimiter` that never refuses.

    Satisfies `app.application.chat.ports.RateLimiter`. The production
    `SlidingWindowRateLimiter` is 3 chat requests per 60 seconds - the right
    ceiling for one signed-in person, and enough to make a 25-case eval suite
    take nine minutes waiting on a throttle that exists to stop a different
    problem than the one this run has. An eval run is a fixed, known workload
    run deliberately by whoever pays for the API keys; the budget it needs is
    exactly as many calls as it was asked to make.
    """

    async def hit(self, key: str) -> None:
        return None

    async def reset(self, key: str) -> None:
        return None


@dataclass(slots=True)
class CaseRun:
    """What one case produced, ready for `evals.metrics` to score."""

    case: EvalCase
    events: list[ReplyEvent] = field(default_factory=list)
    usage: ReplyUsage | None = None
    failed: ReplyFailed | None = None
    latency_ms: float = 0.0
    # Which sample of this case this is, under `--repeat`. 1 unless the driver
    # says otherwise. A model at temperature 0 still varies between calls, so
    # one sample cannot tell a regression from a coin flip - see
    # `evals.__main__._verdicts`, which is what reads this.
    sample: int = 1
    # How many times the case was run before this result stood. 1 is the
    # normal case; anything higher means a provider rate limit or a transient
    # error was retried, and is worth seeing in the report rather than being
    # smoothed away.
    attempts: int = 1
    # Set only if the call raised outright - a bug in the harness or the use
    # case, not a modelled failure. A modelled failure (a dead provider, a
    # rate limit) comes back as `failed` instead, because `GenerateReply`
    # itself never raises those; see its own docstring.
    error: str | None = None

    @property
    def answer(self) -> str:
        return "".join(event.text for event in self.events if isinstance(event, ReplyDelta))

    @property
    def broke(self) -> bool:
        """Did this case fail for a reason that is not the model's answer?

        The two are worth separating everywhere downstream: a rate limit is
        something to re-run, a wrong tool choice is something to fix.
        """
        return self.error is not None or self.failed is not None


class _Shelf:
    """What each eval account has "uploaded", as `GenerateReply` sees it.

    Mutable and shared, because the two halves of a fixture are written at
    different times: the vector store gets the chunks in `index_fixtures`, and
    the reply path needs the matching list of names and summaries on every
    request afterwards. One object holds both so they cannot disagree.
    """

    def __init__(self) -> None:
        self.documents: dict[int, list[UploadedDocument]] = {}

    def put(self, owner_id: int, documents: list[UploadedDocument]) -> None:
        self.documents[owner_id] = documents

    def get(self, owner_id: int) -> list[UploadedDocument]:
        return list(self.documents.get(owner_id, []))


class _ShelfDocumentRepository:
    """The half of `DocumentRepository` a reply actually reads: the list.

    Both lists answer from the same shelf, and the conversation is ignored on
    purpose. In production a document belongs to one thread; here the shelf is
    the whole point of the run and every case must see it, while each case
    still gets a thread of its own so no case remembers another's turns.
    """

    def __init__(self, shelf: _Shelf) -> None:
        self._shelf = shelf

    async def list_by_owner(self, owner_id: int) -> list[UploadedDocument]:
        return self._shelf.get(owner_id)

    async def list_by_conversation(
        self, owner_id: int, conversation_id: int
    ) -> list[UploadedDocument]:
        return self._shelf.get(owner_id)


class _PinnedConversationIndex:
    """A `VectorIndex` that files and reads every passage under one thread.

    The eval store holds one shelf per owner, and every case must be able to
    reach it - but each case runs in a thread of its own so nothing it says is
    remembered by the next. Those two facts pull against the production rule
    that a search only sees the thread it was asked in, so the rule is pinned
    here rather than bent: the wrapper supplies the conversation on both sides,
    and the real filter still runs underneath.
    """

    def __init__(self, index: QdrantVectorIndex, conversation_id: int) -> None:
        self._index = index
        self._conversation_id = conversation_id

    async def ensure(self, dimensions: int) -> None:
        await self._index.ensure(dimensions)

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        owner_id: int,
        document_id: int,
        conversation_id: int | None = None,
    ) -> None:
        await self._index.upsert(chunks, vectors, owner_id, document_id, self._conversation_id)

    async def search(
        self,
        vector: list[float],
        limit: int,
        score_threshold: float,
        owner_id: int,
        conversation_id: int | None = None,
    ) -> list[Chunk]:
        return await self._index.search(
            vector, limit, score_threshold, owner_id, self._conversation_id
        )

    async def delete_document(self, document_id: int, owner_id: int) -> None:
        await self._index.delete_document(document_id, owner_id)


class _ShelfUnitOfWork:
    """A unit of work that only knows how to list documents.

    Structural, like every other port in this codebase: `GenerateReply` reaches
    for `uow.documents` and nothing else on the reply path, so nothing else is
    implemented. Anything that did reach further would fail loudly here rather
    than silently reading an empty stand-in.
    """

    def __init__(self, shelf: _Shelf) -> None:
        self.documents = _ShelfDocumentRepository(shelf)

    async def __aenter__(self) -> "_ShelfUnitOfWork":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


@dataclass(slots=True)
class EvalHarness:
    """Every collaborator one eval run needs, built once and shared by every case."""

    generate_reply: GenerateReply
    judge: Judge
    embedder: GeminiEmbedder
    vector_index: QdrantVectorIndex
    model: str
    # The prompt the agent under test is actually running with, so a leak probe
    # can be derived from it instead of pinned as a literal in a dataset file.
    # Pinned, it would go stale the first time `LLM_SYSTEM_PROMPT` was set in an
    # environment - and go stale silently, still passing, still claiming to
    # prove the prompt does not leak.
    system_prompt: str
    # How many distinct source documents a retrieval metric grades. Not
    # `retrieval_limit`: that counts passages, and several passages routinely
    # come from one file, which `RetrieveDocuments` deduplicates before it
    # cites anything.
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    embedding_dimensions: int
    _shelf: _Shelf

    async def index_fixtures(self) -> int:
        """Chunk, embed and upsert every file in `evals/fixtures/`, then record
        what was indexed so the reply path can be told the same thing.

        Bypasses `IndexDocument` on purpose: that use case also writes a
        bookkeeping row through a `UnitOfWork`, and this harness runs with no
        Postgres at all - see the module docstring. It does not bypass
        production's chunking, which it used to: the sizes come off `Settings`,
        so a retrieval score here is measured over the same windows a real
        upload produces. Hard-coded 500/50 windows meant this suite could pass
        while production's 1400/200 ones failed, or the reverse.
        """
        await self.vector_index.ensure(self.embedding_dimensions)
        if not _FIXTURES_DIR.is_dir():
            logger.warning(
                f"No fixtures directory at {_FIXTURES_DIR}; rag suite will retrieve nothing"
            )
            return 0

        total_chunks = 0
        shelf: list[UploadedDocument] = []
        for document_id, fixture_path in enumerate(sorted(_fixture_files()), start=1):
            document = Document(
                name=DocumentName(fixture_path.name),
                content=fixture_path.read_text(encoding="utf-8"),
            )
            chunks = list(document.chunks(self.chunk_size, self.chunk_overlap))
            if not chunks:
                continue
            vectors = [await self.embedder.embed(chunk.text) for chunk in chunks]
            await self.vector_index.upsert(
                chunks, vectors, EVAL_OWNER_ID, document_id, EVAL_CONVERSATION_ID
            )
            total_chunks += len(chunks)
            shelf.append(
                UploadedDocument(
                    name=fixture_path.name,
                    reference=fixture_path.name,
                    owner_id=EVAL_OWNER_ID,
                    id=document_id,
                    chunks_indexed=len(chunks),
                    summary=_opening_paragraph(document.content),
                )
            )

        self._shelf.put(EVAL_OWNER_ID, shelf)
        # Written explicitly rather than left absent. `GenerateReply` treats a
        # read that never answered as "unknown" and leaves the document tool
        # open; the empty account has to be a read that answered "none", which
        # is the state the gate actually closes on.
        self._shelf.put(EMPTY_OWNER_ID, [])
        logger.info(f"Indexed {total_chunks} chunk(s) from {len(shelf)} fixture(s)")
        return total_chunks

    async def run_case(self, case: EvalCase, *, temperature: float = 0.0) -> CaseRun:
        """Run one case to completion, retrying a transient failure.

        Retries cover a provider rate limit *and* an outright exception. The
        old version retried only the former, which meant one dropped
        connection turned into a permanent red case in a report that costs
        money to regenerate. Neither is a statement about the model, so
        neither should end the run.
        """
        run = await self._run_once(case, temperature=temperature)
        for attempt in range(2, _MAX_ATTEMPTS + 1):
            if not _worth_retrying(run):
                break
            logger.warning(f"Case {case.id!r} failed transiently; attempt {attempt}")
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt - 1))
            run = await self._run_once(case, temperature=temperature)
            run.attempts = attempt
        return run

    async def _run_once(self, case: EvalCase, *, temperature: float) -> CaseRun:
        started = time.perf_counter()
        run = CaseRun(case=case)
        try:
            events = await self.generate_reply(
                GenerateReplyInput(
                    model=self.model,
                    user_input=case.input,
                    owner_id=owner_id_for(case.owner),
                    temperature=temperature,
                    # Every case is its own thread: cross-case state would make
                    # one case's answer depend on run order, which is exactly
                    # what an eval must not do. A number rather than `None`,
                    # because a turn with no thread owns no documents and the
                    # rag suite would then have nothing to retrieve.
                    conversation_id=next(_case_threads),
                )
            )
            async for event in events:
                run.events.append(event)
                if isinstance(event, ReplyUsage):
                    run.usage = event
                elif isinstance(event, ReplyFailed):
                    run.failed = event
        except Exception as error:
            run.error = str(error)
        run.latency_ms = (time.perf_counter() - started) * 1000
        return run


def fixture_text(name: str) -> str | None:
    """One fixture's raw text, or `None` if it is not there.

    Read by the scorer to give the judge the source text an answer was
    supposed to be drawn from. `None` rather than an empty string, so a
    missing file is a reportable problem instead of a groundedness score
    quietly computed against nothing.
    """
    path = _FIXTURES_DIR / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _fixture_files() -> Iterator[Path]:
    return (path for path in sorted(_FIXTURES_DIR.iterdir()) if path.is_file())


def _opening_paragraph(content: str) -> str:
    """The first paragraph of a fixture, standing in for a written summary."""
    for block in content.split("\n\n"):
        collapsed = " ".join(block.split())
        # Skip a bare title line: it is a heading, not a description, and a
        # digest entry reading "- aurora_robotics.txt: Aurora Robotics
        # Overview" tells the model nothing it did not get from the filename.
        if len(collapsed) > 40:
            return collapsed[:_DIGEST_SUMMARY_CHARS]
    return ""


def _worth_retrying(run: CaseRun) -> bool:
    if run.error is not None:
        return True
    return run.failed is not None and _RATE_LIMIT_MARKER in run.failed.detail


def _require_local_qdrant(url: str) -> None:
    """Refuse to run against anything but a local Qdrant.

    Qdrant Cloud's free tier is 1 GB, and a scratch `eval_<run_id>` collection
    per run is exactly the kind of thing that fills it over a few afternoons
    of eval iteration. This is a hard stop, not a warning: nothing downstream
    should be trusted to remember to pass `--local` or equivalent.
    """
    host = urlsplit(url).hostname or ""
    if host not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing to run evals against non-local Qdrant at {url!r}. Point "
            "QDRANT_URL at a local instance first (docker compose up -d qdrant)."
        )


@asynccontextmanager
async def build_harness(
    *, run_id: str | None = None, model: str | None = None
) -> AsyncIterator[EvalHarness]:
    """Build one real container for a run, and tear every client down after -
    the scratch Qdrant collection included, even when the run is interrupted.

    Settings are read directly from the environment/`.env`, the same way
    `app.settings.get_settings` does, but without its `lru_cache`: a fresh
    `Settings()` is fine here, since a CLI process builds a container exactly
    once and exits.
    """
    settings = Settings()
    _require_local_qdrant(settings.qdrant_url)

    resolved_run_id = run_id or uuid4().hex[:12]
    collection = f"eval_{resolved_run_id}"
    answering_model = model or settings.llm_models[0]

    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
        check_compatibility=False,
    )
    genai_client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds)
    )
    tavily_client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    # Unset Langfuse keys, same as production, is the normal case: no key held
    # locally, nothing sent, `NullTracer` everywhere.
    tracer: Tracer = NullTracer()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        candidate = LangfuseTracer(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            capture_content=settings.langfuse_capture_content,
            flush_timeout_seconds=settings.langfuse_flush_timeout_seconds,
        )
        if await candidate.credentials_valid():
            tracer = candidate
            logger.info("Langfuse tracing enabled for this eval run")

    chat_model = LangChainChatModel(
        provider=settings.llm_provider,
        # `answering_model` first, and included at all: `--model` names a model
        # that is very often not in `LLM_MODELS`, which is the entire point of
        # the flag. Left out, the registry has no client for it and every case
        # in the run fails on the first call with an unknown-model error.
        models=list(
            dict.fromkeys([answering_model, *settings.llm_models, settings.agent_condenser_model])
        ),
        api_key=settings.llm_api_key.get_secret_value(),
        max_tokens=settings.llm_max_tokens,
    )
    token_counter = ApproximateTokenCounter()
    condenser = Condenser(
        model=chat_model,
        model_name=settings.agent_condenser_model,
        max_chars=settings.agent_condenser_max_chars,
        tool_condense_prompt=settings.agent_tool_condense_prompt,
        summary_prompt=settings.agent_summary_prompt,
        document_summary_prompt=settings.agent_document_summary_prompt,
        tracer=tracer,
    )
    vector_index = QdrantVectorIndex(client=qdrant_client, collection=collection)
    embedder = GeminiEmbedder(
        client=genai_client,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    knowledge = EmbeddedKnowledgeRetriever(
        embedder=embedder,
        # Pinned, so a case's own thread id never decides which shelf it reads.
        index=_PinnedConversationIndex(vector_index, EVAL_CONVERSATION_ID),
        limit=settings.retrieval_limit,
        score_threshold=settings.retrieval_score_threshold,
    )
    web = AiohttpWebContentFetcher(session=http_session)
    searcher = TavilyWebSearcher(client=tavily_client)
    tool_output_guard = ToolOutputGuard(
        strip_instructions=settings.guardrail_strip_tool_instructions,
        max_scan_chars=settings.guardrail_max_scan_chars,
    )
    input_guard = InputGuardPolicy(
        redact_pii=settings.guardrail_pii_redaction_enabled,
        block_on_injection=settings.guardrail_block_on_injection,
        max_scan_chars=settings.guardrail_max_scan_chars,
    )
    limiter = AlwaysAllowRateLimiter()
    checkpointer = InMemorySaver()
    shelf = _Shelf()

    graph = build_agent_graph(
        model=chat_model,
        tools=ToolRegistry(
            [
                RetrieveDocuments(knowledge),
                FetchWebPages(web),
                SearchWeb(searcher, max_results=settings.tavily_max_results),
            ],
            guard=tool_output_guard,
            # Without this, a case scores on the model's unassisted tool
            # choice - a different, easier-to-fail question than the one
            # production actually answers, where `composition.py` wires this
            # same gate in front of every request. Mirrored here so `tools`
            # eval numbers mean what they claim to mean.
            routing=ToolRoutingPolicy(
                document_tool=RetrieveDocuments.name,
                web_search_tool=SearchWeb.name,
            ),
        ),
        max_iterations=settings.agent_max_tool_iterations,
        checkpointer=checkpointer,
        condenser=condenser,
        counter=token_counter,
        history_token_budget=settings.agent_history_token_budget,
        recent_token_budget=settings.agent_recent_token_budget,
        tool_output_token_budget=settings.agent_tool_output_token_budget,
        max_request_tokens=settings.agent_max_request_tokens,
        tracer=tracer,
    )

    generate_reply = GenerateReply(
        graph=graph,
        system_prompt=settings.llm_system_prompt,
        max_iterations=settings.agent_max_tool_iterations,
        limiter=limiter,
        daily_budget=limiter,
        guards=input_guard,
        tracer=tracer,
        # Serves the same list `index_fixtures` wrote, per owner. Typed through
        # `cast` because the real port is a SQLAlchemy-backed unit of work and
        # this satisfies only the slice of it a reply touches - see
        # `_ShelfUnitOfWork`.
        uow_factory=cast(Any, lambda: _ShelfUnitOfWork(shelf)),
        tool_output_guard=tool_output_guard,
        max_digest_documents=settings.chat_digest_max_documents,
        max_digest_summary_chars=settings.chat_digest_max_summary_chars,
    )
    judge = Judge(model=chat_model, model_name=settings.agent_condenser_model, tracer=tracer)

    harness = EvalHarness(
        generate_reply=generate_reply,
        judge=judge,
        embedder=embedder,
        vector_index=vector_index,
        model=answering_model,
        system_prompt=settings.llm_system_prompt,
        retrieval_k=3,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_dimensions=settings.embedding_dimensions,
        _shelf=shelf,
    )

    try:
        yield harness
    finally:
        # Runs even on Ctrl-C: a `KeyboardInterrupt` raised inside the `async
        # with build_harness(...)` block propagates through this generator as
        # an exception, which is what drives execution into this `finally`.
        try:
            await qdrant_client.delete_collection(collection)
            logger.info(f"Dropped scratch collection {collection!r}")
        except Exception as error:
            logger.warning(f"Could not drop scratch collection {collection!r}: {error}")
        await tracer.flush()
        await chat_model.aclose()
        await vector_index.aclose()
        await web.aclose()
        await searcher.aclose()
        await http_session.close()
