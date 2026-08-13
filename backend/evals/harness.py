"""Builds a real container for one eval run, against real APIs.

Mirrors `app.composition.lifespan` rather than importing from it: importing
the real lifespan would drag in the FastAPI app object and its Postgres
engine, neither of which an eval run has any use for. This module is the
`evals/` package's own composition root - the one place it constructs a
client - built to the same shape for the same reason `app.composition` is:
so a run is exactly the collaborators production would build, not a
simplified stand-in that could hide a real regression.

No Postgres, on purpose. `GenerateReply` never touches a `UnitOfWork` - only
the conversation-history use cases do, and this harness never calls those -
so the checkpointer is LangGraph's in-memory `InMemorySaver` and fixtures are
written straight to the vector store rather than through `IndexDocument`
(which also writes a bookkeeping row through a `UnitOfWork` this harness has
no database for).
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
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
from app.application.chat.tools.base import ToolRegistry
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.domain.ingestion.entities import Document
from app.domain.ingestion.value_objects import DocumentName
from app.domain.usage.value_objects import CostBook, ModelPrice
from app.settings import Settings
from evals.cases import EvalCase
from evals.judge import Judge

EVAL_OWNER_ID = 999_999_001
"""The one owner every eval case runs as. Fixtures are indexed under this id,
and `retrieve_documents` filters by owner - see
`app.application.chat.ports.KnowledgeRetriever` - so this is what lets a run
see its own fixtures at all."""

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

_RATE_LIMIT_MARKER = "busy right now"
"""Substring of `GenerateReply`'s own friendly rate-limit message - see
`_friendly_error` in `app.application.chat.use_cases.generate_reply`. A
provider 429 never reaches this harness as a raised exception: `GenerateReply`
catches it inside the open stream and yields `ReplyFailed` with this text
instead, so a retry has to recognise the text, not an exception type."""

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


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
    # Set only if the call raised outright - a bug in the harness or the use
    # case, not a modelled failure. A modelled failure (a dead provider, a
    # rate limit) comes back as `failed` instead, because `GenerateReply`
    # itself never raises those; see its own docstring.
    error: str | None = None

    @property
    def answer(self) -> str:
        return "".join(event.text for event in self.events if isinstance(event, ReplyDelta))


@dataclass(slots=True)
class EvalHarness:
    """Every collaborator one eval run needs, built once and shared by every case."""

    generate_reply: GenerateReply
    judge: Judge
    embedder: GeminiEmbedder
    vector_index: QdrantVectorIndex
    model: str

    async def index_fixtures(
        self, *, chunk_size: int = 500, chunk_overlap: int = 50, dimensions: int = 768
    ) -> int:
        """Chunk, embed and upsert every file in `evals/fixtures/`.

        Bypasses `IndexDocument` on purpose: that use case also writes a
        bookkeeping row through a `UnitOfWork`, and this harness runs with no
        Postgres at all - see the module docstring. `retrieve_documents` only
        ever reads the vector store, so writing straight to it is the whole of
        what a fixture needs to be searchable.
        """
        await self.vector_index.ensure(dimensions)
        if not _FIXTURES_DIR.is_dir():
            logger.warning(
                f"No fixtures directory at {_FIXTURES_DIR}; rag suite will retrieve nothing"
            )
            return 0

        total_chunks = 0
        for document_id, fixture_path in enumerate(sorted(_FIXTURES_DIR.iterdir()), start=1):
            if not fixture_path.is_file():
                continue
            document = Document(
                name=DocumentName(fixture_path.name),
                content=fixture_path.read_text(encoding="utf-8"),
            )
            chunks = list(document.chunks(chunk_size, chunk_overlap))
            if not chunks:
                continue
            vectors = [await self.embedder.embed(chunk.text) for chunk in chunks]
            await self.vector_index.upsert(chunks, vectors, EVAL_OWNER_ID, document_id)
            total_chunks += len(chunks)

        logger.info(f"Indexed {total_chunks} chunk(s) from {_FIXTURES_DIR}")
        return total_chunks

    async def run_case(self, case: EvalCase, *, temperature: float = 0.0) -> CaseRun:
        """Run one case to completion, retrying once on a provider rate limit."""
        run = await self._run_once(case, temperature=temperature)
        if run.failed is not None and _RATE_LIMIT_MARKER in run.failed.detail:
            logger.warning(f"Case {case.id!r} hit a rate limit; retrying once")
            await asyncio.sleep(5.0)
            run = await self._run_once(case, temperature=temperature)
        return run

    async def _run_once(self, case: EvalCase, *, temperature: float) -> CaseRun:
        started = time.perf_counter()
        run = CaseRun(case=case)
        try:
            events = await self.generate_reply(
                GenerateReplyInput(
                    model=self.model,
                    user_input=case.input,
                    owner_id=EVAL_OWNER_ID,
                    temperature=temperature,
                    # Every case is its own thread: cross-case state would make
                    # one case's answer depend on run order, which is exactly
                    # what an eval must not do.
                    conversation_id=None,
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

    cost_book = CostBook(
        {
            name: ModelPrice(input_usd_per_million=prices[0], output_usd_per_million=prices[1])
            for name, prices in settings.model_prices_usd_per_million.items()
        }
    )

    chat_model = LangChainChatModel(
        provider=settings.llm_provider,
        models=list(dict.fromkeys([*settings.llm_models, settings.agent_condenser_model])),
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
        index=vector_index,
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

    graph = build_agent_graph(
        model=chat_model,
        tools=ToolRegistry(
            [
                RetrieveDocuments(knowledge),
                FetchWebPages(web),
                SearchWeb(searcher, max_results=settings.tavily_max_results),
            ],
            guard=tool_output_guard,
        ),
        max_iterations=settings.agent_max_tool_iterations,
        checkpointer=checkpointer,
        condenser=condenser,
        counter=token_counter,
        history_token_budget=settings.agent_history_token_budget,
        recent_token_budget=settings.agent_recent_token_budget,
        tool_output_token_budget=settings.agent_tool_output_token_budget,
        tracer=tracer,
    )

    generate_reply = GenerateReply(
        graph=graph,
        system_prompt=settings.llm_system_prompt,
        max_iterations=settings.agent_max_tool_iterations,
        limiter=limiter,
        daily_budget=limiter,
        guards=input_guard,
        cost_book=cost_book,
        tracer=tracer,
    )
    judge = Judge(model=chat_model, model_name=settings.agent_condenser_model, tracer=tracer)

    harness = EvalHarness(
        generate_reply=generate_reply,
        judge=judge,
        embedder=embedder,
        vector_index=vector_index,
        model=model or settings.llm_models[0],
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
