"""The composition root: the only module that constructs infrastructure.

Everywhere else receives its collaborators already built. That is what lets a
test swap the container for one holding fakes without patching a single import.
"""

import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import aiohttp
from fastapi import FastAPI
from google import genai
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from app.adapters.embedding.gemini_embedder import GeminiEmbedder
from app.adapters.extraction.pypdf_extractor import PypdfTextExtractor
from app.adapters.llm.langchain_chat_model import LangChainChatModel
from app.adapters.persistence.models import Base
from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.adapters.storage.local_file_storage import LocalFileStorage
from app.adapters.transcription.deepgram import DeepgramLiveTranscriber
from app.adapters.vector.knowledge_retriever import EmbeddedKnowledgeRetriever
from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.adapters.web.tavily_search import TavilyWebSearcher
from app.api.dependencies import Container
from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.tools.base import ToolRegistry
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream
from app.settings import Settings

# `postgresql+psycopg://` and friends. SQLAlchemy spells the driver into the
# scheme; libpq-based clients treat the whole scheme as the protocol name and
# reject anything that is not `postgresql://` (or `postgres://`).
_SQLALCHEMY_DRIVER = re.compile(r"^(postgresql|postgres)\+[a-z0-9_]+://")


def _plain_dsn(database_url: str) -> str:
    """Strip SQLAlchemy's `+driver` so psycopg can read the same URL.

    One `DATABASE_URL` configures two clients with different dialects: the
    SQLAlchemy engine wants `postgresql+psycopg://…`, while `AsyncPostgresSaver`
    hands its string straight to psycopg, which fails on the `+psycopg` suffix
    with an unhelpful "missing host". Rewriting here keeps that a one-line
    translation at the composition root instead of a second env var that can
    drift out of sync with the first.
    """
    return _SQLALCHEMY_DRIVER.sub(r"\1://", database_url, count=1)


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Every use case the API is allowed to invoke, and nothing else."""

    generate_reply: GenerateReply
    start_conversation: StartConversation
    get_conversation: GetConversation
    record_exchange: RecordExchange
    delete_conversation: DeleteConversation
    upload_document: UploadDocument
    index_document: IndexDocument
    transcribe_stream: TranscribeStream
    # Not a use case: the list of models this deployment will answer on. It is
    # here because the API has to reject an unknown model with a 4xx and has to
    # tell the frontend what to offer, and the API may not read settings. So
    # settings still enter at exactly one place and arrive everywhere else by
    # injection.
    chat_models: tuple[str, ...]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every client at startup, and close every one of them at shutdown."""
    settings: Settings = app.state.settings

    engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        # Without this, a connection killed by the database or a NAT idle timeout
        # is handed to a request and fails it. The ping costs a round trip.
        pool_pre_ping=True,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    if settings.environment == "local":
        # Convenience for `docker compose up` on a fresh volume, and deliberately
        # local-only: concurrent replicas racing to create schema is how schema
        # state gets corrupted. Anywhere else, migrations run as their own step.
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        check_compatibility=False,
    )
    genai_client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds),
    )
    tavily_client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    # --- adapters ---
    chat_model = LangChainChatModel(
        provider=settings.llm_provider,
        models=settings.llm_models,
        api_key=settings.llm_api_key.get_secret_value(),
        max_tokens=settings.llm_max_tokens,
    )
    vector_index = QdrantVectorIndex(client=qdrant_client, collection=settings.qdrant_collection)
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
    storage = LocalFileStorage(directory=settings.upload_dir)
    extractor = PypdfTextExtractor()
    transcriber = DeepgramLiveTranscriber(
        api_key=settings.deepgram_api_key.get_secret_value(),
        model=settings.deepgram_model,
        utterance_end_ms=settings.utterance_end_ms,
    )

    # A factory, not an instance: an AsyncSession is not safe to share between
    # concurrent requests, so every use case call gets its own unit of work.
    def uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    # `from_conn_string` is an async context manager, not a constructor: it owns
    # a psycopg pool that has to stay open for as long as the app serves
    # requests. An exit stack is what holds it open across the single `yield`
    # below and still unwinds it in the right order at shutdown.
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(_plain_dsn(settings.database_url))
        )
        # Deliberately NOT gated on `environment == "local"`, unlike the
        # `create_all` above. These tables belong to LangGraph, not to us: they
        # are not in our Alembic history, and `setup()` is the only supported way
        # to create and migrate them. Gating it would mean a fresh production
        # database has no checkpoint tables and every chat request fails. It is
        # safe to run on every boot - it is idempotent, tracks its own schema
        # version, and uses `CREATE TABLE IF NOT EXISTS`. Replicas racing it can
        # at worst collide on one deploy; if that ever bites, the fix is to run
        # `setup()` from the migration step, not to skip it here.
        await checkpointer.setup()

        # Compiled exactly once, at startup. Compiling per request would rebuild
        # every node and re-attach the checkpointer on every message the user
        # sends.
        graph = build_agent_graph(
            model=chat_model,
            # The one place a capability is added to the agent: a new tool is a
            # new line here and nothing anywhere else.
            tools=ToolRegistry(
                [
                    RetrieveDocuments(knowledge),
                    FetchWebPages(web),
                    SearchWeb(searcher, max_results=settings.tavily_max_results),
                ]
            ),
            max_iterations=settings.agent_max_tool_iterations,
            checkpointer=checkpointer,
        )

        # Annotated as the API's Protocol so mypy proves, here at the one place
        # that knows both sides, that the concrete container provides everything
        # the routes ask for.
        container: Container = AppContainer(
            generate_reply=GenerateReply(
                graph=graph,
                system_prompt=settings.llm_system_prompt,
                max_iterations=settings.agent_max_tool_iterations,
            ),
            start_conversation=StartConversation(uow_factory),
            get_conversation=GetConversation(uow_factory),
            record_exchange=RecordExchange(uow_factory),
            delete_conversation=DeleteConversation(uow_factory),
            upload_document=UploadDocument(storage=storage, max_bytes=settings.max_upload_bytes),
            index_document=IndexDocument(
                extractor=extractor,
                embedder=embedder,
                index=vector_index,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embedding_dimensions=settings.embedding_dimensions,
            ),
            transcribe_stream=TranscribeStream(
                transcriber=transcriber,
                finalize_timeout=settings.finalize_timeout_seconds,
            ),
            chat_models=tuple(settings.llm_models),
        )
        app.state.container = container

        logger.info(f"Application started in {settings.environment!r}")
        try:
            yield
        finally:
            # Without this block a redeploy leaks connections until the database
            # refuses new ones. The checkpointer's own pool is closed by the exit
            # stack unwinding immediately after.
            await chat_model.aclose()
            await vector_index.aclose()
            await web.aclose()
            await searcher.aclose()
            await engine.dispose()
            logger.info("Application shut down cleanly")
