"""The composition root: the only module that builds infrastructure.

Everything else gets collaborators already built, so tests can swap in fakes.
"""

import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import aiohttp
from fastapi import FastAPI
from google import genai
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from loguru import logger
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from app.adapters.concurrency.in_process_lock import InProcessDocumentLock
from app.adapters.embedding.gemini_embedder import GeminiEmbedder
from app.adapters.extraction.composite_extractor import CompositeTextExtractor
from app.adapters.extraction.docx_extractor import DocxTextExtractor
from app.adapters.extraction.plain_text_extractor import PlainTextExtractor
from app.adapters.extraction.pypdf_extractor import PypdfTextExtractor
from app.adapters.llm.approximate_token_counter import ApproximateTokenCounter
from app.adapters.llm.langchain_chat_model import LangChainChatModel
from app.adapters.observability.langfuse_tracer import LangfuseTracer
from app.adapters.observability.null_tracer import NullTracer
from app.adapters.persistence.readiness import PostgresProbe
from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.adapters.security.argon2_hasher import Argon2PasswordHasher
from app.adapters.security.jwt_codec import JwtAccessTokenCodec
from app.adapters.security.memory_rate_limiter import SlidingWindowRateLimiter
from app.adapters.security.refresh_tokens import Sha256RefreshTokenFactory
from app.adapters.storage.local_file_storage import LocalFileStorage
from app.adapters.time.system_clock import SystemClock
from app.adapters.transcription.deepgram import DeepgramLiveTranscriber
from app.adapters.vector.knowledge_retriever import EmbeddedKnowledgeRetriever
from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.adapters.vector.readiness import QdrantProbe
from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.adapters.web.tavily_search import TavilyWebSearcher
from app.api.dependencies import Container
from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.guardrails.policy import InputGuardPolicy
from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.models import ChatMessage
from app.application.chat.ports import Tracer
from app.application.chat.tools.base import ToolRegistry, ToolRoutingPolicy
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.list_conversations import ListConversations
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.application.health.use_cases.check_readiness import CheckReadiness
from app.application.identity.dto import RefreshCookiePolicy
from app.application.identity.sessions import SessionMinter
from app.application.identity.use_cases.authenticate_user import AuthenticateUser
from app.application.identity.use_cases.identify_request import IdentifyRequest
from app.application.identity.use_cases.refresh_session import RefreshSession
from app.application.identity.use_cases.register_user import RegisterUser
from app.application.identity.use_cases.revoke_session import RevokeSession
from app.application.ingestion.use_cases.delete_document import DeleteDocument
from app.application.ingestion.use_cases.get_conversation_document import GetConversationDocument
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.list_documents import ListDocuments
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream
from app.settings import Settings

# Matches `postgresql+psycopg://` and similar. libpq rejects the `+driver` part.
_SQLALCHEMY_DRIVER = re.compile(r"^(postgresql|postgres)\+[a-z0-9_]+://")


def _plain_dsn(database_url: str) -> str:
    """Strip SQLAlchemy's `+driver` so psycopg can read the same URL.

    Avoids a second env var that could drift from `DATABASE_URL`.
    """
    return _SQLALCHEMY_DRIVER.sub(r"\1://", database_url, count=1)


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Every use case the API is allowed to invoke, and nothing else."""

    generate_reply: GenerateReply
    start_conversation: StartConversation
    get_conversation: GetConversation
    list_conversations: ListConversations
    record_exchange: RecordExchange
    delete_conversation: DeleteConversation
    upload_document: UploadDocument
    index_document: IndexDocument
    list_documents: ListDocuments
    get_conversation_document: GetConversationDocument
    delete_document: DeleteDocument
    transcribe_stream: TranscribeStream
    register_user: RegisterUser
    authenticate_user: AuthenticateUser
    refresh_session: RefreshSession
    revoke_session: RevokeSession
    identify_request: IdentifyRequest
    check_readiness: CheckReadiness
    # The four fields below are config, not use cases. The API needs them
    # and may not read settings.
    refresh_cookie: RefreshCookiePolicy
    # Whether the forwarded client IP can be trusted as a rate-limit key.
    trust_forwarded_client_ip: bool
    # Models the API accepts and offers to the frontend.
    chat_models: tuple[str, ...]
    # Checked at the WebSocket handshake, before any use case runs.
    websocket_origins: tuple[str, ...]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every client at startup, and close every one of them at shutdown."""
    settings: Settings = app.state.settings

    engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        # Test a connection before use, so a dead one does not fail a request.
        pool_pre_ping=True,
        # Neon drops idle connections, so retire ours first.
        pool_recycle=300,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # No `create_all`. `alembic upgrade head` owns the schema.
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
        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds),
    )
    tavily_client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    # --- observability ---
    # Built first. Without keys, `NullTracer` sends nothing.
    tracer: Tracer = NullTracer()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        candidate = LangfuseTracer(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            capture_content=settings.langfuse_capture_content,
            flush_timeout_seconds=settings.langfuse_flush_timeout_seconds,
        )
        # Checked here, not on `/ready`, so a tracing outage never causes a 503.
        if await candidate.credentials_valid():
            tracer = candidate
            logger.info("Langfuse tracing enabled")
        else:
            logger.warning("Langfuse credentials rejected; tracing disabled")

    input_guard = InputGuardPolicy(
        redact_pii=settings.guardrail_pii_redaction_enabled,
        block_on_injection=settings.guardrail_block_on_injection,
        max_scan_chars=settings.guardrail_max_scan_chars,
    )
    tool_output_guard = ToolOutputGuard(
        strip_instructions=settings.guardrail_strip_tool_instructions,
        max_scan_chars=settings.guardrail_max_scan_chars,
    )

    # --- adapters ---
    # The condenser model shares this client but is not offered in the UI.
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
        document_summary_prompt=settings.agent_document_summary_prompt,
        tracer=tracer,
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
    web = AiohttpWebContentFetcher(
        session=http_session, max_page_chars=settings.fetch_web_max_page_chars
    )
    searcher = TavilyWebSearcher(client=tavily_client)
    storage = LocalFileStorage(directory=settings.upload_dir)
    plain_text_extractor = PlainTextExtractor()
    extractor = CompositeTextExtractor(
        {
            ".pdf": PypdfTextExtractor(),
            ".txt": plain_text_extractor,
            ".md": plain_text_extractor,
            ".docx": DocxTextExtractor(),
        }
    )
    clock = SystemClock()
    password_hasher = Argon2PasswordHasher()
    access_tokens = JwtAccessTokenCodec(
        secret=settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        ttl_seconds=settings.access_token_ttl_seconds,
    )
    refresh_token_factory = Sha256RefreshTokenFactory()
    # Shared by register, login and refresh. Keys are namespaced per route.
    auth_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.auth_rate_limit_attempts,
        window_seconds=settings.auth_rate_limit_window_seconds,
    )
    # One limiter per budget, since chat and upload have different limits.
    chat_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.chat_rate_limit_requests,
        window_seconds=settings.chat_rate_limit_window_seconds,
    )
    upload_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.upload_rate_limit_requests,
        window_seconds=settings.upload_rate_limit_window_seconds,
    )
    # One global key for all paid calls, so new accounts cannot get around it.
    daily_budget_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.global_daily_call_budget,
        window_seconds=settings.global_daily_call_budget_window_seconds,
    )
    transcriber = DeepgramLiveTranscriber(
        api_key=settings.deepgram_api_key.get_secret_value(),
        model=settings.deepgram_model,
        utterance_end_ms=settings.utterance_end_ms,
    )

    # Probes share the request pool, so an exhausted pool shows as not ready.
    check_readiness = CheckReadiness(
        probes=[PostgresProbe(engine), QdrantProbe(qdrant_client)],
        timeout_seconds=settings.readiness_timeout_seconds,
    )

    # A factory, because an AsyncSession cannot be shared across requests.
    def uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    session_minter = SessionMinter(
        access_tokens=access_tokens,
        refresh_tokens=refresh_token_factory,
        clock=clock,
        refresh_ttl_seconds=settings.refresh_token_ttl_seconds,
    )

    # A pool, because one raw connection serializes reads and dies for good on
    # a drop. `prepare_threshold=0` because Neon's pooler loses prepared statements.
    async with AsyncExitStack() as stack:
        checkpointer_pool = await stack.enter_async_context(
            AsyncConnectionPool[AsyncConnection[DictRow]](
                _plain_dsn(settings.database_url),
                min_size=settings.checkpointer_pool_min_size,
                max_size=settings.checkpointer_pool_max_size,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=False,
                # Same as the engine: ping before use, retire before Neon drops them.
                check=AsyncConnectionPool.check_connection,
                max_lifetime=300,
            )
        )
        # Allow-list `ChatMessage`, which future LangGraph versions will require.
        checkpointer = AsyncPostgresSaver(
            conn=checkpointer_pool,
            serde=JsonPlusSerializer(allowed_msgpack_modules=(ChatMessage,)),
        )
        # LangGraph's own tables, outside Alembic. Safe to run on every boot.
        await checkpointer.setup()

        # Compiled once at startup, not per request.
        graph = build_agent_graph(
            model=chat_model,
            # Add new agent tools here.
            tools=ToolRegistry(
                [
                    RetrieveDocuments(knowledge),
                    FetchWebPages(web, max_urls=settings.fetch_web_max_urls_per_call),
                    SearchWeb(searcher, max_results=settings.tavily_max_results),
                ],
                guard=tool_output_guard,
                # Named here, where the tools are chosen. The registry only enforces order.
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

        # One shared remover, so no caller forgets the vectors. One shared lock,
        # so a delete cannot race indexing. Two locks would serialize nothing.
        document_lock = InProcessDocumentLock()
        delete_document = DeleteDocument(
            uow_factory, index=vector_index, storage=storage, lock=document_lock
        )

        # Typed as the API's Protocol, so mypy checks the container is complete.
        container: Container = AppContainer(
            generate_reply=GenerateReply(
                graph=graph,
                system_prompt=settings.llm_system_prompt,
                max_iterations=settings.agent_max_tool_iterations,
                limiter=chat_limiter,
                daily_budget=daily_budget_limiter,
                guards=input_guard,
                tracer=tracer,
                uow_factory=uow_factory,
                # Same guard as tools, because a digest is external content too.
                tool_output_guard=tool_output_guard,
                max_digest_documents=settings.chat_digest_max_documents,
                max_digest_summary_chars=settings.chat_digest_max_summary_chars,
            ),
            start_conversation=StartConversation(
                uow_factory, max_per_owner=settings.max_conversations_per_user
            ),
            get_conversation=GetConversation(uow_factory),
            list_conversations=ListConversations(uow_factory),
            record_exchange=RecordExchange(uow_factory),
            # Both share `delete_document` with the delete route.
            delete_conversation=DeleteConversation(uow_factory, remove_document=delete_document),
            upload_document=UploadDocument(
                storage=storage,
                max_bytes=settings.max_upload_bytes,
                limiter=upload_limiter,
                uow_factory=uow_factory,
                daily_budget=daily_budget_limiter,
                remove_document=delete_document,
            ),
            index_document=IndexDocument(
                extractor=extractor,
                embedder=embedder,
                index=vector_index,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embedding_dimensions=settings.embedding_dimensions,
                uow_factory=uow_factory,
                # The condenser also fits ingestion's `DocumentSummarizer` port.
                summarizer=condenser,
                # Same lock as `delete_document`.
                lock=document_lock,
            ),
            list_documents=ListDocuments(uow_factory),
            get_conversation_document=GetConversationDocument(uow_factory),
            delete_document=delete_document,
            transcribe_stream=TranscribeStream(
                transcriber=transcriber,
                finalize_timeout=settings.finalize_timeout_seconds,
                daily_budget=daily_budget_limiter,
            ),
            register_user=RegisterUser(
                uow_factory,
                hasher=password_hasher,
                sessions=session_minter,
                limiter=auth_limiter,
            ),
            authenticate_user=AuthenticateUser(
                uow_factory,
                hasher=password_hasher,
                sessions=session_minter,
                limiter=auth_limiter,
            ),
            refresh_session=RefreshSession(
                uow_factory,
                refresh_tokens=refresh_token_factory,
                sessions=session_minter,
                clock=clock,
                limiter=auth_limiter,
            ),
            revoke_session=RevokeSession(
                uow_factory,
                refresh_tokens=refresh_token_factory,
                clock=clock,
            ),
            identify_request=IdentifyRequest(access_tokens),
            check_readiness=check_readiness,
            refresh_cookie=RefreshCookiePolicy(
                name=settings.refresh_cookie_name,
                path=settings.refresh_cookie_path,
                secure=settings.refresh_cookie_secure,
                samesite=settings.refresh_cookie_samesite,
                domain=settings.refresh_cookie_domain,
            ),
            trust_forwarded_client_ip=settings.trust_forwarded_client_ip,
            chat_models=tuple(settings.llm_models),
            websocket_origins=tuple(settings.allowed_websocket_origins),
        )
        app.state.container = container

        logger.info(f"Application started in {settings.environment!r}")
        try:
            yield
        finally:
            # Close everything, or redeploys leak connections. The exit stack
            # closes the checkpointer pool. Flush traces first.

            await tracer.flush()
            await chat_model.aclose()
            await vector_index.aclose()
            await web.aclose()
            await searcher.aclose()
            await engine.dispose()
            logger.info("Application shut down cleanly")
