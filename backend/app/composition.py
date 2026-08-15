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
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

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
from app.application.chat.ports import Tracer
from app.application.chat.tools.base import ToolRegistry
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
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.ingest_url import IngestUrl
from app.application.ingestion.use_cases.list_documents import ListDocuments
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
    list_conversations: ListConversations
    record_exchange: RecordExchange
    delete_conversation: DeleteConversation
    upload_document: UploadDocument
    ingest_url: IngestUrl
    index_document: IndexDocument
    list_documents: ListDocuments
    delete_document: DeleteDocument
    transcribe_stream: TranscribeStream
    register_user: RegisterUser
    authenticate_user: AuthenticateUser
    refresh_session: RefreshSession
    revoke_session: RevokeSession
    identify_request: IdentifyRequest
    check_readiness: CheckReadiness
    # Not a use case, and here for the same reason as the two below: writing the
    # refresh cookie is a delivery decision that depends on where this deployment
    # is served from, and the API may not read settings to find out.
    refresh_cookie: RefreshCookiePolicy
    # Whether the address on the connection came from a proxy this deployment
    # configured, and can therefore be used as a rate-limit key.
    trust_forwarded_client_ip: bool
    # Not a use case: the list of models this deployment will answer on. It is
    # here because the API has to reject an unknown model with a 4xx and has to
    # tell the frontend what to offer, and the API may not read settings. So
    # settings still enter at exactly one place and arrive everywhere else by
    # injection.
    chat_models: tuple[str, ...]
    # Also not a use case, and here for the same reason: rejecting a WebSocket
    # handshake from an unknown origin is a delivery concern that happens before
    # any use case is reached, and `TranscribeStream` must not learn that it is
    # being driven over a WebSocket at all.
    websocket_origins: tuple[str, ...]


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
        # Managed Postgres closes idle connections on its own schedule - Neon
        # both scales a compute to zero and reaps idle pooler connections - so a
        # long-lived pool fills up with sockets the server has already forgotten.
        # `pool_pre_ping` catches those and reconnects, at the cost of a failed
        # ping first; retiring connections before they get that old avoids paying
        # for the discovery at all.
        pool_recycle=300,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # No `create_all` here, in any environment. Schema is owned by
    # `alembic upgrade head`, which every compose file runs as its own service
    # before the backend starts. Creating tables from application startup means
    # concurrent replicas racing each other, and a schema that drifts from the
    # migration history nobody then trusts.
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
    # Built before anything that traces, because the condenser and the graph
    # both take it. Unset keys are the normal case: local runs and the whole
    # test suite get `NullTracer` and send nothing anywhere, the same way
    # `configure_sentry` no-ops with no DSN.
    tracer: Tracer = NullTracer()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        candidate = LangfuseTracer(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            capture_content=settings.langfuse_capture_content,
            flush_timeout_seconds=settings.langfuse_flush_timeout_seconds,
        )
        # Checked once, here, rather than probed on `/ready`. Readiness is
        # `all(...)` by design, so putting tracing behind it would let a Langfuse
        # outage return 503 and pull this app out of rotation over a dependency
        # that is deliberately not on the critical path. A typo in a key is the
        # only failure this check was ever really for, and one startup log line
        # catches it.
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
    # The condenser model is added to the same adapter so it shares one HTTP
    # client, but it is not offered in the UI: `chat_models` below stays the
    # user-selectable list. Deduplicated in case the condenser model is also a
    # chat model.
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
    # One limiter shared by register, login and refresh: the keys are namespaced
    # per route, so a shared instance is one budget per key, not one for all.
    auth_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.auth_rate_limit_attempts,
        window_seconds=settings.auth_rate_limit_window_seconds,
    )
    # One limiter per budget, not one shared instance with namespaced keys: a
    # chat request and an upload are counted against different ceilings over
    # different windows, and a limiter holds exactly one of each.
    chat_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.chat_rate_limit_requests,
        window_seconds=settings.chat_rate_limit_window_seconds,
    )
    upload_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.upload_rate_limit_requests,
        window_seconds=settings.upload_rate_limit_window_seconds,
    )
    # One instance, one key ("global"), shared by every use case that spends
    # money at a provider - chat replies, uploads, URL ingestion and
    # transcription alike. Unlike the limiters above, this one is not
    # namespaced per caller: the whole point is a ceiling nobody's account can
    # get around by signing up again.
    daily_budget_limiter = SlidingWindowRateLimiter(
        max_attempts=settings.global_daily_call_budget,
        window_seconds=settings.global_daily_call_budget_window_seconds,
    )
    transcriber = DeepgramLiveTranscriber(
        api_key=settings.deepgram_api_key.get_secret_value(),
        model=settings.deepgram_model,
        utterance_end_ms=settings.utterance_end_ms,
    )

    # Probes reuse the engine and the client the requests use, not connections of
    # their own: a readiness check against a private connection reports "up"
    # while an exhausted pool fails every real request behind it.
    check_readiness = CheckReadiness(
        probes=[PostgresProbe(engine), QdrantProbe(qdrant_client)],
        timeout_seconds=settings.readiness_timeout_seconds,
    )

    # A factory, not an instance: an AsyncSession is not safe to share between
    # concurrent requests, so every use case call gets its own unit of work.
    def uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    session_minter = SessionMinter(
        access_tokens=access_tokens,
        refresh_tokens=refresh_token_factory,
        clock=clock,
        refresh_ttl_seconds=settings.refresh_token_ttl_seconds,
    )

    # A pool, not `AsyncPostgresSaver.from_conn_string(...)`: that helper opens
    # exactly one raw connection, so every checkpoint read/write would serialize
    # on it and a single dropped connection would take the checkpointer down
    # until restart. `autocommit`/`prepare_threshold=0`/`dict_row` are the same
    # connection kwargs `from_conn_string` used — `prepare_threshold=0` matters
    # because `database_url` is Neon's transaction-mode pooler, which prepared
    # statements do not survive. An exit stack holds the pool open across the
    # single `yield` below and unwinds it in the right order at shutdown.
    async with AsyncExitStack() as stack:
        checkpointer_pool = await stack.enter_async_context(
            AsyncConnectionPool[AsyncConnection[DictRow]](
                _plain_dsn(settings.database_url),
                min_size=settings.checkpointer_pool_min_size,
                max_size=settings.checkpointer_pool_max_size,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=False,
                # Same Neon reality as the SQLAlchemy engine above, and this pool
                # needs its own defence against it: `check` runs a lightweight
                # ping on a connection before handing it to a caller, so one
                # Neon already killed is discovered and replaced here rather
                # than failing mid-checkpoint-read. `max_lifetime` mirrors
                # `pool_recycle` - connections are retired before they are old
                # enough to be reaped, matching the engine's 300 seconds.
                check=AsyncConnectionPool.check_connection,
                max_lifetime=300,
            )
        )
        checkpointer = AsyncPostgresSaver(conn=checkpointer_pool)
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
                    FetchWebPages(web, max_urls=settings.fetch_web_max_urls_per_call),
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

        # Annotated as the API's Protocol so mypy proves, here at the one place
        # that knows both sides, that the concrete container provides everything
        # the routes ask for.
        container: Container = AppContainer(
            generate_reply=GenerateReply(
                graph=graph,
                system_prompt=settings.llm_system_prompt,
                max_iterations=settings.agent_max_tool_iterations,
                limiter=chat_limiter,
                daily_budget=daily_budget_limiter,
                guards=input_guard,
                tracer=tracer,
            ),
            start_conversation=StartConversation(uow_factory),
            get_conversation=GetConversation(uow_factory),
            list_conversations=ListConversations(uow_factory),
            record_exchange=RecordExchange(uow_factory),
            delete_conversation=DeleteConversation(uow_factory),
            upload_document=UploadDocument(
                storage=storage,
                max_bytes=settings.max_upload_bytes,
                limiter=upload_limiter,
                uow_factory=uow_factory,
                daily_budget=daily_budget_limiter,
            ),
            ingest_url=IngestUrl(
                fetcher=web,
                limiter=upload_limiter,
                uow_factory=uow_factory,
                daily_budget=daily_budget_limiter,
            ),
            index_document=IndexDocument(
                extractor=extractor,
                embedder=embedder,
                index=vector_index,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embedding_dimensions=settings.embedding_dimensions,
                uow_factory=uow_factory,
            ),
            list_documents=ListDocuments(uow_factory),
            delete_document=DeleteDocument(uow_factory, index=vector_index, storage=storage),
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
            # Without this block a redeploy leaks connections until the database
            # refuses new ones. The checkpointer's own pool is closed by the exit
            # stack unwinding immediately after.
            # First, and under its own timeout inside the adapter: anything
            # still queued describes work this process just did, and after the
            # clients close there is nothing left worth tracing.
            await tracer.flush()
            await chat_model.aclose()
            await vector_index.aclose()
            await web.aclose()
            await searcher.aclose()
            await engine.dispose()
            logger.info("Application shut down cleanly")
