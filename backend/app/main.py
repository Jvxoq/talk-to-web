"""Builds the application. No routes, no handlers, no logic."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.v1.routers import (
    auth,
    chat,
    conversations,
    health,
    ingestion,
    models,
    transcription,
)
from app.composition import lifespan
from app.observability.context import REQUEST_ID_HEADER
from app.observability.logging import configure_logging
from app.observability.sentry import configure_sentry
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    A factory rather than a module-level app, so a test can build an instance
    with a substituted container instead of monkey-patching globals.
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.environment != "local")
    # Before the app exists: the SDK's integrations hook Starlette by patching
    # its classes when `init` runs, and anything already constructed misses them.
    configure_sentry(
        dsn=settings.sentry_dsn.get_secret_value() if settings.sentry_dsn else None,
        environment=settings.environment,
        release=settings.sentry_release,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )

    app = FastAPI(title="Talk to the Web", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        # Exact origins from settings. The old wildcard could never carry
        # credentials, and browsers reject the combination outright.
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        # `Retry-After` is not on the CORS safelist, so a cross-origin caller
        # cannot read it unless it is named here. A frontend pointed straight at
        # this host - rather than through the proxy rewrites - needs it to know
        # how long a 429 lasts. The request id is here for the same reason: a
        # correlation id the browser cannot read correlates nothing.
        expose_headers=["Retry-After", REQUEST_ID_HEADER],
    )

    # Added last, so it wraps everything above: Starlette builds the stack with
    # the most recently added middleware outermost. That is what puts an id on a
    # CORS preflight and on a response written by the error handlers, and what
    # makes the id available to every log line either of them emits.
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)

    for router in (
        auth.router,
        chat.router,
        models.router,
        ingestion.router,
        conversations.router,
        transcription.router,
    ):
        app.include_router(router)
    app.include_router(health.router)

    # Mounting a missing directory raises at startup, and this one is optional.
    if settings.static_pages_dir.is_dir():
        app.mount("/pages", StaticFiles(directory=settings.static_pages_dir), name="pages")

    return app


app = create_app()
