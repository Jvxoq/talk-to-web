"""Builds the application.

Importing this has no side effects, so tests and CI can call `create_app`
without an environment. The eager `app = create_app()` is in `app.main`.
"""

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
    """Build the app. A factory, so tests can swap the container without patching."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.environment != "local")
    # Before the app exists, because the SDK patches Starlette at init.
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
        # Exact origins, because a wildcard cannot carry credentials.
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        # Not CORS-safelisted, so cross-origin callers need them exposed.
        expose_headers=["Retry-After", REQUEST_ID_HEADER],
    )

    # Added last, so it is outermost and tags preflights and error responses too.
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

    # Optional, and mounting a missing directory raises.

    if settings.static_pages_dir.is_dir():
        app.mount("/pages", StaticFiles(directory=settings.static_pages_dir), name="pages")

    return app
