"""Builds the application. No routes, no handlers, no logic."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.v1.routers import chat, conversations, health, ingestion, models, transcription
from app.composition import lifespan
from app.observability.logging import configure_logging
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    A factory rather than a module-level app, so a test can build an instance
    with a substituted container instead of monkey-patching globals.
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.environment != "local")

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
    )

    register_exception_handlers(app)

    for router in (
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
