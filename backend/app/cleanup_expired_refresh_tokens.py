"""Entry point for the refresh-token cleanup job. Not served by the API.

Its own small composition root, since no request should trigger this. Run it
on a schedule or via the `cleanup-refresh-tokens` compose service:

    uv run python -m app.cleanup_expired_refresh_tokens
"""

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.adapters.time.system_clock import SystemClock
from app.application.identity.use_cases.cleanup_expired_refresh_tokens import (
    CleanupExpiredRefreshTokens,
)
from app.observability.logging import configure_logging
from app.settings import get_settings


async def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.environment != "local")

    # Short-lived: one query, then exit.

    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    def uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    cleanup = CleanupExpiredRefreshTokens(
        uow_factory,
        clock=SystemClock(),
        retention_seconds=settings.refresh_token_cleanup_retention_seconds,
    )

    try:
        await cleanup()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
