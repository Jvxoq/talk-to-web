"""Entry point for the refresh-token cleanup job. Not served by the API.

`app/composition.py`'s `AppContainer` is "every use case the API is allowed to
invoke, and nothing else" - this use case is deliberately not in it, since
nothing about deleting old session rows is a thing a request should trigger.
This module is its own small composition root instead, the same way
`migrations/env.py` is one for `alembic upgrade head`: a second, narrow
entrypoint gets to build only the handful of collaborators its one use case
needs, rather than the whole application.

Run it on a schedule - cron, a Kubernetes CronJob, or the `cleanup-refresh-tokens`
one-shot service in the compose files:

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

    # A short-lived engine, not the pooled, long-lived one `composition.py`
    # builds for the running server: this process runs one query and exits.
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
