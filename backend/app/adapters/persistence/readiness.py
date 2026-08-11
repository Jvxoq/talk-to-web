"""Is Postgres answering?"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresProbe:
    """
    `SELECT 1` through the application's own engine.

    Structurally satisfies `app.application.health.ports.ReadinessProbe`.

    The engine is the point. A probe on its own connection would answer "the
    database is up" while every request queued behind an exhausted pool - which
    is the outage a rollout gate most needs to notice. Borrowing from the pool
    the requests use means an exhausted pool fails the probe too.
    """

    name = "postgres"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def check(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
