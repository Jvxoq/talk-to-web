"""Is Postgres answering?"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresProbe:
    """`SELECT 1` through the app's own engine. Implements `ReadinessProbe`.

    Uses the request pool, so an exhausted pool fails the probe too.
    """

    name = "postgres"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def check(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
