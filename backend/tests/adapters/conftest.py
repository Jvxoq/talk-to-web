"""An in-memory SQLite database for the persistence adapters.

SQLite, so `uv run pytest` needs nothing running. It proves the SQL we write:
owner filters, flush, revoke, FK cascade, rollback. It cannot prove Postgres
behaviour: timezones, unique violations, or isolation between sessions (they
share one connection). `tests/integration/` covers those.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.adapters.persistence.models import Base, UserModel
from app.domain.identity.value_objects import Email


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """A fresh in-memory database per test, with foreign keys actually enforced."""
    # Every connection shares this in-memory database and its schema.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # SQLite ignores FK cascades without this pragma.
    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """One session for the repository under test to write through."""
    async with session_factory() as session:
        yield session


@pytest.fixture
async def owner(session: AsyncSession) -> UserModel:
    """A saved user, since conversations have a real FK to `users`."""
    return await make_user(session, "owner@example.com")


async def make_user(session: AsyncSession, email: str) -> UserModel:
    row = UserModel(email=Email.sanitize(email).value, password_hash="$argon2id$stub")
    session.add(row)
    await session.flush()
    return row
