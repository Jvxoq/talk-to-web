"""A real database for the persistence adapters, with no infrastructure to start.

SQLite rather than Postgres because `uv run pytest` is meant to need nothing
running. That trade is worth naming, because it is not free:

- What these tests *do* prove is everything the repositories actually author -
  the owner predicate on every read and delete, `flush()` populating an id
  without committing, the partial `WHERE revoked_at IS NULL`, the FK cascade,
  and the unit of work's rollback-unless-committed rule. All of that is
  dialect-independent SQL and SQLAlchemy behaviour.
- What they cannot prove is anything Postgres-specific: `DateTime(timezone=True)`
  is a real timestamptz there and a naive string here, so nothing below asserts
  on the tzinfo of a value that has been round-tripped through the database.
  Unique-violation semantics and index usage are likewise Postgres's business.
- Every session here shares one connection, because that is what an in-memory
  SQLite database is. So "a second session cannot see it yet" is not a claim
  these tests can make; isolation between concurrent transactions is not
  testable this way, and no test below pretends otherwise. What a second
  session does still distinguish is committed from rolled back, which is the
  unit of work's actual contract.
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
    # `StaticPool` is implied by the shared in-memory URL SQLAlchemy builds for
    # aiosqlite; every connection sees the same database, so a session opened
    # later in the test still finds the schema created here.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # SQLite ignores `ondelete="CASCADE"` unless this pragma is on, and the
    # cascade is the reason deleting a conversation does not strand its
    # messages. Without this the test would pass while proving nothing.
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
    """A persisted user to hang conversations off.

    Conversations carry a real FK to `users`, so a test that invents an owner id
    would be testing against a row the database would have refused.
    """
    return await make_user(session, "owner@example.com")


async def make_user(session: AsyncSession, email: str) -> UserModel:
    row = UserModel(email=Email.sanitize(email).value, password_hash="$argon2id$stub")
    session.add(row)
    await session.flush()
    return row
