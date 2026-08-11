"""Real Postgres and real Qdrant, for the claims the rest of the suite cannot make.

Everything here is deselected by default (`-m "not integration"` in
`pyproject.toml`), so `uv run pytest` still needs nothing running. Opt in with:

    docker compose up -d postgres qdrant
    INTEGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres \
    INTEGRATION_QDRANT_URL=http://localhost:6333 \
    uv run pytest -m integration

Two deliberate choices about that URL:

*Its own variables, not `DATABASE_URL`.* These tests create databases, drop
them, and run `alembic downgrade base`. Reading the same variable the
application reads is how a stray `.env` gets a developer's real database
dropped, and the whole point of an integration suite is that it is destructive.

*It names a maintenance database, not the one under test.* Each run creates a
scratch database of its own and drops it at the end, so two runs on one server
- a laptop and a container, or two CI jobs - never share a schema.
"""

import os
import re
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.adapters.persistence.models import UserModel
from app.domain.identity.value_objects import Email

BACKEND_ROOT = Path(__file__).resolve().parents[2]

_DATABASE_URL_VAR = "INTEGRATION_DATABASE_URL"
_QDRANT_URL_VAR = "INTEGRATION_QDRANT_URL"

# Same rewrite as `app.composition._plain_dsn`, and duplicated rather than
# imported on purpose: importing the composition root would make this fixture
# module depend on every adapter and client the application builds.
_SQLALCHEMY_DRIVER = re.compile(r"^(postgresql|postgres)\+[a-z0-9_]+://")

# The settings a subprocess needs before `Settings` will construct at all. None
# of them are used by a migration; they are here so `migrations/env.py` can read
# the URL out of the same settings object the application uses.
DUMMY_SECRETS = {
    "ENVIRONMENT": "test",
    "LLM_API_KEY": "test",
    "TAVILY_API_KEY": "test",
    "GEMINI_API_KEY": "test",
    "DEEPGRAM_API_KEY": "test",
    "JWT_SECRET": "test",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything in this package, so no test here can be forgotten.

    A per-file marker is one new file away from a test that runs in the default
    suite, fails on a missing Postgres, and gets blamed on the change that
    happened to land beside it.

    The path check is not optional: pytest hands this hook every collected item
    in the run, not just the ones under the conftest that defines it, so without
    it a single `-m integration` would mark the entire suite.
    """
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


def plain_dsn(url: str) -> str:
    """`postgresql+psycopg://…` as psycopg wants it."""
    return _SQLALCHEMY_DRIVER.sub(r"\1://", url, count=1)


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get(_DATABASE_URL_VAR)
    if not url:
        pytest.skip(f"{_DATABASE_URL_VAR} is not set")
    return url


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    url = os.environ.get(_QDRANT_URL_VAR)
    if not url:
        pytest.skip(f"{_QDRANT_URL_VAR} is not set")
    return url


def _with_database(url: str, name: str) -> str:
    """The same URL, pointed at a different database."""
    base, _, tail = url.rpartition("/")
    query = f"?{tail.split('?', 1)[1]}" if "?" in tail else ""
    return f"{base}/{name}{query}"


@pytest.fixture
def scratch_database(admin_url: str) -> Iterator[str]:
    """A brand-new, empty database, dropped when the test ends.

    Function-scoped and cheap: `CREATE DATABASE` on a local Postgres costs
    milliseconds, and a migration test that starts from an empty database is
    the only kind whose result means anything.
    """
    name = f"tttw_it_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(plain_dsn(admin_url), autocommit=True) as connection:
        # Not parameterised because an identifier cannot be: psycopg's
        # `sql.Identifier` is what quotes it safely, and `name` is generated
        # here from a uuid rather than taken from anywhere a caller controls.
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield _with_database(admin_url, name)
    finally:
        with psycopg.connect(plain_dsn(admin_url), autocommit=True) as connection:
            # A connection this suite forgot to close would otherwise make the
            # DROP hang until the test run's own pool is garbage collected.
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


def alembic(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    """Run the real `alembic` CLI against `database_url`.

    A subprocess rather than `alembic.command`, because the thing worth testing
    is the command compose and CI actually run - `alembic.ini`, `env.py`, the
    settings lookup and all - not a Python API that bypasses most of it.
    """
    environment = {
        **os.environ,
        **DUMMY_SECRETS,
        "DATABASE_URL": database_url,
        # Set explicitly rather than left to fall back, so a developer's own
        # `DATABASE_MIGRATION_URL` in the environment cannot redirect the
        # migration at their real database.
        "DATABASE_MIGRATION_URL": database_url,
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )


@pytest.fixture
def migrated_url(scratch_database: str) -> str:
    """An empty database with `alembic upgrade head` applied to it."""
    alembic("upgrade", "head", database_url=scratch_database)
    return scratch_database


@pytest.fixture
async def engine(migrated_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_url)
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
    async with session_factory() as session:
        yield session


@pytest.fixture
async def owner(session: AsyncSession) -> UserModel:
    """A committed user to hang conversations off.

    Committed rather than flushed, because the tests below open a *second*
    session to prove isolation, and a second session in real Postgres cannot see
    an uncommitted row - which is exactly the difference from the SQLite suite,
    where every session shares one connection.
    """
    row = UserModel(email=Email.sanitize("owner@example.com").value, password_hash="$argon2id$stub")
    session.add(row)
    await session.commit()
    return row


@pytest.fixture
async def qdrant(qdrant_url: str) -> AsyncIterator[AsyncQdrantClient]:
    """A client against the real server, and a collection name unique to the test."""
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def collection() -> str:
    return f"tttw_it_{uuid.uuid4().hex[:12]}"


async def table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        return {row[0] for row in result}
