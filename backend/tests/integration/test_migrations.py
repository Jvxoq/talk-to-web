"""The Alembic history, against a real Postgres it has never seen before.

Migrations own the schema in every environment, so nothing else checks any of
this. `Base.metadata.create_all` - which the SQLite adapter suite uses - builds
the schema from the models and therefore agrees with them by construction; it
cannot notice that a migration was never written for the column somebody added
last week.
"""

import os
import subprocess
import sys
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.adapters.persistence.models import Base
from tests.integration.conftest import (
    BACKEND_ROOT,
    DUMMY_SECRETS,
    alembic,
    table_names,
)


def include_name(name: str | None, type_: str, _parents: object) -> bool:
    """The same filter `migrations/env.py` installs.

    Copied rather than imported: `env.py` is a script Alembic executes, not a
    module, and importing it would run a migration as a side effect.
    """
    if type_ == "table":
        return name in Base.metadata.tables
    return True


def schema_diff(url: str, **opts: object) -> list[object]:
    """What autogenerate would propose against the database at `url`."""
    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"include_name": include_name, **opts}
            )
            return list(compare_metadata(context, Base.metadata))
    finally:
        engine.dispose()


# Every table the application owns. Named here rather than derived from
# `Base.metadata` on purpose: the question is whether the *migrations* produce
# them, and reading the answer off the models would make the test agree with
# itself.
APPLICATION_TABLES = {"conversations", "messages", "users", "refresh_tokens", "documents"}


@pytest.fixture
def upgraded(scratch_database: str) -> str:
    alembic("upgrade", "head", database_url=scratch_database)
    return scratch_database


class TestUpgrade:
    async def test_a_fresh_database_gets_every_table(self, upgraded: str) -> None:
        engine = create_async_engine(upgraded)
        try:
            assert await table_names(engine) >= APPLICATION_TABLES
        finally:
            await engine.dispose()

    def test_it_lands_on_a_single_head(self, upgraded: str) -> None:
        """Two heads is a merge nobody noticed, and `upgrade head` then fails.

        It fails at deploy time rather than at review time, which is the worst
        moment to discover that two branches each added a migration.
        """
        heads = alembic("heads", database_url=upgraded).stdout.strip().splitlines()

        assert len(heads) == 1, f"expected one head, found: {heads}"

    def test_the_recorded_version_is_that_head(self, upgraded: str) -> None:
        current = alembic("current", database_url=upgraded).stdout
        head = alembic("heads", database_url=upgraded).stdout.split()[0]

        assert head in current


class TestRoundTrip:
    def test_downgrade_to_base_then_upgrade_again(self, upgraded: str) -> None:
        """The claim that makes a rollback survivable.

        A `downgrade` nobody runs is a `downgrade` that does not work, and the
        moment it matters is the moment a bad release is already live.
        """
        alembic("downgrade", "base", database_url=upgraded)
        alembic("upgrade", "head", database_url=upgraded)

    async def test_downgrade_to_base_leaves_no_application_tables(self, upgraded: str) -> None:
        alembic("downgrade", "base", database_url=upgraded)

        engine = create_async_engine(upgraded)
        try:
            remaining = await table_names(engine)
        finally:
            await engine.dispose()

        # `alembic_version` survives, and should: it is Alembic's bookkeeping,
        # not ours.
        assert not (APPLICATION_TABLES & remaining), (
            f"left behind: {APPLICATION_TABLES & remaining}"
        )

    async def test_the_second_upgrade_rebuilds_the_same_schema(self, upgraded: str) -> None:
        """A round trip that ends somewhere else is not a round trip.

        The usual cause is a downgrade that drops a table but not its index or
        its enum type, so the next upgrade either fails or builds something
        subtly different.
        """
        engine = create_async_engine(upgraded)
        try:
            before = await table_names(engine)
            alembic("downgrade", "base", database_url=upgraded)
            alembic("upgrade", "head", database_url=upgraded)
            after = await table_names(engine)
        finally:
            await engine.dispose()

        assert before == after


class TestDrift:
    """Do the migrations and the models still describe the same schema?"""

    def test_autogenerate_against_a_migrated_database_finds_nothing(self, upgraded: str) -> None:
        """The check that catches a model change nobody wrote a migration for.

        Run through Alembic's comparison API rather than by generating a
        revision file, because the assertion is about the diff, and a test that
        writes a file into `migrations/versions/` is a test that eventually
        leaves one there.

        `compare_type` and `compare_server_default` are on, matching `env.py`:
        without them a widened column or a changed default is invisible here and
        therefore invisible in review.
        """
        differences = schema_diff(upgraded, compare_type=True, compare_server_default=True)

        assert differences == [], f"models and migrations disagree: {differences}"


class TestOfflineMode:
    def test_head_can_be_emitted_as_sql(self, scratch_database: str) -> None:
        """`--sql` is how a DBA reviews a migration before it touches production.

        It breaks the first time a migration branches on live data, and it breaks
        silently, because nobody runs it until the release that needs it.
        """
        emitted = alembic("upgrade", "head", "--sql", database_url=scratch_database).stdout

        assert "CREATE TABLE" in emitted.upper()


class TestCheckpointTables:
    """LangGraph's tables are not Alembic's, and `env.py` has to keep it that way."""

    @pytest.fixture
    def with_a_foreign_table(self, upgraded: str) -> Iterator[str]:
        """Stand in for `checkpoint_*` with a table Alembic has never heard of.

        The real `checkpointer.setup()` would drag a psycopg pool and LangGraph's
        own schema version into a test whose subject is one `include_name`
        callback, and the callback cannot tell the difference.
        """
        engine = create_engine(upgraded, poolclass=pool.NullPool)
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE checkpoint_blobs_stub (id integer PRIMARY KEY)"))
        engine.dispose()
        yield upgraded

    def test_autogenerate_does_not_propose_dropping_them(self, with_a_foreign_table: str) -> None:
        """Without `include_name`, every autogenerate would offer to delete them.

        And someone would eventually accept, because an autogenerated diff looks
        authoritative - which would take the agent's entire memory with it.
        """
        differences = schema_diff(with_a_foreign_table)

        assert "checkpoint_blobs_stub" not in str(differences)


def test_the_migration_url_is_the_one_that_is_used(scratch_database: str) -> None:
    """`DATABASE_MIGRATION_URL` overrides `DATABASE_URL`, and it has to.

    Neon publishes a pooled endpoint and a direct one, and DDL through a
    transaction pooler loses the session state a migration depends on. If this
    precedence ever inverted, every deployment would silently run its migrations
    down the wrong pipe.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env={
            **os.environ,
            **DUMMY_SECRETS,
            # Unreachable on purpose: if this one were used, the command would
            # fail to connect, and the assertion below would never be reached.
            "DATABASE_URL": "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nowhere",
            "DATABASE_MIGRATION_URL": scratch_database,
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert completed.returncode == 0

    # The SQLAlchemy URL, not the plain DSN: `postgresql://` alone resolves to
    # the psycopg2 dialect, which this project does not install.
    engine = create_engine(scratch_database, poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            }
    finally:
        engine.dispose()

    assert tables >= APPLICATION_TABLES
