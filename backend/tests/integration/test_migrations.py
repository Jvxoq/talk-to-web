"""The Alembic history, against a fresh real Postgres.

`create_all` agrees with the models by construction, so only this catches a
missing migration.
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
    """The same filter as `migrations/env.py`. Copied, since importing it runs a migration."""
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


# Listed by hand, not read from the models, so the test cannot agree with itself.
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
        """Two heads would make `upgrade head` fail at deploy time."""
        heads = alembic("heads", database_url=upgraded).stdout.strip().splitlines()

        assert len(heads) == 1, f"expected one head, found: {heads}"

    def test_the_recorded_version_is_that_head(self, upgraded: str) -> None:
        current = alembic("current", database_url=upgraded).stdout
        head = alembic("heads", database_url=upgraded).stdout.split()[0]

        assert head in current


class TestRoundTrip:
    def test_downgrade_to_base_then_upgrade_again(self, upgraded: str) -> None:
        """Downgrade must work, or a bad release cannot be rolled back."""
        alembic("downgrade", "base", database_url=upgraded)
        alembic("upgrade", "head", database_url=upgraded)

    async def test_downgrade_to_base_leaves_no_application_tables(self, upgraded: str) -> None:
        alembic("downgrade", "base", database_url=upgraded)

        engine = create_async_engine(upgraded)
        try:
            remaining = await table_names(engine)
        finally:
            await engine.dispose()

        # `alembic_version` is Alembic's own and stays.
        assert not (APPLICATION_TABLES & remaining), (
            f"left behind: {APPLICATION_TABLES & remaining}"
        )

    async def test_the_second_upgrade_rebuilds_the_same_schema(self, upgraded: str) -> None:
        """Upgrade, downgrade, upgrade must end at the same schema."""
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
        """Catches a model change with no migration.

        Uses the comparison API, so no revision file is written. Type and
        default comparison are on, matching `env.py`.
        """
        differences = schema_diff(upgraded, compare_type=True, compare_server_default=True)

        assert differences == [], f"models and migrations disagree: {differences}"


class TestOfflineMode:
    def test_head_can_be_emitted_as_sql(self, scratch_database: str) -> None:
        """Offline `--sql` mode must keep working for review."""
        emitted = alembic("upgrade", "head", "--sql", database_url=scratch_database).stdout

        assert "CREATE TABLE" in emitted.upper()


class TestCheckpointTables:
    """LangGraph's tables are not Alembic's, and `env.py` has to keep it that way."""

    @pytest.fixture
    def with_a_foreign_table(self, upgraded: str) -> Iterator[str]:
        """A foreign table standing in for LangGraph's `checkpoint_*` tables."""
        engine = create_engine(upgraded, poolclass=pool.NullPool)
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE checkpoint_blobs_stub (id integer PRIMARY KEY)"))
        engine.dispose()
        yield upgraded

    def test_autogenerate_does_not_propose_dropping_them(self, with_a_foreign_table: str) -> None:
        """Autogenerate must not offer to drop tables we do not own."""
        differences = schema_diff(with_a_foreign_table)

        assert "checkpoint_blobs_stub" not in str(differences)


def test_the_migration_url_is_the_one_that_is_used(scratch_database: str) -> None:
    """`DATABASE_MIGRATION_URL` must win over `DATABASE_URL`, so DDL skips the pooler."""
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env={
            **os.environ,
            **DUMMY_SECRETS,
            # Unreachable on purpose, so using it would fail.
            "DATABASE_URL": "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nowhere",
            "DATABASE_MIGRATION_URL": scratch_database,
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert completed.returncode == 0

    # Plain `postgresql://` would pick psycopg2, which is not installed.
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
