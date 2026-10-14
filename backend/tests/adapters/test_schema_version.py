"""The startup gate that refuses a database older than the code.

SQLite stands in for Postgres here: `alembic_version` is a plain two-line table
and `MigrationContext` reads it the same way on both. The real migration
history against real Postgres is covered in `tests/integration/`.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.adapters.persistence.schema_version import (
    MIGRATIONS_DIR,
    SchemaOutOfDateError,
    code_heads,
    database_revisions,
    ensure_schema_current,
    verify_schema,
)

HEAD = "bbb222"


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """A two-step history whose single head is `HEAD`."""
    versions = tmp_path / "scripts" / "versions"
    versions.mkdir(parents=True)
    (versions / "aaa111_first.py").write_text('revision = "aaa111"\ndown_revision = None\n')
    (versions / "bbb222_second.py").write_text('revision = "bbb222"\ndown_revision = "aaa111"\n')
    return tmp_path / "scripts"


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    try:
        yield engine
    finally:
        await engine.dispose()


async def stamp(engine: AsyncEngine, *revisions: str) -> None:
    """Write `alembic_version` the way `alembic stamp` does."""
    async with engine.begin() as connection:
        await connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
        )
        for revision in revisions:
            await connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                {"rev": revision},
            )


class TestComparison:
    def test_equal_revisions_pass(self) -> None:
        ensure_schema_current((HEAD,), (HEAD,))

    def test_behind_names_both_revisions(self) -> None:
        with pytest.raises(
            SchemaOutOfDateError,
            match=r"database schema is at revision aaa111 but the code expects bbb222\. "
            r"Run `alembic upgrade head`",
        ):
            ensure_schema_current(("aaa111",), (HEAD,))

    def test_unmigrated_database_fails(self) -> None:
        with pytest.raises(
            SchemaOutOfDateError,
            match=r"at revision none \(no alembic_version row\) but the code expects bbb222",
        ):
            ensure_schema_current((), (HEAD,))

    def test_database_ahead_of_code_fails(self) -> None:
        """A revision this code has never heard of is as wrong as an old one."""
        with pytest.raises(
            SchemaOutOfDateError, match=r"at revision ccc333 but the code expects bbb222"
        ):
            ensure_schema_current(("ccc333",), (HEAD,))


class TestCodeHeads:
    def test_reads_head_from_scripts(self, migrations_dir: Path) -> None:
        assert code_heads(migrations_dir) == (HEAD,)

    def test_repo_history_has_one_head(self) -> None:
        """The real `migrations/` resolves from this file's location and has one head."""
        heads = code_heads(MIGRATIONS_DIR)

        assert len(heads) == 1
        assert any(
            f'"{heads[0]}"' in path.read_text()
            for path in (MIGRATIONS_DIR / "versions").glob("*.py")
        )


class TestDatabaseRevisions:
    async def test_missing_table_gives_no_revisions(self, engine: AsyncEngine) -> None:
        assert await database_revisions(engine) == ()

    async def test_empty_table_gives_no_revisions(self, engine: AsyncEngine) -> None:
        await stamp(engine)

        assert await database_revisions(engine) == ()

    async def test_stamped_revision_is_read(self, engine: AsyncEngine) -> None:
        await stamp(engine, "aaa111")

        assert await database_revisions(engine) == ("aaa111",)


class TestVerifySchema:
    async def test_current_database_passes(self, engine: AsyncEngine, migrations_dir: Path) -> None:
        await stamp(engine, HEAD)

        await verify_schema(engine, migrations_dir)

    async def test_old_database_blocks_boot(
        self, engine: AsyncEngine, migrations_dir: Path
    ) -> None:
        await stamp(engine, "aaa111")

        with pytest.raises(
            SchemaOutOfDateError, match=r"at revision aaa111 but the code expects bbb222"
        ):
            await verify_schema(engine, migrations_dir)

    async def test_missing_table_blocks_boot(
        self, engine: AsyncEngine, migrations_dir: Path
    ) -> None:
        with pytest.raises(SchemaOutOfDateError, match=r"at revision none .* expects bbb222"):
            await verify_schema(engine, migrations_dir)
