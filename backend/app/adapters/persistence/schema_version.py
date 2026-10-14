"""Does the database sit at the revision the code was written for?

Migrations own the schema, and nothing else notices when a deploy ships code
without the migration that goes with it. The failure then shows up one request
at a time, as a missing column. This module lets startup find out once, and stop.
"""

from pathlib import Path

from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

# app/adapters/persistence/schema_version.py -> the backend root, which holds
# `migrations/` in the repo and in the image (the Dockerfile copies the whole
# context). Taken from this file, not the working directory, so it holds however
# the process was started.
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


class SchemaOutOfDateError(RuntimeError):
    """The database revision differs from the code's Alembic head."""


def code_heads(migrations_dir: Path = MIGRATIONS_DIR) -> tuple[str, ...]:
    """The head revisions in the migration scripts, read with Alembic itself."""
    script = ScriptDirectory(str(migrations_dir))
    return tuple(sorted(script.get_heads()))


def _current_revisions(connection: Connection) -> tuple[str, ...]:
    context = MigrationContext.configure(connection)
    return tuple(sorted(context.get_current_heads()))


async def database_revisions(engine: AsyncEngine) -> tuple[str, ...]:
    """The revisions recorded in `alembic_version`. Empty when there are none.

    A database with no `alembic_version` table has never been migrated, and
    Alembic reports that as no revisions rather than an error.
    """
    async with engine.connect() as connection:
        return await connection.run_sync(_current_revisions)


def ensure_schema_current(current: tuple[str, ...], head: tuple[str, ...]) -> None:
    """Raise `SchemaOutOfDateError` unless the database is exactly at the code's head."""
    if current == head:
        return
    found = ", ".join(current) or "none (no alembic_version row)"
    wanted = ", ".join(head) or "none (no migration scripts)"
    raise SchemaOutOfDateError(
        f"database schema is at revision {found} but the code expects {wanted}. "
        "Run `alembic upgrade head` against this database before starting the app."
    )


async def verify_schema(engine: AsyncEngine, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Startup gate: refuse to boot against a schema that is not at the code's head."""
    ensure_schema_current(await database_revisions(engine), code_heads(migrations_dir))
