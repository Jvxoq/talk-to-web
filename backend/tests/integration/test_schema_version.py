"""The startup schema gate, against the real migration history on real Postgres."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.adapters.persistence.schema_version import SchemaOutOfDateError, verify_schema
from tests.integration.conftest import alembic


async def test_migrated_database_passes(migrated_url: str) -> None:
    engine = create_async_engine(migrated_url)
    try:
        await verify_schema(engine)
    finally:
        await engine.dispose()


async def test_one_step_behind_blocks_boot(migrated_url: str) -> None:
    alembic("downgrade", "-1", database_url=migrated_url)
    engine = create_async_engine(migrated_url)
    try:
        with pytest.raises(
            SchemaOutOfDateError, match=r"database schema is at revision \w+ but the code"
        ):
            await verify_schema(engine)
    finally:
        await engine.dispose()


async def test_unmigrated_database_blocks_boot(scratch_database: str) -> None:
    engine = create_async_engine(scratch_database)
    try:
        with pytest.raises(SchemaOutOfDateError, match=r"no alembic_version row"):
            await verify_schema(engine)
    finally:
        await engine.dispose()
