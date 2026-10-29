"""Alembic environment.

An entrypoint outside `app`, so it may read settings directly. The engine is
synchronous: migrations run in their own process, strictly in order.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.adapters.persistence.models import Base
from app.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The URL comes from settings. `DATABASE_MIGRATION_URL` points DDL at the direct
# endpoint, since a pooler can lose session state between statements.
# `%` is escaped for configparser.
database_url = get_settings().migration_url
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))


def include_name(name: str | None, type_: str, _parent_names: object) -> bool:
    """Hide tables we do not own, like LangGraph's `checkpoint_*`, from autogenerate."""
    if type_ == "table":
        return name in target_metadata.tables
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (`alembic upgrade head --sql`)."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply migrations."""
    # One connection, then exit. No pool needed.
    engine = create_engine(database_url, poolclass=pool.NullPool)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
            # Off by default, which misses type and default changes.
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
