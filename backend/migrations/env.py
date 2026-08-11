"""Alembic environment.

This module is exempt from the "settings enter only at the composition root"
rule: alembic has its own entrypoint, `migrations/` is outside the `app`
package, and there is no container here to read a URL from.

The engine is deliberately synchronous. `alembic upgrade head` runs as its own
process - a pre-deploy step, never inside the running server's event loop - and
applies statements strictly in order, so there is nothing for async to
interleave. `postgresql+psycopg` is the same URL either way: psycopg 3 serves
`create_engine` and `create_async_engine` alike.
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

# One source of truth for the URL. `alembic.ini` ships a placeholder; the real
# value comes from the same settings the application reads.
#
# `migration_url` is `DATABASE_URL` unless `DATABASE_MIGRATION_URL` overrides it.
# Managed Postgres tends to publish two endpoints for one database — Neon gives a
# pooled one for the app and a direct one — and DDL belongs on the direct
# endpoint, because a transaction pooler can hand the connection to another
# client between statements and lose the session state a migration depends on.
#
# The `%` escape is for configparser interpolation, which would otherwise choke
# on a password containing a percent sign.
database_url = get_settings().migration_url
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))


def include_name(name: str | None, type_: str, _parent_names: object) -> bool:
    """Keep autogenerate off tables this application does not own.

    LangGraph creates its own `checkpoint_*` tables in the same database. They
    are not in `Base.metadata`, so without this filter every autogenerate would
    propose dropping them.
    """
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
    # NullPool: this process opens one connection, uses it, and exits. Pooling
    # would only leave a socket to clean up.
    engine = create_engine(database_url, poolclass=pool.NullPool)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
            # Off by default, and off is how autogenerate silently misses a
            # column whose type or server default changed.
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
