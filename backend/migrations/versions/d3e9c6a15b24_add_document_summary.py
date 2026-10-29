"""add document summary

Revision ID: d3e9c6a15b24
Revises: c2d8b5f04a13
Create Date: 2026-08-21 00:00:00.000000

A short per-document summary, shown to the model each turn so it knows what
the user's files cover.

No backfill: a model call per row would be slow, billed, and could fail a
deploy. Existing rows read as "" and stay searchable.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d3e9c6a15b24"
down_revision: str | Sequence[str] | None = "c2d8b5f04a13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Existing rows get "" instead of NULL. The default stays on the column.

    op.add_column(
        "documents",
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "summary")
