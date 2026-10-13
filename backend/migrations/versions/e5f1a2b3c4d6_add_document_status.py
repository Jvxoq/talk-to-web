"""add document status

Revision ID: e5f1a2b3c4d6
Revises: b7acce1e43d7
Create Date: 2026-10-13 00:00:00.000000

The upload request answers before indexing runs, and indexing used to leave no
trace of how it ended. A document whose indexing failed looked exactly like one
that worked: the row was there, so the client showed the file as attached, and
the model could not read a word of it.

This column records which of the two happened. Existing rows read as "ready"
through the server default, which is how the app already treated them.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f1a2b3c4d6"
down_revision: str | Sequence[str] | None = "b7acce1e43d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "documents",
        sa.Column("status", sa.String(length=16), nullable=False, server_default="ready"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "status")
