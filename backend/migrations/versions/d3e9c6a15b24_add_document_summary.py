"""add document summary

Revision ID: d3e9c6a15b24
Revises: c2d8b5f04a13
Create Date: 2026-08-21 00:00:00.000000

The agent could search a user's uploads but had no way to know they existed. It
guessed - from the wording of the question alone - whether a private lookup was
worth a call, and a guess that went the wrong way sent a question about
someone's own file to a web search instead.

This column is what that guess is replaced with: a few sentences per document,
written by the condenser when the upload is indexed, put in front of the model
on every turn. Knowing what is in the files is what makes retrieval the obvious
call rather than a gamble.

Backfill is deliberately not attempted. Summarizing every existing upload would
mean a model call per row inside a migration - slow, billable, and able to fail
a deploy over an optional enhancement. Existing rows read as "" and are still
fully searchable; they gain a summary when re-uploaded.

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
    # `server_default` and NOT NULL together, so the rows that already exist get
    # "" rather than NULL and the entity never has to tell an absent summary
    # apart from an empty one. The default stays on the column afterwards: it
    # costs nothing and keeps a hand-written INSERT honest.
    op.add_column(
        "documents",
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "summary")
