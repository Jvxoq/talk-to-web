"""add documents table

Revision ID: c2d8b5f04a13
Revises: b1c7a4e93d52
Create Date: 2026-08-09 00:00:00.000000

Uploads had no row of their own before this: `IndexDocument` wrote straight to
the vector store and nothing recorded what a person had uploaded, so a second
upload could only ever replace the first. This table is what makes an upload a
thing with an identity - listable, and deletable on its own - instead of a side
effect of indexing.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2d8b5f04a13"
down_revision: str | Sequence[str] | None = "b1c7a4e93d52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("reference", sa.String(), nullable=False),
        sa.Column("chunks_indexed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_owner_id_id", "documents", ["owner_id", "id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_documents_owner_id_id", table_name="documents")
    op.drop_table("documents")
