"""attach documents to a conversation

Revision ID: b7acce1e43d7
Revises: d3e9c6a15b24
Create Date: 2026-08-25 18:20:56.265209

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7acce1e43d7"
down_revision: str | Sequence[str] | None = "d3e9c6a15b24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Named, so the downgrade can drop it.
_FK = "fk_documents_conversation_id_conversations"


def upgrade() -> None:
    """Upgrade schema.

    Nullable on purpose. Older rows match no thread, so retrieval never sees
    them. No backfill, since that would invent attachments.
    """

    op.add_column("documents", sa.Column("conversation_id", sa.Integer(), nullable=True))
    op.create_index(
        "ix_documents_owner_id_conversation_id",
        "documents",
        ["owner_id", "conversation_id"],
        unique=False,
    )
    op.create_foreign_key(
        _FK, "documents", "conversations", ["conversation_id"], ["id"], ondelete="CASCADE"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(_FK, "documents", type_="foreignkey")
    op.drop_index("ix_documents_owner_id_conversation_id", table_name="documents")
    op.drop_column("documents", "conversation_id")
