"""add users, refresh tokens and conversation ownership

Revision ID: b1c7a4e93d52
Revises: f04f8e245fd0
Create Date: 2026-08-08 23:40:00.000000

`conversations.owner_id` is added nullable, ownerless rows are deleted, then
NOT NULL is applied. The DELETE is deliberate: those rows have no owner to
assign, and keeping them would leave rows anybody can read.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1c7a4e93d52"
down_revision: str | Sequence[str] | None = "f04f8e245fd0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_refresh_tokens_fingerprint"), "refresh_tokens", ["fingerprint"], unique=True
    )
    op.create_index(op.f("ix_refresh_tokens_user_id"), "refresh_tokens", ["user_id"], unique=False)

    # Nullable first: the constraint cannot be satisfied by rows that exist now.
    op.add_column("conversations", sa.Column("owner_id", sa.Integer(), nullable=True))
    # See the module docstring. This is the destructive step.
    op.execute(sa.text("DELETE FROM conversations WHERE owner_id IS NULL"))
    op.alter_column("conversations", "owner_id", nullable=False)

    op.create_foreign_key(
        "fk_conversations_owner_id_users",
        "conversations",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # Composite, matching the queries. Also covers owner-only lookups.
    op.create_index(
        "ix_conversations_owner_id_id", "conversations", ["owner_id", "id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema. Deleted conversations are not restored."""

    op.drop_index("ix_conversations_owner_id_id", table_name="conversations")
    op.drop_constraint("fk_conversations_owner_id_users", "conversations", type_="foreignkey")
    op.drop_column("conversations", "owner_id")

    op.drop_index(op.f("ix_refresh_tokens_user_id"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_fingerprint"), table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
