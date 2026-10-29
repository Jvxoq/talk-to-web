"""SQLAlchemy table definitions only. Importing this opens no connection."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    return datetime.now(UTC)


_TIMESTAMP = DateTime(timezone=True)


class Base(DeclarativeBase):
    """Declarative base for every mapped table in this application."""


class UserModel(Base):
    """The `users` row. Mapped to `app.domain.identity.entities.User`."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Bounded so it can be unique-indexed. 254 is the SMTP maximum, as in `Email.sanitize`.
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now, onupdate=_utc_now)


class RefreshTokenModel(Base):
    """The `refresh_tokens` row. Stores a fingerprint, never the secret itself."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # A hex digest. Unique so rotation is unambiguous. Every refresh looks it up.
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(_TIMESTAMP)
    revoked_at: Mapped[datetime | None] = mapped_column(_TIMESTAMP)
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)


class ConversationModel(Base):
    """The `conversations` row. Mapped to `app.domain.chat.entities.Conversation`."""

    __tablename__ = "conversations"

    # Matches every query's shape. Also covers plain `owner_id` lookups.
    __table_args__ = (Index("ix_conversations_owner_id_id", "owner_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column()
    model_type: Mapped[str] = mapped_column(index=True)
    # A callable, so the time is taken per row, not once at import.
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now, onupdate=_utc_now)

    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class MessageModel(Base):
    """The `messages` row. Mapped to `app.domain.chat.entities.Message`."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column()
    # Same callable-default fix as above.
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now, onupdate=_utc_now)

    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="messages"
    )


class DocumentModel(Base):
    """The `documents` row. Mapped to `app.domain.ingestion.entities.UploadedDocument`."""

    __tablename__ = "documents"

    # The second index serves the per-turn query: this owner's documents in this thread.
    __table_args__ = (
        Index("ix_documents_owner_id_id", "owner_id", "id"),
        Index("ix_documents_owner_id_conversation_id", "owner_id", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # Nullable for older rows, which then match no thread. `DeleteConversation`
    # removes vectors and files before the cascade removes the row.
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), default=None
    )
    name: Mapped[str] = mapped_column()
    # Opaque storage reference. Only `FileStorage` and `TextExtractor` read it.
    reference: Mapped[str] = mapped_column()
    chunks_indexed: Mapped[int] = mapped_column(default=0)
    # Written at indexing. `server_default` makes older rows "" instead of NULL.

    summary: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
