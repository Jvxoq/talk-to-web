"""SQLAlchemy table definitions.

Tables only: no engine, no session factory. Importing this module must not open
a connection, which is what lets tests and migrations import it freely.
"""

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
    # Bounded, unlike the other text columns: this one is unique, and Postgres
    # cannot index an unbounded value beyond its page limit. 254 is the longest
    # address SMTP will carry, which is also what `Email.sanitize` enforces.
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now, onupdate=_utc_now)


class RefreshTokenModel(Base):
    """The `refresh_tokens` row. Mapped to `app.domain.identity.entities.RefreshToken`.

    Stores a fingerprint, never the secret the client holds. Anyone who can read
    this table learns which sessions exist and nothing they could present.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # Fixed width because it is a hex digest, unique because two sessions sharing
    # a fingerprint would make rotation ambiguous, and indexed because every
    # refresh is a lookup by exactly this column.
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(_TIMESTAMP)
    revoked_at: Mapped[datetime | None] = mapped_column(_TIMESTAMP)
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)


class ConversationModel(Base):
    """The `conversations` row. Mapped to `app.domain.chat.entities.Conversation`."""

    __tablename__ = "conversations"

    # Every read and delete is "this conversation, if it is this owner's", so the
    # composite index is the one the queries actually use. The plain `owner_id`
    # index is left off deliberately - this covers it, being a prefix.
    __table_args__ = (Index("ix_conversations_owner_id_id", "owner_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column()
    model_type: Mapped[str] = mapped_column(index=True)
    # Callable default: `datetime.now(UTC)` would be evaluated once at import and
    # freeze every row's timestamp to the moment the process started.
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

    # Every read, list and delete is "this owner's documents", the same shape
    # as the conversations index above.
    __table_args__ = (Index("ix_documents_owner_id_id", "owner_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column()
    # The storage adapter's opaque reference (a filesystem path today). Only the
    # matching `FileStorage`/`TextExtractor` pair may interpret it.
    reference: Mapped[str] = mapped_column()
    chunks_indexed: Mapped[int] = mapped_column(default=0)
    # What this document is about, in a few sentences, written once when the
    # upload was indexed. `server_default` rather than a Python-side default so
    # the rows that existed before this column did read as "" instead of NULL,
    # and the domain entity never has to model an absent summary separately
    # from an empty one.
    summary: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(_TIMESTAMP, default=_utc_now)
