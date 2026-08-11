"""SQLAlchemy implementations of the repository ports."""

from datetime import datetime

from loguru import logger
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.persistence.mappers import (
    conversation_summary_to_domain,
    conversation_to_domain,
    conversation_to_model,
    document_to_domain,
    document_to_model,
    message_to_domain,
    message_to_model,
    refresh_token_to_domain,
    refresh_token_to_model,
    user_to_domain,
    user_to_model,
)
from app.adapters.persistence.models import (
    ConversationModel,
    DocumentModel,
    RefreshTokenModel,
    UserModel,
)
from app.domain.chat.entities import Conversation, Message
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.value_objects import Email
from app.domain.ingestion.entities import UploadedDocument


class SqlAlchemyConversationRepository:
    """
    Persists conversations with SQLAlchemy.

    Structurally satisfies `app.application.chat.ports.ConversationRepository`;
    it deliberately does not import or inherit from it, so the dependency arrow
    keeps pointing inward.

    Nothing here commits. The session is owned by the unit of work, which is the
    only thing that knows where the transaction boundary is; a repository that
    committed would make a multi-step use case impossible to roll back.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, conversation_id: int, owner_id: int) -> Conversation | None:
        """Load this owner's conversation with its messages, or `None`.

        The owner is a predicate in the query, not a check on the result. Reading
        the row first and comparing afterwards works right up until someone adds
        an early return, and the failure mode is handing a stranger the whole
        thread.
        """
        statement = (
            select(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.owner_id == owner_id,
            )
            # Eager load: the mapper reads `row.messages`, and a lazy load from
            # async code raises instead of quietly issuing a second query.
            .options(selectinload(ConversationModel.messages))
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is None:
            logger.debug(f"Conversation {conversation_id} not found for owner {owner_id}")
            return None
        return conversation_to_domain(row)

    async def list_by_owner(self, owner_id: int) -> list[Conversation]:
        """This owner's conversations, newest activity first, messages omitted.

        No `selectinload` here on purpose - a sidebar render has no use for
        every message in every thread, and the summary mapper never touches
        the relationship this leaves unloaded.
        """
        statement = (
            select(ConversationModel)
            .where(ConversationModel.owner_id == owner_id)
            .order_by(ConversationModel.updated_at.desc())
        )
        rows = (await self._session.execute(statement)).scalars().all()
        return [conversation_summary_to_domain(row) for row in rows]

    async def add(self, conversation: Conversation) -> Conversation:
        """Insert a conversation and return it with its database-assigned id."""
        row = conversation_to_model(conversation)
        self._session.add(row)
        # Flush, not commit: the INSERT runs so the id and column defaults are
        # populated, but the transaction stays open for the unit of work to close.
        await self._session.flush()
        logger.debug(f"Inserted conversation {row.id}")
        # `row.messages` was assigned by the mapper, so the collection is already
        # loaded here - reading it does not trigger a lazy load.
        return conversation_to_domain(row)

    async def add_message(self, conversation_id: int, message: Message) -> Message:
        """Insert one exchange against a conversation and return it with its id."""
        message.conversation_id = conversation_id
        row = message_to_model(message)
        row.conversation_id = conversation_id
        self._session.add(row)
        await self._session.flush()
        logger.debug(f"Inserted message {row.id} on conversation {conversation_id}")
        return message_to_domain(row)

    async def delete(self, conversation_id: int, owner_id: int) -> None:
        """Remove this owner's conversation. Its messages go with it via the FK cascade."""
        await self._session.execute(
            delete(ConversationModel).where(
                ConversationModel.id == conversation_id,
                ConversationModel.owner_id == owner_id,
            )
        )
        logger.debug(f"Deleted conversation {conversation_id} for owner {owner_id}")


class SqlAlchemyDocumentRepository:
    """
    Persists uploaded documents with SQLAlchemy.

    Structurally satisfies `app.application.ingestion.ports.DocumentRepository`.
    Every method is scoped by owner for the same reason as
    `SqlAlchemyConversationRepository`: a document a stranger can name by id is
    a document a stranger can delete.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: int, owner_id: int) -> UploadedDocument | None:
        statement = select(DocumentModel).where(
            DocumentModel.id == document_id, DocumentModel.owner_id == owner_id
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return document_to_domain(row) if row is not None else None

    async def list_by_owner(self, owner_id: int) -> list[UploadedDocument]:
        statement = (
            select(DocumentModel)
            .where(DocumentModel.owner_id == owner_id)
            .order_by(DocumentModel.created_at.desc())
        )
        rows = (await self._session.execute(statement)).scalars().all()
        return [document_to_domain(row) for row in rows]

    async def add(self, document: UploadedDocument) -> UploadedDocument:
        row = document_to_model(document)
        self._session.add(row)
        await self._session.flush()
        logger.debug(f"Inserted document {row.id} for owner {row.owner_id}")
        return document_to_domain(row)

    async def set_chunks_indexed(self, document_id: int, owner_id: int, count: int) -> None:
        await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id, DocumentModel.owner_id == owner_id)
            .values(chunks_indexed=count)
        )

    async def delete(self, document_id: int, owner_id: int) -> None:
        await self._session.execute(
            delete(DocumentModel).where(
                DocumentModel.id == document_id, DocumentModel.owner_id == owner_id
            )
        )
        logger.debug(f"Deleted document {document_id} for owner {owner_id}")


class SqlAlchemyUserRepository:
    """
    Persists users with SQLAlchemy.

    Structurally satisfies `app.application.identity.ports.UserRepository`. Like
    every repository here it flushes rather than commits - the unit of work owns
    the transaction, which is what lets registration write a user and their first
    session atomically.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: int) -> User | None:
        row = await self._session.get(UserModel, user_id)
        return user_to_domain(row) if row is not None else None

    async def get_by_email(self, email: Email) -> User | None:
        """Find a user by their normalised address, or `None`."""
        # `email.value` is already lowercased by `Email.sanitize`, so this is an
        # exact match on the unique index rather than a function over the column.
        statement = select(UserModel).where(UserModel.email == email.value)
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return user_to_domain(row) if row is not None else None

    async def add(self, user: User) -> User:
        """Insert a user and return them with their database-assigned id."""
        row = user_to_model(user)
        self._session.add(row)
        await self._session.flush()
        logger.debug(f"Inserted user {row.id}")
        return user_to_domain(row)


class SqlAlchemyRefreshTokenRepository:
    """
    Persists refresh sessions with SQLAlchemy.

    Structurally satisfies `app.application.identity.ports.RefreshTokenRepository`.
    Revocation is an UPDATE rather than a DELETE: a row that is gone cannot be
    told apart from one that never existed, and that difference is exactly what
    reuse detection is built on.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, token: RefreshToken) -> RefreshToken:
        row = refresh_token_to_model(token)
        self._session.add(row)
        await self._session.flush()
        logger.debug(f"Opened session {row.id} for user {row.user_id}")
        return refresh_token_to_domain(row)

    async def get_by_fingerprint(self, fingerprint: str) -> RefreshToken | None:
        statement = select(RefreshTokenModel).where(RefreshTokenModel.fingerprint == fingerprint)
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return refresh_token_to_domain(row) if row is not None else None

    async def revoke(self, token_id: int, at: datetime) -> None:
        """End one session, leaving an already-revoked one where it is."""
        await self._session.execute(
            update(RefreshTokenModel)
            .where(
                RefreshTokenModel.id == token_id,
                # Without this, rotating twice would move the timestamp and
                # misreport when the session actually ended.
                RefreshTokenModel.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        logger.debug(f"Revoked session {token_id}")

    async def revoke_all_for_user(self, user_id: int, at: datetime) -> None:
        """End every live session a user has. Called when a token turns up twice."""
        await self._session.execute(
            update(RefreshTokenModel)
            .where(
                RefreshTokenModel.user_id == user_id,
                RefreshTokenModel.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        logger.warning(f"Revoked every session for user {user_id}")

    async def delete_expired_before(self, cutoff: datetime) -> int:
        result = await self._session.execute(
            delete(RefreshTokenModel).where(RefreshTokenModel.expires_at < cutoff)
        )
        # A DELETE always yields a `CursorResult`, which is where `.rowcount`
        # actually lives - `Result[Any]` is just the interface mypy sees it as.
        return int(result.rowcount)  # type: ignore[attr-defined]
