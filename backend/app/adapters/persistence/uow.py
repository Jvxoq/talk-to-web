"""The SQLAlchemy unit of work: one session, one transaction, one use case call."""

from typing import Self

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.persistence.repositories import (
    SqlAlchemyConversationRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyRefreshTokenRepository,
    SqlAlchemyUserRepository,
)
from app.application.chat.ports import ConversationRepository
from app.application.identity.ports import RefreshTokenRepository, UserRepository
from app.application.ingestion.ports import DocumentRepository


class SqlAlchemyUnitOfWork:
    """Owns one `AsyncSession` per `async with` block. Implements `UnitOfWork`.

    Leaving without `commit()` rolls back.
    """

    # Bound on enter. Typed as the ports, or this would not satisfy `UnitOfWork`.
    conversations: ConversationRepository
    users: UserRepository
    refresh_tokens: RefreshTokenRepository
    documents: DocumentRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._committed = False

    async def __aenter__(self) -> Self:
        # AsyncSession is not task-safe, so re-entry fails loudly.
        if self._session is not None:
            raise RuntimeError("This unit of work is already open")

        self._session = self._session_factory()
        self._committed = False
        self.conversations = SqlAlchemyConversationRepository(self._session)
        self.users = SqlAlchemyUserRepository(self._session)
        self.refresh_tokens = SqlAlchemyRefreshTokenRepository(self._session)
        self.documents = SqlAlchemyDocumentRepository(self._session)
        return self

    async def __aexit__(self, *exc: object) -> None:
        session = self._session
        if session is None:
            return

        try:
            if not self._committed:
                await session.rollback()
        finally:
            await session.close()
            self._session = None
            self._committed = False

    async def commit(self) -> None:
        """Make this unit of work's writes durable."""
        session = self._require_session()
        try:
            await session.commit()
        except Exception:
            # Roll back so the transaction is not left stranded.

            await session.rollback()
            logger.exception("Commit failed, rolled back")
            raise
        self._committed = True

    async def rollback(self) -> None:
        """Discard everything written in this unit of work."""
        await self._require_session().rollback()
        self._committed = False

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("This unit of work is not open")
        return self._session
