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
    """
    Owns an `AsyncSession` for the duration of one `async with` block.

    Structurally satisfies `app.application.common.uow.UnitOfWork`. Leaving the
    block without calling `commit()` rolls back, so a use case that raises
    halfway through never leaves a half-written conversation behind.
    """

    # Only bound once the unit of work is entered - a repository cannot exist
    # without the session it writes through. Annotated as the port rather than
    # the concrete class: a protocol's attributes are invariant, so narrowing it
    # here would stop this class satisfying `UnitOfWork` at all.
    conversations: ConversationRepository
    users: UserRepository
    refresh_tokens: RefreshTokenRepository
    documents: DocumentRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._committed = False

    async def __aenter__(self) -> Self:
        # An AsyncSession is not task-safe, so a unit of work must never be
        # re-entered or shared: fail loudly instead of interleaving statements.
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
            # Leave the session usable and the data consistent before the caller
            # sees the failure; a failed commit otherwise strands the transaction.
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
