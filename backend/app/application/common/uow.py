"""The transaction boundary, expressed as a port."""

from collections.abc import Callable
from typing import Protocol, Self

from app.application.chat.ports import ConversationRepository
from app.application.identity.ports import RefreshTokenRepository, UserRepository
from app.application.ingestion.ports import DocumentRepository


class UnitOfWork(Protocol):
    """One atomic piece of work and its repositories. Leaving without `commit()` rolls back."""

    conversations: ConversationRepository
    users: UserRepository
    # With `users`, so registration writes the user and session together.
    refresh_tokens: RefreshTokenRepository
    documents: DocumentRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
"""A fresh unit of work per call, since `AsyncSession` cannot be shared."""
