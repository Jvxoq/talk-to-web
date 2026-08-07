"""The transaction boundary, expressed as a port."""

from collections.abc import Callable
from typing import Protocol, Self

from app.application.chat.ports import ConversationRepository


class UnitOfWork(Protocol):
    """
    One atomic piece of work, holding the repositories it spans.

    Entering opens the session; leaving without an explicit `commit()` rolls
    back, so a use case that raises halfway never half-writes.
    """

    conversations: ConversationRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
"""
Use cases receive a *factory*, not a unit of work.

A single long-lived unit of work would share one database session across
concurrent requests, and `AsyncSession` is not task-safe. Asking for a fresh one
per call makes that mistake unrepresentable.
"""
