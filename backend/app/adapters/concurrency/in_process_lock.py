"""An in-process mutex per document, satisfied as `DocumentLock`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class InProcessDocumentLock:
    """One `asyncio.Lock` per document id, in the worker's own memory.

    This is correct only for a single process, and this deployment runs exactly
    one: one uvicorn worker, which is the same posture as the in-memory rate
    limiters. Two workers would each keep their own locks and the
    indexing-versus-deletion race this closes would reopen between them; the
    day the deployment scales out, this adapter is what has to become a
    database-backed lease (a row lock or advisory lock keyed by document id)
    with the same `hold` shape.

    Entries are pruned once nothing holds or waits on them, so a server that
    has indexed and deleted many documents does not accumulate a lock per id
    forever.
    """

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}
        # How many coroutines are inside or waiting on each lock. Pruning needs
        # to tell "released and forgotten" from "released while someone queued":
        # `lock.locked()` alone cannot, because a waiter does not hold the lock
        # yet and a released lock reads as free even with waiters queued.
        self._holders: dict[int, int] = {}

    @asynccontextmanager
    async def hold(self, document_id: int) -> AsyncIterator[None]:
        lock = self._locks.setdefault(document_id, asyncio.Lock())
        self._holders[document_id] = self._holders.get(document_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._holders[document_id] -= 1
            if self._holders[document_id] == 0:
                # Nobody inside or queued. The `setdefault` above and this
                # decrement are both synchronous blocks - no await between
                # entry and count - so no other coroutine can have taken the
                # lock or registered as a waiter in between.
                self._locks.pop(document_id, None)
                self._holders.pop(document_id, None)
