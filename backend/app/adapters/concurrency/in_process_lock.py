"""An in-process mutex per document, satisfied as `DocumentLock`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class InProcessDocumentLock:
    """One `asyncio.Lock` per document id, in process memory.

    Correct only for one worker, which is what this deployment runs. More
    workers need a database lease with the same `hold` shape. Unused locks are
    pruned.
    """

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}
        # Holders plus waiters per lock. `locked()` cannot see waiters.
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
                # Nobody inside or waiting. No await separates the checks, so this is safe.

                self._locks.pop(document_id, None)
                self._holders.pop(document_id, None)
