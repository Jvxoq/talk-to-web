"""Tracer that records nothing.

Composition always wires this in. Tracing is a no-op for now: the eval suite
uses LangSmith, and production tracing is not decided yet. The chat use cases
are built the same way as before, so the call shape and the collaborator count
do not change. Nothing is sent anywhere and no key is needed. There is nothing
to buffer, so `flush()` has nothing to wait for.

Every span is a no-op async context manager yielding a no-op `Span`. Both
satisfy `app.application.chat.ports.Tracer` and `Span` structurally - this
module imports neither, by design, the same way `qdrant_index.py` never
imports the `VectorIndex` protocol it satisfies.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal


class NullSpan:
    """Discards everything set on it and remembers nothing."""

    def set(self, **attributes: object) -> None:
        return None

    def record_error(self, error: BaseException) -> None:
        return None


# One instance, shared by every span this tracer opens. Safe because it is
# stateless - `set` and `record_error` both no-op - so nothing about handing
# the same object to concurrent callers can race.
_NULL_SPAN = NullSpan()


class NullTracer:
    """No-op `Tracer`. Composition always builds this one."""

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: Literal["span", "generation"] = "span",
        **attributes: object,
    ) -> AsyncIterator[NullSpan]:
        yield _NULL_SPAN

    async def flush(self) -> None:
        return None
