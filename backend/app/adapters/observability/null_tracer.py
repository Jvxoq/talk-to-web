"""No-op tracer, used when Langfuse keys are unset or rejected.

Local runs and tests wire the use cases the same way as production but send
nothing. Matches `Tracer` and `Span` by shape.
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


# Stateless, so one shared instance is safe.
_NULL_SPAN = NullSpan()


class NullTracer:
    """No-op `Tracer`, used when Langfuse credentials are missing or rejected."""

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
