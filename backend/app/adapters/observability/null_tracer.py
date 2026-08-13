"""Tracer used when Langfuse keys are unset.

Mirrors `configure_sentry` returning `False` with no DSN
(`app/observability/sentry.py`): with no Langfuse keys, composition wires this
in instead of `LangfuseTracer`, so a local run and the whole test suite build
the chat use cases exactly as they run in production - same collaborator
count, same call shape - and send nothing anywhere and need no key. There is
nothing to buffer, so `flush()` has nothing to wait for.

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
    """No-op `Tracer`. Wired in whenever Langfuse credentials are absent or
    fail the startup `auth_check` in `LangfuseTracer.credentials_valid`.
    """

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
