"""Langfuse-backed tracer.

Wraps Langfuse's Python SDK (v3, OpenTelemetry-based), satisfying
`app.application.chat.ports.Tracer` and `Span` structurally - this module
imports neither protocol, the same way `qdrant_index.py` never imports
`VectorIndex`.

Nesting is the SDK's problem, not ours: `start_as_current_span` /
`start_as_current_generation` register the new span as the *current* OTel
span via `contextvars`, which `asyncio` copies into every child task at
creation. A node opened from inside another node's span therefore nests
correctly with no parent parameter passed anywhere - exactly the shape
`Tracer.span`'s docstring in `ports.py` requires, since a LangGraph node is
handed its state and nothing else.

Every public method on this class follows the same rule as `Condenser`
(`app/application/chat/agent/condenser.py`): tracing is not on the critical
path, so nothing here may cost a user their reply. Failures are logged at
warning and swallowed; `span()` still yields a working span - falling back to
`NullSpan` - even when Langfuse itself is unreachable.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langfuse import Langfuse
from loguru import logger

from app.adapters.observability.null_tracer import NullSpan
from app.observability.context import get_request_id

# The subset of `LangfuseSpan.update`/`LangfuseGeneration.update`'s keyword
# arguments (both wrappers share one signature) that Langfuse gives a
# dedicated field to - `model`, `usage_details`, `cost_details` and so on are
# what let its UI total tokens and cost across a trace instead of burying them
# in an opaque blob. An attribute passed to `Span.set` under one of these
# names is forwarded as that field; anything else is folded into `metadata`.
_KNOWN_FIELDS = frozenset(
    {
        "input",
        "output",
        "name",
        "version",
        "level",
        "status_message",
        "completion_start_time",
        "model",
        "model_parameters",
        "usage_details",
        "cost_details",
        "prompt",
    }
)

# The two fields that carry raw prompt/completion text rather than shape or
# accounting - the ones `capture_content=False` exists to withhold. Everything
# else `set()` accepts (a model name, a token count, which tool was picked)
# describes the call without repeating what the user typed or the model said,
# so it is kept regardless.
_CONTENT_FIELDS = frozenset({"input", "output"})


def _partition(attributes: dict[str, object], *, capture_content: bool) -> dict[str, Any]:
    """Reshape a `Span.set`-style attribute bag into `update()`'s kwargs.

    Recognised keys pass through by name; everything else lands in
    `metadata`. Content fields are dropped first, before either bucket sees
    them, so a caller cannot smuggle prompt text through under an
    unrecognised key.
    """
    if not capture_content:
        attributes = {k: v for k, v in attributes.items() if k not in _CONTENT_FIELDS}

    known: dict[str, Any] = {k: v for k, v in attributes.items() if k in _KNOWN_FIELDS}
    extra = {k: v for k, v in attributes.items() if k not in _KNOWN_FIELDS}
    if extra:
        known["metadata"] = extra
    return known


class _RecordingSpan:
    """Adapts a live `LangfuseSpan` / `LangfuseGeneration` to `Span`.

    Every method swallows its own failures for the same reason the tracer
    does - a dropped attribute or a bad error report must not be what breaks a
    reply that otherwise succeeded.
    """

    def __init__(self, observation: Any, *, capture_content: bool) -> None:
        self._observation = observation
        self._capture_content = capture_content

    def set(self, **attributes: object) -> None:
        fields = _partition(attributes, capture_content=self._capture_content)
        if not fields:
            return
        try:
            self._observation.update(**fields)
        except Exception as error:
            logger.warning(f"Langfuse span.set failed: {error}")

    def record_error(self, error: BaseException) -> None:
        try:
            self._observation.update(level="ERROR", status_message=str(error))
        except Exception as update_error:
            logger.warning(f"Langfuse record_error failed: {update_error}")


class LangfuseTracer:
    """Tracer backed by a live Langfuse client.

    Constructor arguments are explicit values, never `Settings` - only
    `composition.py` may import `app.settings`, and this class has to be
    constructible from a unit test with no environment at all.
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        capture_content: bool,
        flush_timeout_seconds: float,
    ) -> None:
        self._capture_content = capture_content
        self._flush_timeout_seconds = flush_timeout_seconds
        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    async def credentials_valid(self) -> bool:
        """Round-trips `public_key`/`secret_key` against Langfuse's auth endpoint.

        Meant to be called once, from `composition.py`, before this tracer is
        handed to any use case: on `False` the caller should wire in
        `NullTracer` instead of keeping a tracer that will fail to open every
        single span for the rest of the process's life. `auth_check` is a
        blocking HTTP call, so it runs off the event loop the same way
        `flush` does below.
        """
        try:
            return bool(await asyncio.to_thread(self._client.auth_check))
        except Exception as error:
            logger.warning(f"Langfuse credential check failed: {error}")
            return False

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: Literal["span", "generation"] = "span",
        **attributes: object,
    ) -> AsyncIterator[_RecordingSpan | NullSpan]:
        fields = _partition(attributes, capture_content=self._capture_content)
        context_manager, observation = self._open(name, kind, fields)

        if observation is None:
            yield NullSpan()
            return

        # Correlate with the rest of the system: the request id is what a log
        # line, a Sentry event and now a Langfuse trace all carry, so the
        # three can be found from any one of them. Set on every span rather
        # than once per trace because a request id is only in scope for the
        # spans opened while serving it - there is no single earlier place to
        # set it that would still be correct for a later request reusing the
        # same trace object.
        request_id = get_request_id()
        if request_id is not None:
            try:
                self._client.update_current_trace(
                    tags=[request_id], metadata={"request_id": request_id}
                )
            except Exception as error:
                logger.warning(f"Langfuse trace correlation failed: {error}")

        try:
            yield _RecordingSpan(observation, capture_content=self._capture_content)
        finally:
            # Always exits with (None, None, None), never the real exception
            # info. Forwarding it correctly means re-implementing the
            # generator-based context manager protocol `contextlib` already
            # gives us for free, and a mistake there risks the far worse
            # failure of swallowing the caller's real exception. A node that
            # wants the span marked as failed calls `record_error` before
            # letting the exception propagate - that is what the port gives
            # callers the method for.
            try:
                context_manager.__exit__(None, None, None)
            except Exception as error:
                logger.warning(f"Langfuse span '{name}' failed to close: {error}")

    def _open(
        self, name: str, kind: Literal["span", "generation"], fields: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Start a span or generation, or fail quietly and hand back nothing.

        Returns `(None, None)` on any failure, which `span()` reads as "fall
        back to a no-op span for this step" rather than letting a Langfuse
        outage take down whatever the app was doing.
        """
        try:
            starter = (
                self._client.start_as_current_generation
                if kind == "generation"
                else self._client.start_as_current_span
            )
            context_manager = starter(name=name, **fields)
            observation = context_manager.__enter__()
        except Exception as error:
            logger.warning(f"Langfuse span '{name}' failed to open: {error}")
            return None, None
        return context_manager, observation

    async def flush(self) -> None:
        """Send anything still queued, never blocking longer than `flush_timeout_seconds`.

        The SDK's `flush` is synchronous - it talks to the network on the
        calling thread until the export finishes or its own internal timeout
        elapses. Called directly from an async `finally` at shutdown, that
        would stall the event loop for as long as a dying Langfuse host takes
        to stop responding, which is exactly the moment shutdown cannot
        afford to wait. Running it in a worker thread under `asyncio.timeout`
        lets shutdown move on regardless; Python has no way to force the
        thread to stop early, so a truly hung call leaks one thread rather
        than one hung process.
        """
        try:
            async with asyncio.timeout(self._flush_timeout_seconds):
                await asyncio.to_thread(self._client.flush)
        except Exception as error:
            logger.warning(f"Langfuse flush failed or timed out: {error}")
