"""Langfuse tracer, matching the `Tracer` and `Span` ports by shape.

The SDK nests spans through `contextvars`, which asyncio copies into child
tasks, so no parent is passed. Every failure is logged and swallowed, and
`span()` falls back to `NullSpan`, so tracing never costs a reply.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langfuse import Langfuse
from loguru import logger

from app.adapters.observability.null_tracer import NullSpan
from app.observability.context import get_request_id

# Keys Langfuse has its own fields for, so its UI can total tokens and cost.
# Any other attribute goes into `metadata`.
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

# Raw prompt and completion text, dropped when `capture_content=False`.
_CONTENT_FIELDS = frozenset({"input", "output"})


def _partition(attributes: dict[str, object], *, capture_content: bool) -> dict[str, Any]:
    """Turn `Span.set` attributes into `update()` kwargs.

    Content is dropped first, so it cannot leak through `metadata`.
    """
    if not capture_content:
        attributes = {k: v for k, v in attributes.items() if k not in _CONTENT_FIELDS}

    known: dict[str, Any] = {k: v for k, v in attributes.items() if k in _KNOWN_FIELDS}
    extra = {k: v for k, v in attributes.items() if k not in _KNOWN_FIELDS}
    if extra:
        known["metadata"] = extra
    return known


class _RecordingSpan:
    """Adapts a live Langfuse span or generation to `Span`. Never raises."""

    def __init__(self, observation: Any, *, capture_content: bool) -> None:
        self._observation = observation
        self._capture_content = capture_content

    def set(self, **attributes: object) -> None:
        fields = _partition(attributes, capture_content=self._capture_content)
        if not fields:
            return
        try:
            self._observation.update(**fields)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Langfuse span.set failed: {error}")

    def record_error(self, error: BaseException) -> None:
        try:
            self._observation.update(level="ERROR", status_message=str(error))
        except Exception as update_error:  # noqa: BLE001
            logger.warning(f"Langfuse record_error failed: {update_error}")


class LangfuseTracer:
    """Tracer backed by a live Langfuse client. Takes plain values, not `Settings`."""

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
        """Check the keys once at startup. On `False`, use `NullTracer`.

        `auth_check` blocks, so it runs in a thread.
        """
        try:
            return bool(await asyncio.to_thread(self._client.auth_check))
        except Exception as error:  # noqa: BLE001
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

        # Tag the trace with the request id, which logs and Sentry also carry.
        # Set per span, because the id is only in scope while serving the request.
        request_id = get_request_id()
        if request_id is not None:
            try:
                self._client.update_current_trace(
                    tags=[request_id], metadata={"request_id": request_id}
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Langfuse trace correlation failed: {error}")

        try:
            yield _RecordingSpan(observation, capture_content=self._capture_content)
        finally:
            # Always exits with no exception info, so we can never swallow the
            # caller's exception. Callers use `record_error` to mark failures.
            try:
                context_manager.__exit__(None, None, None)
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Langfuse span '{name}' failed to close: {error}")

    def _open(
        self, name: str, kind: Literal["span", "generation"], fields: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Start a span or generation. Returns `(None, None)` on failure."""
        try:
            starter = (
                self._client.start_as_current_generation
                if kind == "generation"
                else self._client.start_as_current_span
            )
            context_manager = starter(name=name, **fields)
            observation = context_manager.__enter__()
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Langfuse span '{name}' failed to open: {error}")
            return None, None
        return context_manager, observation

    async def flush(self) -> None:
        """Send queued traces, waiting at most `flush_timeout_seconds`.

        The SDK's flush blocks, so it runs in a thread under a timeout. A hung
        call leaks one thread instead of hanging shutdown.
        """

        try:
            async with asyncio.timeout(self._flush_timeout_seconds):
                await asyncio.to_thread(self._client.flush)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Langfuse flush failed or timed out: {error}")
