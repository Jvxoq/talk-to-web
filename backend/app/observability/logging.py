"""Logging setup. Called once, from the composition root."""

import sys
from typing import Any

from loguru import logger

from app.observability.context import NO_REQUEST_ID, get_request_id

# The default loguru format with the request id spliced in before the message.
# Only the human-readable sink uses it; the JSON sink serializes `extra` itself.
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[request_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _add_request_id(record: Any) -> None:
    """Stamp every record with the request being served, if there is one.

    A patcher rather than `logger.bind()` at each call site: the whole value of
    a correlation id is that nobody has to remember to attach it. `setdefault`
    so an explicit `bind(request_id=...)` - a background job replaying one, say
    - still wins.
    """
    record["extra"].setdefault("request_id", get_request_id() or NO_REQUEST_ID)


def configure_logging(level: str, json_logs: bool) -> None:
    logger.remove()
    logger.configure(patcher=_add_request_id)
    logger.add(
        sys.stderr,
        level=level.upper(),
        serialize=json_logs,
        format=_CONSOLE_FORMAT,
        # `diagnose` prints the local variables of every frame in a traceback,
        # which happily includes API keys and connection strings.
        diagnose=False,
        backtrace=False,
    )
