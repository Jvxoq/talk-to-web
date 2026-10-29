"""Logging setup. Called once, from the composition root."""

import sys
from typing import Any

from loguru import logger

from app.observability.context import NO_REQUEST_ID, get_request_id

# Loguru's default format plus the request id. JSON logs ignore it.
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[request_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _add_request_id(record: Any) -> None:
    """Add the current request id to every record. An explicit `bind` still wins."""
    record["extra"].setdefault("request_id", get_request_id() or NO_REQUEST_ID)


def configure_logging(level: str, json_logs: bool) -> None:
    logger.remove()
    logger.configure(patcher=_add_request_id)
    logger.add(
        sys.stderr,
        level=level.upper(),
        serialize=json_logs,
        format=_CONSOLE_FORMAT,
        # Off: it prints local variables, including secrets.
        diagnose=False,
        backtrace=False,
    )
