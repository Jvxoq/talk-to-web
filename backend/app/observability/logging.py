"""Logging setup. Called once, from the composition root."""

import sys

from loguru import logger


def configure_logging(level: str, json_logs: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        serialize=json_logs,
        # `diagnose` prints the local variables of every frame in a traceback,
        # which happily includes API keys and connection strings.
        diagnose=False,
        backtrace=False,
    )
