"""The custom-stream key for token spend, and a safe way to report it.

This lives in its own module rather than on `nodes.py` to break an import
cycle: `nodes.py` imports `condenser.py`, and the condenser also needs to
report usage, so the helper cannot live on the module that imports the thing
that needs it.
"""

from typing import Final

from langgraph.config import get_stream_writer
from loguru import logger

from app.application.chat.models import TokenUsage

USAGE: Final = "usage"


def emit_usage(*, model: str, usage: TokenUsage) -> None:
    """Report one model call's spend onto the custom stream, if there is one.

    The condenser calls this outside any graph run in its own unit tests, and
    `get_stream_writer()` raises there - there is no run config to pull a
    writer out of. A metrics helper must never be the reason a reply dies, so
    the failure is swallowed and logged at debug rather than propagated.
    """
    try:
        writer = get_stream_writer()
    except Exception as error:
        logger.debug(f"No stream writer available to report usage for {model}: {error}")
        return
    writer(
        {
            "type": USAGE,
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
    )
