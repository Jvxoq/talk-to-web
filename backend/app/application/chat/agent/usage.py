"""Reports token spend on the custom stream.

Its own module to avoid an import cycle between `nodes.py` and `condenser.py`.
"""

from typing import Final

from langgraph.config import get_stream_writer
from loguru import logger

from app.application.chat.models import TokenUsage

USAGE: Final = "usage"


def emit_usage(*, model: str, usage: TokenUsage) -> None:
    """Report one model call's spend, if a stream writer exists.

    Outside a graph run there is no writer, so this logs and returns.
    """

    try:
        writer = get_stream_writer()
    except Exception as error:  # noqa: BLE001
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
