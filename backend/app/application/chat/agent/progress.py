"""Custom-stream events for silent work, like summarization.

A summary is a model call with no text, so a long thread looks stalled
without this. Its own module to avoid an import cycle with `nodes.py`.
"""

from typing import Final, Literal

from langgraph.config import get_stream_writer
from loguru import logger

SUMMARIZE: Final = "summarize"

# A Literal, so a typo fails in mypy. The reader silently drops unknown statuses.
SummarizeStatus = Literal["start", "done"]


def emit_summarizing(
    *, status: SummarizeStatus, tokens_before: int, tokens_after: int | None = None
) -> None:
    """Report a summarization step on the custom stream, if one exists.

    `tokens_after` is `None` on "start". Outside a graph run there is no
    writer, so this logs and returns.
    """

    try:
        writer = get_stream_writer()
    except Exception as error:  # noqa: BLE001
        logger.debug(f"No stream writer available to report summarization: {error}")
        return
    writer(
        {
            "type": SUMMARIZE,
            "status": status,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        }
    )
