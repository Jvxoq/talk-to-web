"""The custom-stream key for work the user waits on but never sees text from.

Summarization is the first of those. It is a model call like any other, made
in the middle of a reply, and until it finishes nothing arrives on the stream -
so a long thread looks like a stalled one. Reporting it turns that silence into
a reason on screen, the same way `TOOL_START` does for a slow fetch.

In its own module rather than on `nodes.py` for the same reason `usage.py` is:
`nodes.py` imports `condenser.py`, and the summarization node imports
`nodes.py`, so a helper both need cannot live on either.
"""

from typing import Final, Literal

from langgraph.config import get_stream_writer
from loguru import logger

SUMMARIZE: Final = "summarize"

# The two states this reports, spelled out as a type rather than a bare `str`.
# The reader (`GenerateReply._to_event`) drops a payload whose status it does
# not recognise, silently and by design - so a typo here would cost the feature
# with no error anywhere. This makes that typo fail in the type checker instead.
SummarizeStatus = Literal["start", "done"]


def emit_summarizing(
    *, status: SummarizeStatus, tokens_before: int, tokens_after: int | None = None
) -> None:
    """Report one summarization step onto the custom stream, if there is one.

    `status` is `"start"` when the node decides the thread is over budget and
    `"done"` when the shortened history is ready. `tokens_after` is unknown at
    the start, which is what `None` means - not zero.

    The writer is missing whenever this node is called outside a graph run,
    which its unit tests do. Progress reporting must never be the reason a
    reply dies, so the failure is swallowed and logged at debug - the same
    posture as `emit_usage`.
    """
    try:
        writer = get_stream_writer()
    except Exception as error:
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
