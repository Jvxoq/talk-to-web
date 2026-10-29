"""The Server-Sent Events wire format. The only module that knows frame shapes.

Each frame is JSON, because a raw newline would break an SSE `data:` line.
"""

import json
from collections.abc import AsyncIterator

from app.application.chat.dto import (
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplySummarizing,
    ReplyToolFinished,
    ReplyToolStarted,
    ReplyUsage,
)


def format_frame(**payload: object) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def to_sse(events: AsyncIterator[ReplyEvent]) -> AsyncIterator[str]:
    async for event in events:
        match event:
            case ReplyDelta(text=text):
                yield format_frame(delta=text)
            # Its own key, so the client shows a chip instead of mixing it into the text.
            case ReplyToolStarted(name=name, summary=summary):
                yield format_frame(tool={"name": name, "status": "start", "summary": summary})
            case ReplyToolFinished(name=name, ok=ok, sources=sources):
                # "failed" is not an error frame. The model still answers.
                tool: dict[str, object] = {"name": name, "status": "ok" if ok else "failed"}
                # Omitted when empty. The client treats a missing key as none.
                if sources:
                    tool["sources"] = [
                        {"label": source.label, "url": source.url} for source in sources
                    ]
                yield format_frame(tool=tool)
            # Its own key, like tool activity. Unknown keys are ignored by old clients.
            case ReplySummarizing(
                status=status, tokens_before=tokens_before, tokens_after=tokens_after
            ):
                summarizing: dict[str, object] = {
                    "status": status,
                    "tokens_before": tokens_before,
                }
                # Omitted on the start frame, where it is not known yet.
                if tokens_after is not None:
                    summarizing["tokens_after"] = tokens_after
                yield format_frame(summarizing=summarizing)
            # Its own frame, so older clients can ignore it.

            case ReplyUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_ms=elapsed_ms,
                model=model,
            ):
                yield format_frame(
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "elapsed_ms": elapsed_ms,
                        "model": model,
                    }
                )
            case ReplyFailed(detail=detail):
                yield format_frame(error=detail)
            case ReplyCompleted():
                yield format_frame(done=True)
