"""The Server-Sent Events wire format.

The only module that knows what a frame looks like. Each frame carries one JSON
object rather than bare text because a model token can contain newlines, and a
newline in an SSE `data:` line means "next field", not "next character".
"""

import json
from collections.abc import AsyncIterator

from app.application.chat.dto import (
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
)


def format_frame(**payload: object) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def to_sse(events: AsyncIterator[ReplyEvent]) -> AsyncIterator[str]:
    async for event in events:
        match event:
            case ReplyDelta(text=text):
                yield format_frame(delta=text)
            # Tool activity rides in its own key rather than as more `delta`
            # text, so the client can render it as a chip beside the answer
            # instead of splicing status lines into what the model said.
            case ReplyToolStarted(name=name, summary=summary):
                yield format_frame(tool={"name": name, "status": "start", "summary": summary})
            case ReplyToolFinished(name=name, ok=ok):
                # "failed" is not an error frame: a tool that came back empty
                # costs the answer some grounding, and the model answers anyway.
                yield format_frame(tool={"name": name, "status": "ok" if ok else "failed"})
            case ReplyFailed(detail=detail):
                yield format_frame(error=detail)
            case ReplyCompleted():
                yield format_frame(done=True)
