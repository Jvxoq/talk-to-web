"""The browser end of the transcription bridge.

This class is why `ClientTransport` exists in the application layer. For
transcription the WebSocket *is* the delivery mechanism, so the use case has to
speak to the client mid-flight — and a port is the only way to let it do that
without the application layer importing Starlette. The port is declared there;
the Starlette-shaped half lives here, in the delivery layer, where it belongs.
"""

from collections.abc import MutableMapping
from typing import Any

from fastapi import WebSocket

from app.application.transcription.ports import ClientFrame
from app.domain.transcription.entities import Transcript


class WebSocketTransport:
    """Implements the `ClientTransport` port over a FastAPI WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def receive(self) -> ClientFrame:
        frame: MutableMapping[str, Any] = await self._websocket.receive()

        if frame["type"] == "websocket.disconnect":
            return ClientFrame(kind="disconnect")

        audio = frame.get("bytes")
        if audio is not None:
            return ClientFrame(kind="audio", audio=audio)

        return ClientFrame(kind="text", text=frame.get("text") or "")

    async def ready(self) -> None:
        await self._websocket.send_json({"type": "ready"})

    async def transcript(self, transcript: Transcript) -> None:
        await self._websocket.send_json(
            {
                "type": "transcript",
                "text": transcript.text,
                "is_final": transcript.is_final,
                "speech_final": transcript.speech_final,
            }
        )

    async def done(self) -> None:
        await self._websocket.send_json({"type": "done"})

    async def error(self, detail: str) -> None:
        await self._websocket.send_json({"type": "error", "detail": detail})
