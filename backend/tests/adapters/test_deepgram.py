"""The Deepgram session's error callback, reached the way the SDK reaches it."""

from collections.abc import Callable
from typing import Any

from deepgram.core.events import EventType

from app.adapters.transcription.deepgram import _DeepgramSession


class RecordingConnection:
    """Keeps the handlers the session registers, so a test can fire them."""

    def __init__(self) -> None:
        self.handlers: dict[EventType, Callable[[Any], Any]] = {}

    def on(self, event: EventType, handler: Callable[[Any], Any]) -> None:
        self.handlers[event] = handler


async def test_provider_error_text_stays_off_the_client() -> None:
    sent: list[str] = []

    async def on_error(detail: str) -> None:
        sent.append(detail)

    connection = RecordingConnection()
    session = _DeepgramSession(connection, on_error)
    session.register()

    await connection.handlers[EventType.ERROR]("Invalid API key sk-live-abc123 see https://dash")

    assert sent == ["Transcription failed"]
