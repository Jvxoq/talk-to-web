"""What the transcription use case needs from both sides of the bridge.

Two directions meet here. `ClientTransport` is the browser side, implemented by
the API layer (the WebSocket *is* the delivery mechanism). `LiveTranscriber` is
the provider side, implemented by an adapter. The use case in between never
names WebSockets or Deepgram.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from app.domain.transcription.entities import AudioFormat, Transcript


@dataclass(frozen=True, slots=True)
class ClientFrame:
    """One frame off the browser socket. Exactly one payload field is set."""

    kind: Literal["text", "audio", "disconnect"]
    text: str | None = None
    audio: bytes | None = None


class ClientTransport(Protocol):
    """The browser end of the bridge."""

    async def receive(self) -> ClientFrame: ...

    async def ready(self) -> None: ...

    async def transcript(self, transcript: Transcript) -> None: ...

    async def done(self) -> None: ...

    async def error(self, detail: str) -> None: ...


class RateLimiter(Protocol):
    """Counts requests against a key and refuses the ones over budget.

    Declared here rather than imported from chat or ingestion, even though the
    shape is identical: a port belongs to the layer that consumes it. One
    adapter satisfies all three, structurally, without knowing the others exist.
    """

    async def hit(self, key: str) -> None:
        """Record one request, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None: ...


class TranscriptionSession(Protocol):
    """An open connection to the speech-to-text provider."""

    async def send_audio(self, chunk: bytes) -> None: ...

    async def finalize(self) -> None:
        """Ask the provider to emit whatever it is still holding."""
        ...

    async def wait_flushed(self, timeout: float) -> bool:  # noqa: ASYNC109
        """True once the finalized transcript arrived, False on timeout."""
        ...

    async def end_stream(self) -> None:
        """Close `transcripts()` once everything already received has been read.

        This is what lets a caller stop relaying without discarding transcripts
        the provider has already delivered: the iterator finishes rather than
        being cancelled mid-flight.
        """
        ...

    def transcripts(self) -> AsyncIterator[Transcript]: ...


class LiveTranscriber(Protocol):
    """Opens provider sessions."""

    def open(
        self,
        audio_format: AudioFormat,
        on_error: Callable[[str], Awaitable[None]],
    ) -> AbstractAsyncContextManager[TranscriptionSession]: ...
