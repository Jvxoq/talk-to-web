"""Deepgram live speech-to-text."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1Results
from loguru import logger

from app.domain.transcription.entities import AudioFormat, Transcript

# Raw 16-bit PCM has no container, so Deepgram must be told the format.
DEEPGRAM_ENCODING = "linear16"
DEEPGRAM_CHANNELS = 1


class _DeepgramSession:
    """One open Deepgram socket, as a `TranscriptionSession`.

    A queue turns Deepgram's callbacks into an `async for` stream.
    """

    def __init__(self, connection: Any, on_error: Callable[[str], Awaitable[None]]) -> None:
        self._connection = connection
        self._on_error = on_error
        self._queue: asyncio.Queue[Transcript | None] = asyncio.Queue()
        # Set when the finalized transcript arrives, so closing is safe.
        self._flushed = asyncio.Event()

    def register(self) -> None:
        """Attach the provider callbacks. Called once, inside the context manager."""
        self._connection.on(EventType.MESSAGE, self._handle_message)
        self._connection.on(EventType.ERROR, self._handle_error)

    async def _handle_message(self, message: Any) -> None:
        if not isinstance(message, ListenV1Results):
            return

        from_finalize = bool(getattr(message, "from_finalize", False))
        try:
            if not message.channel or not message.channel.alternatives:
                return

            text = message.channel.alternatives[0].transcript
            if not text.strip():
                return

            await self._queue.put(
                Transcript(
                    text=text,
                    is_final=bool(message.is_final),
                    speech_final=bool(getattr(message, "speech_final", False)),
                )
            )
        finally:
            # Set after queueing, so the final transcript is not lost on teardown.
            if from_finalize:
                self._flushed.set()

    async def _handle_error(self, error: Any) -> None:
        logger.warning(f"Deepgram stream error: {error}")
        await self._on_error(str(error))

    async def transcripts(self) -> AsyncIterator[Transcript]:
        """Yield transcripts as they arrive, until the session is torn down."""
        while True:
            transcript = await self._queue.get()
            # `None` marks the end of the stream.
            if transcript is None:
                return
            yield transcript

    async def send_audio(self, chunk: bytes) -> None:
        """Forward one PCM frame to the provider."""
        await self._connection.send_media(chunk)

    async def finalize(self) -> None:
        """Make Deepgram emit whatever it is holding, without waiting for silence."""
        await self._connection.send_finalize()

    async def wait_flushed(self, timeout: float) -> bool:  # noqa: ASYNC109
        """True once the finalized transcript arrived, False if it never did."""
        try:
            await asyncio.wait_for(self._flushed.wait(), timeout=timeout)
        except TimeoutError:
            logger.debug("Timed out waiting for Deepgram to flush")
            return False
        return True

    async def end_stream(self) -> None:
        """Let `transcripts()` finish once the queued transcripts have been read."""
        await self._queue.put(None)

    async def shutdown(self) -> None:
        """Unblock `transcripts()` and ask the provider to close politely."""
        await self._queue.put(None)
        try:
            await self._connection.send_close_stream()
        except Exception as error:  # noqa: BLE001
            # Best effort: the socket is often already gone.
            logger.debug(f"Ignoring Deepgram close failure: {error}")


class DeepgramLiveTranscriber:
    """Opens Deepgram live sessions. Implements `LiveTranscriber`.

    Sessions are short-lived, so a client is built per `open()`.
    """

    def __init__(self, api_key: str, model: str, utterance_end_ms: int) -> None:
        self._api_key = api_key
        self._model = model
        self._utterance_end_ms = utterance_end_ms

    def open(
        self,
        audio_format: AudioFormat,
        on_error: Callable[[str], Awaitable[None]],
    ) -> AbstractAsyncContextManager[_DeepgramSession]:
        """Open a provider session for one speaker's audio stream."""
        return self._open(audio_format, on_error)

    @asynccontextmanager
    async def _open(
        self,
        audio_format: AudioFormat,
        on_error: Callable[[str], Awaitable[None]],
    ) -> AsyncIterator[_DeepgramSession]:
        client = AsyncDeepgramClient(api_key=self._api_key)
        logger.debug(f"Opening Deepgram session at {audio_format.sample_rate} Hz")

        async with client.listen.v1.connect(
            model=self._model,
            encoding=audio_format.encoding or DEEPGRAM_ENCODING,
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels or DEEPGRAM_CHANNELS,
            # Punctuation, capitalisation and number formatting.
            smart_format=True,
            # Partial transcripts while the user is still speaking.
            interim_results=True,
            utterance_end_ms=self._utterance_end_ms,
        ) as connection:
            session = _DeepgramSession(connection, on_error)
            session.register()

            # Deepgram's receive loop runs beside the caller's send loop.

            listener = asyncio.create_task(connection.start_listening())
            try:
                yield session
            finally:
                listener.cancel()
                await session.shutdown()
