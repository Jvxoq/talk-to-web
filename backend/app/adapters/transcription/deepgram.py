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

# The browser sends raw 16-bit PCM, so encoding and channel count must be
# declared explicitly - there is no container for Deepgram to read them from.
DEEPGRAM_ENCODING = "linear16"
DEEPGRAM_CHANNELS = 1


class _DeepgramSession:
    """
    One open Deepgram socket, exposed as a `TranscriptionSession`.

    Deepgram pushes results through callbacks, but the use case wants to pull
    them with `async for`; the queue in the middle is what bridges those two
    shapes without either side knowing about the other.
    """

    def __init__(self, connection: Any, on_error: Callable[[str], Awaitable[None]]) -> None:
        self._connection = connection
        self._on_error = on_error
        self._queue: asyncio.Queue[Transcript | None] = asyncio.Queue()
        # Set once Deepgram flushes the transcript we asked for with
        # send_finalize(), which is how we know it is safe to close.
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
            # Signalled only once the transcript is on the queue. Setting it
            # first meant the caller could be told "flushed" and tear the
            # session down while the finalized transcript was still unread -
            # the whole point of finalizing was to receive that one.
            if from_finalize:
                self._flushed.set()

    async def _handle_error(self, error: Any) -> None:
        logger.warning(f"Deepgram stream error: {error}")
        await self._on_error(str(error))

    async def transcripts(self) -> AsyncIterator[Transcript]:
        """Yield transcripts as they arrive, until the session is torn down."""
        while True:
            transcript = await self._queue.get()
            # `None` is the sentinel the context manager pushes on teardown; a
            # queue has no other way to say "no more items are coming".
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
        except Exception as error:
            # Best effort: the socket is very often already gone by the time we
            # get here, and failing to say goodbye is not worth an exception.
            logger.debug(f"Ignoring Deepgram close failure: {error}")


class DeepgramLiveTranscriber:
    """
    Opens Deepgram live-transcription sessions.

    Satisfies `app.application.transcription.ports.LiveTranscriber`. A session
    is per-speaker and short-lived, so the client is built inside `open()`
    rather than held on the adapter - only the credentials and settings are.
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

            # start_listening() is Deepgram's receive loop; it has to run
            # concurrently with the caller's own send loop.
            listener = asyncio.create_task(connection.start_listening())
            try:
                yield session
            finally:
                listener.cancel()
                await session.shutdown()
