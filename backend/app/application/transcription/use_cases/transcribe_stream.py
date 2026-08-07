"""Bridge a browser audio stream to a live speech-to-text provider."""

import asyncio
import json

from loguru import logger

from app.application.transcription.ports import (
    ClientTransport,
    LiveTranscriber,
    TranscriptionSession,
)
from app.domain.transcription.entities import AudioFormat

_START_FRAME_SHAPE = '{"type": "start", "sample_rate": <int>}'


def _parse_json_object(raw: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_frame_type(raw: str) -> str | None:
    """The `type` of a control frame, or None when the frame cannot be read."""
    payload = _parse_json_object(raw)
    if payload is None:
        return None
    frame_type = payload.get("type")
    return frame_type if isinstance(frame_type, str) else None


class TranscribeStream:
    """
    Carry audio one way and transcripts the other for as long as the client talks.

    The two directions run as concurrent tasks because neither may block the
    other: audio arriving must not wait on a transcript being delivered, or the
    provider starves and the transcript drifts behind the speaker.
    """

    def __init__(self, transcriber: LiveTranscriber, finalize_timeout: float) -> None:
        self._transcriber = transcriber
        self._finalize_timeout = finalize_timeout

    async def __call__(self, transport: ClientTransport) -> None:
        audio_format = await self._negotiate(transport)
        if audio_format is None:
            return

        async with self._transcriber.open(audio_format, on_error=transport.error) as session:
            await transport.ready()
            async with asyncio.TaskGroup() as tasks:
                relay = tasks.create_task(self._relay(session, transport))
                tasks.create_task(self._pump(session, transport, relay))

    async def _negotiate(self, transport: ClientTransport) -> AudioFormat | None:
        """Read the opening handshake, or tell the client why we cannot start."""
        frame = await transport.receive()
        if frame.kind != "text" or frame.text is None:
            await transport.error(f"Expected a start frame {_START_FRAME_SHAPE} before any audio")
            return None

        payload = _parse_json_object(frame.text)
        if payload is None or payload.get("type") != "start":
            await transport.error(f"First frame must be {_START_FRAME_SHAPE}")
            return None

        sample_rate = payload.get("sample_rate")
        # bool is an int subclass, and `True` is not a sample rate.
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
            await transport.error("Start frame needs an integer 'sample_rate'")
            return None

        try:
            return AudioFormat(sample_rate=sample_rate)
        except ValueError as exc:
            await transport.error(str(exc))
            return None

    async def _pump(
        self,
        session: TranscriptionSession,
        transport: ClientTransport,
        relay: asyncio.Task[None],
    ) -> None:
        """Forward client frames to the provider until the client stops or leaves."""
        try:
            while True:
                frame = await transport.receive()

                if frame.kind == "disconnect":
                    return

                if frame.kind == "audio":
                    if frame.audio:
                        await session.send_audio(frame.audio)
                    continue

                if _parse_frame_type(frame.text or "") == "stop":
                    await session.finalize()
                    if not await session.wait_flushed(self._finalize_timeout):
                        logger.debug(
                            "Provider did not flush within {}s; ending anyway",
                            self._finalize_timeout,
                        )
                    # Let the relay deliver what the provider already sent before
                    # the socket is torn down. Dropping straight through to the
                    # `finally` cancelled it instead, which threw away the very
                    # finalized transcript the client had just waited for.
                    await self._drain(relay, session)
                    await transport.done()
                    return
        finally:
            # Nothing more will arrive, so the relay would hang on an iterator
            # that never ends. A child cancelled from inside the group is not
            # reported as a group error, so this CancelledError stops here.
            relay.cancel()

    async def _drain(self, relay: asyncio.Task[None], session: TranscriptionSession) -> None:
        """Wait for the relay to finish delivering, but never indefinitely."""
        await session.end_stream()
        try:
            await asyncio.wait_for(relay, self._finalize_timeout)
        except TimeoutError:
            # `wait_for` has already cancelled the relay; the session is ending
            # regardless, so a slow client costs it the tail of the transcript
            # rather than holding the socket open.
            logger.debug("Relay did not drain within {}s", self._finalize_timeout)

    async def _relay(self, session: TranscriptionSession, transport: ClientTransport) -> None:
        """Forward provider transcripts back to the client."""
        async for transcript in session.transcripts():
            if transcript.is_worth_sending():
                await transport.transcript(transcript)
