import asyncio
import json
from typing import Any, MutableMapping

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1Results
from fastapi import WebSocket
from loguru import logger

from config import DEEPGRAM_API_KEY

# Deepgram live transcription settings. The browser sends raw 16-bit PCM, so we
# must declare encoding/sample_rate/channels explicitly - there is no container
# for Deepgram to read them from.
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_ENCODING = "linear16"
DEEPGRAM_CHANNELS = 1
# Silence (ms) after which Deepgram decides the speaker finished an utterance.
UTTERANCE_END_MS = 1500
# How long to wait for Deepgram to flush its last transcript after we finalize.
FINALIZE_TIMEOUT_S = 3.0


class WSConnectionManager:
    """
    Wraps the WebSocket lifecycle for the transcription endpoint.

    Deliberately stateless: transcription is one private socket per speaker,
    so there is no connection registry to keep. Add one back if a fan-out
    feature (presence, broadcast) ever needs it.
    """

    # Open web socket connections using accept()
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()

    # Close web socket connections using close()
    async def disconnect(self, websocket: WebSocket) -> None:
        # The client may already be gone, in which case close() raises.
        try:
            await websocket.close()
        except RuntimeError:
            pass

    # Send data. Everything this endpoint returns is JSON.
    async def send(self, message: dict, websocket: WebSocket) -> None:
        await websocket.send_json(message)

    # Receive a raw frame, which may carry either a text or a binary payload.
    # The transcription protocol mixes both, so neither receive_text() nor
    # receive_bytes() alone is enough.
    async def receive_frame(self, websocket: WebSocket) -> MutableMapping[str, Any]:
        return await websocket.receive()


async def transcribe_audio_stream(
    websocket: WebSocket,
    manager: WSConnectionManager,
) -> None:
    """
    Bridge a browser microphone stream to Deepgram live speech-to-text.

    Protocol, browser -> server:
        {"type": "start", "sample_rate": 16000}   opening handshake (text)
        <binary frames>                           raw linear16 PCM audio
        {"type": "stop"}                          flush and finish (text)

    Protocol, server -> browser:
        {"type": "ready"}                         Deepgram socket is open
        {"type": "transcript", "text": ..., "is_final": ...}
        {"type": "done"}                          final transcript flushed
        {"type": "error", "detail": ...}
    """
    sample_rate = await _read_start_frame(websocket, manager)
    if sample_rate is None:
        return

    client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    async with client.listen.v1.connect(
        model=DEEPGRAM_MODEL,
        encoding=DEEPGRAM_ENCODING,
        sample_rate=sample_rate,
        channels=DEEPGRAM_CHANNELS,
        # Punctuation, capitalisation and number formatting.
        smart_format=True,
        # Partial transcripts while the user is still speaking.
        interim_results=True,
        utterance_end_ms=UTTERANCE_END_MS,
    ) as deepgram:
        # Set once Deepgram has flushed the transcript we asked for with
        # send_finalize(), which is how we know it is safe to close.
        flushed = asyncio.Event()

        async def on_message(message: Any) -> None:
            if not isinstance(message, ListenV1Results):
                return
            if not message.channel or not message.channel.alternatives:
                return

            transcript = message.channel.alternatives[0].transcript
            if transcript.strip():
                await manager.send(
                    {
                        "type": "transcript",
                        "text": transcript,
                        # False -> interim guess, will be revised.
                        # True  -> committed, safe to append to the chat box.
                        "is_final": bool(message.is_final),
                        # True -> the speaker also paused (end of utterance).
                        "speech_final": bool(getattr(message, "speech_final", False)),
                    },
                    websocket,
                )

            if getattr(message, "from_finalize", False):
                flushed.set()

        async def on_error(error: Any) -> None:
            logger.warning(f"Deepgram stream error: {error}")
            await manager.send({"type": "error", "detail": str(error)}, websocket)

        deepgram.on(EventType.MESSAGE, on_message)
        deepgram.on(EventType.ERROR, on_error)

        # start_listening() is the Deepgram receive loop; it has to run
        # concurrently with our own browser receive loop below.
        listener = asyncio.create_task(deepgram.start_listening())
        await manager.send({"type": "ready"}, websocket)

        try:
            await _pump_audio(websocket, manager, deepgram, flushed)
        finally:
            listener.cancel()
            try:
                await deepgram.send_close_stream()
            except Exception:
                pass


async def _read_start_frame(
    websocket: WebSocket,
    manager: WSConnectionManager,
) -> int | None:
    """
    Read the opening frame and return the client's PCM sample rate.

    The browser reports the rate its AudioContext actually settled on rather
    than us assuming 16 kHz - a mismatch here does not raise, it just produces
    transcripts of chipmunks.
    """
    frame = await manager.receive_frame(websocket)
    raw = frame.get("text")
    if raw is None:
        await manager.send(
            {"type": "error", "detail": "Expected a JSON start frame"}, websocket
        )
        return None

    try:
        sample_rate = int(json.loads(raw)["sample_rate"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        await manager.send(
            {"type": "error", "detail": "Malformed start frame"}, websocket
        )
        return None

    logger.debug(f"Transcription session starting at {sample_rate} Hz")
    return sample_rate


async def _pump_audio(
    websocket: WebSocket,
    manager: WSConnectionManager,
    deepgram: Any,
    flushed: asyncio.Event,
) -> None:
    """
    Forward browser audio frames to Deepgram until the client stops or leaves.
    """
    while True:
        frame = await manager.receive_frame(websocket)

        if frame["type"] == "websocket.disconnect":
            return

        if (chunk := frame.get("bytes")) is not None:
            await deepgram.send_media(chunk)
            continue

        if frame.get("text") and _is_stop_frame(frame["text"]):
            # Force Deepgram to emit whatever it is still holding, instead of
            # waiting for its own silence-based endpointing to fire.
            await deepgram.send_finalize()
            try:
                await asyncio.wait_for(flushed.wait(), timeout=FINALIZE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.debug("Timed out waiting for Deepgram to flush")
            await manager.send({"type": "done"}, websocket)
            return


def _is_stop_frame(raw: str) -> bool:
    try:
        return json.loads(raw).get("type") == "stop"
    except (json.JSONDecodeError, AttributeError):
        return False
