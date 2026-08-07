"""The transcription bridge, driven by a scripted client."""

import asyncio
import json

import pytest

from app.application.transcription.ports import ClientFrame
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream
from app.domain.transcription.entities import Transcript
from tests.fakes import FakeClientTransport, FakeLiveTranscriber


def start(sample_rate: int | object = 16_000) -> ClientFrame:
    return ClientFrame(kind="text", text=json.dumps({"type": "start", "sample_rate": sample_rate}))


STOP = ClientFrame(kind="text", text=json.dumps({"type": "stop"}))


async def run(transport: FakeClientTransport, transcriber: FakeLiveTranscriber) -> None:
    # The relay never ends on its own — a real provider socket stays open — so a
    # hang here is a genuine bug, not a slow test.
    await asyncio.wait_for(TranscribeStream(transcriber, finalize_timeout=0.1)(transport), 2.0)


class TestHandshake:
    @pytest.mark.parametrize(
        "first",
        [
            ClientFrame(kind="audio", audio=b"\x00\x01"),
            ClientFrame(kind="text", text="not json"),
            ClientFrame(kind="text", text=json.dumps({"type": "stop"})),
            ClientFrame(kind="text", text=json.dumps({"type": "start"})),
            ClientFrame(kind="text", text=json.dumps({"type": "start", "sample_rate": "16000"})),
            ClientFrame(kind="text", text=json.dumps({"type": "start", "sample_rate": True})),
            ClientFrame(kind="text", text=json.dumps({"type": "start", "sample_rate": 3})),
        ],
    )
    async def test_a_bad_opening_frame_is_refused_with_an_error(self, first: ClientFrame) -> None:
        transport = FakeClientTransport([first])
        transcriber = FakeLiveTranscriber()

        await run(transport, transcriber)

        assert transport.kinds() == ["error"]
        assert transcriber.opened_with is None, "no provider session for a bad handshake"

    async def test_a_good_handshake_opens_the_provider_and_says_ready(self) -> None:
        transport = FakeClientTransport([start(), STOP])
        transcriber = FakeLiveTranscriber()

        await run(transport, transcriber)

        assert transcriber.opened_with is not None
        assert transcriber.opened_with.sample_rate == 16_000
        assert transport.kinds() == ["ready", "done"]


class TestStreaming:
    async def test_forwards_audio_and_finalizes_on_stop(self) -> None:
        transport = FakeClientTransport(
            [
                start(),
                ClientFrame(kind="audio", audio=b"aaa"),
                ClientFrame(kind="audio", audio=b"bbb"),
                STOP,
            ]
        )
        transcriber = FakeLiveTranscriber()

        await run(transport, transcriber)

        assert transcriber.session.audio == [b"aaa", b"bbb"]
        assert transcriber.session.finalized
        assert transport.kinds() == ["ready", "done"]

    async def test_relays_transcripts_but_skips_blank_ones(self) -> None:
        transport = FakeClientTransport([start(), STOP])
        transcriber = FakeLiveTranscriber(
            transcripts=[
                Transcript("hello", is_final=False),
                Transcript("   ", is_final=True),
                Transcript("hello world", is_final=True, speech_final=True),
            ]
        )

        await run(transport, transcriber)

        sent = [payload for kind, payload in transport.sent if kind == "transcript"]
        assert sent == ["hello", "hello world"]

    async def test_client_disconnect_ends_the_session_without_a_done_frame(self) -> None:
        transport = FakeClientTransport(
            [start(), ClientFrame(kind="audio", audio=b"aaa"), ClientFrame(kind="disconnect")]
        )
        transcriber = FakeLiveTranscriber()

        await run(transport, transcriber)

        assert transport.kinds() == ["ready"]
        assert not transcriber.session.finalized

    async def test_empty_audio_frames_are_not_forwarded(self) -> None:
        transport = FakeClientTransport([start(), ClientFrame(kind="audio", audio=b""), STOP])
        transcriber = FakeLiveTranscriber()

        await run(transport, transcriber)

        assert transcriber.session.audio == []
