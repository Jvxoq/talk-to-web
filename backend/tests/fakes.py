"""In-memory stand-ins for every port.

Fakes rather than mocks: these implement the behaviour a use case depends on, so
a test asserts what happened rather than which methods were called. Nothing here
inherits from a Protocol — structural typing is the whole point of the seam.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self

from app.application.chat.models import ChatMessage, ModelChunk
from app.application.chat.ports import ConversationRepository
from app.application.chat.tools.base import ToolOutcome, ToolSpec
from app.application.transcription.ports import ClientFrame, TranscriptionSession
from app.domain.chat.entities import Conversation, Message
from app.domain.ingestion.value_objects import Chunk, DocumentName
from app.domain.transcription.entities import AudioFormat, Transcript


class FakeConversationRepository:
    def __init__(self) -> None:
        self.rows: dict[int, Conversation] = {}
        self._next_id = 1

    async def get(self, conversation_id: int) -> Conversation | None:
        return self.rows.get(conversation_id)

    async def add(self, conversation: Conversation) -> Conversation:
        conversation.id = self._next_id
        conversation.created_at = conversation.updated_at = datetime.now(UTC)
        self.rows[self._next_id] = conversation
        self._next_id += 1
        return conversation

    async def add_message(self, conversation_id: int, message: Message) -> Message:
        message.id = self._next_id
        message.conversation_id = conversation_id
        message.created_at = datetime.now(UTC)
        self._next_id += 1
        return message

    async def delete(self, conversation_id: int) -> None:
        self.rows.pop(conversation_id, None)


class FakeUnitOfWork:
    """Records whether the use case actually committed."""

    # Annotated as the port: protocol attributes are invariant, so declaring the
    # concrete fake here would stop this class satisfying `UnitOfWork`.
    conversations: ConversationRepository

    def __init__(self, repository: FakeConversationRepository) -> None:
        self.conversations = repository
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if not self.committed:
            self.rolled_back = True

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


@dataclass
class UnitOfWorkSpy:
    """A factory that hands out units of work and remembers each one."""

    repository: FakeConversationRepository = field(default_factory=FakeConversationRepository)
    issued: list[FakeUnitOfWork] = field(default_factory=list)

    def __call__(self) -> FakeUnitOfWork:
        uow = FakeUnitOfWork(self.repository)
        self.issued.append(uow)
        return uow


class FakeChatModel:
    """
    Replays scripted turns, so a test can script the agent's decisions.

    One entry in `turns` is one model turn: the chunks it streams. A turn whose
    chunks carry `tool_calls` sends the graph round the tool loop; a turn of
    plain text ends it. Scripting two turns is therefore how a test says "call a
    tool, then answer".
    """

    def __init__(
        self,
        turns: Sequence[Sequence[ModelChunk]] = (),
        fail_with: Exception | None = None,
    ) -> None:
        self.turns = [list(turn) for turn in turns]
        self.fail_with = fail_with
        # What the graph handed us, turn by turn - this is how a test proves a
        # tool result actually made it back into the conversation.
        self.seen_messages: list[list[ChatMessage]] = []
        self.seen_tools: list[list[str]] = []
        self._call = 0

    async def stream(
        self,
        *,
        model: str,
        temperature: float,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ModelChunk]:
        self.seen_messages.append(list(messages))
        self.seen_tools.append([tool.name for tool in tools])

        # Past the end of the script, keep answering with plain text. That is
        # what lets a test drive the loop to its iteration ceiling without
        # scripting every lap.
        turn = self.turns[self._call] if self._call < len(self.turns) else [ModelChunk(text="done")]
        self._call += 1

        for chunk in turn:
            yield chunk

        if self.fail_with is not None:
            raise self.fail_with

    @property
    def calls(self) -> int:
        return self._call


class FakeAgentTool:
    """A tool that records its calls and can be told to fail."""

    def __init__(
        self,
        name: str = "fake_tool",
        result: str = "TOOL-RESULT",
        fail_with: Exception | None = None,
    ) -> None:
        self._name = name
        self.result = result
        self.fail_with = fail_with
        self.calls: list[dict[str, object]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"fake tool {self._name}",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, arguments: Mapping[str, object]) -> ToolOutcome:
        self.calls.append(dict(arguments))
        if self.fail_with is not None:
            # Deliberately raised, not returned: real tools go through
            # `BaseTool.run`, which catches. This one proves `ToolRegistry`
            # survives a tool that does not.
            raise self.fail_with
        return ToolOutcome(content=self.result)


class FakeWebSearcher:
    def __init__(self, results: str = "", fail_with: Exception | None = None) -> None:
        self.results = results
        self.fail_with = fail_with
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> str:
        self.queries.append((query, max_results))
        if self.fail_with is not None:
            raise self.fail_with
        return self.results


class FakeWebContentFetcher:
    def __init__(self, content: str = "", fail_with: Exception | None = None) -> None:
        self.content = content
        self.fail_with = fail_with
        self.calls: list[tuple[str, ...]] = []

    async def fetch_all(self, urls: Sequence[str]) -> str:
        self.calls.append(tuple(urls))
        if self.fail_with is not None:
            raise self.fail_with
        return self.content


class FakeKnowledgeRetriever:
    def __init__(self, passages: Sequence[str] = (), fail_with: Exception | None = None) -> None:
        self.passages = list(passages)
        self.fail_with = fail_with

    async def retrieve(self, query: str) -> list[str]:
        if self.fail_with is not None:
            raise self.fail_with
        return list(self.passages)


class FakeFileStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes]] = []

    async def save(self, name: DocumentName, stream: AsyncIterator[bytes], max_bytes: int) -> str:
        body = b"".join([chunk async for chunk in stream])
        self.saved.append((name.value, body))
        return f"uploads/{name.value}"


class FakeTextExtractor:
    def __init__(self, text: str = "") -> None:
        self.text = text

    async def extract(self, reference: str) -> str:
        return self.text


class FakeEmbedder:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.embedded: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [float(len(text))] * self.dimensions


class FakeVectorIndex:
    def __init__(self, hits: Sequence[str] = ()) -> None:
        self.hits = list(hits)
        self.resets: list[int] = []
        self.chunks: list[Chunk] = []

    async def reset(self, dimensions: int) -> None:
        self.resets.append(dimensions)
        self.chunks.clear()

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.chunks.extend(chunks)

    async def search(self, vector: list[float], limit: int, score_threshold: float) -> list[str]:
        return self.hits[:limit]


class FakeClientTransport:
    """Replays a scripted client and records everything sent back."""

    def __init__(self, frames: Sequence[ClientFrame]) -> None:
        self._frames = list(frames)
        self.sent: list[tuple[str, object]] = []

    async def receive(self) -> ClientFrame:
        if not self._frames:
            return ClientFrame(kind="disconnect")
        return self._frames.pop(0)

    async def ready(self) -> None:
        self.sent.append(("ready", None))

    async def transcript(self, transcript: Transcript) -> None:
        self.sent.append(("transcript", transcript.text))

    async def done(self) -> None:
        self.sent.append(("done", None))

    async def error(self, detail: str) -> None:
        self.sent.append(("error", detail))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.sent]


class FakeTranscriptionSession:
    def __init__(self, transcripts: Sequence[Transcript]) -> None:
        self._transcripts = list(transcripts)
        self.audio: list[bytes] = []
        self.finalized = False
        self._ended = asyncio.Event()

    async def send_audio(self, chunk: bytes) -> None:
        self.audio.append(chunk)

    async def finalize(self) -> None:
        self.finalized = True

    async def wait_flushed(self, timeout: float) -> bool:  # noqa: ASYNC109
        return True

    async def end_stream(self) -> None:
        self._ended.set()

    async def transcripts(self) -> AsyncIterator[Transcript]:
        for transcript in self._transcripts:
            yield transcript
        # Then hang, the way a live provider does between utterances, so the
        # test proves the pump is what ends the session — and finish, rather
        # than hang forever, once end_stream() says nothing more is coming.
        await self._ended.wait()


class FakeLiveTranscriber:
    def __init__(self, transcripts: Sequence[Transcript] = ()) -> None:
        self.session = FakeTranscriptionSession(transcripts)
        self.opened_with: AudioFormat | None = None

    @asynccontextmanager
    async def _open(self, audio_format: AudioFormat) -> AsyncIterator[FakeTranscriptionSession]:
        self.opened_with = audio_format
        yield self.session

    def open(
        self,
        audio_format: AudioFormat,
        on_error: Callable[[str], Awaitable[None]],
    ) -> AbstractAsyncContextManager[TranscriptionSession]:
        return self._open(audio_format)
