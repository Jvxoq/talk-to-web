"""In-memory stand-ins for every port.

Fakes rather than mocks: these implement the behaviour a use case depends on, so
a test asserts what happened rather than which methods were called. Nothing here
inherits from a Protocol — structural typing is the whole point of the seam.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Self

from app.application.chat.models import ChatMessage, ModelChunk, Passage, SearchResult, Source
from app.application.chat.ports import ConversationRepository
from app.application.chat.tools.base import ToolContext, ToolOutcome, ToolSpec
from app.application.identity.dto import IssuedAccessToken, TokenClaims
from app.application.identity.ports import RefreshTokenRepository, UserRepository
from app.application.ingestion.ports import DocumentRepository
from app.application.transcription.ports import ClientFrame, TranscriptionSession
from app.domain.chat.entities import Conversation, Message
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.errors import InvalidToken
from app.domain.identity.value_objects import Email
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.value_objects import Chunk, DocumentName
from app.domain.transcription.entities import AudioFormat, Transcript
from app.domain.usage.errors import RateLimited


class FakeConversationRepository:
    """In-memory conversations, scoped by owner exactly as the SQL is.

    The owner check lives inside `get` and `delete` rather than in the test that
    calls them, because that is where the real repository puts it - a fake that
    ignored the argument would let a use case forget to pass it and still pass.
    """

    def __init__(self) -> None:
        self.rows: dict[int, Conversation] = {}
        self._next_id = 1

    async def get(self, conversation_id: int, owner_id: int) -> Conversation | None:
        row = self.rows.get(conversation_id)
        if row is None or row.owner_id != owner_id:
            return None
        return row

    async def list_by_owner(self, owner_id: int) -> list[Conversation]:
        matches = [row for row in self.rows.values() if row.owner_id == owner_id]
        return sorted(
            matches, key=lambda c: c.updated_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

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

    async def delete(self, conversation_id: int, owner_id: int) -> None:
        row = self.rows.get(conversation_id)
        if row is not None and row.owner_id == owner_id:
            del self.rows[conversation_id]


class FakeUserRepository:
    def __init__(self) -> None:
        self.rows: dict[int, User] = {}
        self._next_id = 1

    async def get(self, user_id: int) -> User | None:
        return self.rows.get(user_id)

    async def get_by_email(self, email: Email) -> User | None:
        return next((u for u in self.rows.values() if u.email == email), None)

    async def add(self, user: User) -> User:
        user.id = self._next_id
        user.created_at = user.updated_at = datetime.now(UTC)
        self.rows[self._next_id] = user
        self._next_id += 1
        return user


class FakeRefreshTokenRepository:
    def __init__(self) -> None:
        self.rows: dict[int, RefreshToken] = {}
        self._next_id = 1

    async def add(self, token: RefreshToken) -> RefreshToken:
        token.id = self._next_id
        token.created_at = datetime.now(UTC)
        self.rows[self._next_id] = token
        self._next_id += 1
        return token

    async def get_by_fingerprint(self, fingerprint: str) -> RefreshToken | None:
        return next((t for t in self.rows.values() if t.fingerprint == fingerprint), None)

    async def revoke(self, token_id: int, at: datetime) -> None:
        row = self.rows.get(token_id)
        if row is not None:
            # `revoke` is idempotent on the entity, which is what stops a second
            # rotation from moving the timestamp.
            row.revoke(at)

    async def revoke_all_for_user(self, user_id: int, at: datetime) -> None:
        for row in self.rows.values():
            if row.user_id == user_id:
                row.revoke(at)

    async def delete_expired_before(self, cutoff: datetime) -> int:
        expired = [token_id for token_id, row in self.rows.items() if row.expires_at < cutoff]
        for token_id in expired:
            del self.rows[token_id]
        return len(expired)


class FakePasswordHasher:
    """Reversible on purpose: a test asserts on behaviour, not on Argon2.

    The real hasher is exercised by its own test; here the point is only that
    the same password verifies and a different one does not.
    """

    def hash(self, plain: str) -> str:
        return f"hashed:{plain}"

    def verify(self, plain: str, hashed: str) -> bool:
        return hashed == f"hashed:{plain}"

    def dummy_hash(self) -> str:
        return "hashed:\x00nobody"


class FakeAccessTokenCodec:
    """Issues readable tokens, and can be told to reject one."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self.rejected: set[str] = set()

    def issue(self, user_id: int, email: str, now: datetime) -> IssuedAccessToken:
        return IssuedAccessToken(token=f"access:{user_id}:{email}", expires_at=now + self._ttl)

    def decode(self, token: str) -> TokenClaims:
        if token in self.rejected:
            raise InvalidToken("rejected by the fake")
        parts = token.split(":")
        if len(parts) != 3 or parts[0] != "access":
            raise InvalidToken()
        return TokenClaims(user_id=int(parts[1]), email=parts[2])


class FakeRefreshTokenFactory:
    """Hands out predictable secrets so a test can name the one it means."""

    def __init__(self) -> None:
        self.issued = 0

    def new_secret(self) -> str:
        self.issued += 1
        return f"secret-{self.issued}"

    def fingerprint(self, secret: str) -> str:
        return f"fp:{secret}"


class FakeClock:
    """A clock a test moves by hand, so expiry needs no sleeping."""

    def __init__(self, now: datetime | None = None) -> None:
        self.current = now or datetime(2026, 1, 1, tzinfo=UTC)

    async def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class FakeRateLimiter:
    """Counts hits per key and refuses past a budget. No clock, no window."""

    def __init__(self, max_attempts: int = 1_000) -> None:
        self.max_attempts = max_attempts
        self.hits: dict[str, int] = {}

    async def hit(self, key: str) -> None:
        if self.hits.get(key, 0) >= self.max_attempts:
            raise RateLimited(60)
        self.hits[key] = self.hits.get(key, 0) + 1

    async def reset(self, key: str) -> None:
        self.hits.pop(key, None)


class FakeDocumentRepository:
    """In-memory documents, scoped by owner exactly as the SQL is."""

    def __init__(self) -> None:
        self.rows: dict[int, UploadedDocument] = {}
        self._next_id = 1

    async def get(self, document_id: int, owner_id: int) -> UploadedDocument | None:
        row = self.rows.get(document_id)
        if row is None or row.owner_id != owner_id:
            return None
        return row

    async def list_by_owner(self, owner_id: int) -> list[UploadedDocument]:
        matches = [row for row in self.rows.values() if row.owner_id == owner_id]
        return sorted(
            matches, key=lambda d: d.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

    async def add(self, document: UploadedDocument) -> UploadedDocument:
        document.id = self._next_id
        document.created_at = datetime.now(UTC)
        self.rows[self._next_id] = document
        self._next_id += 1
        return document

    async def set_chunks_indexed(self, document_id: int, owner_id: int, count: int) -> None:
        row = self.rows.get(document_id)
        if row is not None and row.owner_id == owner_id:
            row.chunks_indexed = count

    async def delete(self, document_id: int, owner_id: int) -> None:
        row = self.rows.get(document_id)
        if row is not None and row.owner_id == owner_id:
            del self.rows[document_id]


class FakeUnitOfWork:
    """Records whether the use case actually committed."""

    # Annotated as the ports: protocol attributes are invariant, so declaring the
    # concrete fakes here would stop this class satisfying `UnitOfWork`.
    conversations: ConversationRepository
    users: UserRepository
    refresh_tokens: RefreshTokenRepository
    documents: DocumentRepository

    def __init__(
        self,
        repository: FakeConversationRepository,
        users: FakeUserRepository | None = None,
        refresh_tokens: FakeRefreshTokenRepository | None = None,
        documents: FakeDocumentRepository | None = None,
    ) -> None:
        self.conversations = repository
        self.users = users or FakeUserRepository()
        self.refresh_tokens = refresh_tokens or FakeRefreshTokenRepository()
        self.documents = documents or FakeDocumentRepository()
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
    users: FakeUserRepository = field(default_factory=FakeUserRepository)
    refresh_tokens: FakeRefreshTokenRepository = field(default_factory=FakeRefreshTokenRepository)
    documents: FakeDocumentRepository = field(default_factory=FakeDocumentRepository)
    issued: list[FakeUnitOfWork] = field(default_factory=list)

    def __call__(self) -> FakeUnitOfWork:
        # The repositories are shared across every unit of work this hands out,
        # which is what makes them behave like one database rather than one per
        # call: a use case that opens a second unit of work must see the first
        # one's writes.
        uow = FakeUnitOfWork(self.repository, self.users, self.refresh_tokens, self.documents)
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


class FakeTokenCounter:
    """Counts a fixed number of tokens per message, so a test can force a threshold.

    The real counter is approximate anyway; what matters here is that a test can
    say "every message is 1,000 tokens" and drive the summarization node over or
    under its budget without building a long thread.
    """

    def __init__(self, tokens_per_message: int = 1) -> None:
        self.tokens_per_message = tokens_per_message

    def count(self, messages: Sequence[ChatMessage]) -> int:
        return self.tokens_per_message * len(messages)


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
        self.contexts: list[ToolContext] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"fake tool {self._name}",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, arguments: Mapping[str, object], context: ToolContext) -> ToolOutcome:
        self.calls.append(dict(arguments))
        self.contexts.append(context)
        if self.fail_with is not None:
            # Deliberately raised, not returned: real tools go through
            # `BaseTool.run`, which catches. This one proves `ToolRegistry`
            # survives a tool that does not.
            raise self.fail_with
        return ToolOutcome(content=self.result)


class FakeWebSearcher:
    def __init__(
        self,
        results: str = "",
        sources: Sequence[Source] = (),
        fail_with: Exception | None = None,
    ) -> None:
        self.results = results
        self.sources = tuple(sources)
        self.fail_with = fail_with
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> SearchResult:
        self.queries.append((query, max_results))
        if self.fail_with is not None:
            raise self.fail_with
        return SearchResult(text=self.results, sources=self.sources)


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
    def __init__(
        self, passages: Sequence[Passage] = (), fail_with: Exception | None = None
    ) -> None:
        self.passages = list(passages)
        self.fail_with = fail_with
        # Recorded so a test can assert *whose* documents were searched, which is
        # the whole isolation guarantee.
        self.owners: list[int] = []

    async def retrieve(self, query: str, owner_id: int) -> list[Passage]:
        self.owners.append(owner_id)
        if self.fail_with is not None:
            raise self.fail_with
        return list(self.passages)


class FakeFileStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes]] = []
        self.owners: list[int] = []
        self.deleted: list[str] = []

    async def save(
        self,
        name: DocumentName,
        stream: AsyncIterator[bytes],
        max_bytes: int,
        owner_id: int,
    ) -> str:
        body = b"".join([chunk async for chunk in stream])
        self.saved.append((name.value, body))
        self.owners.append(owner_id)
        # Mirrors the real adapter's per-owner directory, so a test can see that
        # two owners uploading the same filename get two references.
        return f"uploads/{owner_id}/{name.value}"

    async def delete(self, reference: str) -> None:
        self.deleted.append(reference)


class FakeUrlContentFetcher:
    def __init__(self, text: str = "", fail_with: Exception | None = None) -> None:
        self.text = text
        self.fail_with = fail_with
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        if self.fail_with is not None:
            raise self.fail_with
        return self.text


class FakeTextExtractor:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[str] = []

    async def extract(self, reference: str) -> str:
        self.calls.append(reference)
        return self.text


class FakeEmbedder:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.embedded: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [float(len(text))] * self.dimensions


class FakeVectorIndex:
    """Passages keyed by owner and by document, so a test can prove one owner
    cannot see another's, and one document's delete cannot touch another's.

    Chunks are stored per (owner, document) rather than in one list, which is
    what makes "deleting document A left document B's passages alone" an
    assertion about behaviour instead of an assertion about which arguments
    were passed.
    """

    def __init__(self, hits: Sequence[str] = ()) -> None:
        self.hits = list(hits)
        self.ensured: list[int] = []
        self.deleted_documents: list[tuple[int, int]] = []
        self.by_owner: dict[int, list[Chunk]] = {}
        self._by_document: dict[tuple[int, int], list[Chunk]] = {}

    @property
    def chunks(self) -> list[Chunk]:
        """Everything indexed, across every owner."""
        return [chunk for chunks in self.by_owner.values() for chunk in chunks]

    async def ensure(self, dimensions: int) -> None:
        self.ensured.append(dimensions)

    async def delete_document(self, document_id: int, owner_id: int) -> None:
        self.deleted_documents.append((document_id, owner_id))
        removed = self._by_document.pop((owner_id, document_id), [])
        remaining = self.by_owner.get(owner_id, [])
        for chunk in removed:
            if chunk in remaining:
                remaining.remove(chunk)
        if remaining:
            self.by_owner[owner_id] = remaining
        else:
            self.by_owner.pop(owner_id, None)

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        owner_id: int,
        document_id: int,
    ) -> None:
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.by_owner.setdefault(owner_id, []).extend(chunks)
        self._by_document.setdefault((owner_id, document_id), []).extend(chunks)

    async def search(
        self, vector: list[float], limit: int, score_threshold: float, owner_id: int
    ) -> list[Chunk]:
        # `hits` is the scripted answer, but only for an owner who has something
        # indexed - an owner with nothing uploaded must come back empty.
        if not self.by_owner.get(owner_id):
            return []
        return [Chunk(text=hit, source="fake") for hit in self.hits[:limit]]


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


class FakeReadinessProbe:
    """A dependency that answers, refuses or stalls, on request.

    `delay` and `error` are what the readiness tests are actually about: a probe
    is interesting when it is slow or broken, and neither is reproducible
    against a real Postgres.
    """

    def __init__(self, name: str, *, delay: float = 0.0, error: Exception | None = None) -> None:
        self.name = name
        self.checks = 0
        self._delay = delay
        self._error = error

    async def check(self) -> None:
        self.checks += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error


@dataclass
class RecordedSpan:
    """One span a test can assert on, after the fact."""

    name: str
    kind: str
    attributes: dict[str, object] = field(default_factory=dict)
    errors: list[BaseException] = field(default_factory=list)

    def set(self, **attributes: object) -> None:
        self.attributes.update(attributes)

    def record_error(self, error: BaseException) -> None:
        self.errors.append(error)


class RecordingTracer:
    """Collects spans in memory, so a test can prove a reply was traced.

    Records rather than asserts: a test that wanted "the agent opened a
    generation span carrying token counts" should be able to say exactly that,
    without a network, a Langfuse key, or a mock's did-you-call-me bookkeeping.

    Deliberately not nesting. The real adapter nests through OpenTelemetry
    context, which is machinery no fake should try to reproduce; every span
    lands in one flat list in the order it was opened, which is enough to assert
    what ran and what it carried.
    """

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self.flushes = 0

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: str = "span",
        **attributes: object,
    ) -> AsyncIterator[RecordedSpan]:
        recorded = RecordedSpan(name=name, kind=kind, attributes=dict(attributes))
        self.spans.append(recorded)
        yield recorded

    async def flush(self) -> None:
        self.flushes += 1

    def named(self, name: str) -> list[RecordedSpan]:
        """Every span opened under this name, in order."""
        return [span for span in self.spans if span.name == name]
