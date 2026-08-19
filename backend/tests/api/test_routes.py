"""API tests: status codes, serialization and error mapping — not business rules.

The app is built with a container full of stubs and the lifespan never runs, so
these exercise the delivery layer without a database, a vector store or a key.
"""

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from app.api.dependencies import get_container
from app.application.chat.dto import (
    GenerateReplyInput,
    RecordExchangeInput,
    ReplyCompleted,
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
    StartConversationInput,
)
from app.application.chat.models import Source
from app.application.health.use_cases.check_readiness import CheckReadiness
from app.application.identity.dto import (
    RefreshCookiePolicy,
    UserContext,
)
from app.application.ingestion.dto import (
    IndexDocumentResult,
    UploadDocumentInput,
    UploadDocumentResult,
)
from app.domain.chat.entities import Conversation, Message
from app.domain.chat.errors import ConversationNotFound
from app.domain.identity.errors import InvalidToken
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import DocumentNotFound, DocumentTooLarge, UnsupportedDocumentType
from app.domain.usage.errors import RateLimited
from app.factory import create_app
from app.settings import Settings
from tests.fakes import FakeReadinessProbe

NOW = datetime(2026, 8, 6, tzinfo=UTC)

# What the stub container reports as configured. The API reads the model list off
# the container, never off settings, so this is the only place the tests need it.
MODELS: tuple[str, ...] = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b")

# Likewise for the WebSocket allow-list.
WS_ORIGINS: tuple[str, ...] = ("https://app.example.com",)

# Every authenticated request in this file carries this token and resolves to
# this person. The token is opaque here on purpose - `StubIdentifyRequest` below
# stands in for the codec, so these tests never depend on the JWT format.
TOKEN = "a-valid-access-token"
USER = UserContext(user_id=7, email="owner@example.com")
AUTH: dict[str, str] = {"Authorization": f"Bearer {TOKEN}"}

# The WebSocket carries its token as an offered subprotocol rather than a query
# parameter: browsers give no way to set a header on a handshake, but they do
# let JavaScript choose `Sec-WebSocket-Protocol`. The fixed marker comes first,
# the token second - see `_token_from_subprotocols` in the transcription router.
WS_URL = "/ws/transcribe/"
WS_SUBPROTOCOLS = ["access_token", TOKEN]


def settings() -> Settings:
    """Explicit dummies, so a developer's local .env can never change a result."""
    return Settings(
        environment="test",
        database_url="postgresql+psycopg://u:p@localhost:5432/test",
        llm_api_key=SecretStr("test"),
        tavily_api_key=SecretStr("test"),
        gemini_api_key=SecretStr("test"),
        deepgram_api_key=SecretStr("test"),
        jwt_secret=SecretStr("test"),
    )


class StubGenerateReply:
    def __init__(self, events: list[ReplyEvent]) -> None:
        self.events = events
        self.seen: list[GenerateReplyInput] = []

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        # A coroutine returning an iterator, matching the real use case: it
        # spends the rate-limit budget eagerly and only then hands back the
        # stream, so a refusal happens before the response starts.
        self.seen.append(data)
        return self._events()

    async def _events(self) -> AsyncIterator[ReplyEvent]:
        for event in self.events:
            yield event


class StubUseCase:
    """Returns a canned value, or raises a canned error."""

    def __init__(self, returns: Any = None, raises: Exception | None = None) -> None:
        self.returns = returns
        self.raises = raises
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args or tuple(kwargs.values()))
        if self.raises is not None:
            raise self.raises
        return self.returns


class StubIdentifyRequest:
    """Accepts one token and rejects everything else.

    Deliberately not a signature check: these tests are about the delivery layer,
    and the real codec has its own tests. What matters here is that a route with
    no usable token never reaches its use case.
    """

    def __init__(self, user: UserContext = USER, accepts: str = TOKEN) -> None:
        self.user = user
        self.accepts = accepts

    async def __call__(self, token: str) -> UserContext:
        if token != self.accepts:
            raise InvalidToken()
        return self.user


@dataclass
class StubContainer:
    generate_reply: Any = None
    start_conversation: Any = None
    get_conversation: Any = None
    list_conversations: Any = None
    record_exchange: Any = None
    delete_conversation: Any = None
    upload_document: Any = None
    index_document: Any = None
    list_documents: Any = None
    delete_document: Any = None
    transcribe_stream: Any = None
    register_user: Any = None
    authenticate_user: Any = None
    refresh_session: Any = None
    revoke_session: Any = None
    identify_request: Any = field(default_factory=StubIdentifyRequest)
    # The real use case, not a stub: it is pure orchestration over its probes,
    # so a test says what it means by supplying the probes rather than by
    # replacing the thing under test. No probes is a ready deployment, which is
    # what every test in this file that is not about `/ready` wants.
    check_readiness: Any = field(
        default_factory=lambda: CheckReadiness(probes=[], timeout_seconds=1.0)
    )
    refresh_cookie: RefreshCookiePolicy = field(
        # `secure=False` so `httpx` and `TestClient` keep the cookie over the
        # plain-http base URL these tests use; a Secure cookie would be dropped
        # and every refresh test would fail for the wrong reason.
        default_factory=lambda: RefreshCookiePolicy(
            name="refresh_token", path="/auth", secure=False, samesite="lax"
        )
    )
    trust_forwarded_client_ip: bool = False
    chat_models: Sequence[str] = field(default_factory=lambda: list(MODELS))
    websocket_origins: Sequence[str] = field(default_factory=lambda: list(WS_ORIGINS))


def frames(body: str) -> list[dict[str, Any]]:
    """Every SSE frame's JSON payload, in order.

    Parsed rather than string-matched because the assertions below are about
    what the frontend will read out of each frame, not about how json.dumps
    happened to space it.
    """
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]


def client(
    container: StubContainer, *, reraise: bool = True, authenticated: bool = True
) -> httpx.AsyncClient:
    """
    Build the real app around a stub container, without running the lifespan.

    `reraise=False` is for the 500 path only: Starlette hands the exception to
    our handler, sends that response, and then re-raises so the server still
    logs it. Only by not re-raising here can a test see the body a browser would
    actually receive.

    The bearer header is attached by default so every test below reads as what
    it is about. `authenticated=False` is how a test asks the opposite question:
    what an anonymous caller gets.
    """
    app = create_app(settings())
    app.dependency_overrides[get_container] = lambda: container
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=reraise),
        base_url="http://test",
        headers=dict(AUTH) if authenticated else {},
    )


@asynccontextmanager
async def _no_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def ws_client(container: StubContainer) -> TestClient:
    """A WebSocket-capable client over the real app and a stub container.

    `httpx.ASGITransport` cannot speak WebSocket, so these tests need
    `TestClient` — and `TestClient` only opens the portal `websocket_connect`
    needs when it is entered as a context manager, which is also what runs the
    lifespan. The real lifespan builds Postgres, Qdrant and Deepgram clients, so
    it is replaced with one that does nothing; the container arrives through the
    dependency override instead, exactly as in the HTTP tests above.
    """
    app = create_app(settings())
    app.dependency_overrides[get_container] = lambda: container
    app.router.lifespan_context = _no_lifespan
    return TestClient(app)


def conversation(**overrides: Any) -> Conversation:
    defaults: dict[str, Any] = {
        "id": 1,
        "title": "T",
        "model_type": MODELS[0],
        "owner_id": USER.user_id,
        "created_at": NOW,
        "updated_at": NOW,
    }
    return Conversation(**(defaults | overrides))


class TestGenerateText:
    async def test_streams_sse_frames_in_the_shape_the_frontend_parses(self) -> None:
        stub = StubGenerateReply([ReplyDelta(text="Hel\nlo"), ReplyCompleted()])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/",
                json={"model": MODELS[0], "user_input": "hi", "temperature": 0},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # JSON-encoded per frame, because a token can contain the newline that
        # would otherwise terminate the frame early.
        assert response.text == 'data: {"delta": "Hel\\nlo"}\n\ndata: {"done": true}\n\n'

    async def test_tool_activity_reaches_the_client_as_its_own_frames(self) -> None:
        # Asserted on content, not on frame count. `to_sse` matches these events
        # with class patterns that capture fields by name, so a renamed field
        # would stop matching and silently drop the frame - which a "three frames
        # arrived" assertion would happily pass.
        stub = StubGenerateReply(
            [
                ReplyToolStarted(name="search_web", summary="tallest building"),
                ReplyToolFinished(name="search_web", ok=True),
                ReplyDelta(text="It is Burj Khalifa."),
                ReplyCompleted(),
            ]
        )

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[0], "user_input": "hi"}
            )

        assert frames(response.text) == [
            {"tool": {"name": "search_web", "status": "start", "summary": "tallest building"}},
            {"tool": {"name": "search_web", "status": "ok"}},
            {"delta": "It is Burj Khalifa."},
            {"done": True},
        ]

    async def test_a_tools_sources_ride_along_in_its_finished_frame(self) -> None:
        stub = StubGenerateReply(
            [
                ReplyToolFinished(
                    name="search_web",
                    ok=True,
                    sources=(Source(label="Burj Khalifa", url="https://example.com/burj"),),
                ),
                ReplyCompleted(),
            ]
        )

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[0], "user_input": "hi"}
            )

        assert frames(response.text)[0] == {
            "tool": {
                "name": "search_web",
                "status": "ok",
                "sources": [{"label": "Burj Khalifa", "url": "https://example.com/burj"}],
            }
        }

    async def test_no_sources_means_the_key_is_left_off_the_frame(self) -> None:
        stub = StubGenerateReply([ReplyToolFinished(name="search_web", ok=True), ReplyCompleted()])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[0], "user_input": "hi"}
            )

        assert frames(response.text)[0] == {"tool": {"name": "search_web", "status": "ok"}}

    async def test_a_failed_tool_is_reported_as_failed_not_as_an_error(self) -> None:
        # The distinction matters to the client: `error` ends the answer, a
        # failed tool only costs it some grounding.
        stub = StubGenerateReply(
            [ReplyToolFinished(name="retrieve_documents", ok=False), ReplyCompleted()]
        )

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[0], "user_input": "hi"}
            )

        assert frames(response.text) == [
            {"tool": {"name": "retrieve_documents", "status": "failed"}},
            {"done": True},
        ]

    async def test_a_mid_stream_failure_arrives_as_an_error_frame_not_a_500(self) -> None:
        stub = StubGenerateReply([ReplyDelta(text="part"), ReplyFailed(detail="rate limited")])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[1], "user_input": "hi"}
            )

        assert response.status_code == 200
        assert response.text.endswith('data: {"error": "rate limited"}\n\n')

    async def test_a_spent_chat_budget_is_a_429_before_the_stream_opens(self) -> None:
        # The whole reason the use case is awaited rather than iterated: a limit
        # discovered mid-stream could only truncate a body that already sent 200,
        # and the client would read a refusal as a broken answer.
        stub = StubUseCase(raises=RateLimited(retry_after_seconds=30))

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": MODELS[0], "user_input": "hi"}
            )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "30"
        assert not response.headers["content-type"].startswith("text/event-stream")
        # The same number in the body, because a browser cannot read the header
        # cross-origin and the frontend counts down from this.
        assert response.json()["retry_after_seconds"] == 30

    async def test_the_conversation_id_is_forwarded_as_the_agents_memory_key(self) -> None:
        stub = StubGenerateReply([ReplyCompleted()])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/",
                json={"model": MODELS[0], "user_input": "hi", "conversation_id": 42},
            )

        assert response.status_code == 200
        assert stub.seen[0].conversation_id == 42

    @pytest.mark.parametrize(
        "body",
        [
            {"model": MODELS[0], "user_input": "hi"},
            # The frontend sends the key explicitly and sets it to null until the
            # conversation exists server-side, so `extra="forbid"` must not choke
            # on it.
            {"model": MODELS[0], "user_input": "hi", "conversation_id": None},
        ],
    )
    async def test_no_conversation_id_means_a_one_off_turn(self, body: dict[str, Any]) -> None:
        stub = StubGenerateReply([ReplyCompleted()])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post("/generate/text/", json=body)

        assert response.status_code == 200
        assert stub.seen[0].conversation_id is None

    async def test_an_unconfigured_model_is_a_4xx_not_a_500_or_an_error_frame(self) -> None:
        # The model list is deployment configuration, so this cannot be caught by
        # the request schema alone. It still has to fail before the stream opens,
        # while a status code can still say so.
        stub = StubGenerateReply([ReplyCompleted()])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post(
                "/generate/text/", json={"model": "gpt-9-ultra", "user_input": "hi"}
            )

        assert 400 <= response.status_code < 500
        assert response.json()["type"] == "UnsupportedModel"
        assert stub.seen == [], "an unknown model must never reach the use case"

    @pytest.mark.parametrize(
        "body",
        [
            {"model": MODELS[0]},
            {"model": MODELS[0], "user_input": ""},
            {"model": MODELS[0], "user_input": "   "},
            {"model": MODELS[0], "user_input": "hi", "temperature": 9},
            {"model": MODELS[0], "user_input": "hi", "surprise": 1},
            {"model": MODELS[0], "user_input": "hi", "conversation_id": "seven"},
            {"model": "", "user_input": "hi"},
        ],
    )
    async def test_bad_requests_are_rejected_before_the_stream_opens(
        self, body: dict[str, Any]
    ) -> None:
        stub = StubGenerateReply([])

        async with client(StubContainer(generate_reply=stub)) as http:
            response = await http.post("/generate/text/", json=body)

        assert response.status_code == 422
        assert stub.seen == [], "a rejected request must never reach the use case"


class TestModels:
    async def test_lists_the_configured_models_and_defaults_to_the_first(self) -> None:
        async with client(StubContainer()) as http:
            response = await http.get("/models/")

        assert response.status_code == 200
        assert response.json() == {"models": list(MODELS), "default": MODELS[0]}

    async def test_the_list_follows_the_deployment_not_a_hardcoded_literal(self) -> None:
        container = StubContainer(chat_models=["only-model"])

        async with client(container) as http:
            response = await http.get("/models/")

        assert response.json() == {"models": ["only-model"], "default": "only-model"}


class TestUpload:
    async def test_stores_the_file_and_queues_exactly_one_index_task(self) -> None:
        upload = StubUseCase(
            returns=UploadDocumentResult(reference="uploads/a.pdf", name="a.pdf", document_id=1)
        )
        index = StubUseCase(returns=IndexDocumentResult(chunks_indexed=3))

        async with client(StubContainer(upload_document=upload, index_document=index)) as http:
            response = await http.post(
                "/upload/file/", files={"file": ("a.pdf", b"%PDF-1.4 body", "application/pdf")}
            )

        assert response.status_code == 200
        assert response.json() == {
            "message": "File uploaded successfully",
            "file_path": "uploads/a.pdf",
        }
        # The owner rides along to the background task: indexing happens after
        # the response, so it cannot look the caller up again.
        assert index.calls == [("uploads/a.pdf", "a.pdf", 1, USER.user_id)]

    async def test_the_uploaded_bytes_reach_the_use_case(self) -> None:
        captured: list[bytes] = []

        class Capturing:
            async def __call__(self, data: UploadDocumentInput) -> UploadDocumentResult:
                captured.append(b"".join([c async for c in data.stream]))
                return UploadDocumentResult(reference="uploads/a.pdf", name="a.pdf", document_id=1)

        container = StubContainer(upload_document=Capturing(), index_document=StubUseCase())
        async with client(container) as http:
            await http.post(
                "/upload/file/", files={"file": ("a.pdf", b"%PDF-1.4 body", "application/pdf")}
            )

        assert captured == [b"%PDF-1.4 body"]

    @pytest.mark.parametrize(
        ("error", "expected_status"),
        [
            (UnsupportedDocumentType("text/html"), 400),
            (DocumentTooLarge(1024), 413),
        ],
    )
    async def test_domain_errors_map_to_their_status(
        self, error: Exception, expected_status: int
    ) -> None:
        container = StubContainer(
            upload_document=StubUseCase(raises=error), index_document=StubUseCase()
        )
        async with client(container) as http:
            response = await http.post(
                "/upload/file/", files={"file": ("a.html", b"<html>", "text/html")}
            )

        assert response.status_code == expected_status
        assert response.json()["type"] == type(error).__name__
        # Only a rate limit carries a wait; every other failure would be telling
        # the client to retry something that will fail again.
        assert "retry_after_seconds" not in response.json()

    async def test_a_spent_upload_budget_is_a_429_that_says_for_how_long(self) -> None:
        container = StubContainer(
            upload_document=StubUseCase(raises=RateLimited(retry_after_seconds=90)),
            index_document=StubUseCase(),
        )
        async with client(container) as http:
            response = await http.post(
                "/upload/file/", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
            )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "90"
        assert response.json()["retry_after_seconds"] == 90


class TestDocuments:
    async def test_list_returns_every_document_this_owner_has(self) -> None:
        documents = [
            UploadedDocument(
                name="a.pdf", reference="uploads/a.pdf", owner_id=USER.user_id, id=1, created_at=NOW
            )
        ]
        stub = StubUseCase(returns=documents)

        async with client(StubContainer(list_documents=stub)) as http:
            response = await http.get("/documents/")

        assert response.status_code == 200
        body = response.json()
        assert [
            {"id": d["id"], "name": d["name"], "chunks_indexed": d["chunks_indexed"]} for d in body
        ] == [{"id": 1, "name": "a.pdf", "chunks_indexed": 0}]
        assert stub.calls[0][0] == USER.user_id

    async def test_delete_is_a_post_returning_204(self) -> None:
        stub = StubUseCase()

        async with client(StubContainer(delete_document=stub)) as http:
            response = await http.post("/documents/1/delete")

        assert response.status_code == 204
        assert response.content == b""

    async def test_a_missing_document_is_a_404(self) -> None:
        stub = StubUseCase(raises=DocumentNotFound(99))

        async with client(StubContainer(delete_document=stub)) as http:
            response = await http.post("/documents/99/delete")

        assert response.status_code == 404
        assert response.json()["type"] == "DocumentNotFound"


class TestConversations:
    async def test_create_returns_201_and_the_new_id(self) -> None:
        stub = StubUseCase(returns=conversation(id=7))

        async with client(StubContainer(start_conversation=stub)) as http:
            response = await http.post(
                "/conversations/", json={"title": "T", "model_type": MODELS[0]}
            )

        assert response.status_code == 201
        assert response.json()["id"] == 7
        assert isinstance(stub.calls[0][0], StartConversationInput)

    async def test_get_includes_the_message_log(self) -> None:
        stored = conversation()
        stored.messages = [
            Message(
                prompt_content="p", response_content="r", id=1, conversation_id=1, created_at=NOW
            )
        ]
        async with client(StubContainer(get_conversation=StubUseCase(returns=stored))) as http:
            response = await http.get("/conversations/1")

        body = response.json()
        assert response.status_code == 200
        assert [m["prompt_content"] for m in body["messages"]] == ["p"]

    async def test_list_returns_every_conversation_this_owner_has(self) -> None:
        stub = StubUseCase(returns=[conversation(id=2), conversation(id=1)])

        async with client(StubContainer(list_conversations=stub)) as http:
            response = await http.get("/conversations/")

        assert response.status_code == 200
        assert [c["id"] for c in response.json()] == [2, 1]
        assert stub.calls[0][0] == USER.user_id

    async def test_a_missing_conversation_is_a_404(self) -> None:
        stub = StubUseCase(raises=ConversationNotFound(99))

        async with client(StubContainer(get_conversation=stub)) as http:
            response = await http.get("/conversations/99")

        assert response.status_code == 404
        assert response.json()["type"] == "ConversationNotFound"

    async def test_append_message_returns_201(self) -> None:
        stored = Message(
            prompt_content="p", response_content="r", id=5, conversation_id=1, created_at=NOW
        )

        async with client(StubContainer(record_exchange=StubUseCase(returns=stored))) as http:
            response = await http.post(
                "/conversations/1/messages",
                json={"prompt_content": "p", "response_content": "r"},
            )

        assert response.status_code == 201
        assert response.json()["id"] == 5

    async def test_record_exchange_receives_the_path_id_not_a_body_id(self) -> None:
        stub = StubUseCase(
            returns=Message(prompt_content="p", response_content="r", id=5, created_at=NOW)
        )
        async with client(StubContainer(record_exchange=stub)) as http:
            await http.post(
                "/conversations/42/messages",
                json={"prompt_content": "p", "response_content": "r"},
            )

        data = stub.calls[0][0]
        assert isinstance(data, RecordExchangeInput)
        assert data.conversation_id == 42

    async def test_delete_is_a_post_returning_204(self) -> None:
        # POST rather than DELETE: CORS only allows GET, POST and OPTIONS.
        stub = StubUseCase()

        async with client(StubContainer(delete_conversation=stub)) as http:
            response = await http.post("/conversations/1/delete")

        assert response.status_code == 204
        assert response.content == b""


class TestTranscriptionHandshake:
    """The origin check on `/ws/transcribe/`.

    CORS middleware does not run on a WebSocket handshake, so this route is the
    only thing standing between a public deployment and anyone who knows the URL
    opening a billed Deepgram session. These tests are sync because `TestClient`
    is; pytest-asyncio's auto mode leaves non-async tests alone.
    """

    class StubTranscribe:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, transport: Any) -> None:
            self.calls += 1
            await transport.ready()

    def test_an_allowed_origin_reaches_the_use_case(self) -> None:
        stub = self.StubTranscribe()
        allowed = {"origin": WS_ORIGINS[0]}

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as client,
            client.websocket_connect(
                WS_URL, subprotocols=WS_SUBPROTOCOLS, headers=allowed
            ) as socket,
        ):
            assert socket.receive_json() == {"type": "ready"}

        assert stub.calls == 1

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({"origin": "https://evil.example"}, id="wrong-origin"),
            # A browser always sends Origin on a handshake, so a request without
            # one is not the frontend — and the frontend is the only caller this
            # route has. Refusing it costs nothing and closes the scripted case.
            pytest.param({}, id="no-origin"),
            # Not a prefix match: an attacker controls their own subdomain, and
            # `https://app.example.com.evil.test` starts with the allowed value.
            pytest.param({"origin": f"{WS_ORIGINS[0]}.evil.test"}, id="suffix-attack"),
        ],
    )
    def test_a_disallowed_origin_is_refused_before_the_socket_opens(
        self, headers: dict[str, str]
    ) -> None:
        stub = self.StubTranscribe()

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as client,
            pytest.raises(WebSocketDisconnect) as refusal,
            client.websocket_connect(WS_URL, subprotocols=WS_SUBPROTOCOLS, headers=headers),
        ):
            pass  # pragma: no cover - the handshake never gets this far

        # 1008 is "policy violation", and it arrives instead of an accept, so the
        # client never has an open socket to send audio down.
        assert refusal.value.code == 1008
        assert stub.calls == 0, "a refused handshake must never start a session"


class TestTranscriptionAuth:
    """The token check on `/ws/transcribe/`, offered as a subprotocol rather
    than a query parameter — see `_token_from_subprotocols` in the router."""

    class StubTranscribe:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, transport: Any) -> None:
            self.calls += 1
            await transport.ready()

    def test_the_accepted_subprotocol_never_echoes_the_token(self) -> None:
        stub = self.StubTranscribe()

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as client,
            client.websocket_connect(
                WS_URL, subprotocols=WS_SUBPROTOCOLS, headers={"origin": WS_ORIGINS[0]}
            ) as socket,
        ):
            assert socket.accepted_subprotocol == "access_token"

    @pytest.mark.parametrize(
        "subprotocols",
        [
            pytest.param(None, id="no-subprotocols-offered"),
            pytest.param(["access_token"], id="marker-with-no-token"),
            pytest.param([TOKEN], id="token-with-no-marker"),
        ],
    )
    def test_a_missing_or_malformed_offer_is_refused_before_the_socket_opens(
        self, subprotocols: list[str] | None
    ) -> None:
        stub = self.StubTranscribe()

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as client,
            pytest.raises(WebSocketDisconnect) as refusal,
            client.websocket_connect(
                WS_URL, subprotocols=subprotocols, headers={"origin": WS_ORIGINS[0]}
            ),
        ):
            pass  # pragma: no cover - the handshake never gets this far

        assert refusal.value.code == 1008
        assert stub.calls == 0, "a refused handshake must never start a session"


class TestTranscriptionDailyBudget:
    """`RateLimited` raised mid-session gets its own branch in the router,
    distinct from the generic `except Exception` catch-all."""

    def test_a_spent_daily_budget_sends_a_busy_error_not_a_generic_failure(self) -> None:
        stub = StubUseCase(raises=RateLimited(retry_after_seconds=30))

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as client,
            client.websocket_connect(
                WS_URL, subprotocols=WS_SUBPROTOCOLS, headers={"origin": WS_ORIGINS[0]}
            ) as socket,
        ):
            assert socket.receive_json() == {
                "type": "error",
                "detail": "Service is busy right now - please try again later.",
            }


class TestErrorsAndHealth:
    async def test_an_unexpected_failure_never_leaks_its_message(self) -> None:
        stub = StubUseCase(raises=RuntimeError("postgresql://user:hunter2@db:5432"))

        async with client(StubContainer(get_conversation=stub), reraise=False) as http:
            response = await http.get("/conversations/1")

        assert response.status_code == 500
        assert "hunter2" not in response.text
        assert response.json()["detail"] == "Internal server error"

    async def test_health_does_not_touch_dependencies(self) -> None:
        """Liveness stays static even when every dependency is down.

        The two endpoints answer different questions on purpose: a failing
        `/health` gets the process restarted, and restarting is never the cure
        for a database that is briefly unreachable.
        """
        container = StubContainer(
            check_readiness=CheckReadiness(
                probes=[FakeReadinessProbe("postgres", error=RuntimeError("refused"))],
                timeout_seconds=1.0,
            )
        )

        async with client(container) as http:
            assert (await http.get("/health")).status_code == 200

    async def test_ready_reports_every_dependency_by_name(self) -> None:
        container = StubContainer(
            check_readiness=CheckReadiness(
                probes=[FakeReadinessProbe("postgres"), FakeReadinessProbe("qdrant")],
                timeout_seconds=1.0,
            )
        )

        async with client(container) as http:
            response = await http.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "checks": {"postgres": "ok", "qdrant": "ok"}}

    async def test_a_down_dependency_makes_ready_a_503(self) -> None:
        """The status code is what a rollout gate reads, so it has to move."""
        container = StubContainer(
            check_readiness=CheckReadiness(
                probes=[
                    FakeReadinessProbe("postgres"),
                    FakeReadinessProbe("qdrant", error=RuntimeError("connection refused")),
                ],
                timeout_seconds=1.0,
            )
        )

        async with client(container) as http:
            response = await http.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "checks": {"postgres": "ok", "qdrant": "down"},
        }

    async def test_ready_never_describes_the_failure_it_saw(self) -> None:
        """It needs no credentials, so the body must not carry a DSN or a driver error."""
        container = StubContainer(
            check_readiness=CheckReadiness(
                probes=[
                    FakeReadinessProbe(
                        "postgres",
                        error=RuntimeError("postgresql://user:hunter2@db:5432 refused"),
                    )
                ],
                timeout_seconds=1.0,
            )
        )

        async with client(container, authenticated=False) as http:
            response = await http.get("/ready")

        assert response.status_code == 503
        assert "hunter2" not in response.text
