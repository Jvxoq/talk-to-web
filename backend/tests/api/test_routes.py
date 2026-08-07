"""API tests: status codes, serialization and error mapping — not business rules.

The app is built with a container full of stubs and the lifespan never runs, so
these exercise the delivery layer without a database, a vector store or a key.
"""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

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
from app.application.ingestion.dto import (
    IndexDocumentResult,
    UploadDocumentInput,
    UploadDocumentResult,
)
from app.domain.chat.entities import Conversation, Message
from app.domain.chat.errors import ConversationNotFound
from app.domain.ingestion.errors import DocumentTooLarge, UnsupportedDocumentType
from app.main import create_app
from app.settings import Settings

NOW = datetime(2026, 8, 6, tzinfo=UTC)

# What the stub container reports as configured. The API reads the model list off
# the container, never off settings, so this is the only place the tests need it.
MODELS: tuple[str, ...] = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b")


def settings() -> Settings:
    """Explicit dummies, so a developer's local .env can never change a result."""
    return Settings(
        environment="test",
        database_url="postgresql+psycopg://u:p@localhost:5432/test",
        llm_api_key=SecretStr("test"),
        tavily_api_key=SecretStr("test"),
        gemini_api_key=SecretStr("test"),
        deepgram_api_key=SecretStr("test"),
    )


class StubGenerateReply:
    def __init__(self, events: list[ReplyEvent]) -> None:
        self.events = events
        self.seen: list[GenerateReplyInput] = []

    async def __call__(self, data: GenerateReplyInput) -> AsyncIterator[ReplyEvent]:
        self.seen.append(data)
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


@dataclass
class StubContainer:
    generate_reply: Any = None
    start_conversation: Any = None
    get_conversation: Any = None
    record_exchange: Any = None
    delete_conversation: Any = None
    upload_document: Any = None
    index_document: Any = None
    transcribe_stream: Any = None
    chat_models: Sequence[str] = field(default_factory=lambda: list(MODELS))


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


def client(container: StubContainer, *, reraise: bool = True) -> httpx.AsyncClient:
    """
    Build the real app around a stub container, without running the lifespan.

    `reraise=False` is for the 500 path only: Starlette hands the exception to
    our handler, sends that response, and then re-raises so the server still
    logs it. Only by not re-raising here can a test see the body a browser would
    actually receive.
    """
    app = create_app(settings())
    app.dependency_overrides[get_container] = lambda: container
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=reraise),
        base_url="http://test",
    )


def conversation(**overrides: Any) -> Conversation:
    defaults: dict[str, Any] = {
        "id": 1,
        "title": "T",
        "model_type": MODELS[0],
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
        upload = StubUseCase(returns=UploadDocumentResult(reference="uploads/a.pdf", name="a.pdf"))
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
        assert index.calls == [("uploads/a.pdf", "a.pdf")]

    async def test_the_uploaded_bytes_reach_the_use_case(self) -> None:
        captured: list[bytes] = []

        class Capturing:
            async def __call__(self, data: UploadDocumentInput) -> UploadDocumentResult:
                captured.append(b"".join([c async for c in data.stream]))
                return UploadDocumentResult(reference="uploads/a.pdf", name="a.pdf")

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
        # POST rather than DELETE because navigator.sendBeacon can only POST.
        stub = StubUseCase()

        async with client(StubContainer(delete_conversation=stub)) as http:
            response = await http.post("/conversations/1/delete")

        assert response.status_code == 204
        assert response.content == b""


class TestErrorsAndHealth:
    async def test_an_unexpected_failure_never_leaks_its_message(self) -> None:
        stub = StubUseCase(raises=RuntimeError("postgresql://user:hunter2@db:5432"))

        async with client(StubContainer(get_conversation=stub), reraise=False) as http:
            response = await http.get("/conversations/1")

        assert response.status_code == 500
        assert "hunter2" not in response.text
        assert response.json()["detail"] == "Internal server error"

    async def test_health_does_not_touch_dependencies(self) -> None:
        async with client(StubContainer()) as http:
            assert (await http.get("/health")).status_code == 200
            assert (await http.get("/ready")).status_code == 200
