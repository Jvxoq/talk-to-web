"""Ports the chat use cases need.

Structural `Protocol`s: adapters match by shape, never by import.
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Literal, Protocol

from app.application.chat.models import ChatMessage, ModelChunk, Passage, SearchResult
from app.application.chat.tools.base import ToolSpec
from app.domain.chat.entities import Conversation, Message


class ConversationRepository(Protocol):
    """Conversations, always filtered by owner in the query itself.

    A check left to the use case is one forgotten `if` from a data leak.
    """

    async def get(self, conversation_id: int, owner_id: int) -> Conversation | None: ...

    async def count_by_owner(self, owner_id: int) -> int:
        """How many conversations this owner holds, counted in SQL."""
        ...

    async def list_by_owner(self, owner_id: int) -> list[Conversation]:
        """This owner's conversations, newest update first, without messages.

        Use `get` for one conversation's messages.
        """
        ...

    async def add(self, conversation: Conversation) -> Conversation: ...

    async def add_message(self, conversation_id: int, message: Message) -> Message: ...

    async def delete(self, conversation_id: int, owner_id: int) -> None: ...


class ChatModel(Protocol):
    """A streaming, tool-calling chat model. Names no vendor.

    Switching provider is an adapter change. The graph never notices.
    """

    def stream(
        self,
        *,
        model: str,
        temperature: float,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ModelChunk]: ...


class TokenCounter(Protocol):
    """Estimates a conversation's token count.

    Runs on every reply to decide whether to condense, so cheap beats exact.
    """

    def count(self, messages: Sequence[ChatMessage]) -> int: ...


class WebContentFetcher(Protocol):
    """Fetches and flattens the readable text of web pages."""

    async def fetch_all(self, urls: Sequence[str]) -> str: ...


class WebSearcher(Protocol):
    """Answers an open question with passages from the live web.

    `WebContentFetcher` reads given pages. This one finds the pages.
    """

    async def search(self, query: str, max_results: int) -> SearchResult: ...


class KnowledgeRetriever(Protocol):
    """Finds passages relevant to a query, each naming its document.

    Owner and conversation scope the query itself, not the results.
    A `None` conversation owns no documents.
    """

    async def retrieve(
        self, query: str, owner_id: int, conversation_id: int | None
    ) -> list[Passage]: ...


class DocumentRemover(Protocol):
    """Removes one document completely: vectors, file and row.

    Also declared in ingestion, because a port belongs to the layer that uses
    it. A database cascade cannot reach Qdrant or the disk, so deleting a
    conversation needs this.
    """

    async def __call__(self, document_id: int, owner_id: int) -> None: ...


class Span(Protocol):
    """One timed step in a reply.

    Attributes are set after opening, because most are known only after the step runs.
    """

    def set(self, **attributes: object) -> None: ...

    def record_error(self, error: BaseException) -> None: ...


class Tracer(Protocol):
    """Opens spans. Says nothing about where they go.

    Spans nest through context, because LangGraph nodes get only their state.
    `kind="generation"` marks model calls so a backend can total tokens.
    Never raises: tracing must not cost a reply.
    """

    def span(
        self,
        name: str,
        *,
        kind: Literal["span", "generation"] = "span",
        **attributes: object,
    ) -> AbstractAsyncContextManager[Span]: ...

    async def flush(self) -> None:
        """Send anything still queued. Called once, at shutdown, under a timeout."""
        ...


class RateLimiter(Protocol):
    """Counts requests per key and refuses those over budget.

    Same shape as the identity port. Declared here so chat does not depend on identity.
    """

    async def hit(self, key: str) -> None:
        """Record one request, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None: ...
