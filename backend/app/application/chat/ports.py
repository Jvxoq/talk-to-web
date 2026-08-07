"""What the chat use cases need from the outside world.

Structural `Protocol`s: an adapter satisfies one by having the methods, without
importing or inheriting from anything here. That is what keeps the arrow
pointing inward.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.application.chat.models import ChatMessage, ModelChunk
from app.application.chat.tools.base import ToolSpec
from app.domain.chat.entities import Conversation, Message


class ConversationRepository(Protocol):
    async def get(self, conversation_id: int) -> Conversation | None: ...

    async def add(self, conversation: Conversation) -> Conversation: ...

    async def add_message(self, conversation_id: int, message: Message) -> Message: ...

    async def delete(self, conversation_id: int) -> None: ...


class ChatModel(Protocol):
    """
    A streaming, tool-calling chat model - and the seam that makes the agent
    provider-agnostic.

    Says nothing about which vendor answers: it takes a conversation and a list
    of tools the model may ask for, and yields the turn one step at a time.
    Swapping Groq for Anthropic is an adapter change, and the graph never
    notices.
    """

    def stream(
        self,
        *,
        model: str,
        temperature: float,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ModelChunk]: ...


class WebContentFetcher(Protocol):
    """Fetches and flattens the readable text of web pages."""

    async def fetch_all(self, urls: Sequence[str]) -> str: ...


class WebSearcher(Protocol):
    """
    Answers an open question with passages from the live web.

    Distinct from `WebContentFetcher`, which is told exactly which pages to
    read: this one is asked what to read in the first place.
    """

    async def search(self, query: str, max_results: int) -> str: ...


class KnowledgeRetriever(Protocol):
    """
    Finds passages relevant to a query.

    Deliberately says nothing about embeddings or vector stores: the chat
    context asks a question and gets text back.
    """

    async def retrieve(self, query: str) -> list[str]: ...
