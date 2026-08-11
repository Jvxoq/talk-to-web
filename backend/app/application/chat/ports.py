"""What the chat use cases need from the outside world.

Structural `Protocol`s: an adapter satisfies one by having the methods, without
importing or inheriting from anything here. That is what keeps the arrow
pointing inward.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.application.chat.models import ChatMessage, ModelChunk, Passage, SearchResult
from app.application.chat.tools.base import ToolSpec
from app.domain.chat.entities import Conversation, Message


class ConversationRepository(Protocol):
    """
    Conversations, always seen through their owner.

    `owner_id` is a parameter rather than something the caller filters on
    afterwards, because the isolation has to live in the query. A repository
    that returned any conversation and left the ownership check to a use case
    would be one forgotten `if` away from handing over someone else's thread,
    and `delete` would issue the DELETE before anyone had the chance to forget.
    """

    async def get(self, conversation_id: int, owner_id: int) -> Conversation | None: ...

    async def list_by_owner(self, owner_id: int) -> list[Conversation]:
        """This owner's conversations, most recently updated first, with no messages loaded.

        A list view has no use for the transcript of every thread - loading it
        would turn a sidebar render into fetching everything the owner ever
        said. Callers that need messages ask for one conversation at a time
        through `get`.
        """
        ...

    async def add(self, conversation: Conversation) -> Conversation: ...

    async def add_message(self, conversation_id: int, message: Message) -> Message: ...

    async def delete(self, conversation_id: int, owner_id: int) -> None: ...


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


class TokenCounter(Protocol):
    """
    Counts how many tokens a conversation would cost a model.

    The agent uses this to decide whether to condense, so it runs on the hot
    path of every reply. Approximate is the right posture: being 10% out only
    moves the threshold a little, and a cheap estimate is what keeps the check
    itself from costing a model call.
    """

    def count(self, messages: Sequence[ChatMessage]) -> int: ...


class WebContentFetcher(Protocol):
    """Fetches and flattens the readable text of web pages."""

    async def fetch_all(self, urls: Sequence[str]) -> str: ...


class WebSearcher(Protocol):
    """
    Answers an open question with passages from the live web.

    Distinct from `WebContentFetcher`, which is told exactly which pages to
    read: this one is asked what to read in the first place.
    """

    async def search(self, query: str, max_results: int) -> SearchResult: ...


class KnowledgeRetriever(Protocol):
    """
    Finds passages relevant to a query.

    Deliberately says nothing about embeddings or vector stores: the chat
    context asks a question and gets passages back, each still naming the
    document it came from - that name is what lets the UI cite it.

    It does say who is asking. "The user's documents" is only meaningful if the
    search is confined to them, and that confinement belongs in the query the
    store runs, not in a filter applied to results that were already read.
    """

    async def retrieve(self, query: str, owner_id: int) -> list[Passage]: ...


class RateLimiter(Protocol):
    """Counts requests against a key and refuses the ones over budget.

    Declared here rather than imported from the identity context, even though
    the shape is identical: a port belongs to the layer that consumes it, and
    chat needing a limiter is not chat depending on sign-in. One adapter
    satisfies both, structurally, without knowing either exists.
    """

    async def hit(self, key: str) -> None:
        """Record one request, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None: ...
