"""The seam between the chat context and the vector store."""

from typing import TYPE_CHECKING

from loguru import logger

from app.application.chat.models import Passage

if TYPE_CHECKING:
    from app.application.ingestion.ports import Embedder, VectorIndex


class EmbeddedKnowledgeRetriever:
    """Returns passages for a query. Implements `KnowledgeRetriever`.

    Chat sends text and gets text back. Embeddings and the vector store stay here.
    """

    def __init__(
        self,
        embedder: "Embedder",
        index: "VectorIndex",
        limit: int,
        score_threshold: float,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._limit = limit
        self._score_threshold = score_threshold

    async def retrieve(
        self, query: str, owner_id: int, conversation_id: int | None
    ) -> list[Passage]:
        """Embed the query and return this thread's closest passages.

        No conversation skips the paid embedding call. Converts `Chunk` to `Passage`.
        """

        if conversation_id is None:
            return []

        vector = await self._embedder.embed(query)
        chunks = await self._index.search(
            vector=vector,
            limit=self._limit,
            score_threshold=self._score_threshold,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
        logger.debug(f"Retrieved {len(chunks)} passages for a {len(query)} char query")
        return [Passage(text=chunk.text, source=chunk.source) for chunk in chunks]
