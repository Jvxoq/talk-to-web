"""The seam between the chat context and the vector store."""

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.application.ingestion.ports import Embedder, VectorIndex


class EmbeddedKnowledgeRetriever:
    """
    Answers "what do we know about this?" with plain passages.

    Satisfies `app.application.chat.ports.KnowledgeRetriever`. This is the seam
    that keeps the chat context ignorant of embeddings: chat asks a question in
    text and gets text back, while the choice of embedding model, vector store
    and thresholds lives entirely on this side of the line. Swapping Qdrant for
    keyword search would not change a line of chat code.
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

    async def retrieve(self, query: str) -> list[str]:
        """Embed the query and return the passages that come back closest to it."""
        vector = await self._embedder.embed(query)
        passages = await self._index.search(
            vector=vector,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )
        logger.debug(f"Retrieved {len(passages)} passages for a {len(query)} char query")
        return passages
