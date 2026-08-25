"""The seam between the chat context and the vector store."""

from typing import TYPE_CHECKING

from loguru import logger

from app.application.chat.models import Passage

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

    async def retrieve(
        self, query: str, owner_id: int, conversation_id: int | None
    ) -> list[Passage]:
        """Embed the query and return this thread's passages closest to it.

        No conversation means no attached documents, so the embedding call is
        skipped entirely - it would be paid for and then filtered to nothing.

        Translates the ingestion context's `Chunk` into chat's own `Passage`
        here, at the seam - the one place allowed to know both shapes - so
        that a source name earned by a vector search reaches the tool above it
        as chat's own vocabulary, not ingestion's.
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
