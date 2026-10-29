"""Qdrant-backed vector index."""

from collections.abc import Sequence
from uuid import uuid4

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.domain.ingestion.value_objects import Chunk

_OWNER_FIELD = "owner_id"
"""The payload key every point is tagged with, and every search filters on."""

_DOCUMENT_FIELD = "document_id"
"""The payload key for the source document. Used by delete, not by search."""

_CONVERSATION_FIELD = "conversation_id"
"""The payload key for the thread. Search filters on it, so chats do not share files.

Older points without it match no filter, which is intended.
"""


class QdrantVectorIndex:
    """One Qdrant collection, partitioned by owner. Implements `VectorIndex`.

    Returns domain `Chunk`s, never Qdrant types. One filtered collection is
    cheaper than one collection per user.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def ensure(self, dimensions: int) -> None:
        """Create the collection and payload indexes if missing. Never deletes."""
        try:
            response = await self._client.get_collections()
            exists = any(collection.name == self._collection for collection in response.collections)
            if not exists:
                logger.debug(f"Creating a new collection {self._collection}")
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
                )

            # Idempotent. Without payload indexes, filtered queries scan every point.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_OWNER_FIELD,
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            # Used by delete.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_DOCUMENT_FIELD,
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            # Used by every search.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_CONVERSATION_FIELD,
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
        except (ApiException, ResponseHandlingException) as error:
            logger.error(f"Failed to prepare collection {self._collection}: {error}")
            raise RuntimeError(f"Vector index preparation failed: {error}") from error

    async def delete_document(self, document_id: int, owner_id: int) -> None:
        """Delete one document's points, leaving the owner's other documents in place."""
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=_owned_document(document_id, owner_id)),
                wait=True,
            )
        except (ApiException, ResponseHandlingException) as error:
            logger.error(
                f"Failed to delete document {document_id} from {self._collection}: {error}"
            )
            raise RuntimeError(f"Vector index delete failed: {error}") from error
        logger.debug(f"Deleted document {document_id} (owner {owner_id}) from {self._collection}")

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        owner_id: int,
        document_id: int,
        conversation_id: int | None,
    ) -> None:
        """Write every chunk and its vector to the collection in one call."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk needs exactly one vector")
        if not chunks:
            return

        # Random ids. Count-based ids collide after deletes and concurrent uploads.
        points = [
            PointStruct(
                id=uuid4().hex,
                vector=vector,
                payload={
                    _OWNER_FIELD: owner_id,
                    _DOCUMENT_FIELD: document_id,
                    _CONVERSATION_FIELD: conversation_id,
                    "source": chunk.source,
                    "original_text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        logger.debug(f"Upserting {len(points)} points into {self._collection}")
        try:
            # One batched call, not one per chunk.
            await self._client.upsert(collection_name=self._collection, points=points)
        except (ApiException, ResponseHandlingException) as error:
            logger.error(f"Failed to upsert into {self._collection}: {error}")
            raise RuntimeError(f"Vector index upsert failed: {error}") from error

    async def search(
        self,
        vector: list[float],
        limit: int,
        score_threshold: float,
        owner_id: int,
        conversation_id: int | None,
    ) -> list[Chunk]:
        """This thread's passages nearest to `vector`, best first.

        A `None` conversation returns `[]` without searching.
        """
        if conversation_id is None:
            return []

        logger.debug(
            f"Searching {self._collection} for owner {owner_id}, conversation {conversation_id}"
        )
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
                # Filtered during the search, so other owners never use up `limit`.
                query_filter=_owned_by(owner_id, conversation_id),
            )
        except Exception as error:  # noqa: BLE001
            # Broad on purpose: a missing collection raises different types per
            # transport, and chat must still answer without context.
            logger.warning(
                f"Search against {self._collection} failed ({error}), returning empty results"
            )
            return []

        return [
            Chunk(text=str(point.payload["original_text"]), source=str(point.payload["source"]))
            for point in response.points
            if point.payload
        ]

    async def aclose(self) -> None:
        """Close the Qdrant connection."""
        await self._client.close()


def _owned_by(owner_id: int, conversation_id: int) -> Filter:
    """Owner filter (security) plus conversation filter (scoping)."""
    return Filter(
        must=[
            FieldCondition(key=_OWNER_FIELD, match=MatchValue(value=owner_id)),
            FieldCondition(key=_CONVERSATION_FIELD, match=MatchValue(value=conversation_id)),
        ]
    )


def _owned_document(document_id: int, owner_id: int) -> Filter:
    """Narrow a delete to one document, also checking the owner."""

    return Filter(
        must=[
            FieldCondition(key=_OWNER_FIELD, match=MatchValue(value=owner_id)),
            FieldCondition(key=_DOCUMENT_FIELD, match=MatchValue(value=document_id)),
        ]
    )
