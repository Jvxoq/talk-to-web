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
"""The payload key that scopes a point to the one upload it came from.

Search never filters on this - a query should draw from every document the
thread holds - but a delete does, which is the whole reason it exists:
without it, removing one document could only ever mean removing all of an
owner's passages.
"""

_CONVERSATION_FIELD = "conversation_id"
"""The payload key that scopes a point to the thread the file was attached to.

Search filters on this as well as on the owner. Without it a question in one
chat retrieved passages from a file uploaded in another, which is bleed the
user never asked for and cannot see. Points written before this field existed
carry no value for it, so they match no filter and are unreachable by search -
that is the intended outcome, not a gap: they belong to no thread.
"""


class QdrantVectorIndex:
    """
    One Qdrant collection, partitioned by owner, exposed as
    `app.application.ingestion.ports.VectorIndex`.

    Returns plain strings, never `ScoredPoint`s: the use cases stay free of the
    vendor's result types.

    Every point carries an `owner_id` payload field and every read filters on
    it. One collection with a filter rather than a collection per user because
    Qdrant collections are not free - each one allocates its own segments and
    index structures - and the filter is exact-match on an indexed keyword,
    which is the case it is fastest at.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def ensure(self, dimensions: int) -> None:
        """Create the collection and its owner index if they are not there yet.

        Never deletes anything. Its predecessor did: it dropped and recreated the
        collection on every single upload, which with more than one account meant
        each upload destroyed everybody else's documents.
        """
        try:
            response = await self._client.get_collections()
            exists = any(collection.name == self._collection for collection in response.collections)
            if not exists:
                logger.debug(f"Creating a new collection {self._collection}")
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
                )

            # Idempotent, and worth doing every time rather than only on
            # creation: without a payload index Qdrant still answers the filtered
            # search correctly, but by scanning, so isolation would quietly cost
            # more the more documents the deployment holds.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_OWNER_FIELD,
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            # Same reasoning, one level down: a delete filters on both fields,
            # and without an index on this one it would fall back to scanning
            # every point the owner has to find the one document's worth.
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=_DOCUMENT_FIELD,
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            # Every search filters on this one, so it carries the same argument
            # as the owner index above: correct without it, but scanned.
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

        # Random ids, not the collection's current `count()`. Counting to pick an
        # id overwrites points whenever anything was deleted, and two concurrent
        # uploads read the same count and clobber each other.
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
            # One batched call: a request per chunk turns a 200-chunk document
            # into 200 round trips.
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
        """Return this thread's stored passages nearest to `vector`, best first.

        A `None` conversation owns no documents, so the search is not run at
        all. Returning early rather than filtering on NULL is the safe
        direction: a filter that fails to match nothing would match everything.
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
                # Applied by Qdrant during the search, not to its results. A
                # post-filter would let another owner's chunks consume the `limit`
                # nearest slots and return fewer of this owner's than asked for -
                # a correctness bug on top of the obvious one.
                query_filter=_owned_by(owner_id, conversation_id),
            )
        except Exception as error:
            # Deliberately broad, preserving the legacy behaviour: a chat request
            # before anything has been uploaded must still answer, just without
            # retrieved context. Qdrant signals "no such collection" through
            # several different exception types depending on transport, so
            # narrowing here would reintroduce a 500 on an empty knowledge base.
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
    """The predicate that keeps one thread's documents out of another's answers.

    Two conditions, both required. The owner one is the security boundary: it
    is what stops one account reading another's files. The conversation one is
    the scoping boundary: it is what stops a thread reading a file the same
    person attached somewhere else.
    """
    return Filter(
        must=[
            FieldCondition(key=_OWNER_FIELD, match=MatchValue(value=owner_id)),
            FieldCondition(key=_CONVERSATION_FIELD, match=MatchValue(value=conversation_id)),
        ]
    )


def _owned_document(document_id: int, owner_id: int) -> Filter:
    """The predicate that narrows a delete to one document, within one owner.

    Both fields are checked, not just `document_id`: an id from the database is
    trustworthy, but the point still carries `owner_id` as the primary
    isolation boundary, and a delete should never rely on a single field being
    right.
    """
    return Filter(
        must=[
            FieldCondition(key=_OWNER_FIELD, match=MatchValue(value=owner_id)),
            FieldCondition(key=_DOCUMENT_FIELD, match=MatchValue(value=document_id)),
        ]
    )
