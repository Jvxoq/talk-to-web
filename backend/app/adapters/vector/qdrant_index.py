"""Qdrant-backed vector index."""

from collections.abc import Sequence
from uuid import uuid4

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.domain.ingestion.value_objects import Chunk


class QdrantVectorIndex:
    """
    One Qdrant collection, exposed as `app.application.ingestion.ports.VectorIndex`.

    Returns plain strings, never `ScoredPoint`s: the use cases stay free of the
    vendor's result types.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def reset(self, dimensions: int) -> None:
        """Drop the collection if present and recreate it empty at `dimensions`."""
        vectors_config = VectorParams(size=dimensions, distance=Distance.COSINE)
        try:
            response = await self._client.get_collections()
            exists = any(collection.name == self._collection for collection in response.collections)
            if exists:
                logger.debug(f"Collection {self._collection} already exists - recreating it")
                await self._client.delete_collection(self._collection)
            else:
                logger.debug(f"Creating a new collection {self._collection}")

            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=vectors_config,
            )
        except (ApiException, ResponseHandlingException) as error:
            logger.error(f"Failed to reset collection {self._collection}: {error}")
            raise RuntimeError(f"Vector index reset failed: {error}") from error

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
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
                payload={"source": chunk.source, "original_text": chunk.text},
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

    async def search(self, vector: list[float], limit: int, score_threshold: float) -> list[str]:
        """Return the stored passages nearest to `vector`, best first."""
        logger.debug(f"Searching for relevant items in the {self._collection} collection")
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
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

        return [str(point.payload["original_text"]) for point in response.points if point.payload]

    async def aclose(self) -> None:
        """Close the Qdrant connection."""
        await self._client.close()
