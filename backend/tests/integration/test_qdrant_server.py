"""The Qdrant adapter against a real server.

Local mode ignores payload indexes, so this is where `ensure` is shown to
build them. Without them, filtered search scans every point.
"""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.adapters.vector.qdrant_index import QdrantVectorIndex

DIMENSIONS = 4
OWNER_FIELD = "owner_id"
CONVERSATION_FIELD = "conversation_id"


@pytest.fixture
async def index(qdrant: AsyncQdrantClient, collection: str) -> AsyncIterator[QdrantVectorIndex]:
    """A collection for this test only, deleted afterwards."""
    try:
        yield QdrantVectorIndex(qdrant, collection)
    finally:
        await qdrant.delete_collection(collection)


class TestEnsure:
    async def test_it_creates_the_collection_with_the_right_dimensions(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        await index.ensure(DIMENSIONS)

        info = await qdrant.get_collection(collection)
        vectors = info.config.params.vectors
        assert vectors is not None
        assert getattr(vectors, "size", None) == DIMENSIONS

    async def test_it_indexes_the_owner_field(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """The claim local mode cannot make, and the reason this file exists."""
        await index.ensure(DIMENSIONS)

        info = await qdrant.get_collection(collection)

        assert OWNER_FIELD in (info.payload_schema or {})

    async def test_it_indexes_the_conversation_field(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """Every search filters on conversation, so it needs an index too."""
        await index.ensure(DIMENSIONS)

        info = await qdrant.get_collection(collection)

        assert CONVERSATION_FIELD in (info.payload_schema or {})

    async def test_it_is_idempotent_against_a_live_server(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """`ensure` runs on every upload, so a repeat must not fail on a real server."""
        await index.ensure(DIMENSIONS)
        await index.ensure(DIMENSIONS)

        assert await qdrant.collection_exists(collection)

    async def test_it_never_drops_an_existing_collection(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """`ensure` keeps existing points on a real server."""
        await index.ensure(DIMENSIONS)
        await qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={OWNER_FIELD: 1})],
            wait=True,
        )

        await index.ensure(DIMENSIONS)

        assert (await qdrant.count(collection, exact=True)).count == 1
