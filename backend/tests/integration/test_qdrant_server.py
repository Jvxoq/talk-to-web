"""The Qdrant adapter against a real server.

`tests/adapters/test_qdrant_index.py` runs the vendor's in-process local mode,
which is enough for the isolation claims - one owner's documents never reach
another owner's answers - and its docstring names what it cannot do. Payload
index creation is top of that list: local mode accepts the call and does
nothing, so the only place `ensure` can be shown to build the index is here.

The index is not a performance detail. Without it the owner filter still returns
the right passages, by scanning every point in the collection, so a deployment
would get slower the more documents it holds and no test would notice.
"""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.adapters.vector.qdrant_index import QdrantVectorIndex

DIMENSIONS = 4
OWNER_FIELD = "owner_id"


@pytest.fixture
async def index(qdrant: AsyncQdrantClient, collection: str) -> AsyncIterator[QdrantVectorIndex]:
    """A collection of this test's own, deleted afterwards.

    A shared name would make two runs against one server - a laptop and CI, or
    two CI jobs - fight over the same points.
    """
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

    async def test_it_is_idempotent_against_a_live_server(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """Called on every upload, so a second call must not be an error.

        Local mode would accept a repeated `create_payload_index` silently; a
        real server is the one that can reject it.
        """
        await index.ensure(DIMENSIONS)
        await index.ensure(DIMENSIONS)

        assert await qdrant.collection_exists(collection)

    async def test_it_never_drops_an_existing_collection(
        self, index: QdrantVectorIndex, qdrant: AsyncQdrantClient, collection: str
    ) -> None:
        """The regression its predecessor caused: every upload wiped the store.

        Checked here as well as in local mode because "the collection was not
        recreated" is a statement about server state, and a point count is the
        only thing that can distinguish "kept" from "rebuilt empty".
        """
        await index.ensure(DIMENSIONS)
        await qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={OWNER_FIELD: 1})],
            wait=True,
        )

        await index.ensure(DIMENSIONS)

        assert (await qdrant.count(collection, exact=True)).count == 1
