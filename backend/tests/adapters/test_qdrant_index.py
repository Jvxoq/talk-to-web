"""The Qdrant index against in-process Qdrant (`location=":memory:"`).

Real collections and filters, so owner isolation is tested for real. Not
covered: payload indexes and `ApiException` paths, which need a server.
"""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.domain.ingestion.value_objects import Chunk

COLLECTION = "documents"
DIMENSIONS = 4

ALICE = 1
BOB = 2

# Distinct from owner ids, so a filter on the wrong field fails.
DOC_A = 10
DOC_B = 20
DOC_C = 30

# Conversation ids, distinct from both of the above.
THREAD = 100
OTHER_THREAD = 200

# Axis-aligned, so "nearest" is obvious.
NORTH = [1.0, 0.0, 0.0, 0.0]
EAST = [0.0, 1.0, 0.0, 0.0]
SOUTH = [-1.0, 0.0, 0.0, 0.0]


@pytest.fixture
async def index() -> AsyncIterator[QdrantVectorIndex]:
    client = AsyncQdrantClient(location=":memory:")
    index = QdrantVectorIndex(client, COLLECTION)
    await index.ensure(DIMENSIONS)
    try:
        yield index
    finally:
        await index.aclose()


def chunk(text: str, source: str = "doc.pdf") -> Chunk:
    return Chunk(text=text, source=source)


async def texts(
    index: QdrantVectorIndex,
    vector: list[float],
    owner_id: int,
    limit: int = 10,
    conversation_id: int = THREAD,
) -> list[str]:
    """The text of each passage a search found."""
    found = await index.search(
        vector,
        limit=limit,
        score_threshold=-1.0,
        owner_id=owner_id,
        conversation_id=conversation_id,
    )
    return [chunk.text for chunk in found]


class TestEnsure:
    async def test_it_creates_the_collection(self, index: QdrantVectorIndex) -> None:
        # Upserting proves `ensure` created the collection.
        await index.upsert([chunk("hello")], [NORTH], ALICE, DOC_A, THREAD)

        assert await texts(index, NORTH, ALICE) == ["hello"]

    async def test_calling_it_again_keeps_the_existing_documents(self) -> None:
        # `ensure` must never wipe existing points.
        client = AsyncQdrantClient(location=":memory:")
        index = QdrantVectorIndex(client, COLLECTION)
        await index.ensure(DIMENSIONS)
        await index.upsert([chunk("survives")], [NORTH], ALICE, DOC_A, THREAD)

        await index.ensure(DIMENSIONS)

        assert await texts(index, NORTH, ALICE) == ["survives"]
        await index.aclose()


class TestUpsert:
    async def test_it_stores_the_original_text_of_every_chunk(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.upsert([chunk("north"), chunk("east")], [NORTH, EAST], ALICE, DOC_A, THREAD)

        found = await texts(index, NORTH, ALICE)

        assert sorted(found) == ["east", "north"]

    async def test_two_upserts_of_the_same_text_both_survive(
        self, index: QdrantVectorIndex
    ) -> None:
        # Random ids, so identical uploads do not overwrite each other.
        await index.upsert([chunk("same")], [NORTH], ALICE, DOC_A, THREAD)
        await index.upsert([chunk("same")], [NORTH], ALICE, DOC_A, THREAD)

        found = await texts(index, NORTH, ALICE)

        assert found == ["same", "same"]

    async def test_it_refuses_a_mismatched_number_of_vectors(
        self, index: QdrantVectorIndex
    ) -> None:
        # A mismatch must raise, not silently drop chunks.
        with pytest.raises(ValueError, match="exactly one vector"):
            await index.upsert([chunk("a"), chunk("b")], [NORTH], ALICE, DOC_A, THREAD)

    async def test_the_source_file_survives_the_round_trip(self, index: QdrantVectorIndex) -> None:
        # The source round-trips through the payload.
        await index.upsert(
            [chunk("a passage", source="handbook.pdf")], [NORTH], ALICE, DOC_A, THREAD
        )

        found = await index.search(
            NORTH, limit=1, score_threshold=-1.0, owner_id=ALICE, conversation_id=THREAD
        )

        assert found == [Chunk(text="a passage", source="handbook.pdf")]

    async def test_upserting_nothing_is_not_a_request(self, index: QdrantVectorIndex) -> None:
        await index.upsert([], [], ALICE, DOC_A, THREAD)

        assert await texts(index, NORTH, ALICE) == []


class TestSearch:
    async def test_it_returns_the_nearest_passages_best_first(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.upsert(
            [chunk("opposite"), chunk("adjacent"), chunk("identical")],
            [SOUTH, EAST, NORTH],
            ALICE,
            DOC_A,
            THREAD,
        )

        found = await texts(index, NORTH, ALICE, limit=3)

        assert found == ["identical", "adjacent", "opposite"]

    async def test_it_honours_the_limit(self, index: QdrantVectorIndex) -> None:
        await index.upsert(
            [chunk("a"), chunk("b"), chunk("c")], [NORTH, EAST, SOUTH], ALICE, DOC_A, THREAD
        )

        assert len(await texts(index, NORTH, ALICE, limit=2)) == 2

    async def test_the_score_threshold_drops_the_irrelevant(self, index: QdrantVectorIndex) -> None:
        await index.upsert([chunk("near"), chunk("far")], [NORTH, SOUTH], ALICE, DOC_A, THREAD)

        # Cosine: 1.0 for NORTH, -1.0 for SOUTH.
        found = await index.search(
            NORTH, limit=10, score_threshold=0.5, owner_id=ALICE, conversation_id=THREAD
        )

        assert [chunk.text for chunk in found] == ["near"]

    async def test_one_owners_documents_never_appear_in_anothers_results(
        self, index: QdrantVectorIndex
    ) -> None:
        # The single most consequential assertion about this adapter.
        await index.upsert([chunk("alice's private notes")], [NORTH], ALICE, DOC_A, THREAD)

        assert await texts(index, NORTH, BOB) == []

    async def test_another_owners_nearer_chunks_do_not_consume_the_limit(
        self, index: QdrantVectorIndex
    ) -> None:
        # Filtered during search, so Bob's matches cannot use up Alice's slots.
        await index.upsert(
            [chunk("bob 1"), chunk("bob 2"), chunk("bob 3")], [NORTH] * 3, BOB, DOC_B, THREAD
        )
        await index.upsert([chunk("alice")], [EAST], ALICE, DOC_A, THREAD)

        found = await texts(index, NORTH, ALICE, limit=3)

        assert found == ["alice"]

    async def test_a_search_against_a_collection_that_does_not_exist_answers_empty(self) -> None:
        # A missing collection returns [] so chat still answers.
        client = AsyncQdrantClient(location=":memory:")
        index = QdrantVectorIndex(client, "never-created")

        assert (
            await index.search(
                NORTH, limit=3, score_threshold=0.0, owner_id=ALICE, conversation_id=THREAD
            )
            == []
        )

        await index.aclose()


class TestDeleteDocument:
    async def test_it_removes_that_documents_passages(self, index: QdrantVectorIndex) -> None:
        await index.upsert([chunk("gone")], [NORTH], ALICE, DOC_A, THREAD)

        await index.delete_document(DOC_A, ALICE)

        assert await texts(index, NORTH, ALICE) == []

    async def test_it_leaves_the_owners_other_documents_alone(
        self, index: QdrantVectorIndex
    ) -> None:
        # Deleting one document leaves the owner's others.
        await index.upsert([chunk("deleted doc")], [NORTH], ALICE, DOC_A, THREAD)
        await index.upsert([chunk("kept doc")], [EAST], ALICE, DOC_C, THREAD)

        await index.delete_document(DOC_A, ALICE)

        assert await texts(index, EAST, ALICE) == ["kept doc"]

    async def test_it_cannot_reach_another_owners_document(self, index: QdrantVectorIndex) -> None:
        # Delete also checks the owner, not just the id.
        await index.upsert([chunk("bob's")], [NORTH], BOB, DOC_B, THREAD)

        await index.delete_document(DOC_B, ALICE)

        assert await texts(index, NORTH, BOB) == ["bob's"]

    async def test_deleting_a_document_that_is_not_there_is_fine(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.delete_document(DOC_C, BOB)


class TestConversationIsolation:
    """One thread must not read a file attached to another thread of the same user."""

    async def test_a_search_never_reads_another_threads_document(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.upsert([chunk("the tax return")], [NORTH], ALICE, DOC_A, THREAD)
        await index.upsert([chunk("the lease")], [NORTH], ALICE, DOC_C, OTHER_THREAD)

        assert await texts(index, NORTH, ALICE, conversation_id=THREAD) == ["the tax return"]
        assert await texts(index, NORTH, ALICE, conversation_id=OTHER_THREAD) == ["the lease"]

    async def test_a_turn_with_no_thread_reads_nothing(self, index: QdrantVectorIndex) -> None:
        """`None` is not a wildcard. No thread owns no documents."""
        await index.upsert([chunk("the tax return")], [NORTH], ALICE, DOC_A, THREAD)

        found = await index.search(
            NORTH, limit=10, score_threshold=-1.0, owner_id=ALICE, conversation_id=None
        )

        assert found == []
