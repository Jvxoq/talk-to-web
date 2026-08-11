"""The Qdrant index against real Qdrant.

`AsyncQdrantClient(location=":memory:")` runs the vendor's own local
implementation in-process: real collections, real payload filters, real vector
search. So the claim that matters here - one owner's documents never appear in
another owner's answers - is checked against the thing that will enforce it in
production, not against a dictionary that filters in Python.

Not covered, because local mode does not implement them: payload index creation
is a no-op here, so `ensure` is exercised for its collection handling rather
than its indexing, and the `ApiException` translation paths need a server that
can fail.
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

# Document ids, distinct from owner ids on purpose: a payload filter that
# matched on the wrong field would still pass if the two ever coincided.
DOC_A = 10
DOC_B = 20
DOC_C = 30

# Deliberately axis-aligned, so "nearest" is obvious by inspection rather than
# an artefact of whatever numbers happened to be typed.
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
    index: QdrantVectorIndex, vector: list[float], owner_id: int, limit: int = 10
) -> list[str]:
    """The passages a search found, as plain text.

    `search` returns `Chunk`s so a citation can name the file a passage came
    from; most assertions here are about which passages came back, not about
    that, so they read the text off rather than restating the source every time.
    One test below asserts on the source itself.
    """
    found = await index.search(vector, limit=limit, score_threshold=-1.0, owner_id=owner_id)
    return [chunk.text for chunk in found]


class TestEnsure:
    async def test_it_creates_the_collection(self, index: QdrantVectorIndex) -> None:
        # The fixture already called `ensure`; upserting proves the collection
        # it was supposed to create is really there.
        await index.upsert([chunk("hello")], [NORTH], ALICE, DOC_A)

        assert await texts(index, NORTH, ALICE) == ["hello"]

    async def test_calling_it_again_keeps_the_existing_documents(self) -> None:
        # This is the regression the docstring on `ensure` describes: its
        # predecessor dropped and recreated the collection on every upload, so
        # with more than one account each upload destroyed everyone else's
        # documents.
        client = AsyncQdrantClient(location=":memory:")
        index = QdrantVectorIndex(client, COLLECTION)
        await index.ensure(DIMENSIONS)
        await index.upsert([chunk("survives")], [NORTH], ALICE, DOC_A)

        await index.ensure(DIMENSIONS)

        assert await texts(index, NORTH, ALICE) == ["survives"]
        await index.aclose()


class TestUpsert:
    async def test_it_stores_the_original_text_of_every_chunk(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.upsert([chunk("north"), chunk("east")], [NORTH, EAST], ALICE, DOC_A)

        found = await texts(index, NORTH, ALICE)

        assert sorted(found) == ["east", "north"]

    async def test_two_upserts_of_the_same_text_both_survive(
        self, index: QdrantVectorIndex
    ) -> None:
        # Point ids are random rather than derived from the current `count()`.
        # Counting to pick an id overwrites points whenever anything was
        # deleted, and two concurrent uploads read the same count and clobber
        # each other.
        await index.upsert([chunk("same")], [NORTH], ALICE, DOC_A)
        await index.upsert([chunk("same")], [NORTH], ALICE, DOC_A)

        found = await texts(index, NORTH, ALICE)

        assert found == ["same", "same"]

    async def test_it_refuses_a_mismatched_number_of_vectors(
        self, index: QdrantVectorIndex
    ) -> None:
        # Zipping these without checking would silently drop the tail, and the
        # missing chunks would only surface as an answer that omitted them.
        with pytest.raises(ValueError, match="exactly one vector"):
            await index.upsert([chunk("a"), chunk("b")], [NORTH], ALICE, DOC_A)

    async def test_the_source_file_survives_the_round_trip(self, index: QdrantVectorIndex) -> None:
        # `search` returns `Chunk`s rather than bare strings so an answer can
        # say which document a passage came from. That only works if the source
        # is written into the payload and read back out of it.
        await index.upsert([chunk("a passage", source="handbook.pdf")], [NORTH], ALICE, DOC_A)

        found = await index.search(NORTH, limit=1, score_threshold=-1.0, owner_id=ALICE)

        assert found == [Chunk(text="a passage", source="handbook.pdf")]

    async def test_upserting_nothing_is_not_a_request(self, index: QdrantVectorIndex) -> None:
        await index.upsert([], [], ALICE, DOC_A)

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
        )

        found = await texts(index, NORTH, ALICE, limit=3)

        assert found == ["identical", "adjacent", "opposite"]

    async def test_it_honours_the_limit(self, index: QdrantVectorIndex) -> None:
        await index.upsert([chunk("a"), chunk("b"), chunk("c")], [NORTH, EAST, SOUTH], ALICE, DOC_A)

        assert len(await texts(index, NORTH, ALICE, limit=2)) == 2

    async def test_the_score_threshold_drops_the_irrelevant(self, index: QdrantVectorIndex) -> None:
        await index.upsert([chunk("near"), chunk("far")], [NORTH, SOUTH], ALICE, DOC_A)

        # Cosine: 1.0 for NORTH, -1.0 for SOUTH.
        found = await index.search(NORTH, limit=10, score_threshold=0.5, owner_id=ALICE)

        assert [chunk.text for chunk in found] == ["near"]

    async def test_one_owners_documents_never_appear_in_anothers_results(
        self, index: QdrantVectorIndex
    ) -> None:
        # The single most consequential assertion about this adapter.
        await index.upsert([chunk("alice's private notes")], [NORTH], ALICE, DOC_A)

        assert await texts(index, NORTH, BOB) == []

    async def test_another_owners_nearer_chunks_do_not_consume_the_limit(
        self, index: QdrantVectorIndex
    ) -> None:
        # The filter is applied by Qdrant *during* the search, not to its
        # results. Under a post-filter, Bob's three exact matches would take
        # every slot and Alice would get nothing back - a correctness bug on top
        # of the isolation one.
        await index.upsert(
            [chunk("bob 1"), chunk("bob 2"), chunk("bob 3")], [NORTH] * 3, BOB, DOC_B
        )
        await index.upsert([chunk("alice")], [EAST], ALICE, DOC_A)

        found = await texts(index, NORTH, ALICE, limit=3)

        assert found == ["alice"]

    async def test_a_search_against_a_collection_that_does_not_exist_answers_empty(self) -> None:
        # A chat request before anything has been uploaded must still answer,
        # just without retrieved context. This is the reason the `except` in
        # `search` is deliberately broad.
        client = AsyncQdrantClient(location=":memory:")
        index = QdrantVectorIndex(client, "never-created")

        assert await index.search(NORTH, limit=3, score_threshold=0.0, owner_id=ALICE) == []

        await index.aclose()


class TestDeleteDocument:
    async def test_it_removes_that_documents_passages(self, index: QdrantVectorIndex) -> None:
        await index.upsert([chunk("gone")], [NORTH], ALICE, DOC_A)

        await index.delete_document(DOC_A, ALICE)

        assert await texts(index, NORTH, ALICE) == []

    async def test_it_leaves_the_owners_other_documents_alone(
        self, index: QdrantVectorIndex
    ) -> None:
        # The reason points carry a document id at all: without it, removing one
        # upload could only ever mean removing everything the owner had.
        await index.upsert([chunk("deleted doc")], [NORTH], ALICE, DOC_A)
        await index.upsert([chunk("kept doc")], [EAST], ALICE, DOC_C)

        await index.delete_document(DOC_A, ALICE)

        assert await texts(index, EAST, ALICE) == ["kept doc"]

    async def test_it_cannot_reach_another_owners_document(self, index: QdrantVectorIndex) -> None:
        # Both fields are in the filter. An id alone would let one account
        # delete another's upload by naming its number.
        await index.upsert([chunk("bob's")], [NORTH], BOB, DOC_B)

        await index.delete_document(DOC_B, ALICE)

        assert await texts(index, NORTH, BOB) == ["bob's"]

    async def test_deleting_a_document_that_is_not_there_is_fine(
        self, index: QdrantVectorIndex
    ) -> None:
        await index.delete_document(DOC_C, BOB)
