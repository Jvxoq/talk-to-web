"""The upload-index-delete lifecycle against real Postgres and real Qdrant.

After a delete, however it raced indexing, a real filtered search finds nothing.
"""

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.concurrency.in_process_lock import InProcessDocumentLock
from app.adapters.persistence.models import UserModel
from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.application.ingestion.use_cases.delete_document import DeleteDocument
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.domain.chat.entities import Conversation
from app.domain.ingestion.entities import UploadedDocument
from tests.fakes import (
    FakeDocumentSummarizer,
    FakeEmbedder,
    FakeFileStorage,
    FakeTextExtractor,
)

DIMENSIONS = 4
QUERY = [100.0] * DIMENSIONS
THRESHOLD = 0.5
# The fake embedder's vectors all point the same way, so similarity is always 1.0.


class GatedEmbedder(FakeEmbedder):
    """Pauses in its first embed call until released, to make the race deterministic."""

    def __init__(self) -> None:
        super().__init__(dimensions=DIMENSIONS)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, text: str) -> list[float]:
        self.entered.set()
        await self.release.wait()
        return await super().embed(text)


@pytest.fixture
async def index(qdrant: AsyncQdrantClient, collection: str) -> AsyncIterator[QdrantVectorIndex]:
    """A collection of this test's own, deleted afterwards."""
    try:
        yield QdrantVectorIndex(qdrant, collection)
    finally:
        await qdrant.delete_collection(collection)


@pytest.fixture
def uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[], SqlAlchemyUnitOfWork]:
    """A unit of work factory, as `composition.py` builds it."""

    def factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return factory


@pytest.fixture
async def conversation_id(uow_factory: Callable[[], SqlAlchemyUnitOfWork], owner: UserModel) -> int:
    """A committed thread for the owner, since a document must name one."""
    async with uow_factory() as uow:
        stored = await uow.conversations.add(
            Conversation(title="t", model_type="m", owner_id=owner.id)
        )
        await uow.commit()
    assert stored.id is not None
    return stored.id


@pytest.fixture
async def document_id(
    uow_factory: Callable[[], SqlAlchemyUnitOfWork], owner: UserModel, conversation_id: int
) -> int:
    """The committed row `IndexDocument`'s in-lock re-check reads."""
    async with uow_factory() as uow:
        stored = await uow.documents.add(
            UploadedDocument(
                name="a.pdf",
                reference="uploads/a.pdf",
                owner_id=owner.id,
                conversation_id=conversation_id,
            )
        )
        await uow.commit()
    assert stored.id is not None
    return stored.id


def indexer(
    index: QdrantVectorIndex,
    uow_factory: Callable[[], SqlAlchemyUnitOfWork],
    lock: InProcessDocumentLock,
    embedder: FakeEmbedder | None = None,
) -> IndexDocument:
    return IndexDocument(
        extractor=FakeTextExtractor("word " * 200),
        embedder=embedder or FakeEmbedder(dimensions=DIMENSIONS),
        index=index,
        chunk_size=100,
        chunk_overlap=20,
        embedding_dimensions=DIMENSIONS,
        uow_factory=uow_factory,
        summarizer=FakeDocumentSummarizer(),
        lock=lock,
    )


def deleter(
    index: QdrantVectorIndex,
    uow_factory: Callable[[], SqlAlchemyUnitOfWork],
    lock: InProcessDocumentLock,
) -> DeleteDocument:
    return DeleteDocument(uow_factory, index=index, storage=FakeFileStorage(), lock=lock)


async def searchable(index: QdrantVectorIndex, owner_id: int, conversation_id: int) -> list[str]:
    return [
        chunk.text for chunk in await index.search(QUERY, 5, THRESHOLD, owner_id, conversation_id)
    ]


class TestDocumentLifecycle:
    async def test_a_deleted_document_is_not_searchable(
        self,
        index: QdrantVectorIndex,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        owner: UserModel,
        conversation_id: int,
        document_id: int,
    ) -> None:
        lock = InProcessDocumentLock()

        result = await indexer(index, uow_factory, lock)(
            "uploads/a.pdf", "a.pdf", document_id, owner.id, conversation_id
        )
        assert result.chunks_indexed > 0
        assert await searchable(index, owner.id, conversation_id), (
            "the positive control: an indexed document must be retrievable"
        )

        await deleter(index, uow_factory, lock)(document_id, owner.id)

        assert await searchable(index, owner.id, conversation_id) == []

    async def test_a_delete_during_indexing_leaves_nothing_searchable(
        self,
        index: QdrantVectorIndex,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        owner: UserModel,
        conversation_id: int,
        document_id: int,
    ) -> None:
        """A delete during embedding: the indexer must never write its points."""
        embedder = GatedEmbedder()
        lock = InProcessDocumentLock()

        indexing = asyncio.create_task(
            indexer(index, uow_factory, lock, embedder)(
                "uploads/a.pdf", "a.pdf", document_id, owner.id, conversation_id
            )
        )
        await embedder.entered.wait()

        await deleter(index, uow_factory, lock)(document_id, owner.id)

        embedder.release.set()
        result = await indexing

        assert result.chunks_indexed == 0
        assert await searchable(index, owner.id, conversation_id) == []
