"""Ingestion use cases against fakes."""

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.application.ingestion.dto import UploadDocumentInput
from app.application.ingestion.ports import DocumentSummarizer
from app.application.ingestion.use_cases.delete_document import DeleteDocument
from app.application.ingestion.use_cases.get_conversation_document import GetConversationDocument
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.list_documents import ListDocuments
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.domain.chat.entities import Conversation
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import (
    DocumentNotFound,
    DocumentNotIndexable,
    UnknownConversation,
    UnsupportedDocumentType,
)
from app.domain.ingestion.value_objects import Chunk
from app.domain.usage.errors import RateLimited
from tests.fakes import (
    FakeDocumentLock,
    FakeDocumentRemover,
    FakeDocumentSummarizer,
    FakeEmbedder,
    FakeFileStorage,
    FakeRateLimiter,
    FakeTextExtractor,
    FakeVectorIndex,
    UnitOfWorkSpy,
)

OWNER = 7
STRANGER = 8
# The thread every upload here attaches to.
CONVERSATION = 42
# The stranger's own thread.
STRANGER_CONVERSATION = 43
# A second thread of the same account.
OTHER_CONVERSATION = 44


async def bytes_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def upload_document(
    storage: FakeFileStorage,
    max_bytes: int,
    limiter: FakeRateLimiter,
    factory: UnitOfWorkSpy | None = None,
    daily_budget: FakeRateLimiter | None = None,
    remove_document: "FakeDocumentRemover | None" = None,
) -> UploadDocument:
    factory = factory or UnitOfWorkSpy()
    # Uploads need an owned thread, so seed one.
    seed_conversation(factory)
    seed_conversation(factory, STRANGER, STRANGER_CONVERSATION)
    return UploadDocument(
        storage,
        max_bytes=max_bytes,
        limiter=limiter,
        uow_factory=factory,
        daily_budget=daily_budget or FakeRateLimiter(),
        remove_document=remove_document or FakeDocumentRemover(),
    )


def seed_conversation(
    factory: UnitOfWorkSpy, owner_id: int = OWNER, conversation_id: int = CONVERSATION
) -> None:
    """Give `factory` a conversation an upload in this file can attach to."""
    factory.repository.rows.setdefault(
        conversation_id,
        Conversation(title="t", model_type="m", owner_id=owner_id, id=conversation_id),
    )


def existing_document(document_id: int, owner_id: int) -> UploadedDocument:
    return UploadedDocument(
        name="a.pdf", reference="uploads/a.pdf", owner_id=owner_id, id=document_id
    )


class TestUploadDocument:
    async def test_stores_a_pdf_under_a_safe_name(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                filename="../../etc/report.pdf",
                content_type="application/pdf",
                stream=bytes_stream(b"%PDF", b"-data"),
                owner_id=OWNER,
                conversation_id=CONVERSATION,
            )
        )

        assert storage.saved == [("report.pdf", b"%PDF-data")]
        assert result.name == "report.pdf"
        assert result.reference == "uploads/7/report.pdf"
        assert result.document_id == 1
        assert storage.owners == [OWNER]

    async def test_records_a_document_for_every_accepted_upload(self) -> None:
        factory = UnitOfWorkSpy()
        upload = upload_document(FakeFileStorage(), 1024, FakeRateLimiter(), factory)

        result = await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )

        stored = factory.documents.rows[result.document_id]
        assert stored.name == "a.pdf"
        assert stored.owner_id == OWNER
        assert stored.chunks_indexed == 0

    async def test_accepts_a_content_type_with_parameters(self) -> None:
        # Browsers append parameters like charset.
        result = await upload_document(FakeFileStorage(), 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "a.pdf",
                "application/pdf; charset=binary",
                bytes_stream(b"%PDF-1.7"),
                OWNER,
                CONVERSATION,
            )
        )
        assert result.name == "a.pdf"

    @pytest.mark.parametrize("content_type", ["text/html", "image/png", None, ""])
    async def test_rejects_everything_that_is_not_a_pdf(self, content_type: str | None) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "a.exe", content_type, bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
                )
            )
        assert storage.saved == [], "nothing may be written before the type check passes"

    async def test_rejects_a_file_that_only_claims_to_be_a_pdf(self) -> None:
        # Type and name are client-chosen. The first bytes decide.
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "payload.pdf",
                    "application/pdf",
                    bytes_stream(b"MZ\x90\x00"),
                    OWNER,
                    CONVERSATION,
                )
            )
        assert storage.saved == []

    @pytest.mark.parametrize("chunks", [(), (b"",), (b"%PD",)])
    async def test_rejects_a_file_too_short_to_prove_anything(self, chunks: tuple[bytes]) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "a.pdf", "application/pdf", bytes_stream(*chunks), OWNER, CONVERSATION
                )
            )
        assert storage.saved == []

    async def test_rejects_a_name_without_the_pdf_extension(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "report.txt", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
                )
            )
        assert storage.saved == []

    async def test_the_signature_check_does_not_eat_the_bytes_it_read(self) -> None:
        # Buffering the opening bytes must not change the stored file.
        storage = FakeFileStorage()
        await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "a.pdf",
                "application/pdf",
                bytes_stream(b"%P", b"DF", b"-1.7", b"body"),
                OWNER,
                CONVERSATION,
            )
        )
        assert storage.saved == [("a.pdf", b"%PDF-1.7body")]

    async def test_an_upload_past_the_budget_is_refused_before_anything_is_stored(self) -> None:
        # The limit applies before any bytes are accepted.
        storage = FakeFileStorage()
        limiter = FakeRateLimiter(max_attempts=1)
        upload = upload_document(storage, 1024, limiter)

        await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )
        with pytest.raises(RateLimited):
            await upload(
                UploadDocumentInput(
                    "b.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
                )
            )

        assert len(storage.saved) == 1

    async def test_accepts_a_txt_file(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "notes.txt", "text/plain", bytes_stream(b"just some text"), OWNER, CONVERSATION
            )
        )
        assert result.name == "notes.txt"
        assert storage.saved == [("notes.txt", b"just some text")]

    async def test_accepts_a_markdown_file(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "notes.md", "text/markdown", bytes_stream(b"# heading\n\nbody"), OWNER, CONVERSATION
            )
        )
        assert result.name == "notes.md"
        assert storage.saved == [("notes.md", b"# heading\n\nbody")]

    async def test_accepts_a_docx_file(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                bytes_stream(b"PK\x03\x04", b"therest"),
                OWNER,
                CONVERSATION,
            )
        )
        assert result.name == "report.docx"
        assert storage.saved == [("report.docx", b"PK\x03\x04therest")]

    async def test_rejects_a_txt_file_with_the_wrong_extension(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "notes.pdf", "text/plain", bytes_stream(b"just some text"), OWNER, CONVERSATION
                )
            )
        assert storage.saved == []

    async def test_rejects_a_docx_that_only_claims_to_be_one(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    bytes_stream(b"not a zip at all"),
                    OWNER,
                    CONVERSATION,
                )
            )
        assert storage.saved == []

    async def test_one_user_hitting_the_limit_does_not_block_another(self) -> None:
        limiter = FakeRateLimiter(max_attempts=1)
        upload = upload_document(FakeFileStorage(), 1024, limiter)

        await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )
        # Keyed per account, so the stranger still has their own budget.
        await upload(
            UploadDocumentInput(
                "a.pdf",
                "application/pdf",
                bytes_stream(b"%PDF-1.7"),
                STRANGER,
                STRANGER_CONVERSATION,
            )
        )

        assert limiter.hits == {f"upload:{OWNER}": 1, f"upload:{STRANGER}": 1}

    async def test_a_spent_daily_budget_refuses_before_anything_is_stored(self) -> None:
        storage = FakeFileStorage()
        limiter = FakeRateLimiter()
        upload = upload_document(
            storage, 1024, limiter, daily_budget=FakeRateLimiter(max_attempts=0)
        )

        with pytest.raises(RateLimited):
            await upload(
                UploadDocumentInput(
                    "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
                )
            )

        assert storage.saved == []
        assert limiter.hits == {}, "the per-user limit must not be spent on a refused request"


class TestOneAttachmentPerConversation:
    """A thread holds one document. Uploading a second replaces the first."""

    async def test_a_second_upload_removes_the_first(self) -> None:
        factory = UnitOfWorkSpy()
        remover = FakeDocumentRemover()
        upload = upload_document(
            FakeFileStorage(), 1024, FakeRateLimiter(), factory=factory, remove_document=remover
        )

        first = await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )
        await upload(
            UploadDocumentInput(
                "b.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )

        assert remover.removed == [(first.document_id, OWNER)]

    async def test_it_leaves_another_threads_attachment_alone(self) -> None:
        factory = UnitOfWorkSpy()
        remover = FakeDocumentRemover()
        upload = upload_document(
            FakeFileStorage(), 1024, FakeRateLimiter(), factory=factory, remove_document=remover
        )
        seed_conversation(factory, OWNER, OTHER_CONVERSATION)

        await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, OTHER_CONVERSATION
            )
        )
        await upload(
            UploadDocumentInput(
                "b.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )

        assert remover.removed == [], "the other thread's file is not this thread's to replace"

    async def test_a_failed_removal_does_not_cost_the_new_upload(self) -> None:
        # The new file is stored, so a failed old delete must not fail the upload.
        factory = UnitOfWorkSpy()
        remover = FakeDocumentRemover(fail_with=RuntimeError("qdrant is down"))
        upload = upload_document(
            FakeFileStorage(), 1024, FakeRateLimiter(), factory=factory, remove_document=remover
        )

        await upload(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )
        result = await upload(
            UploadDocumentInput(
                "b.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )

        assert result.name == "b.pdf"

    async def test_it_refuses_a_thread_this_account_does_not_own(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnknownConversation):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), STRANGER, CONVERSATION
                )
            )

        assert storage.saved == [], "nothing may be written for a thread that is not theirs"

    async def test_the_stored_row_names_the_thread(self) -> None:
        factory = UnitOfWorkSpy()
        result = await upload_document(FakeFileStorage(), 1024, FakeRateLimiter(), factory=factory)(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER, CONVERSATION
            )
        )

        assert factory.documents.rows[result.document_id].conversation_id == CONVERSATION


class TestIndexDocument:
    def build(
        self,
        text: str,
        index: FakeVectorIndex | None = None,
        factory: UnitOfWorkSpy | None = None,
        summarizer: DocumentSummarizer | None = None,
        embedder: FakeEmbedder | None = None,
        lock: FakeDocumentLock | None = None,
        document_id: int = 1,
        owner_id: int = OWNER,
    ) -> IndexDocument:
        factory = factory or UnitOfWorkSpy()
        # In production the row always exists before indexing, so seed it.
        factory.documents.rows.setdefault(document_id, existing_document(document_id, owner_id))
        return IndexDocument(
            extractor=FakeTextExtractor(text),
            embedder=embedder or FakeEmbedder(),
            index=index or FakeVectorIndex(),
            chunk_size=100,
            chunk_overlap=20,
            embedding_dimensions=3,
            uow_factory=factory,
            summarizer=summarizer or FakeDocumentSummarizer(),
            lock=lock or FakeDocumentLock(),
        )

    async def test_indexes_every_chunk_exactly_once(self) -> None:
        index = FakeVectorIndex()
        result = await self.build("word " * 200, index)(
            reference="uploads/a.pdf",
            name="a.pdf",
            document_id=1,
            owner_id=OWNER,
            conversation_id=CONVERSATION,
        )

        assert result.chunks_indexed == len(index.chunks)
        assert result.chunks_indexed > 1
        assert all(chunk.source == "a.pdf" for chunk in index.chunks)

    async def test_records_the_chunk_count_against_the_document(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)

        result = await self.build("word " * 200, factory=factory)(
            "uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION
        )

        assert factory.documents.rows[1].chunks_indexed == result.chunks_indexed

    async def test_batches_larger_than_the_embed_window(self) -> None:
        # More than one batch. Each chunk still gets one vector.
        index = FakeVectorIndex()
        embedder = FakeEmbedder()
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        use_case = IndexDocument(
            extractor=FakeTextExtractor("x" * 2_000),
            embedder=embedder,
            index=index,
            chunk_size=50,
            chunk_overlap=10,
            embedding_dimensions=3,
            uow_factory=factory,
            summarizer=FakeDocumentSummarizer(),
            lock=FakeDocumentLock(),
        )

        result = await use_case("uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION)

        assert len(embedder.embedded) == result.chunks_indexed > 20

    async def test_ensures_the_collection_without_clearing_it(self) -> None:
        index = FakeVectorIndex()
        await self.build("hello world", index)("uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION)

        assert index.ensured == [3], "the collection is created, never recreated"

    async def test_a_second_upload_does_not_wipe_the_first(self) -> None:
        """A second upload must not erase the first one's passages."""
        index = FakeVectorIndex()
        await self.build("first document text", index, document_id=1)(
            "uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION
        )
        first_count = len(index.chunks)
        assert first_count > 0

        await self.build("second document text", index, document_id=2)(
            "uploads/b.pdf", "b.pdf", 2, OWNER, CONVERSATION
        )

        assert len(index.chunks) > first_count, "the first document's passages must survive"
        assert {chunk.source for chunk in index.chunks} == {"a.pdf", "b.pdf"}

    async def test_one_upload_does_not_wipe_another_account(self) -> None:
        index = FakeVectorIndex()
        await self.build("owner text", index, document_id=1)(
            "uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION
        )
        assert index.by_owner[OWNER]

        await self.build("stranger text", index, document_id=2, owner_id=STRANGER)(
            "uploads/b.pdf", "b.pdf", 2, STRANGER, CONVERSATION
        )

        assert index.by_owner[OWNER], "the first owner's passages must survive"
        assert index.by_owner[STRANGER]

    async def test_it_stores_the_digest_alongside_the_chunk_count(self) -> None:
        factory = UnitOfWorkSpy()
        summarizer = FakeDocumentSummarizer("A quarterly travel budget.")
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        await self.build("word " * 200, factory=factory, summarizer=summarizer)(
            reference="uploads/a.pdf",
            name="a.pdf",
            document_id=document.id,
            owner_id=OWNER,
            conversation_id=CONVERSATION,
        )

        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.summary == "A quarterly travel budget."
        assert stored.chunks_indexed > 0

    async def test_the_summarizer_reads_the_extracted_text_not_the_raw_file(self) -> None:
        summarizer = FakeDocumentSummarizer()

        await self.build("the extracted words", summarizer=summarizer)(
            reference="uploads/a.pdf",
            name="a.pdf",
            document_id=1,
            owner_id=OWNER,
            conversation_id=CONVERSATION,
        )

        assert summarizer.calls == [("a.pdf", "the extracted words")]

    async def test_a_summarizer_that_declines_still_indexes_the_document(self) -> None:
        # `None` means no summary. The file is still indexed.
        factory = UnitOfWorkSpy()
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        result = await self.build(
            "word " * 200, factory=factory, summarizer=FakeDocumentSummarizer(fail=True)
        )(
            reference="uploads/a.pdf",
            name="a.pdf",
            document_id=document.id,
            owner_id=OWNER,
            conversation_id=CONVERSATION,
        )

        assert result.chunks_indexed > 0
        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.summary == ""
        assert stored.chunks_indexed == result.chunks_indexed

    async def test_a_summarizer_that_raises_still_indexes_the_document(self) -> None:
        # A summarizer that raises must not leave the row claiming zero chunks.
        class ExplodingSummarizer:
            async def summarize_document(self, name: str, text: str) -> str | None:
                raise RuntimeError("provider down")

        factory = UnitOfWorkSpy()
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        result = await self.build("word " * 200, factory=factory, summarizer=ExplodingSummarizer())(
            reference="uploads/a.pdf",
            name="a.pdf",
            document_id=document.id,
            owner_id=OWNER,
            conversation_id=CONVERSATION,
        )

        assert result.chunks_indexed > 0
        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.chunks_indexed == result.chunks_indexed

    async def test_a_scan_with_no_text_is_a_business_failure(self) -> None:
        index = FakeVectorIndex()
        with pytest.raises(DocumentNotIndexable):
            await self.build("   \n ", index)("uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION)
        assert index.chunks == [], "an unindexable document must not write anything"

    async def test_search_never_crosses_owners(self) -> None:
        index = FakeVectorIndex(hits=["a secret passage"])
        await self.build("owner text", index)("uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION)

        assert [c.text for c in await index.search([0.0], 3, 0.5, OWNER, CONVERSATION)] == [
            "a secret passage"
        ]
        assert await index.search([0.0], 3, 0.5, STRANGER, CONVERSATION) == []

    async def test_a_delete_that_lands_while_embedding_wins(self) -> None:
        """A delete during embedding wins. The indexer re-checks the row and stops."""
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        embedder = GatedEmbedder()
        index = FakeVectorIndex()
        # Both use cases hold the *same* lock - that sharing is the mechanism.
        lock = FakeDocumentLock()

        indexing = asyncio.create_task(
            self.build("word " * 200, index, factory, embedder=embedder, lock=lock)(
                "uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION
            )
        )
        # The indexer is paused in its first embed call, before the lock.
        await embedder.entered.wait()

        await DeleteDocument(factory, index=index, storage=FakeFileStorage(), lock=lock)(1, OWNER)
        assert factory.documents.rows == {}, "the delete ran to completion here"

        embedder.release.set()
        result = await indexing

        assert result.chunks_indexed == 0
        assert index.chunks == [], "nothing may be upserted for a removed document"
        assert factory.documents.rows == {}

    async def test_a_delete_that_arrives_while_the_indexer_writes_waits(self) -> None:
        """A delete during the upsert waits, then removes what was written."""
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        index = GatedIndex()
        lock = FakeDocumentLock()
        indexer = self.build("word " * 200, index, factory, lock=lock)

        indexing = asyncio.create_task(indexer("uploads/a.pdf", "a.pdf", 1, OWNER, CONVERSATION))
        # The indexer is inside its upsert, holding the lock.
        await index.entered_upsert.wait()

        deleting = asyncio.create_task(
            DeleteDocument(factory, index=index, storage=FakeFileStorage(), lock=lock)(1, OWNER)
        )
        # One scheduler turn. Only the lock can keep the delete running now.
        await asyncio.sleep(0)
        assert not deleting.done(), "the delete must wait for the indexer to leave the lock"

        index.release_upsert.set()
        indexed = await indexing
        assert indexed.chunks_indexed > 0

        # The delete now holds the lock, after the indexer's writes finished.
        await index.entered_delete.wait()
        assert factory.documents.rows[1].chunks_indexed == indexed.chunks_indexed

        index.release_delete.set()
        await deleting
        assert index.chunks == [], "the delete removes what the indexer wrote"
        assert index.deleted_documents == [(1, OWNER)]
        assert factory.documents.rows == {}


class GatedEmbedder(FakeEmbedder):
    """Stops inside its first embed call until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, text: str) -> list[float]:
        self.entered.set()
        await self.release.wait()
        return await super().embed(text)


class GatedIndex(FakeVectorIndex):
    """Stops inside its upsert and inside its delete until the test lets go."""

    def __init__(self) -> None:
        super().__init__()
        self.entered_upsert = asyncio.Event()
        self.release_upsert = asyncio.Event()
        self.entered_delete = asyncio.Event()
        self.release_delete = asyncio.Event()

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        owner_id: int,
        document_id: int,
        conversation_id: int | None = None,
    ) -> None:
        self.entered_upsert.set()
        await self.release_upsert.wait()
        await super().upsert(chunks, vectors, owner_id, document_id, conversation_id)

    async def delete_document(self, document_id: int, owner_id: int) -> None:
        self.entered_delete.set()
        await self.release_delete.wait()
        await super().delete_document(document_id, owner_id)


class TestListDocuments:
    async def test_lists_only_this_owners_documents(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        factory.documents.rows[2] = existing_document(2, STRANGER)

        result = await ListDocuments(factory)(OWNER)

        assert [d.id for d in result] == [1]


class TestDeleteDocument:
    def build(
        self,
        factory: UnitOfWorkSpy,
        index: FakeVectorIndex | None = None,
        storage: FakeFileStorage | None = None,
        lock: FakeDocumentLock | None = None,
    ) -> DeleteDocument:
        return DeleteDocument(
            factory,
            index=index or FakeVectorIndex(),
            storage=storage or FakeFileStorage(),
            lock=lock or FakeDocumentLock(),
        )

    async def test_removes_the_vectors_the_file_and_the_row(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        index = FakeVectorIndex()
        storage = FakeFileStorage()

        await self.build(factory, index=index, storage=storage)(1, OWNER)

        assert factory.documents.rows == {}
        assert index.deleted_documents == [(1, OWNER)]
        assert storage.deleted == ["uploads/a.pdf"]
        assert factory.issued[0].committed

    async def test_a_url_ingested_document_never_touches_storage(self) -> None:
        """An old URL-ingested document has no stored file to delete."""
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = UploadedDocument(
            name="example.com-abc123.txt",
            reference="https://example.com/article",
            owner_id=OWNER,
            id=1,
        )
        storage = FakeFileStorage()

        await self.build(factory, storage=storage)(1, OWNER)

        assert factory.documents.rows == {}
        assert storage.deleted == []

    async def test_missing_document_is_reported_not_swallowed(self) -> None:
        delete = self.build(UnitOfWorkSpy())
        with pytest.raises(DocumentNotFound):
            await delete(404, OWNER)

    async def test_a_stranger_cannot_delete_it(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        delete = self.build(factory)

        with pytest.raises(DocumentNotFound, match="Document 1 not found"):
            await delete(1, STRANGER)

        assert 1 in factory.documents.rows


class TestGetConversationDocument:
    async def test_returns_attached_document(self) -> None:
        """The composer chip shows the one file the thread holds."""
        factory = UnitOfWorkSpy()
        seed_conversation(factory)
        attached = await factory.documents.add(
            UploadedDocument(
                name="a.pdf",
                reference="uploads/7/a.pdf",
                owner_id=OWNER,
                conversation_id=CONVERSATION,
            )
        )

        result = await GetConversationDocument(factory)(CONVERSATION, OWNER)

        assert result == attached

    async def test_empty_thread_returns_none(self) -> None:
        """An empty attachment is a normal state, not an error."""
        factory = UnitOfWorkSpy()
        seed_conversation(factory)

        assert await GetConversationDocument(factory)(CONVERSATION, OWNER) is None

    async def test_missing_thread_raises_unknown(self) -> None:
        with pytest.raises(UnknownConversation, match="Conversation 404 not found"):
            await GetConversationDocument(UnitOfWorkSpy())(404, OWNER)

    async def test_stranger_cannot_read_attachment(self) -> None:
        """A stranger's thread id must read as missing, never as another's file."""
        factory = UnitOfWorkSpy()
        seed_conversation(factory)
        await factory.documents.add(
            UploadedDocument(
                name="a.pdf",
                reference="uploads/7/a.pdf",
                owner_id=OWNER,
                conversation_id=CONVERSATION,
            )
        )

        with pytest.raises(UnknownConversation, match="Conversation 42 not found"):
            await GetConversationDocument(factory)(CONVERSATION, STRANGER)
