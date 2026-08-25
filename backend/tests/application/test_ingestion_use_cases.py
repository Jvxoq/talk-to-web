"""Ingestion use cases against fakes."""

from collections.abc import AsyncIterator

import pytest

from app.application.ingestion.dto import UploadDocumentInput
from app.application.ingestion.ports import DocumentSummarizer
from app.application.ingestion.use_cases.delete_document import DeleteDocument
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.list_documents import ListDocuments
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import (
    DocumentNotFound,
    DocumentNotIndexable,
    UnsupportedDocumentType,
)
from app.domain.usage.errors import RateLimited
from tests.fakes import (
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


async def bytes_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def upload_document(
    storage: FakeFileStorage,
    max_bytes: int,
    limiter: FakeRateLimiter,
    factory: UnitOfWorkSpy | None = None,
    daily_budget: FakeRateLimiter | None = None,
) -> UploadDocument:
    return UploadDocument(
        storage,
        max_bytes=max_bytes,
        limiter=limiter,
        uow_factory=factory or UnitOfWorkSpy(),
        daily_budget=daily_budget or FakeRateLimiter(),
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
            UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER)
        )

        stored = factory.documents.rows[result.document_id]
        assert stored.name == "a.pdf"
        assert stored.owner_id == OWNER
        assert stored.chunks_indexed == 0

    async def test_accepts_a_content_type_with_parameters(self) -> None:
        # Browsers append charset and friends; strict equality would reject a
        # perfectly good upload.
        result = await upload_document(FakeFileStorage(), 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "a.pdf", "application/pdf; charset=binary", bytes_stream(b"%PDF-1.7"), OWNER
            )
        )
        assert result.name == "a.pdf"

    @pytest.mark.parametrize("content_type", ["text/html", "image/png", None, ""])
    async def test_rejects_everything_that_is_not_a_pdf(self, content_type: str | None) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput("a.exe", content_type, bytes_stream(b"%PDF-1.7"), OWNER)
            )
        assert storage.saved == [], "nothing may be written before the type check passes"

    async def test_rejects_a_file_that_only_claims_to_be_a_pdf(self) -> None:
        # The declared type and the name are both attacker-chosen. Only the
        # first bytes say what the file really is.
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "payload.pdf", "application/pdf", bytes_stream(b"MZ\x90\x00"), OWNER
                )
            )
        assert storage.saved == []

    @pytest.mark.parametrize("chunks", [(), (b"",), (b"%PD",)])
    async def test_rejects_a_file_too_short_to_prove_anything(self, chunks: tuple[bytes]) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(*chunks), OWNER)
            )
        assert storage.saved == []

    async def test_rejects_a_name_without_the_pdf_extension(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "report.txt", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER
                )
            )
        assert storage.saved == []

    async def test_the_signature_check_does_not_eat_the_bytes_it_read(self) -> None:
        # The opening bytes are buffered across chunk boundaries to be judged;
        # what lands in storage must still be the whole original file.
        storage = FakeFileStorage()
        await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "a.pdf", "application/pdf", bytes_stream(b"%P", b"DF", b"-1.7", b"body"), OWNER
            )
        )
        assert storage.saved == [("a.pdf", b"%PDF-1.7body")]

    async def test_an_upload_past_the_budget_is_refused_before_anything_is_stored(self) -> None:
        # Indexing is what costs money - extraction, chunking, one embedding call
        # per chunk - so the limit has to bite before the bytes are accepted, not
        # after they are on disk.
        storage = FakeFileStorage()
        limiter = FakeRateLimiter(max_attempts=1)
        upload = upload_document(storage, 1024, limiter)

        await upload(
            UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER)
        )
        with pytest.raises(RateLimited):
            await upload(
                UploadDocumentInput("b.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER)
            )

        assert len(storage.saved) == 1

    async def test_accepts_a_txt_file(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput("notes.txt", "text/plain", bytes_stream(b"just some text"), OWNER)
        )
        assert result.name == "notes.txt"
        assert storage.saved == [("notes.txt", b"just some text")]

    async def test_accepts_a_markdown_file(self) -> None:
        storage = FakeFileStorage()
        result = await upload_document(storage, 1024, FakeRateLimiter())(
            UploadDocumentInput(
                "notes.md", "text/markdown", bytes_stream(b"# heading\n\nbody"), OWNER
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
            )
        )
        assert result.name == "report.docx"
        assert storage.saved == [("report.docx", b"PK\x03\x04therest")]

    async def test_rejects_a_txt_file_with_the_wrong_extension(self) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await upload_document(storage, 1024, FakeRateLimiter())(
                UploadDocumentInput(
                    "notes.pdf", "text/plain", bytes_stream(b"just some text"), OWNER
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
                )
            )
        assert storage.saved == []

    async def test_one_user_hitting_the_limit_does_not_block_another(self) -> None:
        limiter = FakeRateLimiter(max_attempts=1)
        upload = upload_document(FakeFileStorage(), 1024, limiter)

        await upload(
            UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER)
        )
        # Keyed per account, so the stranger still has their own budget.
        await upload(
            UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), STRANGER)
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
                UploadDocumentInput("a.pdf", "application/pdf", bytes_stream(b"%PDF-1.7"), OWNER)
            )

        assert storage.saved == []
        assert limiter.hits == {}, "the per-user limit must not be spent on a refused request"


class TestIndexDocument:
    def build(
        self,
        text: str,
        index: FakeVectorIndex | None = None,
        factory: UnitOfWorkSpy | None = None,
        summarizer: DocumentSummarizer | None = None,
    ) -> IndexDocument:
        return IndexDocument(
            extractor=FakeTextExtractor(text),
            embedder=FakeEmbedder(),
            index=index or FakeVectorIndex(),
            chunk_size=100,
            chunk_overlap=20,
            embedding_dimensions=3,
            uow_factory=factory or UnitOfWorkSpy(),
            summarizer=summarizer or FakeDocumentSummarizer(),
        )

    async def test_indexes_every_chunk_exactly_once(self) -> None:
        index = FakeVectorIndex()
        result = await self.build("word " * 200, index)(
            reference="uploads/a.pdf", name="a.pdf", document_id=1, owner_id=OWNER
        )

        assert result.chunks_indexed == len(index.chunks)
        assert result.chunks_indexed > 1
        assert all(chunk.source == "a.pdf" for chunk in index.chunks)

    async def test_records_the_chunk_count_against_the_document(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)

        result = await self.build("word " * 200, factory=factory)(
            "uploads/a.pdf", "a.pdf", 1, OWNER
        )

        assert factory.documents.rows[1].chunks_indexed == result.chunks_indexed

    async def test_batches_larger_than_the_embed_window(self) -> None:
        # 20+ chunks forces more than one embedding batch; every chunk must
        # still get exactly one vector.
        index = FakeVectorIndex()
        embedder = FakeEmbedder()
        use_case = IndexDocument(
            extractor=FakeTextExtractor("x" * 2_000),
            embedder=embedder,
            index=index,
            chunk_size=50,
            chunk_overlap=10,
            embedding_dimensions=3,
            uow_factory=UnitOfWorkSpy(),
            summarizer=FakeDocumentSummarizer(),
        )

        result = await use_case("uploads/a.pdf", "a.pdf", 1, OWNER)

        assert len(embedder.embedded) == result.chunks_indexed > 20

    async def test_ensures_the_collection_without_clearing_it(self) -> None:
        index = FakeVectorIndex()
        await self.build("hello world", index)("uploads/a.pdf", "a.pdf", 1, OWNER)

        assert index.ensured == [3], "the collection is created, never recreated"

    async def test_a_second_upload_does_not_wipe_the_first(self) -> None:
        """The regression 2.3 exists to fix.

        Indexing used to clear the owner's whole namespace before every
        upload, so a second document destroyed the first one's passages. Now
        each document is tagged with its own id, and only `DeleteDocument`
        removes one - never another upload.
        """
        index = FakeVectorIndex()
        await self.build("first document text", index)("uploads/a.pdf", "a.pdf", 1, OWNER)
        first_count = len(index.chunks)
        assert first_count > 0

        await self.build("second document text", index)("uploads/b.pdf", "b.pdf", 2, OWNER)

        assert len(index.chunks) > first_count, "the first document's passages must survive"
        assert {chunk.source for chunk in index.chunks} == {"a.pdf", "b.pdf"}

    async def test_one_upload_does_not_wipe_another_account(self) -> None:
        index = FakeVectorIndex()
        await self.build("owner text", index)("uploads/a.pdf", "a.pdf", 1, OWNER)
        assert index.by_owner[OWNER]

        await self.build("stranger text", index)("uploads/b.pdf", "b.pdf", 2, STRANGER)

        assert index.by_owner[OWNER], "the first owner's passages must survive"
        assert index.by_owner[STRANGER]

    async def test_it_stores_the_digest_alongside_the_chunk_count(self) -> None:
        factory = UnitOfWorkSpy()
        summarizer = FakeDocumentSummarizer("A quarterly travel budget.")
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        await self.build("word " * 200, factory=factory, summarizer=summarizer)(
            reference="uploads/a.pdf", name="a.pdf", document_id=document.id, owner_id=OWNER
        )

        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.summary == "A quarterly travel budget."
        assert stored.chunks_indexed > 0

    async def test_the_summarizer_reads_the_extracted_text_not_the_raw_file(self) -> None:
        summarizer = FakeDocumentSummarizer()

        await self.build("the extracted words", summarizer=summarizer)(
            reference="uploads/a.pdf", name="a.pdf", document_id=1, owner_id=OWNER
        )

        assert summarizer.calls == [("a.pdf", "the extracted words")]

    async def test_a_summarizer_that_declines_still_indexes_the_document(self) -> None:
        # `None` is the port's "could not summarize this". The file is already
        # embedded by then, so it must stay fully searchable with an empty
        # digest rather than losing the upload over an enhancement.
        factory = UnitOfWorkSpy()
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        result = await self.build(
            "word " * 200, factory=factory, summarizer=FakeDocumentSummarizer(fail=True)
        )(reference="uploads/a.pdf", name="a.pdf", document_id=document.id, owner_id=OWNER)

        assert result.chunks_indexed > 0
        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.summary == ""
        assert stored.chunks_indexed == result.chunks_indexed

    async def test_a_summarizer_that_raises_still_indexes_the_document(self) -> None:
        # Belt and braces against a summarizer that breaks the port's contract.
        # The vectors are already upserted at this point, so an exception
        # escaping would leave them stranded behind a row still claiming zero
        # chunks - the one inconsistency this use case must not produce.
        class ExplodingSummarizer:
            async def summarize_document(self, name: str, text: str) -> str | None:
                raise RuntimeError("provider down")

        factory = UnitOfWorkSpy()
        document = await factory.documents.add(existing_document(1, OWNER))
        assert document.id is not None

        result = await self.build("word " * 200, factory=factory, summarizer=ExplodingSummarizer())(
            reference="uploads/a.pdf", name="a.pdf", document_id=document.id, owner_id=OWNER
        )

        assert result.chunks_indexed > 0
        stored = await factory.documents.get(document.id, OWNER)
        assert stored is not None
        assert stored.chunks_indexed == result.chunks_indexed

    async def test_a_scan_with_no_text_is_a_business_failure(self) -> None:
        index = FakeVectorIndex()
        with pytest.raises(DocumentNotIndexable):
            await self.build("   \n ", index)("uploads/a.pdf", "a.pdf", 1, OWNER)
        assert index.chunks == [], "an unindexable document must not write anything"

    async def test_search_never_crosses_owners(self) -> None:
        index = FakeVectorIndex(hits=["a secret passage"])
        await self.build("owner text", index)("uploads/a.pdf", "a.pdf", 1, OWNER)

        assert [c.text for c in await index.search([0.0], 3, 0.5, OWNER)] == ["a secret passage"]
        assert await index.search([0.0], 3, 0.5, STRANGER) == []


class TestListDocuments:
    async def test_lists_only_this_owners_documents(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        factory.documents.rows[2] = existing_document(2, STRANGER)

        result = await ListDocuments(factory)(OWNER)

        assert [d.id for d in result] == [1]


class TestDeleteDocument:
    async def test_removes_the_vectors_the_file_and_the_row(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        index = FakeVectorIndex()
        storage = FakeFileStorage()

        await DeleteDocument(factory, index=index, storage=storage)(1, OWNER)

        assert factory.documents.rows == {}
        assert index.deleted_documents == [(1, OWNER)]
        assert storage.deleted == ["uploads/a.pdf"]
        assert factory.issued[0].committed

    async def test_a_url_ingested_document_never_touches_storage(self) -> None:
        """A document from the removed URL path never had a file - its `reference`
        is the source URL - so deleting it must not ask `FileStorage` to remove
        anything."""
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = UploadedDocument(
            name="example.com-abc123.txt",
            reference="https://example.com/article",
            owner_id=OWNER,
            id=1,
        )
        storage = FakeFileStorage()

        await DeleteDocument(factory, index=FakeVectorIndex(), storage=storage)(1, OWNER)

        assert factory.documents.rows == {}
        assert storage.deleted == []

    async def test_missing_document_is_reported_not_swallowed(self) -> None:
        delete = DeleteDocument(UnitOfWorkSpy(), index=FakeVectorIndex(), storage=FakeFileStorage())
        with pytest.raises(DocumentNotFound):
            await delete(404, OWNER)

    async def test_a_stranger_cannot_delete_it(self) -> None:
        factory = UnitOfWorkSpy()
        factory.documents.rows[1] = existing_document(1, OWNER)
        delete = DeleteDocument(factory, index=FakeVectorIndex(), storage=FakeFileStorage())

        with pytest.raises(DocumentNotFound):
            await delete(1, STRANGER)

        assert 1 in factory.documents.rows
