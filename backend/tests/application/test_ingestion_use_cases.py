"""Ingestion use cases against fakes."""

from collections.abc import AsyncIterator

import pytest

from app.application.ingestion.dto import UploadDocumentInput
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.domain.ingestion.errors import DocumentNotIndexable, UnsupportedDocumentType
from tests.fakes import FakeEmbedder, FakeFileStorage, FakeTextExtractor, FakeVectorIndex


async def bytes_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


class TestUploadDocument:
    async def test_stores_a_pdf_under_a_safe_name(self) -> None:
        storage = FakeFileStorage()
        result = await UploadDocument(storage, max_bytes=1024)(
            UploadDocumentInput(
                filename="../../etc/report.pdf",
                content_type="application/pdf",
                stream=bytes_stream(b"%PDF", b"-data"),
            )
        )

        assert storage.saved == [("report.pdf", b"%PDF-data")]
        assert result.name == "report.pdf"
        assert result.reference == "uploads/report.pdf"

    async def test_accepts_a_content_type_with_parameters(self) -> None:
        # Browsers append charset and friends; strict equality would reject a
        # perfectly good upload.
        result = await UploadDocument(FakeFileStorage(), max_bytes=1024)(
            UploadDocumentInput("a.pdf", "application/pdf; charset=binary", bytes_stream(b"x"))
        )
        assert result.name == "a.pdf"

    @pytest.mark.parametrize("content_type", ["text/html", "image/png", None, ""])
    async def test_rejects_everything_that_is_not_a_pdf(self, content_type: str | None) -> None:
        storage = FakeFileStorage()
        with pytest.raises(UnsupportedDocumentType):
            await UploadDocument(storage, max_bytes=1024)(
                UploadDocumentInput("a.exe", content_type, bytes_stream(b"x"))
            )
        assert storage.saved == [], "nothing may be written before the type check passes"


class TestIndexDocument:
    def build(self, text: str, index: FakeVectorIndex | None = None) -> IndexDocument:
        return IndexDocument(
            extractor=FakeTextExtractor(text),
            embedder=FakeEmbedder(),
            index=index or FakeVectorIndex(),
            chunk_size=100,
            chunk_overlap=20,
            embedding_dimensions=3,
        )

    async def test_indexes_every_chunk_exactly_once(self) -> None:
        index = FakeVectorIndex()
        result = await self.build("word " * 200, index)(reference="uploads/a.pdf", name="a.pdf")

        assert result.chunks_indexed == len(index.chunks)
        assert result.chunks_indexed > 1
        assert all(chunk.source == "a.pdf" for chunk in index.chunks)

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
        )

        result = await use_case("uploads/a.pdf", "a.pdf")

        assert len(embedder.embedded) == result.chunks_indexed > 20

    async def test_resets_the_collection_before_writing(self) -> None:
        index = FakeVectorIndex()
        await self.build("hello world", index)("uploads/a.pdf", "a.pdf")
        assert index.resets == [3]

    async def test_a_scan_with_no_text_is_a_business_failure(self) -> None:
        index = FakeVectorIndex()
        with pytest.raises(DocumentNotIndexable):
            await self.build("   \n ", index)("uploads/a.pdf", "a.pdf")
        assert index.resets == [], "an unindexable document must not wipe the collection"
