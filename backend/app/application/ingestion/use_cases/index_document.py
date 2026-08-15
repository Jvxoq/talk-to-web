"""Turn a stored document into searchable vectors."""

import asyncio

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import IndexDocumentResult
from app.application.ingestion.ports import Embedder, TextExtractor, VectorIndex
from app.domain.ingestion.entities import Document
from app.domain.ingestion.errors import DocumentNotIndexable
from app.domain.ingestion.value_objects import DocumentName

_EMBED_BATCH_SIZE = 16
"""How many chunks may be embedded at once.

Unbounded `gather` over a large PDF would open hundreds of simultaneous
provider calls and earn a rate limit; batching keeps the fan-out predictable
while still overlapping the latency.
"""


class IndexDocument:
    """
    Make an uploaded document answerable.

    Extraction, chunking and embedding are separate steps so an empty scan
    fails as a business outcome ("no text in here") instead of silently
    indexing nothing.
    """

    def __init__(
        self,
        extractor: TextExtractor,
        embedder: Embedder,
        index: VectorIndex,
        chunk_size: int,
        chunk_overlap: int,
        embedding_dimensions: int,
        uow_factory: UnitOfWorkFactory,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_dimensions = embedding_dimensions
        self._uow_factory = uow_factory

    async def __call__(
        self, reference: str, name: str, document_id: int, owner_id: int, text: str | None = None
    ) -> IndexDocumentResult:
        """
        `text`, when given, is already-fetched content (currently only
        `IngestUrl`'s page text) and is indexed as-is - `reference` is then
        just the record's identity, never read from. Left `None`, `reference`
        is a `TextExtractor`-readable location, as it is for an uploaded file.
        """
        logger.debug("Indexing {} from {} for owner {}", name, reference, owner_id)

        content = text if text is not None else await self._extractor.extract(reference)
        document = Document(name=DocumentName(name), content=content)
        if not document.is_indexable():
            raise DocumentNotIndexable(name)

        # Tagged with `document_id` rather than clearing the owner's whole
        # namespace first - that used to be how a second upload wiped out the
        # first. Every owner may now hold several documents at once; only
        # `DeleteDocument` removes one, and only its own passages.
        await self._index.ensure(self._embedding_dimensions)

        chunks = list(document.chunks(self._chunk_size, self._chunk_overlap))
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(await asyncio.gather(*(self._embedder.embed(c.text) for c in batch)))

        await self._index.upsert(chunks, vectors, owner_id, document_id)

        async with self._uow_factory() as uow:
            await uow.documents.set_chunks_indexed(document_id, owner_id, len(chunks))
            await uow.commit()

        logger.debug("Indexed {} chunk(s) from {}", len(chunks), name)
        return IndexDocumentResult(chunks_indexed=len(chunks))
