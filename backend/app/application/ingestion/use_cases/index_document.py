"""Turn a stored document into searchable vectors."""

import asyncio

from loguru import logger

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
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_dimensions = embedding_dimensions

    async def __call__(self, reference: str, name: str) -> IndexDocumentResult:
        logger.debug("Indexing {} from {}", name, reference)

        text = await self._extractor.extract(reference)
        document = Document(name=DocumentName(name), content=text)
        if not document.is_indexable():
            raise DocumentNotIndexable(name)

        # Preserves existing behaviour: every upload replaces the whole
        # collection, so only the most recent document is retrievable. Known
        # limitation, kept deliberately — not an accident.
        await self._index.reset(self._embedding_dimensions)

        chunks = list(document.chunks(self._chunk_size, self._chunk_overlap))
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(await asyncio.gather(*(self._embedder.embed(c.text) for c in batch)))

        await self._index.upsert(chunks, vectors)

        logger.debug("Indexed {} chunk(s) from {}", len(chunks), name)
        return IndexDocumentResult(chunks_indexed=len(chunks))
