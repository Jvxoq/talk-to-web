"""Turn a stored document into searchable vectors."""

import asyncio

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import IndexDocumentResult
from app.application.ingestion.ports import (
    DocumentSummarizer,
    Embedder,
    TextExtractor,
    VectorIndex,
)
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
        summarizer: DocumentSummarizer,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_dimensions = embedding_dimensions
        self._uow_factory = uow_factory
        self._summarizer = summarizer

    async def __call__(
        self,
        reference: str,
        name: str,
        document_id: int,
        owner_id: int,
        conversation_id: int,
    ) -> IndexDocumentResult:
        """`reference` is a `TextExtractor`-readable location of the stored file."""
        logger.debug("Indexing {} from {} for owner {}", name, reference, owner_id)

        content = await self._extractor.extract(reference)
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

        # The conversation is written onto every point, because it is what
        # every search filters on. A point without it is retrievable by nobody.
        await self._index.upsert(chunks, vectors, owner_id, document_id, conversation_id)

        summary = await self._digest(name, document.content)

        async with self._uow_factory() as uow:
            await uow.documents.set_indexed(document_id, owner_id, len(chunks), summary)
            await uow.commit()

        logger.debug("Indexed {} chunk(s) from {}", len(chunks), name)
        return IndexDocumentResult(chunks_indexed=len(chunks))

    async def _digest(self, name: str, content: str) -> str:
        """A few sentences on what this document is, or "" if none could be had.

        Never raises and never blocks the upload. The document is already
        embedded and searchable by the time this runs; the digest only decides
        how readily the agent *reaches* for it, so failing to write one costs
        discoverability, not the file.

        Stored exactly as the summarizer wrote it. It is derived from an
        uploaded file, so a hostile document shapes part of it - but the fence
        that makes that safe belongs where the model reads it, not here.
        `GenerateReply` puts the whole digest block through the same
        `ToolOutputGuard` every tool result goes through, which keeps fencing
        in one place rather than storing pre-fenced text in a column.
        """
        # Caught, even though the port says a summarizer returns `None` rather
        # than raising - the same belt-and-braces `ToolRegistry.invoke` applies
        # to tools that do not extend `BaseTool`. By this point the document is
        # embedded and upserted, so an exception escaping here would abandon the
        # run before the chunk count was ever written: the vectors would exist
        # while the row still claimed nothing had been indexed.
        try:
            summary = await self._summarizer.summarize_document(name, content)
        except Exception as error:
            logger.warning("Summarizer raised for {}; indexing it without one: {}", name, error)
            return ""
        if summary is None:
            logger.info("No summary produced for {}; indexing it without one", name)
            return ""
        return summary
