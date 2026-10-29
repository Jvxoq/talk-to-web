"""Turn a stored document into searchable vectors."""

import asyncio

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import IndexDocumentResult
from app.application.ingestion.ports import (
    DocumentLock,
    DocumentSummarizer,
    Embedder,
    TextExtractor,
    VectorIndex,
)
from app.domain.ingestion.entities import Document
from app.domain.ingestion.errors import DocumentNotIndexable
from app.domain.ingestion.value_objects import DocumentName

_EMBED_BATCH_SIZE = 16
"""Chunks embedded at once. Unbounded fan-out on a big PDF hits rate limits."""


class IndexDocument:
    """Make an uploaded document searchable.

    A file with no text fails with `DocumentNotIndexable` instead of indexing nothing.
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
        lock: DocumentLock,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_dimensions = embedding_dimensions
        self._uow_factory = uow_factory
        self._summarizer = summarizer
        self._lock = lock

    async def __call__(
        self,
        reference: str,
        name: str,
        document_id: int,
        owner_id: int,
        conversation_id: int,
    ) -> IndexDocumentResult:
        """Index the stored file at `reference`.

        Returns zero chunks if the document was deleted during indexing.
        """
        logger.debug("Indexing {} from {} for owner {}", name, reference, owner_id)

        content = await self._extractor.extract(reference)
        document = Document(name=DocumentName(name), content=content)
        if not document.is_indexable():
            raise DocumentNotIndexable(name)

        # Points are tagged by document, so nothing else is cleared here.
        await self._index.ensure(self._embedding_dimensions)

        chunks = list(document.chunks(self._chunk_size, self._chunk_overlap))
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(await asyncio.gather(*(self._embedder.embed(c.text) for c in batch)))

        # Lock only the writes. Re-read the row inside the lock: if a delete
        # won, upserting now would leave orphaned points.
        async with self._lock.hold(document_id):
            async with self._uow_factory() as uow:
                if await uow.documents.get(document_id, owner_id) is None:
                    logger.info(
                        "Document {} was removed while {} was being indexed; not indexing it",
                        document_id,
                        name,
                    )
                    return IndexDocumentResult(chunks_indexed=0)

            await self._index.upsert(chunks, vectors, owner_id, document_id, conversation_id)

            summary = await self._digest(name, document.content)

            async with self._uow_factory() as uow:
                await uow.documents.set_indexed(document_id, owner_id, len(chunks), summary)
                await uow.commit()

        logger.debug("Indexed {} chunk(s) from {}", len(chunks), name)
        return IndexDocumentResult(chunks_indexed=len(chunks))

    async def _digest(self, name: str, content: str) -> str:
        """A short summary of the document, or "" on failure. Never raises.

        Stored unfenced. `GenerateReply` fences it when the model reads it.
        """
        # Caught anyway: the vectors are written, so raising would leave the
        # row saying nothing was indexed.

        try:
            summary = await self._summarizer.summarize_document(name, content)
        except Exception as error:  # noqa: BLE001
            logger.warning("Summarizer raised for {}; indexing it without one: {}", name, error)
            return ""
        if summary is None:
            logger.info("No summary produced for {}; indexing it without one", name)
            return ""
        return summary
