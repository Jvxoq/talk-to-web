"""What the ingestion use cases need from the outside world."""

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.value_objects import Chunk, DocumentName


class FileStorage(Protocol):
    """Stores uploads and returns an opaque reference.

    `owner_id` is part of the path, so two users' `report.pdf` never collide.
    """

    async def save(
        self,
        name: DocumentName,
        stream: AsyncIterator[bytes],
        max_bytes: int,
        owner_id: int,
    ) -> str: ...

    async def delete(self, reference: str) -> None:
        """Remove a stored file. Never raises for a reference already gone."""
        ...


class TextExtractor(Protocol):
    """Pulls plain text out of a stored file."""

    async def extract(self, reference: str) -> str: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    """Embedded passages, split by owner and tagged by document.

    Every read and write names an owner. `document_id` lets one upload be
    deleted without touching the owner's others.
    """

    async def ensure(self, dimensions: int) -> None:
        """Create the collection if it is missing. Never destructive."""
        ...

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        owner_id: int,
        document_id: int,
        conversation_id: int | None,
    ) -> None: ...

    async def search(
        self,
        vector: list[float],
        limit: int,
        score_threshold: float,
        owner_id: int,
        conversation_id: int | None,
    ) -> list[Chunk]: ...

    async def delete_document(self, document_id: int, owner_id: int) -> None:
        """Drop one document's passages, leaving the owner's other documents alone."""
        ...


class DocumentRepository(Protocol):
    """Stored document records, always accessed through their owner."""

    async def get(self, document_id: int, owner_id: int) -> UploadedDocument | None: ...

    async def list_by_owner(self, owner_id: int) -> list[UploadedDocument]: ...

    async def list_by_conversation(
        self, owner_id: int, conversation_id: int
    ) -> list[UploadedDocument]:
        """This owner's documents attached to one conversation, newest first."""
        ...

    async def add(self, document: UploadedDocument) -> UploadedDocument: ...

    async def set_indexed(self, document_id: int, owner_id: int, count: int, summary: str) -> None:
        """Record what indexing produced: the chunk count and the digest."""
        ...

    async def delete(self, document_id: int, owner_id: int) -> None: ...


class DocumentLock(Protocol):
    """Serializes indexing against deletion for one document.

    Without it, a delete during indexing succeeds and the indexer then upserts
    orphaned vectors that no later delete can find. Both use cases hold the
    lock around their writes. The indexer re-checks the row inside the lock
    and stops if it is gone.
    """

    def hold(self, document_id: int) -> AbstractAsyncContextManager[None]:
        """Hold the lock for one document while the `async with` block runs."""
        ...


class DocumentRemover(Protocol):
    """Removes one document completely: vectors, file and row.

    A port over the `DeleteDocument` use case, so no caller forgets a step.
    """

    async def __call__(self, document_id: int, owner_id: int) -> None: ...


class DocumentSummarizer(Protocol):
    """Writes a few sentences about what a document covers.

    `Condenser` matches it by shape. Returns `None` on failure instead of
    raising, because the document is still searchable without a summary.
    """

    async def summarize_document(self, name: str, text: str) -> str | None: ...


class RateLimiter(Protocol):
    """Counts uploads per key and refuses those over budget. Same shape as the chat port."""

    async def hit(self, key: str) -> None:
        """Record one upload, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None: ...
