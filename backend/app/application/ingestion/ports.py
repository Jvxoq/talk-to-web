"""What the ingestion use cases need from the outside world."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.value_objects import Chunk, DocumentName


class FileStorage(Protocol):
    """Somewhere to put an upload. The reference it returns is opaque.

    `owner_id` is not metadata: it is part of where the file goes. Two people
    uploading `report.pdf` into one namespace overwrite each other, and the
    second one silently gets the first one's document back.
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
    """A collection of embedded passages, partitioned by who uploaded them and
    tagged with the document each passage came from.

    Every method that reads or writes passages names an owner. The alternative -
    one shared space, filtered afterwards - is what indexing used to do: a
    single upload wiped the store for everyone, and every search read
    everyone's documents. `document_id` narrows that same discipline one level
    further, to the document: without it, deleting one upload could only ever
    mean deleting every upload an owner had made.
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
    """The persisted record of every document an owner has uploaded.

    `owner_id` shapes every method here the same way it shapes
    `ConversationRepository`: a document is only ever read, listed or deleted
    through its owner, never by id alone.
    """

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


class DocumentRemover(Protocol):
    """Removes one document completely: its vectors, its file and its row.

    A port over a use case rather than over an adapter, which is unusual here
    and deliberate. Two callers need "get rid of this document, entirely", and
    the three steps that make it entire already live together in
    `DeleteDocument`. Re-deriving them at each call site is how one of them
    ends up deleting the row and leaving the vectors behind.
    """

    async def __call__(self, document_id: int, owner_id: int) -> None: ...


class DocumentSummarizer(Protocol):
    """Writes the few sentences that say what a document is about.

    Declared here rather than imported from the chat context, for the same
    reason `RateLimiter` is declared twice: a port belongs to the layer that
    consumes it, and ingestion needing a summary is not ingestion depending on
    the agent. `Condenser` happens to satisfy this structurally, without
    knowing that ingestion exists.

    Returning `None` rather than raising is part of the contract. A digest is an
    enhancement - a document that could not be summarized is still fully
    indexed and fully searchable - so a summarizer having a bad day must cost
    the upload nothing.
    """

    async def summarize_document(self, name: str, text: str) -> str | None: ...


class RateLimiter(Protocol):
    """Counts uploads against a key and refuses the ones over budget.

    Declared here for the same reason as the chat one: the consumer owns its
    port. A single adapter satisfies every copy, because `Protocol` matches on
    methods rather than on ancestry.
    """

    async def hit(self, key: str) -> None:
        """Record one upload, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None: ...
