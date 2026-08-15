"""Inputs and outputs owned by the ingestion use cases."""

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UploadDocumentInput:
    filename: str | None
    content_type: str | None
    stream: AsyncIterator[bytes]
    owner_id: int


@dataclass(frozen=True, slots=True)
class IngestUrlInput:
    url: str
    owner_id: int


@dataclass(frozen=True, slots=True)
class UploadDocumentResult:
    reference: str
    name: str
    document_id: int
    # Set only by `IngestUrl`: the page's text is already in hand, so indexing
    # can use it directly instead of reading it back from wherever `reference`
    # points. Always `None` for an uploaded file, which has no text yet.
    text: str | None = None


@dataclass(frozen=True, slots=True)
class IndexDocumentResult:
    chunks_indexed: int
