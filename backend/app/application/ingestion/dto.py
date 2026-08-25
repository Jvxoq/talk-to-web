"""Inputs and outputs owned by the ingestion use cases."""

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UploadDocumentInput:
    filename: str | None
    content_type: str | None
    stream: AsyncIterator[bytes]
    owner_id: int
    # The thread the file is attached to. Required, not optional: a document
    # with no conversation is one no retrieval can ever reach, so accepting the
    # upload would be accepting a file the user can never ask about.
    conversation_id: int


@dataclass(frozen=True, slots=True)
class UploadDocumentResult:
    reference: str
    name: str
    document_id: int


@dataclass(frozen=True, slots=True)
class IndexDocumentResult:
    chunks_indexed: int
