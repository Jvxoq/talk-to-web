"""Inputs and outputs owned by the ingestion use cases."""

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UploadDocumentInput:
    filename: str | None
    content_type: str | None
    stream: AsyncIterator[bytes]


@dataclass(frozen=True, slots=True)
class UploadDocumentResult:
    reference: str
    name: str


@dataclass(frozen=True, slots=True)
class IndexDocumentResult:
    chunks_indexed: int
