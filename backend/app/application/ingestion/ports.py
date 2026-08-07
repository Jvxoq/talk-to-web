"""What the ingestion use cases need from the outside world."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.domain.ingestion.value_objects import Chunk, DocumentName


class FileStorage(Protocol):
    """Somewhere to put an upload. The reference it returns is opaque."""

    async def save(
        self,
        name: DocumentName,
        stream: AsyncIterator[bytes],
        max_bytes: int,
    ) -> str: ...


class TextExtractor(Protocol):
    """Pulls plain text out of a stored file."""

    async def extract(self, reference: str) -> str: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    """A collection of embedded passages."""

    async def reset(self, dimensions: int) -> None: ...

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None: ...

    async def search(
        self, vector: list[float], limit: int, score_threshold: float
    ) -> list[str]: ...
