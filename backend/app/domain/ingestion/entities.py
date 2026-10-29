"""The ingestion entity: a document, and the rules for turning it into chunks."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from app.domain.ingestion.value_objects import Chunk, DocumentName

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Document:
    """Extracted text from one upload, ready to be sliced for embedding."""

    name: DocumentName
    content: str

    def is_indexable(self) -> bool:
        return bool(self.content.strip())

    @staticmethod
    def clean(text: str) -> str:
        """Collapse whitespace and PDF debris so chunks hold real text."""
        collapsed = _WHITESPACE.sub(" ", text.replace("\n", " "))
        for noisy, fixed in ((". ,", ""), ("..", "."), (". .", ".")):
            collapsed = collapsed.replace(noisy, fixed)
        return collapsed.strip()

    def chunks(self, size: int, overlap: int) -> Iterator[Chunk]:
        """Slice the cleaned text into overlapping windows.

        The overlap keeps a sentence on a boundary whole in one window.
        """
        if size <= 0:
            raise ValueError("Chunk size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("Overlap must be non-negative and smaller than the chunk size")

        text = self.clean(self.content)
        if not text:
            return

        stride = size - overlap
        for start in range(0, len(text), stride):
            window = text[start : start + size]
            if window.strip():
                yield Chunk(text=window, source=self.name.value)
            if start + size >= len(text):
                return


@dataclass(slots=True)
class UploadedDocument:
    """The stored record of one upload. `Document` is the text, never stored."""

    name: str
    reference: str
    owner_id: int
    # The owning thread. `None` only on old rows, which retrieval never matches.
    conversation_id: int | None = None
    id: int | None = None
    chunks_indexed: int = 0
    # Written at indexing, so the agent knows what the file covers. Empty until
    # then, or if the condenser failed. The file is still searchable either way.

    summary: str = ""
    created_at: datetime | None = None
