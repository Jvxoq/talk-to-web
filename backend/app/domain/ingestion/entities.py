"""The ingestion entity: a document, and the rules for turning it into chunks."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

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
        """
        Normalise extracted text so chunk boundaries fall on real words.

        PDF extraction leaves hard-wrapped lines and the ". ," debris of
        column-split sentences; collapsing them first keeps a chunk from
        spending its budget on whitespace.
        """
        collapsed = _WHITESPACE.sub(" ", text.replace("\n", " "))
        for noisy, fixed in ((". ,", ""), ("..", "."), (". .", ".")):
            collapsed = collapsed.replace(noisy, fixed)
        return collapsed.strip()

    def chunks(self, size: int, overlap: int) -> Iterator[Chunk]:
        """
        Slice the cleaned content into overlapping windows.

        The overlap is what stops a sentence that straddles a boundary from
        being unretrievable: it appears whole in one of the two windows.
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
