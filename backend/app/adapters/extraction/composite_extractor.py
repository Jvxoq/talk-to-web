"""Dispatches extraction to the extractor registered for a file's suffix."""

from pathlib import Path

from app.application.ingestion.ports import TextExtractor
from app.domain.ingestion.errors import UnsupportedDocumentType


class CompositeTextExtractor:
    """Routes a file to the extractor for its suffix. Is a `TextExtractor` itself.

    A new file type is one more map entry.
    """

    def __init__(self, extractors: dict[str, TextExtractor]) -> None:
        self._extractors = extractors

    async def extract(self, reference: str) -> str:
        suffix = Path(reference).suffix.lower()
        extractor = self._extractors.get(suffix)
        if extractor is None:
            raise UnsupportedDocumentType(suffix or reference)
        return await extractor.extract(reference)
