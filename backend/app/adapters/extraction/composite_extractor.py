"""Dispatches extraction to the extractor registered for a file's suffix."""

from pathlib import Path

from app.application.ingestion.ports import TextExtractor
from app.domain.ingestion.errors import UnsupportedDocumentType


class CompositeTextExtractor:
    """
    Routes a stored file to the extractor that understands its suffix.

    Satisfies `app.application.ingestion.ports.TextExtractor` itself, so
    `IndexDocument` needs no changes to gain support for a new file type -
    only a new row in the map this is constructed with.
    """

    def __init__(self, extractors: dict[str, TextExtractor]) -> None:
        self._extractors = extractors

    async def extract(self, reference: str) -> str:
        suffix = Path(reference).suffix.lower()
        extractor = self._extractors.get(suffix)
        if extractor is None:
            raise UnsupportedDocumentType(suffix or reference)
        return await extractor.extract(reference)
