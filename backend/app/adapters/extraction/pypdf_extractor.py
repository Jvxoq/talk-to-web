"""PDF text extraction with pypdf."""

import asyncio

from loguru import logger
from pypdf import PdfReader


class PypdfTextExtractor:
    """Reads the text layer of a PDF. Implements `TextExtractor`."""

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored PDF."""
        # pypdf blocks, so it runs in a thread.
        return await asyncio.to_thread(self._read, reference)

    @staticmethod
    def _read(filepath: str) -> str:
        reader = PdfReader(filepath, strict=True)
        pages = [text for page in reader.pages if (text := page.extract_text())]

        if not pages:
            # A scanned PDF has no text. The use case decides if that is an error.

            logger.warning(f"No extractable text in {filepath}")
            return ""

        return "\n\n".join(pages)
