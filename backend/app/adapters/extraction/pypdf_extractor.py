"""PDF text extraction with pypdf."""

import asyncio

from loguru import logger
from pypdf import PdfReader


class PypdfTextExtractor:
    """
    Reads the text layer out of a PDF.

    Satisfies `app.application.ingestion.ports.TextExtractor`. Unlike the legacy
    extractor it writes no sibling `.txt` file - nothing ever read that file, and
    writing it doubled the disk footprint of every upload.
    """

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored PDF."""
        # pypdf is entirely blocking, so it runs off the event loop; a 200-page
        # scan would otherwise stall every other request in the process.
        return await asyncio.to_thread(self._read, reference)

    @staticmethod
    def _read(filepath: str) -> str:
        reader = PdfReader(filepath, strict=True)
        pages = [text for page in reader.pages if (text := page.extract_text())]

        if not pages:
            # A scanned PDF with no text layer is a normal outcome, not a crash.
            # Whether an empty document is an error is the use case's decision.
            logger.warning(f"No extractable text in {filepath}")
            return ""

        return "\n\n".join(pages)
