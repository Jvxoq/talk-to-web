"""DOCX text extraction with python-docx."""

import asyncio

from docx import Document as DocxDocument


class DocxTextExtractor:
    """
    Reads the paragraph text out of a stored `.docx` file.

    Satisfies `app.application.ingestion.ports.TextExtractor`.
    """

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored .docx file."""
        # python-docx is entirely blocking, so it runs off the event loop; a
        # large document would otherwise stall every other request in the
        # process, the same reasoning as the pypdf extractor.
        return await asyncio.to_thread(self._read, reference)

    @staticmethod
    def _read(filepath: str) -> str:
        document = DocxDocument(filepath)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
