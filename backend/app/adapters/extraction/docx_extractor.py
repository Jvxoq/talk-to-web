"""DOCX text extraction with python-docx."""

import asyncio

from docx import Document as DocxDocument


class DocxTextExtractor:
    """Reads the paragraph text of a `.docx` file. Implements `TextExtractor`."""

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored .docx file."""
        # python-docx blocks, so it runs in a thread.

        return await asyncio.to_thread(self._read, reference)

    @staticmethod
    def _read(filepath: str) -> str:
        document = DocxDocument(filepath)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
