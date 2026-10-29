"""Plain-text extraction for .txt and .md uploads."""

import aiofiles


class PlainTextExtractor:
    """Reads a `.txt` or `.md` file as UTF-8. Implements `TextExtractor`.

    Markdown is kept as-is, since its structure helps retrieval.
    """

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored plain-text or markdown file."""
        async with aiofiles.open(reference, encoding="utf-8") as handle:
            content: str = await handle.read()
            return content
