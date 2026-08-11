"""Plain-text extraction for .txt and .md uploads."""

import aiofiles


class PlainTextExtractor:
    """
    Reads a stored `.txt` or `.md` file as UTF-8.

    Satisfies `app.application.ingestion.ports.TextExtractor`. Markdown is
    embedded as-is - stripping its syntax would throw away structure (headings,
    lists) that is still useful signal once chunked and embedded, and the
    extra parsing step buys nothing the chat model needs.
    """

    async def extract(self, reference: str) -> str:
        """Extract the text of a stored plain-text or markdown file."""
        async with aiofiles.open(reference, encoding="utf-8") as handle:
            content: str = await handle.read()
            return content
