"""Stores uploads on the local filesystem."""

from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import aiofiles.os
from loguru import logger

from app.domain.ingestion.errors import DocumentTooLarge
from app.domain.ingestion.value_objects import DocumentName


class LocalFileStorage:
    """
    Writes uploads into one directory on disk.

    Satisfies `app.application.ingestion.ports.FileStorage`. The reference it
    returns is a filesystem path, which only the matching extractor may
    interpret - the use cases treat it as opaque.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def save(
        self,
        name: DocumentName,
        stream: AsyncIterator[bytes],
        max_bytes: int,
    ) -> str:
        """
        Stream an upload to disk, refusing it the moment it exceeds `max_bytes`.

        The size is checked while writing rather than from a Content-Length
        header, because a header is client-supplied and a chunked upload has
        none: the only honest measure is the bytes that actually arrived.
        """
        destination = self._resolve(name)
        await aiofiles.os.makedirs(self._directory, exist_ok=True)

        written = 0
        try:
            async with aiofiles.open(destination, "wb") as handle:
                async for chunk in stream:
                    written += len(chunk)
                    if written > max_bytes:
                        raise DocumentTooLarge(max_bytes)
                    await handle.write(chunk)
        except BaseException:
            # Never leave a half-written file behind: the extractor downstream
            # cannot tell a truncated PDF from a corrupt one.
            await self._discard(destination)
            raise

        logger.debug(f"Saved {written} bytes to {destination}")
        return str(destination)

    def _resolve(self, name: DocumentName) -> Path:
        """Join the name onto the storage directory, refusing anything that escapes."""
        directory = self._directory.resolve()
        destination = (directory / name.value).resolve()
        # `DocumentName` already sanitizes, but path traversal is cheap to check
        # and expensive to get wrong, so it is verified again at the syscall edge.
        if not destination.is_relative_to(directory):
            raise ValueError(f"Refusing to write outside the storage directory: {name.value}")
        return destination

    @staticmethod
    async def _discard(destination: Path) -> None:
        try:
            await aiofiles.os.remove(destination)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(f"Could not remove partial upload {destination}: {error}")
