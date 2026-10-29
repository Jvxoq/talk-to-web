"""Stores uploads on the local filesystem."""

from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import aiofiles.os
from loguru import logger

from app.domain.ingestion.errors import DocumentTooLarge
from app.domain.ingestion.value_objects import DocumentName


class LocalFileStorage:
    """Writes uploads into one directory per owner. Implements `FileStorage`.

    The returned path is opaque to the use cases.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def save(
        self,
        name: DocumentName,
        stream: AsyncIterator[bytes],
        max_bytes: int,
        owner_id: int,
    ) -> str:
        """Stream an upload to disk, stopping once it passes `max_bytes`.

        Counts real bytes, since Content-Length is client-supplied or missing.
        """
        destination = self._resolve(name, owner_id)
        await aiofiles.os.makedirs(destination.parent, exist_ok=True)

        written = 0
        try:
            async with aiofiles.open(destination, "wb") as handle:
                async for chunk in stream:
                    written += len(chunk)
                    if written > max_bytes:
                        raise DocumentTooLarge(max_bytes)
                    await handle.write(chunk)
        except BaseException:
            # Never leave a half-written file behind.
            await self._discard(destination)
            raise

        logger.debug(f"Saved {written} bytes to {destination}")
        return str(destination)

    def _resolve(self, name: DocumentName, owner_id: int) -> Path:
        """Join the name onto the owner's directory, refusing anything that escapes."""
        directory = self._directory.resolve()
        destination = (directory / str(owner_id) / name.value).resolve()
        # `DocumentName` already sanitizes. Check again here, at the filesystem edge.
        if not destination.is_relative_to(directory):
            raise ValueError(f"Refusing to write outside the storage directory: {name.value}")
        return destination

    async def delete(self, reference: str) -> None:
        """Remove a stored file. A missing file is not an error.

        `reference` came from `save`, not from a client, so it is trusted.
        """

        await self._discard(Path(reference))

    @staticmethod
    async def _discard(destination: Path) -> None:
        try:
            await aiofiles.os.remove(destination)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(f"Could not remove partial upload {destination}: {error}")
