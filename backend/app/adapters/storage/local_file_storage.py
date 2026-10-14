"""Stores uploads on the local filesystem."""

from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import aiofiles
import aiofiles.os
from loguru import logger

from app.domain.ingestion.errors import DocumentTooLarge
from app.domain.ingestion.value_objects import DocumentName


class LocalFileStorage:
    """
    Writes uploads into a directory per owner.

    Satisfies `app.application.ingestion.ports.FileStorage`. The reference it
    returns is a filesystem path, which only the matching extractor may
    interpret - the use cases treat it as opaque.

    One flat directory would have every account sharing a namespace it does not
    control the keys to: two people uploading `report.pdf` overwrite each other,
    and the second one's chat answers from the first one's document.

    The same holds for one owner: every save also gets a directory of its own.
    Replacing an attachment stores the new file before the old document is
    removed, so a re-upload of `report.pdf` shares a name with the file about
    to be deleted. Under a name-only path the two were one file, and removing
    the old document deleted the new upload before it could be indexed.
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
        """
        Stream an upload to this owner's directory, refusing it the moment it
        exceeds `max_bytes`.

        The size is checked while writing rather than from a Content-Length
        header, because a header is client-supplied and a chunked upload has
        none: the only honest measure is the bytes that actually arrived.
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
            # Never leave a half-written file behind: the extractor downstream
            # cannot tell a truncated PDF from a corrupt one.
            await self._discard(destination)
            raise

        logger.debug(f"Saved {written} bytes to {destination}")
        return str(destination)

    def _resolve(self, name: DocumentName, owner_id: int) -> Path:
        """Join the name onto the owner's directory, refusing anything that escapes."""
        directory = self._directory.resolve()
        # The owner segment is an integer from a verified token, never a string
        # from a request, so it cannot itself contain a separator. The check
        # below is still made against the *base* directory rather than the
        # owner's, so a traversal cannot land in someone else's folder either.
        destination = (directory / str(owner_id) / uuid4().hex / name.value).resolve()
        # `DocumentName` already sanitizes, but path traversal is cheap to check
        # and expensive to get wrong, so it is verified again at the syscall edge.
        if not destination.is_relative_to(directory):
            raise ValueError(f"Refusing to write outside the storage directory: {name.value}")
        return destination

    async def delete(self, reference: str) -> None:
        """Remove a stored file. A reference already gone is not an error.

        `reference` is the opaque path this adapter itself handed back from
        `save`, never client input, so it is trusted here rather than
        re-resolved through `_resolve`.
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
            return
        # The per-save directory goes with its file. `rmdir` refuses a
        # directory that still holds anything, so this cannot remove more than
        # an empty folder.
        with suppress(OSError):
            await aiofiles.os.rmdir(destination.parent)
