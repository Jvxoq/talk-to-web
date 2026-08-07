"""Accept an uploaded document and put it somewhere the indexer can reach."""

from loguru import logger

from app.application.ingestion.dto import UploadDocumentInput, UploadDocumentResult
from app.application.ingestion.ports import FileStorage
from app.domain.ingestion.errors import UnsupportedDocumentType
from app.domain.ingestion.value_objects import DocumentName


class UploadDocument:
    """
    Take an upload from an untrusted client to a safe stored reference.

    Type and size are checked before anything is written: the size limit is
    handed to the storage port rather than measured here so an oversized file
    is refused while streaming, not after it has landed on disk.
    """

    def __init__(
        self,
        storage: FileStorage,
        max_bytes: int,
        allowed_content_types: frozenset[str] = frozenset({"application/pdf"}),
    ) -> None:
        self._storage = storage
        self._max_bytes = max_bytes
        self._allowed_content_types = allowed_content_types

    async def __call__(self, data: UploadDocumentInput) -> UploadDocumentResult:
        # Browsers may append parameters ("application/pdf; charset=..."), so
        # only the media type itself is compared.
        media_type = (data.content_type or "").split(";")[0].strip().lower()
        if media_type not in self._allowed_content_types:
            raise UnsupportedDocumentType(data.content_type)

        name = DocumentName.sanitize(data.filename)
        reference = await self._storage.save(name, data.stream, self._max_bytes)
        logger.debug("Stored upload {} as {}", name.value, reference)
        return UploadDocumentResult(reference=reference, name=name.value)
