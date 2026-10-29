"""Accept an uploaded document and put it somewhere the indexer can reach."""

from collections.abc import AsyncIterator

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import UploadDocumentInput, UploadDocumentResult
from app.application.ingestion.ports import DocumentRemover, FileStorage, RateLimiter
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import UnknownConversation, UnsupportedDocumentType
from app.domain.ingestion.value_objects import DocumentName

# Accepted media types: required extension and required first bytes.
# Text and markdown have no signature (`None`), so only type and extension are checked.
_ACCEPTED: dict[str, tuple[str, bytes | None]] = {
    "application/pdf": (".pdf", b"%PDF-"),
    "text/plain": (".txt", None),
    "text/markdown": (".md", None),
    # Weak: any zip matches. Still more than a rename can fake.
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ".docx",
        b"PK\x03\x04",
    ),
}


class UploadDocument:
    """Store an untrusted upload safely.

    Storage enforces the size limit while streaming. The content type is a
    client header, so the file's first bytes decide its type.
    """

    def __init__(
        self,
        storage: FileStorage,
        max_bytes: int,
        limiter: RateLimiter,
        uow_factory: UnitOfWorkFactory,
        daily_budget: RateLimiter,
        remove_document: DocumentRemover,
        allowed_content_types: frozenset[str] = frozenset(_ACCEPTED),
    ) -> None:
        self._storage = storage
        self._max_bytes = max_bytes
        self._limiter = limiter
        self._uow_factory = uow_factory
        self._daily_budget = daily_budget
        self._remove_document = remove_document
        self._allowed_content_types = allowed_content_types

    async def __call__(self, data: UploadDocumentInput) -> UploadDocumentResult:
        # Global budget first, since one person can make many accounts.
        await self._daily_budget.hit("global")

        # Counted before validation, so a loop of rejected uploads still hits the limit.
        await self._limiter.hit(f"upload:{data.owner_id}")

        # Refuse a foreign thread before reading any bytes.
        async with self._uow_factory() as uow:
            if await uow.conversations.get(data.conversation_id, data.owner_id) is None:
                raise UnknownConversation(data.conversation_id)

        # Drop parameters like "; charset=...".
        media_type = (data.content_type or "").split(";")[0].strip().lower()
        if media_type not in self._allowed_content_types or media_type not in _ACCEPTED:
            raise UnsupportedDocumentType(data.content_type)

        suffix, signature = _ACCEPTED[media_type]

        name = DocumentName.sanitize(data.filename)
        if not name.value.lower().endswith(suffix):
            raise UnsupportedDocumentType(data.filename)

        # Checked as bytes flow, so nothing is written before the signature is seen.
        stream = _signed(data.stream, signature, data.content_type)
        reference = await self._storage.save(name, stream, self._max_bytes, data.owner_id)
        logger.debug("Stored upload {} as {}", name.value, reference)

        # One attachment per conversation, so remove the old one. Done after the
        # new file is stored, so a failed upload keeps the old document.
        await self._replace_attachments(data.owner_id, data.conversation_id)

        # Recorded here, so the file is listed even if indexing fails.
        record = UploadedDocument(
            name=name.value,
            reference=reference,
            owner_id=data.owner_id,
            conversation_id=data.conversation_id,
        )
        async with self._uow_factory() as uow:
            stored = await uow.documents.add(record)
            await uow.commit()

        return UploadDocumentResult(
            reference=stored.reference, name=stored.name, document_id=_require_id(stored)
        )

    async def _replace_attachments(self, owner_id: int, conversation_id: int) -> None:
        """Remove the documents already attached to this conversation.

        Failures are logged, not raised, so the new upload still succeeds.
        """
        async with self._uow_factory() as uow:
            previous = await uow.documents.list_by_conversation(owner_id, conversation_id)

        for document in previous:
            if document.id is None:
                continue
            try:
                await self._remove_document(document.id, owner_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not replace document {document.id}: {exc}")


def _require_id(document: UploadedDocument) -> int:
    """Narrow a new row's id for mypy."""
    if document.id is None:
        raise RuntimeError("A newly persisted document must have an id")
    return document.id


async def _signed(
    stream: AsyncIterator[bytes],
    signature: bytes | None,
    content_type: str | None,
) -> AsyncIterator[bytes]:
    """Pass a stream through once its first bytes match `signature`.

    Buffers the opening bytes across chunks, then passes them on unchanged.
    A `None` signature skips the check.
    """
    if signature is None:
        async for chunk in stream:
            yield chunk
        return

    head = b""
    async for chunk in stream:
        if len(head) < len(signature):
            head += chunk
            if len(head) < len(signature):
                continue
            if not head.startswith(signature):
                raise UnsupportedDocumentType(content_type)
            yield head
            continue
        yield chunk

    # Too short or empty to match.

    if len(head) < len(signature):
        raise UnsupportedDocumentType(content_type)
