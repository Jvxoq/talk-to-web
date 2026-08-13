"""Accept an uploaded document and put it somewhere the indexer can reach."""

from collections.abc import AsyncIterator

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import UploadDocumentInput, UploadDocumentResult
from app.application.ingestion.ports import FileStorage, RateLimiter
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import UnsupportedDocumentType
from app.domain.ingestion.value_objects import DocumentName

# What each accepted media type must actually look like: the extension the name
# has to carry, and the bytes the file has to start with. Every PDF begins
# "%PDF-", and that is the one claim about the file a client cannot simply
# assert. Widening the accepted set means adding a row here, which is deliberate
# - a type with no signature to check would be accepted on the sender's word.
#
# Plain text and markdown have no reliable magic-byte signature - any byte
# sequence is valid UTF-8 text, so there is nothing to sniff. Those two rows
# carry `None` and are trusted on content-type and extension alone; every
# other row keeps a real signature, and `_signed` only skips the byte-check
# for the `None` case rather than weakening it globally.
_ACCEPTED: dict[str, tuple[str, bytes | None]] = {
    "application/pdf": (".pdf", b"%PDF-"),
    "text/plain": (".txt", None),
    "text/markdown": (".md", None),
    # A .docx is a zip archive, so this is a weak signature - it collides with
    # any zip-based format - but it is still a real claim a plain rename
    # cannot fake, unlike the txt/md rows above.
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ".docx",
        b"PK\x03\x04",
    ),
}


class UploadDocument:
    """
    Take an upload from an untrusted client to a safe stored reference.

    Type and size are checked before anything is written: the size limit is
    handed to the storage port rather than measured here so an oversized file
    is refused while streaming, not after it has landed on disk.

    The declared content type is checked too, but it is not evidence - it is a
    header the client wrote, and a renamed executable arrives with whatever type
    the sender chose. The first bytes of the file are what decide.
    """

    def __init__(
        self,
        storage: FileStorage,
        max_bytes: int,
        limiter: RateLimiter,
        uow_factory: UnitOfWorkFactory,
        daily_budget: RateLimiter,
        allowed_content_types: frozenset[str] = frozenset(_ACCEPTED),
    ) -> None:
        self._storage = storage
        self._max_bytes = max_bytes
        self._limiter = limiter
        self._uow_factory = uow_factory
        self._daily_budget = daily_budget
        self._allowed_content_types = allowed_content_types

    async def __call__(self, data: UploadDocumentInput) -> UploadDocumentResult:
        # One counter shared with every chat reply, URL ingestion and
        # transcription session in the deployment - see
        # `Settings.global_daily_call_budget`. Checked before the per-user limit
        # for the same reason as there: the per-user one alone caps a single
        # account, not the total spend across as many accounts as someone cares
        # to create.
        await self._daily_budget.hit("global")

        # Counted first, before the type check and before a byte is read. An
        # upload is the expensive half of this app - every accepted file is
        # extracted, chunked and embedded at the provider's per-token price - and
        # a limit that only counted the uploads that turned out to be valid would
        # leave a loop of rejected ones free to run forever. Keyed on the account
        # because that is who the bill follows.
        await self._limiter.hit(f"upload:{data.owner_id}")

        # Browsers may append parameters ("application/pdf; charset=..."), so
        # only the media type itself is compared.
        media_type = (data.content_type or "").split(";")[0].strip().lower()
        if media_type not in self._allowed_content_types or media_type not in _ACCEPTED:
            raise UnsupportedDocumentType(data.content_type)

        suffix, signature = _ACCEPTED[media_type]

        name = DocumentName.sanitize(data.filename)
        if not name.value.lower().endswith(suffix):
            raise UnsupportedDocumentType(data.filename)

        # The check is wrapped around the stream rather than done up front so it
        # happens as the bytes flow: nothing is written before the signature has
        # been seen, and the storage adapter's own cleanup removes the partial
        # file if it fails later.
        stream = _signed(data.stream, signature, data.content_type)
        reference = await self._storage.save(name, stream, self._max_bytes, data.owner_id)
        logger.debug("Stored upload {} as {}", name.value, reference)

        # Recorded here, not by the indexer: the file exists and is this
        # owner's the moment it lands on disk, whether or not indexing behind
        # it ever succeeds, and a document manager needs to list it either way.
        record = UploadedDocument(name=name.value, reference=reference, owner_id=data.owner_id)
        async with self._uow_factory() as uow:
            stored = await uow.documents.add(record)
            await uow.commit()

        return UploadDocumentResult(
            reference=stored.reference, name=stored.name, document_id=_require_id(stored)
        )


def _require_id(document: UploadedDocument) -> int:
    """The id a just-inserted row always has - narrowed once, for mypy's sake."""
    if document.id is None:
        raise RuntimeError("A newly persisted document must have an id")
    return document.id


async def _signed(
    stream: AsyncIterator[bytes],
    signature: bytes | None,
    content_type: str | None,
) -> AsyncIterator[bytes]:
    """Pass a stream through, but only once its first bytes prove what it is.

    Chunk boundaries are not guaranteed, so the opening bytes are buffered until
    there are enough of them to judge, then handed on unchanged - the file that
    reaches storage is byte-for-byte the file that arrived.

    A `None` signature (plain text, markdown - nothing to sniff) skips the
    byte-check entirely and passes the stream through unchanged, trusting the
    content-type and extension checks already done by the caller.
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

    # A file shorter than the signature - or empty - never proved anything.
    if len(head) < len(signature):
        raise UnsupportedDocumentType(content_type)
