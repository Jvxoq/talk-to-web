"""Fetch a URL's full text and put it wherever `UploadDocument` puts a file.

Same destination as an uploaded document, reached from a different starting
point: instead of trusting a client-sent stream, this trusts a client-sent
URL and does the fetching itself.
"""

import hashlib
from collections.abc import AsyncIterator

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import IngestUrlInput, UploadDocumentResult
from app.application.ingestion.ports import FileStorage, RateLimiter, UrlContentFetcher
from app.domain.chat.value_objects import FetchableUrl
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.value_objects import DocumentName

# The fetched text is saved with this suffix so `CompositeTextExtractor` (built
# alongside this use case) routes it to the plain-text extractor at index time,
# the same way a hand-uploaded .txt file would be - no coupling to that
# extractor's code, just a shared filename convention.
_SUFFIX = ".txt"

# Long enough that two different pages on the same host essentially never
# collide, short enough that the stored name stays readable.
_HASH_CHARS = 10


class IngestUrl:
    """
    Take a URL from an untrusted client to a safe stored reference.

    Mirrors `UploadDocument`'s shape: rate-limit first, then validate, then
    fetch/store. The validation here is "is this URL safe to open a connection
    to" rather than "does this file start with the bytes it claims to" - the
    equivalent check for something the server itself goes and fetches, not
    something a client handed over already-formed.
    """

    def __init__(
        self,
        fetcher: UrlContentFetcher,
        storage: FileStorage,
        limiter: RateLimiter,
        max_bytes: int,
        uow_factory: UnitOfWorkFactory,
        daily_budget: RateLimiter,
    ) -> None:
        self._fetcher = fetcher
        self._storage = storage
        self._limiter = limiter
        self._max_bytes = max_bytes
        self._uow_factory = uow_factory
        self._daily_budget = daily_budget

    async def __call__(self, data: IngestUrlInput) -> UploadDocumentResult:
        # One counter shared with every chat reply, upload and transcription
        # session in the deployment - see `Settings.global_daily_call_budget`.
        await self._daily_budget.hit("global")

        # Same reasoning as `UploadDocument`: counted before any fetch happens,
        # so a loop of refused requests cannot run for free.
        await self._limiter.hit(f"ingest_url:{data.owner_id}")

        # Raises `UnsafeUrl` for a bad scheme, a missing host, or a literal
        # address that is not on the public internet. The adapter's `fetch`
        # still has to re-check after DNS resolution - a hostname can answer
        # with a private address - but everything checkable without the network
        # is checked here, before a request is ever made.
        fetchable = FetchableUrl.parse(data.url)

        text = await self._fetcher.fetch(fetchable.value)

        name = _synthetic_name(fetchable.host, fetchable.value)
        stream = _byte_stream(text)
        reference = await self._storage.save(name, stream, self._max_bytes, data.owner_id)
        logger.debug("Stored {} as {}", fetchable.value, reference)

        # Recorded the same way `UploadDocument` records a file: the moment the
        # text is safely on disk, it belongs in the document manager whether or
        # not indexing behind it ever succeeds.
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


def _synthetic_name(host: str, url: str) -> DocumentName:
    """Derive a safe `.txt` filename from a URL, so it never collides across hosts."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:_HASH_CHARS]
    return DocumentName.sanitize(f"{host}-{digest}{_SUFFIX}")


async def _byte_stream(text: str) -> AsyncIterator[bytes]:
    """Wrap already-fetched text as the `AsyncIterator[bytes]` `FileStorage.save` expects."""
    yield text.encode("utf-8")
