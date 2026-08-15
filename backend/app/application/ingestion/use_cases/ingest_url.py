"""Fetch a URL's full text and record it as a document, no file involved.

Reached from a different starting point than `UploadDocument` - instead of
trusting a client-sent stream, this trusts a client-sent URL and does the
fetching itself - but unlike an upload there is never a client-supplied blob
that has to land somewhere before it can be read back. The fetcher already
hands back the full page as a string, so that string goes straight to
`IndexDocument`; nothing is written to disk or object storage on this path.
"""

import hashlib

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.dto import IngestUrlInput, UploadDocumentResult
from app.application.ingestion.ports import RateLimiter, UrlContentFetcher
from app.domain.chat.value_objects import FetchableUrl
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.value_objects import DocumentName

# The document name still carries this suffix so it displays and sorts like
# the text file it represents, even though nothing with this name is ever
# written to storage.
_SUFFIX = ".txt"

# Long enough that two different pages on the same host essentially never
# collide, short enough that the stored name stays readable.
_HASH_CHARS = 10


class IngestUrl:
    """
    Take a URL from an untrusted client to an indexable document, in memory.

    Mirrors `UploadDocument`'s shape: rate-limit first, then validate, then
    fetch. The validation here is "is this URL safe to open a connection to"
    rather than "does this file start with the bytes it claims to" - the
    equivalent check for something the server itself goes and fetches, not
    something a client handed over already-formed.

    `reference` on the resulting record is the source URL itself rather than
    a storage path - there is no file behind it for `DeleteDocument` to clean
    up, only the vectors and the row.
    """

    def __init__(
        self,
        fetcher: UrlContentFetcher,
        limiter: RateLimiter,
        uow_factory: UnitOfWorkFactory,
        daily_budget: RateLimiter,
    ) -> None:
        self._fetcher = fetcher
        self._limiter = limiter
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
        logger.debug("Fetched {} as {}", fetchable.value, name.value)

        # Recorded the same way `UploadDocument` records a file: the moment the
        # fetch succeeds, it belongs in the document manager whether or not
        # indexing behind it ever succeeds. `reference` is the source URL - the
        # only thing identifying this document once the text above is gone.
        record = UploadedDocument(
            name=name.value, reference=fetchable.value, owner_id=data.owner_id
        )
        async with self._uow_factory() as uow:
            stored = await uow.documents.add(record)
            await uow.commit()

        return UploadDocumentResult(
            reference=stored.reference,
            name=stored.name,
            document_id=_require_id(stored),
            text=text,
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
