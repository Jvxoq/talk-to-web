"""Fetches and flattens the readable text of web pages with aiohttp."""

import asyncio
from collections.abc import Sequence

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from yarl import URL

from app.domain.chat.errors import UnsafeUrl
from app.domain.chat.value_objects import FetchableUrl, is_blocked_address

# One page's worth of text is capped so a single enormous document cannot eat
# the whole model context window (or the whole prompt budget for the other URLs).
MAX_PAGE_CHARS = 20_000

# `response.text()` reads until the server stops sending. A link to a disk image
# is enough to exhaust memory, so the body is read in slices and abandoned once
# it is clearly larger than any article. This is generous next to MAX_PAGE_CHARS
# on purpose: markup is mostly not text.
MAX_PAGE_BYTES = 2 * 1024 * 1024

# Redirects are followed by hand rather than by aiohttp, because a guard that
# only inspects the first URL is no guard at all - a public host is free to
# answer with "Location: http://169.254.169.254/". Each hop is re-checked, and
# the chain is short because no honest article needs more.
MAX_REDIRECTS = 3

_READABLE_CONTENT_TYPES = ("text/", "application/xhtml+xml", "application/xml")

# aiohttp's default User-Agent ("Python/3.x aiohttp/3.x") is rejected outright by
# a great many sites - Wikimedia answers it with a flat 403 under its user-agent
# policy, which silently cost us the content of every wikipedia.org link. The
# policy asks for a descriptive agent that identifies the client and gives a way
# to make contact, so that is what we send.
REQUEST_HEADERS = {
    "User-Agent": (
        "TalkToWeb/1.0 (+https://github.com/Jvxoq/talk-to-web; "
        "link-summarisation bot) python-aiohttp"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class AiohttpWebContentFetcher:
    """
    Reads the visible text of the pages a user linked to.

    Satisfies `app.application.chat.ports.WebContentFetcher`. The session is
    injected and lives for the process, not the request: creating a
    `ClientSession` per call throws away connection pooling and DNS caching, and
    is the single most common aiohttp performance mistake.
    """

    def __init__(
        self, session: aiohttp.ClientSession, max_page_chars: int = MAX_PAGE_CHARS
    ) -> None:
        self._session = session
        self._max_page_chars = max_page_chars

    async def fetch_all(self, urls: Sequence[str]) -> str:
        """Fetch every URL concurrently and join whatever came back.

        Each page is truncated to `MAX_PAGE_CHARS` here, at the point where the
        result is about to be folded into one shared blob for a chat turn - not
        inside `_fetch`, which also backs `fetch()` below and has to hand back
        the full page for something that is about to be indexed.
        """
        if not urls:
            return ""

        results = await asyncio.gather(
            *(self._fetch(url) for url in urls),
            return_exceptions=True,
        )

        texts: list[str] = []
        for url, result in zip(urls, results, strict=True):
            # One unreachable link must not lose the content of the others, so
            # failures are logged with their URL and dropped rather than raised.
            if isinstance(result, BaseException):
                logger.warning(f"Could not fetch {url}: {result}")
                continue
            if result:
                texts.append(self._truncated(result, url))

        return " ".join(texts)

    async def fetch(self, url: str) -> str:
        """Fetch one URL and return its full, untruncated text.

        Satisfies `app.application.ingestion.ports.UrlContentFetcher`. Unlike
        `fetch_all`, a failure here is not swallowed: there is only one URL, so
        there is nothing else worth returning, and the caller - ingesting a
        document a user is about to pay to embed - needs to know it failed
        rather than silently store nothing.
        """
        return await self._fetch(url)

    async def _fetch(self, url: str) -> str:
        """
        Read one page, refusing any address that is not on the public internet.

        Residual risk worth naming: this resolves the hostname, and then aiohttp
        resolves it again when it connects. A DNS entry that changes between the
        two answers - rebinding - slips through the gap. Closing it means a
        custom connector that dials the address already checked, which is more
        machinery than this earns; the window is small and the record is here.
        """
        current = FetchableUrl.parse(url)

        for _ in range(MAX_REDIRECTS + 1):
            await self._refuse_private_targets(current)

            async with self._session.get(
                current.value,
                headers=REQUEST_HEADERS,
                allow_redirects=False,
            ) as response:
                location = response.headers.get("Location")
                if response.status in (301, 302, 303, 307, 308) and location:
                    # A relative Location is resolved against the hop it came
                    # from, then re-checked from scratch like any other URL.
                    current = FetchableUrl.parse(str(response.url.join(URL(location))))
                    continue

                response.raise_for_status()
                self._refuse_unreadable_type(response, current.value)
                html = await self._read_bounded(response, current.value)

            return self._parse(html, current.value)

        raise UnsafeUrl(url, f"more than {MAX_REDIRECTS} redirects")

    @staticmethod
    async def _refuse_private_targets(url: FetchableUrl) -> None:
        """Resolve the host and refuse if *any* answer points inside the network.

        Any, not the first: a hostname is free to publish one public address and
        one private one, and aiohttp may pick either when it connects.
        """
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(url.host, url.port)
        except OSError as error:
            raise UnsafeUrl(url.value, f"host does not resolve: {error}") from error

        for info in infos:
            address = str(info[4][0])
            if is_blocked_address(address):
                raise UnsafeUrl(url.value, f"{url.host} resolves to {address}")

    @staticmethod
    def _refuse_unreadable_type(response: aiohttp.ClientResponse, url: str) -> None:
        """Only parse things that could plausibly be a page.

        The parser wants markup. Pointing it at a video or a tarball spends the
        whole byte budget to produce nothing.
        """
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not content_type.startswith(_READABLE_CONTENT_TYPES):
            raise UnsafeUrl(url, f"content type {content_type or 'unknown'!r} is not a page")

    @staticmethod
    async def _read_bounded(response: aiohttp.ClientResponse, url: str) -> str:
        """Read at most MAX_PAGE_BYTES, then stop asking for more."""
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.content.iter_chunked(64 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_PAGE_BYTES:
                logger.debug(f"Stopped reading {url} at {size} bytes")
                break

        # `get_encoding()` is not available here: it sniffs the body, and it
        # raises unless aiohttp buffered the whole thing itself - which is
        # exactly what this method exists to avoid. The declared charset is
        # what is left, and utf-8 is the right guess when there is none.
        encoding = response.charset or "utf-8"
        try:
            return b"".join(chunks).decode(encoding, errors="replace")
        except LookupError:
            # A server is free to name an encoding Python has never heard of.
            return b"".join(chunks).decode("utf-8", errors="replace")

    @staticmethod
    def _parse(html: str, url: str) -> str:
        """
        Pull the readable body text out of a page, in full.

        The legacy parser only ever looked for Wikipedia's `div#bodyContent`, so
        every other site on the internet silently produced an empty string. The
        fallbacks below mean a normal article page still yields its text.

        Truncation is not this method's job: `fetch()` wants the whole page for
        indexing, and only `fetch_all` needs a size cap, so that cap is applied
        by its caller instead of here.
        """
        soup = BeautifulSoup(html, "lxml")

        # `select_one` rather than `find` so the result is always a Tag or None.
        element = (
            soup.select_one("div#bodyContent")
            or soup.select_one("main")
            or soup.select_one("article")
        )
        if element is not None:
            text = element.get_text(" ", strip=True)
        else:
            logger.debug(f"No main content element on {url}, falling back to whole document")
            text = soup.get_text(" ", strip=True)

        if not text:
            logger.warning(f"Could not parse any text from {url}")
            return ""

        return text

    async def aclose(self) -> None:
        """Close the shared HTTP session."""
        await self._session.close()

    def _truncated(self, text: str, url: str) -> str:
        """Cap one page's contribution to a `fetch_all` blob at `self._max_page_chars`."""
        if len(text) > self._max_page_chars:
            logger.debug(f"Truncating {url} from {len(text)} to {self._max_page_chars} chars")
            return text[: self._max_page_chars]
        return text
