"""Fetches and flattens the readable text of web pages with aiohttp."""

import asyncio
from collections.abc import Sequence

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from yarl import URL

from app.domain.chat.errors import UnsafeUrl
from app.domain.chat.value_objects import FetchableUrl, is_blocked_address

# Per-page text cap, so one huge page cannot fill the context window.
MAX_PAGE_CHARS = 20_000

# Byte cap on the body, so a link to a huge file cannot exhaust memory.
# Larger than MAX_PAGE_CHARS because markup is mostly not text.
MAX_PAGE_BYTES = 2 * 1024 * 1024

# Redirects are followed by hand so every hop is checked for private addresses.
MAX_REDIRECTS = 3

_READABLE_CONTENT_TYPES = ("text/", "application/xhtml+xml", "application/xml")

# Many sites (Wikimedia included) reject aiohttp's default User-Agent with a 403.
# They ask for a descriptive agent with contact details.
REQUEST_HEADERS = {
    "User-Agent": (
        "TalkToWeb/1.0 (+https://github.com/Jvxoq/talk-to-web; "
        "link-summarisation bot) python-aiohttp"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class AiohttpWebContentFetcher:
    """Reads the visible text of the pages a user linked to.

    The session lives for the whole process, to keep connection pooling.
    """

    def __init__(
        self, session: aiohttp.ClientSession, max_page_chars: int = MAX_PAGE_CHARS
    ) -> None:
        self._session = session
        self._max_page_chars = max_page_chars

    async def fetch_all(self, urls: Sequence[str]) -> str:
        """Fetch every URL concurrently and join the results, each page truncated."""
        if not urls:
            return ""

        results = await asyncio.gather(
            *(self._fetch(url) for url in urls),
            return_exceptions=True,
        )

        texts: list[str] = []
        for url, result in zip(urls, results, strict=True):
            # Log and skip a failed link, so the others still count.
            if isinstance(result, BaseException):
                logger.warning(f"Could not fetch {url}: {result}")
                continue
            if result:
                texts.append(self._truncated(result, url))

        return " ".join(texts)

    async def _fetch(self, url: str) -> str:
        """Read one page, refusing any non-public address.

        Known gap: aiohttp resolves the host again on connect, so DNS rebinding
        between the two lookups is not caught.
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
                    # Resolve a relative Location, then check it like any URL.
                    current = FetchableUrl.parse(str(response.url.join(URL(location))))
                    continue

                response.raise_for_status()
                self._refuse_unreadable_type(response, current.value)
                html = await self._read_bounded(response, current.value)

            return self._parse(html, current.value)

        raise UnsafeUrl(url, f"more than {MAX_REDIRECTS} redirects")

    @staticmethod
    async def _refuse_private_targets(url: FetchableUrl) -> None:
        """Refuse if any resolved address is private, since aiohttp may pick any of them."""
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
        """Refuse content types that are not pages."""
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

        # `get_encoding()` needs the full body, so use the declared charset or utf-8.
        encoding = response.charset or "utf-8"
        try:
            return b"".join(chunks).decode(encoding, errors="replace")
        except LookupError:
            # Unknown encoding name from the server.
            return b"".join(chunks).decode("utf-8", errors="replace")

    @staticmethod
    def _parse(html: str, url: str) -> str:
        """Pull the readable body text out of a page, untruncated.

        Tries Wikipedia's body, then `main`, then `article`, then the whole page.
        """
        soup = BeautifulSoup(html, "lxml")

        # `select_one` always returns a Tag or None.

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
