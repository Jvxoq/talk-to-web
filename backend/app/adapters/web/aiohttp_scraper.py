"""Fetches and flattens the readable text of web pages with aiohttp."""

import asyncio
from collections.abc import Sequence

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

# One page's worth of text is capped so a single enormous document cannot eat
# the whole model context window (or the whole prompt budget for the other URLs).
MAX_PAGE_CHARS = 20_000

# aiohttp's default User-Agent ("Python/3.x aiohttp/3.x") is rejected outright by
# a great many sites - Wikimedia answers it with a flat 403 under its user-agent
# policy, which silently cost us the content of every wikipedia.org link. The
# policy asks for a descriptive agent that identifies the client and gives a way
# to make contact, so that is what we send.
REQUEST_HEADERS = {
    "User-Agent": (
        "TalkToTheWeb/1.0 (+https://github.com/talk-to-the-web; "
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

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def fetch_all(self, urls: Sequence[str]) -> str:
        """Fetch every URL concurrently and join whatever came back."""
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
                texts.append(result)

        return " ".join(texts)

    async def _fetch(self, url: str) -> str:
        async with self._session.get(url, headers=REQUEST_HEADERS) as response:
            response.raise_for_status()
            html = await response.text()
        return self._parse(html, url)

    @staticmethod
    def _parse(html: str, url: str) -> str:
        """
        Pull the readable body text out of a page.

        The legacy parser only ever looked for Wikipedia's `div#bodyContent`, so
        every other site on the internet silently produced an empty string. The
        fallbacks below mean a normal article page still yields its text.
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

        if len(text) > MAX_PAGE_CHARS:
            logger.debug(f"Truncating {url} from {len(text)} to {MAX_PAGE_CHARS} chars")
            return text[:MAX_PAGE_CHARS]
        return text

    async def aclose(self) -> None:
        """Close the shared HTTP session."""
        await self._session.close()
