"""Saved web pages and search results, so an eval run needs no internet.

`fetch_web_pages` needs the live internet and `search_web` needs a Tavily key,
so a task about either could not run in CI. This module stands in for the two
*ports* underneath those tools, not for the tools. `FetchWebPages` and
`SearchWeb` stay the real classes, so the model sees the same names, the same
argument schemas and the same description and failure wording as in
production. Only the I/O below them is replaced:

- `RecordedWebContentFetcher` satisfies `WebContentFetcher`. A URL with a
  recording returns its saved text. A URL with none contributes nothing, like
  a page the real fetcher could not reach, and `FetchWebPages` then says
  "Could not read any text from ...".
- `RecordedSearchClient` stands in for Tavily's client. It returns a saved
  response in Tavily's own JSON shape, and the real `TavilyWebSearcher`
  flattens it, so the text the model reads has the production format. A query
  with no recording returns no results, and `SearchWeb` says so.

Recordings live in `evals/fixtures/web/`. `index.json` lists them:

    pages        URL -> a text file under `fetch/`
    unreachable  URLs that are recorded as failing on purpose
    searches     {terms, file}: the first entry whose every term appears in
                 the query (case-insensitive) answers it. The model words a
                 query differently every trial, so matching on the exact
                 string would fail for reasons that have nothing to do with
                 the agent.

All content is invented. A task about a real page would let the model answer
from memory, and would make the recording go stale.

Hosts are invented names under ordinary domains, not `.example`. The agent
recognises `.example` as reserved and tells the user so, a claim the source
never made. The names were checked not to resolve on 2026-11-26. Under
`--web live` a made-up host could be registered later, so trust live-mode
fetch tasks less.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

WEB_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "web"
INDEX_NAME = "index.json"


class RecordedWebError(ValueError):
    """The recordings are broken. Raised on load, before the first paid call."""


def normalize_url(url: str) -> str:
    """Compare URLs the way a reader does.

    `HttpUrl` turns `https://example.com` into `https://example.com/`, and a
    model may add or drop a trailing slash or a fragment. None of those pick a
    different page, so none may make a recording miss.
    """
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


@dataclass(frozen=True, slots=True)
class SearchRecording:
    terms: tuple[str, ...]
    response: dict[str, Any]


class RecordedWeb:
    """The saved pages and searches, loaded once and read-only."""

    def __init__(
        self,
        pages: dict[str, str],
        unreachable: frozenset[str],
        searches: Sequence[SearchRecording],
    ) -> None:
        self._pages = {normalize_url(url): text for url, text in pages.items()}
        self._unreachable = frozenset(normalize_url(url) for url in unreachable)
        self._searches = tuple(searches)

    @classmethod
    def load(cls, root: Path = WEB_FIXTURES_DIR) -> "RecordedWeb":
        index = _read_json(root / INDEX_NAME)
        pages = {url: _read_text(root, name) for url, name in index.get("pages", {}).items()}
        unreachable = frozenset(index.get("unreachable", []))
        both = {normalize_url(url) for url in pages} & {normalize_url(url) for url in unreachable}
        if both:
            raise RecordedWebError(f"URLs both recorded and unreachable: {sorted(both)}")
        searches = []
        for entry in index.get("searches", []):
            terms = tuple(str(term).lower() for term in entry.get("terms", []))
            if not terms:
                raise RecordedWebError(f"A search recording has no terms: {entry}")
            searches.append(
                SearchRecording(terms=terms, response=_read_json(root / str(entry["file"])))
            )
        return cls(pages, unreachable, searches)

    def page(self, url: str) -> str:
        """The saved text of `url`, or `""` when it is unreachable or unknown."""
        return self._pages.get(normalize_url(url), "")

    def search(self, query: str) -> dict[str, Any] | None:
        lowered = query.lower()
        for recording in self._searches:
            if all(term in lowered for term in recording.terms):
                return recording.response
        return None


class RecordedWebContentFetcher:
    """Satisfies `app.application.chat.ports.WebContentFetcher` from saved pages."""

    def __init__(self, web: RecordedWeb) -> None:
        self._web = web

    async def fetch_all(self, urls: Sequence[str]) -> str:
        # Same join as `AiohttpWebContentFetcher.fetch_all`: pages that could
        # not be read are dropped, the rest are joined with a space.
        texts = [text for url in urls if (text := self._web.page(url))]
        return " ".join(texts)


class RecordedSearchClient:
    """The slice of `AsyncTavilyClient` that `TavilyWebSearcher` calls."""

    def __init__(self, web: RecordedWeb) -> None:
        self._web = web

    async def search(
        self, query: str, *, max_results: int, include_answer: bool = False
    ) -> dict[str, Any]:
        saved = self._web.search(query)
        if saved is None:
            return {"results": []}
        response: dict[str, Any] = {"results": list(saved.get("results", []))[:max_results]}
        if include_answer and "answer" in saved:
            response["answer"] = saved["answer"]
        return response

    async def close(self) -> None:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordedWebError(f"Cannot read recording {path}: {error}") from error
    if not isinstance(data, dict):
        raise RecordedWebError(f"Recording {path} must hold a JSON object")
    return data


def _read_text(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="utf-8")
    except OSError as error:
        raise RecordedWebError(f"Cannot read recording {root / name}: {error}") from error
