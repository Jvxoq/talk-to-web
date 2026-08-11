"""Tavily-backed web search.

Satisfies `app.application.chat.ports.WebSearcher`. Returns flattened text, never
Tavily's JSON: the tool above it prints what comes back straight into the
conversation, so the vendor's result shape stops here.
"""

from typing import Any

from loguru import logger
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from app.application.chat.models import SearchResult, Source

# One result's body is capped so a single verbose page cannot crowd the other
# results (and the rest of the conversation) out of the context window.
MAX_RESULT_CHARS = 4_000

_EMPTY_RESULT = SearchResult(text="")


class TavilyWebSearcher:
    """Asks Tavily what is worth reading, and flattens the answer."""

    def __init__(self, client: AsyncTavilyClient) -> None:
        self._client = client

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Return the best passages for `query`, best first, or an empty result on failure."""
        logger.debug(f"Tavily search: {query!r} (max_results={max_results})")
        try:
            response: dict[str, Any] = await self._client.search(
                query,
                max_results=max_results,
                # Tavily's own one-paragraph synthesis. Cheap, and it gives the
                # model something to anchor on when the individual results
                # disagree.
                include_answer=True,
            )
        except Exception as error:
            # Deliberately broad, matching the posture of the other read paths:
            # search is an enhancement to an answer that is already streaming,
            # so a dead upstream costs grounding, never the reply. The tool
            # above turns an empty result into a sentence the model can act on.
            logger.warning(f"Tavily search for {query!r} failed ({error}), returning no results")
            return _EMPTY_RESULT

        return self._flatten(response)

    @staticmethod
    def _flatten(response: dict[str, Any]) -> SearchResult:
        blocks: list[str] = []
        sources: list[Source] = []

        answer = response.get("answer")
        if isinstance(answer, str) and answer.strip():
            blocks.append(f"Answer: {answer.strip()}")

        results = response.get("results")
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "Untitled")
            url = str(result.get("url") or "unknown")
            content = str(result.get("content") or "").strip()[:MAX_RESULT_CHARS]
            blocks.append(f"Title: {title}\nURL: {url}\n{content}")
            sources.append(Source(label=title, url=url))

        return SearchResult(text="\n\n".join(blocks), sources=tuple(sources))

    async def aclose(self) -> None:
        """Release Tavily's HTTP connection pool."""
        await self._client.close()
