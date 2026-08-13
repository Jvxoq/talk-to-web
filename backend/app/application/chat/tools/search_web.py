"""The agent's window onto the live web, when it does not know where to look."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.application.chat.ports import WebSearcher
from app.application.chat.tools.base import BaseTool, ToolContext, ToolResult


class SearchWebArgs(BaseModel):
    """What the model must supply to search the web."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description=(
            "What to search for, phrased the way a person would type it into a "
            "search engine. Keep it to one topic - run two searches rather than "
            "one query joined by 'and'."
        ),
    )


class SearchWeb(BaseTool[SearchWebArgs]):
    """
    Answers an open question with passages from the live web.

    `max_results` arrives from settings through the constructor rather than from
    the model: how much the search costs is an operator's decision, and a model
    given the dial will ask for twenty results on a question that needed three.
    """

    name: ClassVar[str] = "search_web"
    description: ClassVar[str] = (
        "Search the live web and return the best matching passages with their "
        "source URLs. Use this for anything you cannot answer from your own "
        "knowledge: current events, prices, releases, schedules, statistics, "
        "anything that changed recently, and any question where being out of date "
        "would be wrong. Use it also when you need a page's address before calling "
        "fetch_web_pages. Do not use it when the user already gave you the link. "
        "If the question names a specific person, company, product or topic you "
        "don't recognize, try retrieve_documents first - it may be in a file the "
        "user uploaded - and only reach for this once that comes back empty."
    )
    args_model: ClassVar[type[BaseModel]] = SearchWebArgs

    def __init__(self, searcher: WebSearcher, max_results: int) -> None:
        self._searcher = searcher
        self._max_results = max_results

    async def _run(self, args: SearchWebArgs, context: ToolContext) -> ToolResult:
        result = await self._searcher.search(args.query, self._max_results)
        if not result.text.strip():
            return ToolResult(
                content=(
                    f"The web search for {args.query!r} returned no results. Try different "
                    "wording, or answer from what you already know and say it is unverified."
                )
            )
        return ToolResult(content=result.text, sources=result.sources)
