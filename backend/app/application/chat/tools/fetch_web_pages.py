"""The agent's window onto pages the user named explicitly."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.application.chat.models import Source
from app.application.chat.ports import WebContentFetcher
from app.application.chat.tools.base import BaseTool, ToolContext, ToolResult

# Max pages per call, so a model cannot open thirty invented links.
MAX_URLS_PER_CALL = 10


class FetchWebPagesArgs(BaseModel):
    """What the model must supply to have pages read."""

    model_config = ConfigDict(extra="forbid")

    # `HttpUrl`, so a malformed URL is rejected before any request and the
    # model sees which entry was wrong.
    urls: list[HttpUrl] = Field(
        min_length=1,
        max_length=MAX_URLS_PER_CALL,
        description=(
            "The full http(s) addresses to read, copied exactly as the user wrote "
            "them. Never invent, shorten or guess a URL."
        ),
    )


class FetchWebPages(BaseTool[FetchWebPagesArgs]):
    """Reads the visible text of given pages. `search_web` finds pages instead."""

    name: ClassVar[str] = "fetch_web_pages"
    description: ClassVar[str] = (
        "Read the full text of specific web pages whose addresses you already "
        "have. Use this whenever the user pastes a link and asks you to summarise, "
        "explain, compare or quote it, and to open a promising result returned by "
        "search_web. Only pass URLs that appear in the conversation - if you do "
        "not have an address, use search_web to find one first."
    )
    args_model: ClassVar[type[BaseModel]] = FetchWebPagesArgs

    def __init__(self, web: WebContentFetcher, max_urls: int = MAX_URLS_PER_CALL) -> None:
        self._web = web
        # A configurable limit below the hard `MAX_URLS_PER_CALL`.
        self._max_urls = max_urls

    async def _run(self, args: FetchWebPagesArgs, context: ToolContext) -> ToolResult:
        # The port takes plain strings.
        urls = [str(url) for url in args.urls[: self._max_urls]]
        content = await self._web.fetch_all(urls)
        if not content.strip():
            return ToolResult(
                content=(
                    f"Could not read any text from: {', '.join(urls)}. The pages may be "
                    "unreachable, empty, or blocked to automated readers."
                )
            )
        # Cite the requested URLs. The fetched text is one merged blob.

        sources = tuple(Source(label=url, url=url) for url in urls)
        return ToolResult(content=content, sources=sources)
