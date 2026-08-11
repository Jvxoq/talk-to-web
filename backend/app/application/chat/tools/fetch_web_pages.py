"""The agent's window onto pages the user named explicitly."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.application.chat.models import Source
from app.application.chat.ports import WebContentFetcher
from app.application.chat.tools.base import BaseTool, ToolContext, ToolResult

# A ceiling on how many pages one call may open. Without it, a model that
# hallucinates a list of thirty links turns one turn into thirty outbound
# requests the user is waiting on.
MAX_URLS_PER_CALL = 10


class FetchWebPagesArgs(BaseModel):
    """What the model must supply to have pages read."""

    model_config = ConfigDict(extra="forbid")

    # `HttpUrl`, not `str`, quite deliberately. The model writes these, and a
    # model that half-remembers an address writes something that is not a URL at
    # all. Pydantic rejects it here, before the scraper opens a socket, and
    # `BaseTool.run` feeds the validation error back as a tool result - so the
    # model sees exactly which entry was wrong and can correct it.
    urls: list[HttpUrl] = Field(
        min_length=1,
        max_length=MAX_URLS_PER_CALL,
        description=(
            "The full http(s) addresses to read, copied exactly as the user wrote "
            "them. Never invent, shorten or guess a URL."
        ),
    )


class FetchWebPages(BaseTool[FetchWebPagesArgs]):
    """
    Reads the visible text of specific pages.

    The counterpart to `search_web`: this one is told what to read, that one is
    asked what is worth reading.
    """

    name: ClassVar[str] = "fetch_web_pages"
    description: ClassVar[str] = (
        "Read the full text of specific web pages whose addresses you already "
        "have. Use this whenever the user pastes a link and asks you to summarise, "
        "explain, compare or quote it, and to open a promising result returned by "
        "search_web. Only pass URLs that appear in the conversation - if you do "
        "not have an address, use search_web to find one first."
    )
    args_model: ClassVar[type[BaseModel]] = FetchWebPagesArgs

    def __init__(self, web: WebContentFetcher) -> None:
        self._web = web

    async def _run(self, args: FetchWebPagesArgs, context: ToolContext) -> ToolResult:
        # Back to plain strings at the boundary: `HttpUrl` is a validation type,
        # and the port speaks `Sequence[str]`.
        urls = [str(url) for url in args.urls]
        content = await self._web.fetch_all(urls)
        if not content.strip():
            return ToolResult(
                content=(
                    f"Could not read any text from: {', '.join(urls)}. The pages may be "
                    "unreachable, empty, or blocked to automated readers."
                )
            )
        # The sources are exactly the URLs asked for, not ones parsed back out
        # of `content`: the fetcher already flattens several pages into one
        # blob, and re-deriving per-page addresses from that text would be
        # guesswork `fetch_all`'s own port has no way to settle.
        sources = tuple(Source(label=url, url=url) for url in urls)
        return ToolResult(content=content, sources=sources)
