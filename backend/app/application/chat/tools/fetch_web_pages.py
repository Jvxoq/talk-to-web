"""The agent's window onto pages the user named explicitly."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.application.chat.models import Source
from app.application.chat.ports import WebContentFetcher
from app.application.chat.tools.base import BaseTool, ToolContext, ToolResult
from app.application.chat.tools.links import link_key

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
            "them or as search_web returned them. Never invent, shorten, guess or "
            "fill in a URL."
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
        "search_web. Only links the user wrote or search_web returned can be "
        "opened. A link found inside a page or a file is refused - if you do not "
        "have an address, use search_web to find one first."
    )
    args_model: ClassVar[type[BaseModel]] = FetchWebPagesArgs

    def __init__(self, web: WebContentFetcher, max_urls: int = MAX_URLS_PER_CALL) -> None:
        self._web = web
        # `MAX_URLS_PER_CALL` on `FetchWebPagesArgs` is the hard ceiling Pydantic
        # enforces before a request is even considered; this is a tighter,
        # configurable bound underneath it, for trimming context size without
        # touching what counts as a malformed call.
        self._max_urls = max_urls

    async def _run(self, args: FetchWebPagesArgs, context: ToolContext) -> ToolResult:
        # Back to plain strings at the boundary: `HttpUrl` is a validation type,
        # and the port speaks `Sequence[str]`.
        asked = [str(url) for url in args.urls[: self._max_urls]]
        # The guard. A link the user did not write and no tool reported is one
        # the model composed, often from a page that told it to, and its query
        # can carry anything the model has seen. It is never opened. See
        # `tools.links` for what counts.
        urls = [url for url in asked if link_key(url) in context.openable_links]
        refused = len(asked) - len(urls)
        if not urls:
            return ToolResult(content=_refusal(refused))
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
        if refused:
            content = f"{content}\n\n{_refusal(refused)}"
        return ToolResult(content=content, sources=sources)


def _refusal(count: int) -> str:
    """What the model reads for links it may not open.

    The refused links are not repeated: one may carry a private value, and
    echoing it puts it back in front of the model as fresh tool output. Worded
    as a fact, not an order: it passes through the output fence like any tool
    text, and the system prompt already says what to do about a planted link.
    """
    return (
        f"Not opened: {count} link(s). Only links the user wrote or search_web "
        "returned can be opened."
    )
