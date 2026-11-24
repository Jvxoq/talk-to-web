"""The recorded web tools, checked with no network and no key.

The point of recording is that the model sees what it sees in production. These
tests hold that line from three sides: the saved content comes back, a URL or
query with no recording fails the way the real tool fails (a sentence, never a
crash), and the tool names, schemas and descriptions equal the live ones.
"""

import inspect
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.adapters.web.tavily_search import TavilyWebSearcher
from app.application.chat.tools.base import ToolContext
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.search_web import SearchWeb
from evals.harness import build_harness, build_web_tools
from evals.recorded_web import (
    RecordedSearchClient,
    RecordedWeb,
    RecordedWebContentFetcher,
    RecordedWebError,
    normalize_url,
)

CONTEXT = ToolContext(owner_id=1)
PAGE_URL = "https://quillfeather-observatory.example/visitor-guide"
GONE_URL = "https://harbourlight-festival.example/tickets"


@pytest.fixture
def web() -> RecordedWeb:
    return RecordedWeb.load()


@pytest.fixture
def fetch_tool(web: RecordedWeb) -> FetchWebPages:
    return FetchWebPages(RecordedWebContentFetcher(web))


@pytest.fixture
def search_tool(web: RecordedWeb) -> SearchWeb:
    searcher = TavilyWebSearcher(client=RecordedSearchClient(web))
    return SearchWeb(searcher, max_results=5)


class TestRecordedFetch:
    async def test_returns_the_saved_page(self, fetch_tool: FetchWebPages) -> None:
        outcome = await fetch_tool.run({"urls": [PAGE_URL]}, CONTEXT)

        assert outcome.ok
        assert "Marrowgate Refractor" in outcome.content
        assert [source.url for source in outcome.sources] == [PAGE_URL]

    @pytest.mark.parametrize(
        "url", [PAGE_URL + "/", PAGE_URL + "#top", PAGE_URL.upper()[:8] + PAGE_URL[8:]]
    )
    async def test_trailing_slash_fragment_and_scheme_case_still_match(
        self, fetch_tool: FetchWebPages, url: str
    ) -> None:
        outcome = await fetch_tool.run({"urls": [url]}, CONTEXT)

        assert "Marrowgate" in outcome.content

    @pytest.mark.parametrize("url", [GONE_URL, "https://never-recorded.example/page"])
    async def test_an_unreachable_or_unknown_url_gets_the_real_failure_text(
        self, fetch_tool: FetchWebPages, url: str
    ) -> None:
        outcome = await fetch_tool.run({"urls": [url]}, CONTEXT)

        assert outcome.ok  # the real tool reports this as a normal result, not a crash
        assert outcome.content.startswith(f"Could not read any text from: {url}")
        assert "unreachable, empty, or blocked" in outcome.content
        assert outcome.sources == ()

    async def test_one_missing_page_does_not_hide_the_others(
        self, fetch_tool: FetchWebPages
    ) -> None:
        outcome = await fetch_tool.run({"urls": [GONE_URL, PAGE_URL]}, CONTEXT)

        assert "Marrowgate" in outcome.content
        assert "Could not read" not in outcome.content

    async def test_fetch_all_joins_saved_pages_like_the_real_fetcher(self) -> None:
        web = RecordedWeb(
            {"https://a.example/1": "one", "https://a.example/2": "two"}, frozenset(), []
        )

        text = await RecordedWebContentFetcher(web).fetch_all(
            ["https://a.example/1", "https://a.example/missing", "https://a.example/2"]
        )

        assert text == "one two"


class TestRecordedSearch:
    async def test_returns_the_saved_results_in_the_tavily_format(
        self, search_tool: SearchWeb
    ) -> None:
        outcome = await search_tool.run(
            {"query": "Brindlewick tram line council decision"}, CONTEXT
        )

        assert outcome.ok
        assert outcome.content.startswith("Answer: Tessaly Regional Council voted")
        assert "Title: Council backs Brindlewick tram extension to Verrin Quay" in outcome.content
        assert (
            "URL: https://tessaly-courier.example/news/brindlewick-tram-extension"
            in outcome.content
        )
        assert [source.url for source in outcome.sources] == [
            "https://tessaly-courier.example/news/brindlewick-tram-extension",
            "https://tessaly-courier.example/news/brindlewick-riders-react",
        ]

    async def test_matching_ignores_case_and_word_order(self, search_tool: SearchWeb) -> None:
        outcome = await search_tool.run({"query": "TRAM news: what about brindlewick?"}, CONTEXT)

        assert "Verrin Quay" in outcome.content

    async def test_an_unknown_query_gets_the_real_empty_search_text(
        self, search_tool: SearchWeb
    ) -> None:
        outcome = await search_tool.run({"query": "bus timetable in Paris"}, CONTEXT)

        assert outcome.ok
        assert outcome.content.startswith(
            "The web search for 'bus timetable in Paris' returned no results."
        )
        assert outcome.sources == ()

    async def test_a_query_with_only_some_terms_does_not_match(
        self, search_tool: SearchWeb
    ) -> None:
        outcome = await search_tool.run({"query": "Brindlewick bakery"}, CONTEXT)

        assert "returned no results" in outcome.content

    async def test_max_results_is_honoured(self, web: RecordedWeb) -> None:
        searcher = TavilyWebSearcher(client=RecordedSearchClient(web))

        result = await searcher.search("brindlewick tram", 1)

        assert len(result.sources) == 1


class TestSameTools:
    async def test_specs_match_the_live_tools(self, web: RecordedWeb) -> None:
        async with aiohttp.ClientSession() as session:
            recorded = build_web_tools("recorded", http_session=session, tavily_api_key="x")
            live = build_web_tools("live", http_session=session, tavily_api_key="x")
            try:
                for make in (
                    lambda tools: FetchWebPages(tools.fetcher),
                    lambda tools: SearchWeb(tools.searcher, max_results=5),
                ):
                    assert make(recorded).spec == make(live).spec
            finally:
                await recorded.aclose()
                await live.aclose()

    def test_the_specs_still_carry_the_production_names_and_url_argument(self) -> None:
        fetch = FetchWebPages(RecordedWebContentFetcher(RecordedWeb({}, frozenset(), []))).spec
        assert fetch.name == "fetch_web_pages"
        assert fetch.parameters["properties"]["urls"]["items"]["format"] == "uri"
        assert fetch.parameters["required"] == ["urls"]

    async def test_a_malformed_url_is_still_rejected_by_the_real_schema(
        self, fetch_tool: FetchWebPages
    ) -> None:
        outcome = await fetch_tool.run({"urls": ["not a url"]}, CONTEXT)

        assert not outcome.ok
        assert outcome.content.startswith("Invalid arguments for fetch_web_pages")


class TestWebMode:
    async def test_recorded_mode_builds_recorded_ports_and_needs_no_key(self) -> None:
        async with aiohttp.ClientSession() as session:
            tools = build_web_tools("recorded", http_session=session, tavily_api_key="")
            try:
                assert isinstance(tools.fetcher, RecordedWebContentFetcher)
                assert not isinstance(tools.fetcher, AiohttpWebContentFetcher)
            finally:
                await tools.aclose()

    async def test_live_mode_builds_the_real_fetcher(self) -> None:
        async with aiohttp.ClientSession() as session:
            tools = build_web_tools("live", http_session=session, tavily_api_key="x")
            try:
                assert isinstance(tools.fetcher, AiohttpWebContentFetcher)
            finally:
                await tools.aclose()

    def test_the_harness_defaults_to_recorded(self) -> None:
        default = inspect.signature(build_harness).parameters["web_mode"].default

        assert default == "recorded"


class TestLoadingFailsEarly:
    def _write(self, root: Path, index: dict[str, Any], **files: str) -> Path:
        (root / "index.json").write_text(json.dumps(index))
        for name, text in files.items():
            (root / name).write_text(text)
        return root

    def test_a_missing_page_file_is_an_error(self, tmp_path: Path) -> None:
        self._write(tmp_path, {"pages": {"https://a.example/": "gone.txt"}})

        with pytest.raises(RecordedWebError, match=r"gone\.txt"):
            RecordedWeb.load(tmp_path)

    def test_a_search_with_no_terms_is_an_error(self, tmp_path: Path) -> None:
        self._write(tmp_path, {"searches": [{"terms": [], "file": "s.json"}]}, **{"s.json": "{}"})

        with pytest.raises(RecordedWebError, match="no terms"):
            RecordedWeb.load(tmp_path)

    def test_a_url_cannot_be_both_recorded_and_unreachable(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {"pages": {"https://a.example/x": "p.txt"}, "unreachable": ["https://a.example/x/"]},
            **{"p.txt": "text"},
        )

        with pytest.raises(RecordedWebError, match="both"):
            RecordedWeb.load(tmp_path)


def test_normalize_url_drops_what_does_not_pick_a_page() -> None:
    assert normalize_url("HTTPS://Example.COM/a/#x") == "https://example.com/a"
    assert normalize_url("https://example.com/a?q=1") != normalize_url("https://example.com/a")
