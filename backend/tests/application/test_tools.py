"""The three tools, against fake ports.

Three things are being proved throughout: a tool passes its port's answer
back verbatim, a tool never raises - a failure comes back as text the model
can read, because the response body is already streaming by the time a tool
runs - and every tool reports the sources behind a successful answer.
"""

import pytest

from app.application.chat.models import Passage, Source
from app.application.chat.tools.base import ToolContext
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from tests.fakes import FakeKnowledgeRetriever, FakeWebContentFetcher, FakeWebSearcher

# Who the turn belongs to. The model never supplies this - it arrives from the
# run config - so every call here passes the same one.
CONTEXT = ToolContext(owner_id=42)


class TestRetrieveDocuments:
    async def test_returns_the_passages_the_retriever_found(self) -> None:
        tool = RetrieveDocuments(
            FakeKnowledgeRetriever(
                passages=[
                    Passage(text="first", source="a.pdf"),
                    Passage(text="second", source="a.pdf"),
                ]
            )
        )

        outcome = await tool.run({"query": "what is in my report?"}, CONTEXT)

        assert outcome.ok
        assert "first" in outcome.content
        assert "second" in outcome.content

    async def test_cites_every_distinct_document_once(self) -> None:
        tool = RetrieveDocuments(
            FakeKnowledgeRetriever(
                passages=[
                    Passage(text="first", source="a.pdf"),
                    Passage(text="second", source="a.pdf"),
                    Passage(text="third", source="b.pdf"),
                ]
            )
        )

        outcome = await tool.run({"query": "what is in my report?"}, CONTEXT)

        assert outcome.sources == (Source(label="a.pdf"), Source(label="b.pdf"))

    async def test_says_so_rather_than_returning_nothing_when_no_passage_matches(self) -> None:
        tool = RetrieveDocuments(FakeKnowledgeRetriever(passages=[]))

        outcome = await tool.run({"query": "unindexed"}, CONTEXT)

        # An empty result reads to the model as a broken tool and provokes a
        # retry; a sentence tells it to answer from what it already knows.
        assert outcome.ok
        assert outcome.content.strip()
        assert "unindexed" in outcome.content
        assert outcome.sources == ()

    async def test_a_failing_retriever_is_reported_not_raised(self) -> None:
        tool = RetrieveDocuments(FakeKnowledgeRetriever(fail_with=RuntimeError("qdrant down")))

        outcome = await tool.run({"query": "anything"}, CONTEXT)

        assert not outcome.ok
        assert "retrieve_documents" in outcome.content
        assert outcome.sources == ()

    async def test_missing_query_is_reported_back_to_the_model(self) -> None:
        tool = RetrieveDocuments(FakeKnowledgeRetriever())

        outcome = await tool.run({}, CONTEXT)

        assert not outcome.ok
        assert "query" in outcome.content

    def test_schema_names_the_argument_the_model_must_supply(self) -> None:
        spec = RetrieveDocuments(FakeKnowledgeRetriever()).spec

        assert spec.name == "retrieve_documents"
        assert set(spec.parameters["properties"]) == {"query"}
        assert spec.parameters["required"] == ["query"]


class TestFetchWebPages:
    async def test_returns_the_content_the_fetcher_read(self) -> None:
        fetcher = FakeWebContentFetcher(content="PAGE TEXT")
        tool = FetchWebPages(fetcher)

        outcome = await tool.run({"urls": ["https://example.com/a"]}, CONTEXT)

        assert outcome.ok
        assert outcome.content == "PAGE TEXT"
        # Converted back to plain strings at the boundary: the port takes str.
        assert fetcher.calls == [("https://example.com/a",)]
        assert all(isinstance(url, str) for url in fetcher.calls[0])

    async def test_cites_every_url_it_was_asked_to_read(self) -> None:
        fetcher = FakeWebContentFetcher(content="PAGE TEXT")
        tool = FetchWebPages(fetcher)

        outcome = await tool.run(
            {"urls": ["https://example.com/a", "https://example.com/b"]}, CONTEXT
        )

        assert outcome.sources == (
            Source(label="https://example.com/a", url="https://example.com/a"),
            Source(label="https://example.com/b", url="https://example.com/b"),
        )

    async def test_rejects_a_hallucinated_url_before_opening_a_socket(self) -> None:
        fetcher = FakeWebContentFetcher(content="PAGE TEXT")
        tool = FetchWebPages(fetcher)

        outcome = await tool.run({"urls": ["not a url at all"]}, CONTEXT)

        assert not outcome.ok
        # The message names the field so the model knows what to correct.
        assert "urls" in outcome.content
        assert fetcher.calls == []

    async def test_says_so_when_no_page_yielded_text(self) -> None:
        tool = FetchWebPages(FakeWebContentFetcher(content="   "))

        outcome = await tool.run({"urls": ["https://example.com/empty"]}, CONTEXT)

        assert outcome.ok
        assert "https://example.com/empty" in outcome.content
        assert outcome.sources == ()

    async def test_a_failing_fetcher_is_reported_not_raised(self) -> None:
        tool = FetchWebPages(FakeWebContentFetcher(fail_with=TimeoutError("slow")))

        outcome = await tool.run({"urls": ["https://example.com"]}, CONTEXT)

        assert not outcome.ok
        assert "fetch_web_pages" in outcome.content

    def test_schema_names_the_argument_the_model_must_supply(self) -> None:
        spec = FetchWebPages(FakeWebContentFetcher()).spec

        assert spec.name == "fetch_web_pages"
        assert set(spec.parameters["properties"]) == {"urls"}
        assert spec.parameters["properties"]["urls"]["type"] == "array"


class TestSearchWeb:
    async def test_returns_what_the_searcher_found_using_the_configured_limit(self) -> None:
        searcher = FakeWebSearcher(results="RESULTS")
        tool = SearchWeb(searcher, max_results=7)

        outcome = await tool.run({"query": "who won yesterday"}, CONTEXT)

        assert outcome.ok
        assert outcome.content == "RESULTS"
        # max_results comes from settings, not from the model: how much a search
        # costs is an operator's decision.
        assert searcher.queries == [("who won yesterday", 7)]

    async def test_returns_the_sources_the_searcher_found(self) -> None:
        sources = (Source(label="Result", url="https://example.com/result"),)
        tool = SearchWeb(FakeWebSearcher(results="RESULTS", sources=sources), max_results=3)

        outcome = await tool.run({"query": "who won yesterday"}, CONTEXT)

        assert outcome.sources == sources

    async def test_says_so_rather_than_returning_nothing_on_an_empty_search(self) -> None:
        tool = SearchWeb(FakeWebSearcher(results=""), max_results=3)

        outcome = await tool.run({"query": "obscure"}, CONTEXT)

        assert outcome.ok
        assert "obscure" in outcome.content
        assert outcome.sources == ()

    async def test_a_failing_searcher_is_reported_not_raised(self) -> None:
        tool = SearchWeb(FakeWebSearcher(fail_with=RuntimeError("429")), max_results=3)

        outcome = await tool.run({"query": "anything"}, CONTEXT)

        assert not outcome.ok
        assert "search_web" in outcome.content

    def test_schema_names_the_argument_the_model_must_supply(self) -> None:
        spec = SearchWeb(FakeWebSearcher(), max_results=3).spec

        assert spec.name == "search_web"
        assert set(spec.parameters["properties"]) == {"query"}
        # The model must not be handed the cost dial.
        assert "max_results" not in spec.parameters["properties"]


class TestDescriptions:
    """The description is the only thing telling the model which tool to pick."""

    @pytest.mark.parametrize(
        "tool",
        [
            RetrieveDocuments(FakeKnowledgeRetriever()),
            FetchWebPages(FakeWebContentFetcher()),
            SearchWeb(FakeWebSearcher(), max_results=3),
        ],
    )
    def test_every_tool_explains_itself(
        self, tool: RetrieveDocuments | FetchWebPages | SearchWeb
    ) -> None:
        assert len(tool.spec.description) > 80
