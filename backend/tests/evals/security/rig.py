"""A security harness over the real agent graph, with scripted models and fakes.

The real graph, the real tool registry, the real `GenerateReply` and the real
summarize node run. The agent model and the condenser are scripts. The page
fetcher is the adapter's own. The embedder and the vector store are fakes, so
no Qdrant server and no network.
"""

from collections.abc import Sequence
from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver

from app.adapters.llm.approximate_token_counter import ApproximateTokenCounter
from app.adapters.web.tavily_search import TavilyWebSearcher
from app.application.chat.agent.graph import build_agent_graph
from app.application.chat.models import ModelChunk, Passage, ToolCall
from app.application.chat.tools.base import ToolRegistry, ToolRoutingPolicy
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.retrieve_documents import RetrieveDocuments
from app.application.chat.tools.search_web import SearchWeb
from app.settings import Settings
from evals.harness import EvalHarness, Shelf, shelf_uow_factory
from evals.judge import JudgeName, JudgeVerdict
from evals.recorded_web import RecordedSearchClient, RecordedWeb
from evals.security.adapter import SecurityAdapter, TaskPageFetcher, task_pages
from evals.security.tasks import Task
from tests.application import test_system_layer as layer
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeKnowledgeRetriever,
    FakeVectorIndex,
    RecordingTracer,
)

SUMMARY_TEXT = "SUMMARY OF EARLIER CHAT"


class FakeJudge:
    """Answers each judge question from a script and records what it was asked."""

    model_name = "fake-judge"
    prompt_tokens = 0
    completion_tokens = 0

    def __init__(self, verdicts: dict[str, bool | None] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.calls: list[tuple[str, str, str]] = []

    async def verdict(
        self, judge: JudgeName, *, question: str, answer: str, sources: str
    ) -> JudgeVerdict | None:
        self.calls.append((judge, answer, sources))
        verdict = self.verdicts.get(judge)
        return (
            None if verdict is None else JudgeVerdict(judge=judge, verdict=verdict, reason="fake")
        )


def call(name: str, **arguments: Any) -> list[ModelChunk]:
    """One scripted model turn that asks for one tool."""
    return [ModelChunk(tool_calls=(ToolCall(id=f"c-{name}", name=name, arguments=arguments),))]


def say(text: str) -> list[ModelChunk]:
    """One scripted model turn that answers in text."""
    return [ModelChunk(text=text)]


class SecurityRig:
    """One adapter and everything behind it, for a list of scripted agent turns."""

    def __init__(
        self,
        settings: Settings,
        tasks: Sequence[Task],
        agent_turns: Sequence[Sequence[ModelChunk]],
        *,
        summary: str = SUMMARY_TEXT,
        judge: FakeJudge | None = None,
        passages: Sequence[Passage] = (),
    ) -> None:
        self.model = FakeChatModel(turns=agent_turns)
        self.condenser_model = FakeChatModel(turns=[[ModelChunk(text=summary)]])
        self.retriever = FakeKnowledgeRetriever(passages=passages)
        self.embedder = FakeEmbedder()
        self.vector_index = FakeVectorIndex()
        self.shelf = Shelf()
        self.judge = judge or FakeJudge()
        self.searcher = TavilyWebSearcher(
            client=cast(Any, RecordedSearchClient(RecordedWeb({}, frozenset(), [])))
        )
        self.registry = ToolRegistry(
            [
                RetrieveDocuments(self.retriever),
                FetchWebPages(TaskPageFetcher(task_pages(tasks))),
                SearchWeb(self.searcher, max_results=3),
            ],
            guard=layer.make_guard(),
            routing=ToolRoutingPolicy(
                document_tool=RetrieveDocuments.name, web_search_tool=SearchWeb.name
            ),
        )
        graph = build_agent_graph(
            model=self.model,
            tools=self.registry,
            max_iterations=settings.agent_max_tool_iterations,
            checkpointer=InMemorySaver(),
            condenser=layer.make_condenser(self.condenser_model),
            counter=ApproximateTokenCounter(),
            history_token_budget=settings.agent_history_token_budget,
            recent_token_budget=settings.agent_recent_token_budget,
            tool_output_token_budget=settings.agent_tool_output_token_budget,
            max_request_tokens=settings.agent_max_request_tokens,
            tracer=RecordingTracer(),
        )
        self.harness = EvalHarness(
            generate_reply=layer.build_use_case(
                settings, graph, cast(Any, shelf_uow_factory(self.shelf))
            ),
            judge=cast(Any, self.judge),
            embedder=cast(Any, self.embedder),
            vector_index=cast(Any, self.vector_index),
            model="test-model",
            retrieval_k=3,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_dimensions=3,
            _shelf=self.shelf,
            graph=graph,
            system_prompt=layer.SYSTEM_PROMPT,
            registry=self.registry,
        )
        self.adapter = SecurityAdapter(self.harness)
