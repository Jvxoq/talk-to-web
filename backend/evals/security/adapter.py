"""The adapter for the talk-to-web chat agent: how a task becomes a real turn.

The core of this package knows nothing about the agent. This module is the one
place that does: which tools it has, how its web fetcher works, how its threads
and uploads are stored, and how to read a transcript back out of a finished
thread. A second agent gets a second adapter and reuses the rest.

Everything here runs offline. A page is served from the task file through the
production text extractor. Any other URL answers like a fetch that failed.
Search goes through the recorded search client, which has no results for an
unknown query. No trial reaches the network.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any, cast

from app.adapters.llm.approximate_token_counter import ApproximateTokenCounter
from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.application.chat.models import ChatMessage, ToolCall
from app.application.chat.ports import WebContentFetcher
from app.application.chat.tools.base import ToolContext, ToolRegistry
from app.application.chat.tools.fetch_web_pages import FetchWebPages
from app.application.chat.tools.links import link_key
from evals.cases import HistoryTurn, expanded_history
from evals.harness import (
    EvalHarness,
    TurnRun,
    last_user_index,
    new_thread_id,
    retry_transient,
)
from evals.metrics.tools import tools_completed
from evals.recorded_web import normalize_url
from evals.security.graders import Grade, TrialView, grade_trial
from evals.security.report import (
    JudgeRecord,
    ToolCallRecord,
    ToolResultRecord,
    TranscriptRecord,
    TrialRecord,
)
from evals.security.tasks import Task

_trial_owners = count(start=2_000_000_000)
"""Account ids, one per trial attempt. Far from the main suite's two fixed owners."""

_document_ids = count(start=1)

_FILLER_USER = "Let's talk about something else for a moment."
_FILLER_ASSISTANT = "Sure, happy to chat about anything you like."
_FILLER_CHARS = 2_400
_MAX_FILLER_PAIRS = 500


class TaskPageFetcher(AiohttpWebContentFetcher):
    """The production fetcher with the network cut out and the task pages put in.

    It inherits `fetch_all`, so the join, the per-page cap and the dropping of a
    page that gave no text are production's own code. Only `_fetch`, the one
    method that opens a socket, is replaced. A URL that is a task page goes
    through `AiohttpWebContentFetcher._parse`, the production extractor. Every
    other URL gives `""`, exactly like a fetch that failed.
    """

    def __init__(self, pages: Mapping[str, str]) -> None:
        super().__init__(session=cast(Any, None))
        self._task_pages = {normalize_url(url): html for url, html in pages.items()}

    async def _fetch(self, url: str) -> str:
        html = self._task_pages.get(normalize_url(url))
        return self._parse(html, url) if html is not None else ""

    async def aclose(self) -> None:
        return None


def task_pages(tasks: Sequence[Task]) -> dict[str, str]:
    """Every page the tasks can reach: each `page.url` and each history fetch.

    Raises when one URL is given two different pages, because which one a trial
    saw would then depend on the order the tasks were loaded.
    """
    pages: dict[str, str] = {}

    def add(url: str, html: str) -> None:
        key = normalize_url(url)
        if pages.setdefault(key, html) != html:
            raise ValueError(f"two tasks give {url!r} different pages")

    for task in tasks:
        if task.page is not None:
            add(task.page.url, task.page.html)
        for item in task.history:
            if item.role == "tool_fetch" and item.url is not None and item.html is not None:
                add(item.url, item.html)
    return pages


def page_wrapper(tasks: Sequence[Task]) -> Callable[[WebContentFetcher], WebContentFetcher]:
    """The `wrap_fetcher` argument of `build_harness`: serve the tasks' pages, nothing else."""
    pages = task_pages(tasks)
    return lambda _recorded: TaskPageFetcher(pages)


def pad_with_filler(
    messages: Sequence[ChatMessage], *, system_prompt: str, target_tokens: int
) -> list[ChatMessage]:
    """`messages` plus filler turns until the thread holds `target_tokens` tokens.

    The filler goes after the seeded turns, so what the task planted sits at the
    start of the chat, where the summary will cover it. It is the main suite's
    own filler, plain chatter with no digit, name or code. The thread counts
    with its system prompt, as it will in the checkpointer, and ends on an
    assistant turn so the final input is the next user turn.
    """
    counter = ApproximateTokenCounter()
    system = ChatMessage(role="system", content=system_prompt)
    for pairs in range(_MAX_FILLER_PAIRS + 1):
        turns = [
            HistoryTurn(role=role, text=text, pad_chars=_FILLER_CHARS)
            for _ in range(pairs)
            for role, text in (("user", _FILLER_USER), ("assistant", _FILLER_ASSISTANT))
        ]
        padded = [
            *messages,
            *(ChatMessage(role=role, content=text) for role, text in expanded_history(turns)),
        ]
        if counter.count([system, *padded]) >= target_tokens:
            return padded
    raise ValueError(f"filler could not reach {target_tokens} tokens")


def tool_calls_since_last_user(messages: Sequence[ChatMessage]) -> tuple[tuple[str, str], ...]:
    """Every tool call of the final turn as (name, full argument JSON text).

    Read from the assistant messages in the thread, not from the shortened text
    the reply stream shows the user: a leak at the end of a long argument must
    not be cut off before a grader sees it. A call the routing policy refused
    is included, because the model asked for it.
    """
    calls: list[tuple[str, str]] = []
    for message in messages[last_user_index(messages) :]:
        calls.extend(
            (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
            for call in message.tool_calls
        )
    return tuple(calls)


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One graded trial: the record for the results file and the grade for the metrics."""

    task: Task
    trial: int
    grade: Grade
    record: TrialRecord


class SecurityAdapter:
    """Runs security tasks against one built `EvalHarness`."""

    def __init__(self, harness: EvalHarness) -> None:
        if harness.registry is None:
            raise ValueError("the harness has no tool registry to read tool results from")
        self._harness = harness
        self._registry: ToolRegistry = harness.registry

    async def prepare_store(self) -> None:
        """Make the scratch vector collection ready, once per run, before any trial."""
        await self._harness.vector_index.ensure(self._harness.embedding_dimensions)

    async def run_trial(self, task: Task, trial: int, *, judges: bool) -> TrialResult:
        """Run one trial and grade it. Never raises: a broken trial is an errored one."""
        run = await retry_transient(lambda: self._attempt(task, trial), label=task.id)
        verdicts = await self._judge(task, run) if judges else {}
        view = self.view_of(run, verdicts)
        grade = grade_trial(task, view)
        return TrialResult(task, trial, grade, self._record(task, trial, run, verdicts, grade))

    async def _attempt(self, task: Task, trial: int) -> TurnRun:
        # A fresh account and a fresh thread on every attempt, so nothing a trial
        # saw or uploaded can be seen by another, and a retried trial starts clean.
        owner_id = next(_trial_owners)
        conversation_id = new_thread_id()

        async def prepare() -> list[ChatMessage]:
            await self._set_up_account(task, owner_id)
            return await self._seed_messages(task, owner_id, conversation_id)

        return await self._harness.run_turn(
            TurnRun(),
            owner_id=owner_id,
            conversation_id=conversation_id,
            user_input=task.input,
            metadata={"task_id": task.id, "suite": "security", "line": task.line, "trial": trial},
            prepare=prepare,
            keep_thread=True,
        )

    async def _set_up_account(self, task: Task, owner_id: int) -> None:
        """Index the task's one file for this owner, or tell the reply path it has none.

        An empty list is an answer ("no files"), and it is not the same as saying
        nothing: `GenerateReply` reads a missing answer as unknown and leaves the
        document tool open.
        """
        if task.document is None:
            self._harness.shelve(owner_id, [])
            return
        entry = await self._harness.index_document(
            owner_id, next(_document_ids), task.document.filename, task.document.text
        )
        self._harness.shelve(owner_id, [entry] if entry is not None else [])

    async def _seed_messages(
        self, task: Task, owner_id: int, conversation_id: int
    ) -> list[ChatMessage]:
        """What production would have stored for the task's earlier turns, then filler."""
        messages: list[ChatMessage] = []
        for index, item in enumerate(task.history):
            if item.role != "tool_fetch":
                messages.append(ChatMessage(role=item.role, content=item.content or ""))
                continue
            if item.url is None:
                raise RuntimeError(f"task {task.id}: a tool_fetch history item has no url")
            call = ToolCall(
                id=f"seed-fetch-{index}",
                name=FetchWebPages.name,
                arguments={"urls": [item.url]},
            )
            # The real registry and the real fence, not a hand-written copy of the
            # fence: if the fence changes, the seeded history changes with it.
            outcome = await self._registry.invoke(
                call,
                ToolContext(
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                    document_scoped=False,
                    has_documents=False,
                    # The seeded turn stands for one where the user named this
                    # page, so the fetch guard lets it through.
                    openable_links=frozenset({link_key(item.url)}),
                ),
            )
            # `sources` is set only when the tool read something. A page the fetcher
            # does not know comes back as a polite "could not read" with `ok` true.
            if not outcome.ok or not outcome.sources:
                raise RuntimeError(f"seeding the page {item.url} failed: {outcome.content}")
            messages.append(ChatMessage(role="assistant", tool_calls=(call,)))
            messages.append(ChatMessage(role="tool", content=outcome.content, tool_call_id=call.id))
        if task.pad_history_to_tokens:
            return pad_with_filler(
                messages,
                system_prompt=self._harness.system_prompt,
                target_tokens=task.pad_history_to_tokens,
            )
        return messages

    async def _judge(self, task: Task, run: TurnRun) -> dict[str, JudgeRecord]:
        """Ask the task's one judge, if it has one. A failed call is an unknown verdict."""
        name = task.expect.judge
        if name is None or run.broke:
            return {}
        sources = f"HARMFUL GOAL: {task.goal}" if name == "harmful_compliance" else ""
        result = await self._harness.judge.verdict(
            name, question=task.input, answer=run.answer, sources=sources
        )
        if result is None:
            return {name: JudgeRecord(judge=name, verdict=None, reason="no verdict")}
        return {name: JudgeRecord(judge=name, verdict=result.verdict, reason=result.reason)}

    def view_of(self, run: TurnRun, verdicts: Mapping[str, JudgeRecord]) -> TrialView:
        """The plain data the graders read, from a finished trial."""
        return TrialView(
            answer=run.answer,
            tool_calls=tool_calls_since_last_user(run.thread_messages),
            tools_completed=tuple(tools_completed(run.events)),
            # From the state's own field. The summarize node's message format is
            # not parsed, so a change to it cannot change what is graded.
            summary_text=run.summary or None,
            summarized=run.summarized,
            judge_verdicts={name: record.verdict for name, record in verdicts.items()},
            errored=run.broke,
            error=_error_text(run),
            system_prompt=self._harness.system_prompt,
        )

    def _record(
        self,
        task: Task,
        trial: int,
        run: TurnRun,
        verdicts: Mapping[str, JudgeRecord],
        grade: Grade,
    ) -> TrialRecord:
        return TrialRecord(
            task_id=task.id,
            trial=trial,
            outcome=grade.outcome,
            trace_id=run.trace_id,
            question=task.input,
            transcript=TranscriptRecord(
                answer=run.answer,
                tool_calls=tuple(
                    ToolCallRecord(name=name, arguments=arguments)
                    for name, arguments in tool_calls_since_last_user(run.thread_messages)
                ),
                tool_results=tuple(
                    ToolResultRecord(tool=r.tool, text=r.text) for r in run.tool_results
                ),
                summary_text=run.summary or None,
                error=_error_text(run) or None,
            ),
            judges=tuple(verdicts.values()),
            attack_succeeded=grade.attack_succeeded,
            utility_ok=grade.utility_ok,
            failure_reasons=grade.reasons,
            tokens_in=(run.usage.prompt_tokens if run.usage else 0) + run.retried_tokens_in,
            tokens_out=(run.usage.completion_tokens if run.usage else 0) + run.retried_tokens_out,
            latency_ms=run.latency_ms,
        )


def _error_text(run: TurnRun) -> str:
    if run.error is not None:
        return run.error
    return run.failed.code if run.failed is not None else ""
