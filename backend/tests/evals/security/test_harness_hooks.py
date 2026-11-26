"""The small hooks the security suite added to `evals.harness`."""

from typing import Any

import pytest

from app.application.chat.dto import ReplyFailed, ReplyUsage
from app.application.chat.models import ChatMessage
from app.settings import Settings
from evals import harness as harness_module
from evals.harness import (
    EMPTY_OWNER_ID,
    EVAL_OWNER_ID,
    TurnRun,
    build_harness,
    retry_transient,
)
from tests.evals.security.rig import SecurityRig, say


def _rig(settings: Settings, turns: list[Any] | None = None) -> SecurityRig:
    return SecurityRig(settings, [], turns or [say("an answer")])


# --- build_harness --------------------------------------------------------------


async def test_wrapping_the_fetcher_needs_recorded_mode() -> None:
    with pytest.raises(ValueError, match="recorded web mode only"):
        async with build_harness(web_mode="live", wrap_fetcher=lambda fetcher: fetcher):
            pytest.fail("the harness must not be built")


# --- one indexing path ----------------------------------------------------------


async def test_index_document_writes_chunks_for_one_owner(settings: Settings) -> None:
    rig = _rig(settings)
    harness = rig.harness
    harness.chunk_size, harness.chunk_overlap = 10, 2

    entry = await harness.index_document(77, 5, "notes.txt", "alpha beta gamma delta epsilon zeta")

    chunks = rig.vector_index.by_owner[77]
    assert entry is not None
    assert (entry.name, entry.owner_id, entry.id, entry.chunks_indexed) == (
        "notes.txt",
        77,
        5,
        len(chunks),
    )
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= 10
    assert {chunk.source for chunk in chunks} == {"notes.txt"}
    assert rig.embedder.embedded == [chunk.text for chunk in chunks]


async def test_blank_document_indexes_nothing(settings: Settings) -> None:
    rig = _rig(settings)

    entry = await rig.harness.index_document(77, 5, "empty.txt", "   \n ")

    assert entry is None
    assert rig.vector_index.by_owner == {}


async def test_fixtures_go_through_the_same_path(settings: Settings) -> None:
    rig = _rig(settings)

    total = await rig.harness.index_fixtures()

    shelf = rig.shelf.get(EVAL_OWNER_ID)
    assert total == len(rig.vector_index.by_owner[EVAL_OWNER_ID])
    assert total == sum(entry.chunks_indexed for entry in shelf)
    assert [entry.id for entry in shelf] == list(range(1, len(shelf) + 1))
    assert rig.shelf.get(EMPTY_OWNER_ID) == []
    assert all(entry.summary for entry in shelf)


# --- one turn runner ------------------------------------------------------------


async def test_turn_keeps_the_thread_and_summary_only_when_asked(settings: Settings) -> None:
    rig = _rig(settings, [say("first"), say("second")])
    spec: dict[str, Any] = {"owner_id": 5, "user_input": "hello", "metadata": {"task_id": "t"}}

    plain = await rig.harness.run_turn(TurnRun(), conversation_id=1, **spec)
    kept = await rig.harness.run_turn(TurnRun(), conversation_id=2, keep_thread=True, **spec)

    assert plain.thread_messages == []
    assert [m.role for m in kept.thread_messages] == ["system", "user", "assistant"]
    assert kept.thread_messages[-1].content == "second"
    assert kept.summary == ""


async def test_prepared_messages_are_seeded_before_the_turn(settings: Settings) -> None:
    rig = _rig(settings)

    async def prepare() -> list[ChatMessage]:
        return [
            ChatMessage(role="user", content="earlier"),
            ChatMessage(role="assistant", content="noted"),
        ]

    run = await rig.harness.run_turn(
        TurnRun(), owner_id=5, conversation_id=1, user_input="now", metadata={}, prepare=prepare
    )

    assert run.seeded_turns == 2
    assert [m.content for m in rig.model.seen_messages[0][:3]] == [
        rig.harness.system_prompt,
        "earlier",
        "noted",
    ]


async def test_failed_preparation_is_an_error_not_a_crash(settings: Settings) -> None:
    rig = _rig(settings)

    async def prepare() -> list[ChatMessage]:
        raise RuntimeError("embedding service down")

    run = await rig.harness.run_turn(
        TurnRun(), owner_id=5, conversation_id=1, user_input="now", metadata={}, prepare=prepare
    )

    assert (run.error, run.seeded_turns, rig.model.calls) == ("embedding service down", 0, 0)
    assert run.broke


# --- retry ----------------------------------------------------------------------


async def test_retry_runs_a_fresh_attempt_and_carries_the_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness_module, "_RETRY_BACKOFF_SECONDS", 0.0)
    limited = TurnRun(failed=ReplyFailed(code="overloaded"))
    limited.usage = ReplyUsage(prompt_tokens=40, completion_tokens=4, elapsed_ms=5, model="m")
    good = TurnRun()
    attempts = [limited, good]

    async def attempt() -> TurnRun:
        return attempts.pop(0)

    run = await retry_transient(attempt, label="t")

    assert run is good
    assert (run.attempts, run.retried_tokens_in, run.retried_tokens_out) == (2, 40, 4)


async def test_a_good_attempt_is_not_retried() -> None:
    calls: list[int] = []

    async def attempt() -> TurnRun:
        calls.append(1)
        return TurnRun()

    run = await retry_transient(attempt, label="t")

    assert (len(calls), run.attempts) == (1, 1)
