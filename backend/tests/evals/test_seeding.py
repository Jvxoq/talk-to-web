"""Seeding a task's history into its own thread, with a scripted model and no network.

The real graph, the real `GenerateReply` and the real summarize node run here.
Only the models are fakes. What the final turn's model call received is the
outcome: it shows which messages reached the thread, whether the thread key
matched, and whether anything leaked from one trial to the next.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.application.chat.models import ChatMessage, ModelChunk
from app.settings import Settings
from evals.__main__ import _sample_failures, _transcript
from evals.cases import EvalCase, load_cases
from evals.harness import CaseRun, EvalHarness
from tests.application import test_system_layer as layer
from tests.fakes import FakeChatModel

settings = layer.settings

SYSTEM_PROMPT = layer.SYSTEM_PROMPT
"""The prompt `build_use_case` gives `GenerateReply`, so the harness seeds the same one."""


class Rig:
    """A harness over the real agent graph with scripted models."""

    def __init__(self, settings: Settings, *, answers: int = 4) -> None:
        self.model = FakeChatModel(turns=[[ModelChunk(text=f"answer {n}")] for n in range(answers)])
        self.condenser_model = FakeChatModel(turns=[[ModelChunk(text="SUMMARY OF EARLIER CHAT")]])
        graph = layer.build_graph(
            settings,
            self.model,
            [],
            condenser_model=self.condenser_model,
            checkpointer=InMemorySaver(),
        )
        self.harness = EvalHarness(
            generate_reply=layer.build_use_case(settings, graph),
            judge=cast(Any, None),
            embedder=cast(Any, None),
            vector_index=cast(Any, None),
            model="test-model",
            retrieval_k=3,
            chunk_size=1,
            chunk_overlap=0,
            embedding_dimensions=1,
            _shelf=cast(Any, None),
            graph=graph,
            system_prompt=SYSTEM_PROMPT,
        )

    async def run(self, case: EvalCase) -> CaseRun:
        return await self.harness.run_case(case)


@pytest.fixture
def rig(settings: Settings) -> Rig:
    return Rig(settings)


def _task(
    history: Sequence[tuple[str, str]] = (),
    *,
    question: str = "final question",
    task_id: str = "t-1",
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": task_id,
            "suite": "tools",
            "input": question,
            "owner": "empty",
            "history": [{"role": role, "text": text} for role, text in history],
        }
    )


def _shown(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
    return [(message.role, message.content) for message in messages]


_FACT_CHAT = [("user", "my code is zeta-1"), ("assistant", "noted zeta-1")]


async def test_final_turn_sees_seeded_history(rig: Rig) -> None:
    await rig.run(_task(_FACT_CHAT, question="what is my code?"))

    [seen] = rig.model.seen_messages
    assert _shown(seen[:3]) == [("system", SYSTEM_PROMPT), *_FACT_CHAT]
    assert seen[3].role == "user"
    assert seen[3].content.startswith("what is my code?")
    assert len(seen) == 4


async def test_only_the_final_turn_calls_the_model(rig: Rig) -> None:
    history = [("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four")]

    run = await rig.run(_task(history))

    assert rig.model.calls == 1
    assert rig.condenser_model.calls == 0
    assert run.answer == "answer 0"


async def test_each_trial_starts_in_a_fresh_thread(rig: Rig) -> None:
    case = _task(_FACT_CHAT)

    await rig.run(case)
    await rig.run(case)

    first, second = rig.model.seen_messages
    assert _shown(first) == _shown(second)
    assert len(second) == 4


async def test_history_does_not_leak_into_other_tasks(rig: Rig) -> None:
    await rig.run(_task(_FACT_CHAT, task_id="with-history"))
    await rig.run(_task(question="another question", task_id="one-turn"))

    _, seen = rig.model.seen_messages
    assert [message.role for message in seen] == ["system", "user"]
    assert all("zeta-1" not in message.content for message in seen)


async def test_one_turn_task_is_not_seeded(rig: Rig) -> None:
    run = await rig.run(_task(question="just this"))

    [seen] = rig.model.seen_messages
    assert [message.role for message in seen] == ["system", "user"]
    assert run.seeded_turns == 0


async def test_run_records_how_many_turns_were_seeded(rig: Rig) -> None:
    run = await rig.run(_task(_FACT_CHAT))

    assert run.seeded_turns == 2
    assert _transcript(run).seeded_turns == 2


async def test_failed_seed_ends_the_trial_as_an_error(rig: Rig) -> None:
    class BrokenGraph:
        async def aupdate_state(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("checkpointer down")

    rig.harness.graph = cast(Any, BrokenGraph())

    run = await rig.run(_task(_FACT_CHAT))

    assert (run.error, run.seeded_turns, rig.model.calls) == ("checkpointer down", 0, 0)


def _dataset_task(task_id: str) -> EvalCase:
    [case] = [c for c in load_cases(Path("tools.jsonl")) if c.id == task_id]
    return case


async def test_long_task_fires_a_summary(rig: Rig) -> None:
    run = await rig.run(_dataset_task("tools-018-recall-fact-after-summary"))

    assert run.summarized
    assert _transcript(run).summarized
    assert rig.condenser_model.calls == 1
    assert run.seeded_turns == 16


async def test_short_task_fires_no_summary(rig: Rig) -> None:
    run = await rig.run(_task(_FACT_CHAT))

    assert not run.summarized
    assert not _transcript(run).summarized


async def test_old_fact_is_summarized_away(rig: Rig) -> None:
    await rig.run(_dataset_task("tools-018-recall-fact-after-summary"))

    [seen] = rig.model.seen_messages
    assert any("SUMMARY OF EARLIER CHAT" in m.content for m in seen)
    assert all("marlowe-7731" not in m.content for m in seen)


async def test_recent_correction_stays_verbatim(rig: Rig) -> None:
    await rig.run(_dataset_task("tools-020-recent-update-beats-summary"))

    [seen] = rig.model.seen_messages
    assert any("tavish-2058" in m.content for m in seen)
    assert all("marlowe-7731" not in m.content for m in seen)


async def test_summary_task_is_flagged_when_no_summary_fired(settings: Settings) -> None:
    """A dev's .env sets AGENT_HISTORY_TOKEN_BUDGET=50000. The 14k-token chat is
    never summarized, the model reads marlowe-7731 straight from turn 1, and
    tools-018 passes. `summarized` is stored but never graded."""
    big = settings.model_copy(update={"agent_history_token_budget": 50_000})
    model = FakeChatModel(turns=[[ModelChunk(text="marlowe-7731")]])
    graph = layer.build_graph(big, model, [], checkpointer=InMemorySaver())
    harness = EvalHarness(
        generate_reply=layer.build_use_case(big, graph),
        judge=cast(Any, None),
        embedder=cast(Any, None),
        vector_index=cast(Any, None),
        model="m",
        retrieval_k=3,
        chunk_size=1,
        chunk_overlap=0,
        embedding_dimensions=1,
        _shelf=cast(Any, None),
        graph=graph,
        system_prompt=layer.SYSTEM_PROMPT,
    )
    task = _dataset_task("tools-018-recall-fact-after-summary")

    run = await harness.run_case(task)

    assert not run.summarized
    assert _sample_failures(run, harness), "task tagged `summary` passed with no summary"


async def test_summary_task_is_not_flagged_when_summary_fired(rig: Rig) -> None:
    run = await rig.run(_dataset_task("tools-018-recall-fact-after-summary"))

    assert run.summarized
    assert not any("summary" in reason for reason in _sample_failures(run, rig.harness))
