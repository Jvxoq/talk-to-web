"""The hand-label loop: which traces get queued, what a label row holds, how agreement is counted.

`evals.labels` is the only place a person's answer meets a judge's verdict, so
a wrong row here quietly corrupts every agreement number after it. These tests
use an in-memory store, so nothing talks to LangSmith. The one LangSmith-facing
class is tested against a mock client, to pin the calls it makes.
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import langsmith
import pytest

from evals.judge import QUESTIONS, JudgeName
from evals.labels import (
    HumanLabel,
    LabelRow,
    LangSmithLabelStore,
    _as_yes,
    agreement,
    label_tasks,
    main,
    merge_rows,
    pick,
    pull,
    queue,
    read_rows,
    render_agreement,
    write_rows,
)


class FakeLabelStore:
    """A `LabelStore` in memory. The test pre-fills `stored` with what a person labeled.

    Queue ids are `q-<judge>`, so a test reads `added` by judge name.
    """

    def __init__(self) -> None:
        self.ensured: list[str] = []
        self.added: dict[str, list[str]] = {}
        self.stored: dict[str, dict[str, HumanLabel]] = {}

    def ensure_queue(self, judge: JudgeName) -> str:
        queue_id = f"q-{judge}"
        self.ensured.append(queue_id)
        return queue_id

    def add(self, queue_id: str, trace_ids: Sequence[str]) -> None:
        self.added.setdefault(queue_id, []).extend(trace_ids)

    def labels(self, judge: JudgeName, trace_ids: Sequence[str]) -> dict[str, HumanLabel]:
        known = self.stored.get(judge, {})
        return {trace: known[trace] for trace in trace_ids if trace in known}


def _judge(name: str, verdict: bool | None, reason: str = "because") -> dict[str, Any]:
    return {"judge": name, "verdict": verdict, "reason": reason}


def _trial(
    task_id: str,
    trial: int,
    trace_id: str | None,
    *,
    outcome: str = "passed",
    judges: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "trial": trial,
        "outcome": outcome,
        "question": "What is the fee?",
        "trace_id": trace_id,
        "judges": [_judge("grounded", True), _judge("eligible", True)]
        if judges is None
        else judges,
        "transcript": {"answer": "It is 5 euros.", "tool_results": tool_results or []},
    }


TASKS = [
    {"task_id": "t-b1", "behavior": "b1"},
    {"task_id": "t-b6", "behavior": "b6"},
    {"task_id": "t-b3", "behavior": "b3"},
    {"task_id": "t-a", "behavior": "b1"},
    {"task_id": "t-b", "behavior": "b1"},
]


def _results(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "model": "m-1",
        "prompt_version": "pv-1",
        "tasks": TASKS,
        "trials": trials,
    }


def _row(trace_id: str, judge: str, human: bool, verdict: bool | None, **extra: Any) -> LabelRow:
    fields: dict[str, Any] = {
        "trace_id": trace_id,
        "judge": judge,
        "human": human,
        "judge_verdict": verdict,
        "judge_reason": "reason",
        "comment": "",
        "labeled_at": "2026-01-01T00:00:00",
        "task_id": "t-b1",
        "trial": 1,
        "question": "q?",
        "answer": "a",
        "sources": "",
        "run_id": "run-1",
        "model": "m-1",
        "prompt_version": "pv-1",
    }
    return LabelRow(**(fields | extra))


def _write_results(path: Path, results: dict[str, Any]) -> Path:
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def test_label_tasks_gives_one_task_per_judge_for_a_b1_trial() -> None:
    """A b1 answer is read by two judges, so one trial becomes two tasks to label."""
    tasks = label_tasks(_results([_trial("t-b1", 1, "tr-1")]))

    assert sorted(task.judge for task in tasks) == ["eligible", "grounded"]
    assert {task.trace_id for task in tasks} == {"tr-1"}


def test_label_tasks_skips_errored_trial_and_missing_trace() -> None:
    """An errored trial has no answer to label, and a trial with no trace cannot be opened."""
    results = _results(
        [
            _trial("t-b1", 1, "tr-err", outcome="errored"),
            _trial("t-b1", 2, None),
        ]
    )

    assert label_tasks(results) == []


def test_label_tasks_gives_nothing_for_a_b3_task() -> None:
    """B3 is graded by code alone, so no judge reads it and nobody needs to label it."""
    assert label_tasks(_results([_trial("t-b3", 1, "tr-3")])) == []


def test_label_tasks_joins_sources_with_tool_names() -> None:
    """The judge reads the tool results as one text, each under its tool's name."""
    trial = _trial(
        "t-b1",
        1,
        "tr-1",
        tool_results=[
            {"tool": "retrieve_documents", "text": "doc A"},
            {"tool": "search_web", "text": "page B"},
        ],
    )

    tasks = label_tasks(_results([trial]))

    assert tasks[0].sources == "[retrieve_documents]\ndoc A\n\n[search_web]\npage B"


def test_label_tasks_has_no_verdict_when_the_judge_record_is_missing() -> None:
    """A judge call that failed leaves no record. The task then has no verdict, not a "no"."""
    tasks = label_tasks(_results([_trial("t-b1", 1, "tr-1", judges=[])]))

    assert tasks
    assert all(task.judge_verdict is None and task.judge_reason == "" for task in tasks)


def test_pick_puts_failed_trials_before_passed() -> None:
    """Failed trials are the ones worth a person's time, so they fill the limit first."""
    tasks = label_tasks(
        _results(
            [
                _trial("t-b1", 1, "tr-passed", outcome="passed"),
                _trial("t-b1", 2, "tr-failed", outcome="failed"),
            ]
        )
    )

    chosen = pick(tasks, limit=1)

    assert {task.trace_id for task in chosen} == {"tr-failed"}


def test_pick_counts_traces_not_judge_rows() -> None:
    """A b1 trace makes two judge rows. The limit of one still keeps both rows of that trace."""
    tasks = label_tasks(_results([_trial("t-b1", 1, "tr-1"), _trial("t-b1", 2, "tr-2")]))

    chosen = pick(tasks, limit=1)

    assert len(chosen) == 2
    assert {task.trace_id for task in chosen} == {"tr-1"}
    assert {task.judge for task in chosen} == {"grounded", "eligible"}


def test_pick_takes_trial_one_of_every_task_before_trial_two() -> None:
    """A sample should spread across tasks, not use up two trials of the first task."""
    tasks = label_tasks(
        _results(
            [
                _trial("t-a", 1, "tr-a1"),
                _trial("t-a", 2, "tr-a2"),
                _trial("t-b", 1, "tr-b1"),
                _trial("t-b", 2, "tr-b2"),
            ]
        )
    )

    chosen = pick(tasks, limit=2)

    assert {task.trace_id for task in chosen} == {"tr-a1", "tr-b1"}


def test_queue_sends_each_judge_only_its_traces() -> None:
    """The grounded queue gets only b1 traces. The eligible queue also gets the b6 trace."""
    store = FakeLabelStore()
    results = _results(
        [
            _trial("t-b1", 1, "tr-1"),
            _trial("t-b6", 1, "tr-6", judges=[_judge("eligible", True)]),
            _trial("t-b1", 2, "tr-2"),
        ]
    )

    counts = queue(results, store, limit=100)

    assert counts == {"grounded": 2, "eligible": 3}
    assert sorted(store.added["q-grounded"]) == ["tr-1", "tr-2"]
    assert sorted(store.added["q-eligible"]) == ["tr-1", "tr-2", "tr-6"]


def test_pull_leaves_out_unlabeled_traces_and_fills_the_row() -> None:
    """Only labeled traces become rows, and each row holds the label next to the judge's verdict."""
    store = FakeLabelStore()
    store.stored["grounded"] = {
        "tr-1": HumanLabel(yes=True, comment="fine", labeled_at="2026-01-02T10:00:00")
    }
    results = _results(
        [
            _trial(
                "t-b1",
                1,
                "tr-1",
                judges=[_judge("grounded", True, "cites the page"), _judge("eligible", True)],
                tool_results=[{"tool": "retrieve_documents", "text": "doc A"}],
            ),
            _trial("t-b1", 2, "tr-2"),
        ]
    )

    rows = pull(results, store)

    assert len(rows) == 1
    row = rows[0]
    assert (row.trace_id, row.judge, row.human, row.judge_verdict) == (
        "tr-1",
        "grounded",
        True,
        True,
    )
    assert row.judge_reason == "cites the page"
    assert row.run_id == "run-1"
    assert row.model == "m-1"
    assert row.question == "What is the fee?"
    assert row.answer == "It is 5 euros."
    assert row.sources == "[retrieve_documents]\ndoc A"
    assert row.comment == "fine"


def test_merge_rows_lets_a_new_label_replace_the_old_one() -> None:
    """A relabel in LangSmith must replace the stored row, not add a second one."""
    old = [_row("tr-1", "grounded", human=False, verdict=False)]
    new = [_row("tr-1", "grounded", human=True, verdict=False)]

    merged = merge_rows(old, new)

    assert len(merged) == 1
    assert merged[0].human is True


def test_write_rows_then_read_rows_round_trips(tmp_path: Path) -> None:
    """The labels file must read back to the same rows, or the next run loses labels."""
    rows = [
        _row("tr-1", "grounded", human=True, verdict=True, comment="ok"),
        _row("tr-2", "eligible", human=False, verdict=None),
    ]
    path = tmp_path / "labels.jsonl"

    write_rows(path, rows)

    assert read_rows(path) == rows


def test_agreement_counts_the_four_cells_and_both_rates() -> None:
    """Two true yes, one true no, one false yes, no false no: TPR 1.0, TNR 0.5, 4 compared."""
    rows = [
        _row("a", "grounded", human=True, verdict=True),
        _row("b", "grounded", human=True, verdict=True),
        _row("c", "grounded", human=False, verdict=False),
        _row("d", "grounded", human=False, verdict=True),
    ]

    result = {a.judge: a for a in agreement(rows)}["grounded"]

    assert (result.true_pos, result.true_neg, result.false_pos, result.false_neg) == (2, 1, 1, 0)
    assert result.tpr == 1.0
    assert result.tnr == 0.5
    assert result.compared == 4


def test_agreement_counts_a_missing_verdict_as_labeled_not_compared() -> None:
    """A failed judge call has a label but no verdict to agree with, so it is not compared."""
    rows = [
        _row("a", "grounded", human=True, verdict=True),
        _row("b", "grounded", human=True, verdict=None),
    ]

    result = {a.judge: a for a in agreement(rows)}["grounded"]

    assert result.labeled == 2
    assert result.compared == 1


def test_agreement_gives_no_rates_for_a_judge_without_labels() -> None:
    """With no labels, a rate would be a divide by zero. It must read as n/a, which is None."""
    result = {a.judge: a for a in agreement([])}["eligible"]

    assert result.labeled == 0
    assert result.tpr is None
    assert result.tnr is None


def test_render_agreement_names_the_task_and_trace_of_a_disagreement() -> None:
    """A disagreement line must name the task and trace, so a person can open the trace."""
    rows = [_row("tr-9", "grounded", human=False, verdict=True, task_id="t-fee", trial=2)]

    text = render_agreement(agreement(rows), rows)

    assert "t-fee trial 2" in text
    assert "tr-9" in text
    assert "judge said yes, person said no" in text


@pytest.mark.parametrize(
    ("score", "value", "expected"),
    [
        (1.0, None, True),
        (0, None, False),
        (None, "No", False),
        (None, None, None),
    ],
)
def test_as_yes_reads_score_or_value(score: object, value: object, expected: bool | None) -> None:
    """A score of 0.5 or more is yes. A text value counts only if it says yes or no."""
    assert _as_yes(score, value) is expected


def test_main_pull_writes_labels_file(tmp_path: Path) -> None:
    """`pull` saves the labels to the file and exits 0."""
    store = FakeLabelStore()
    store.stored["grounded"] = {"tr-1": HumanLabel(yes=True, comment="", labeled_at="")}
    results_path = _write_results(tmp_path / "results.json", _results([_trial("t-b1", 1, "tr-1")]))
    labels_path = tmp_path / "labels.jsonl"

    code = main(["pull", "--results", str(results_path), "--labels", str(labels_path)], store=store)

    assert code == 0
    assert labels_path.is_file()
    assert [row.trace_id for row in read_rows(labels_path)] == ["tr-1"]


def test_main_queue_refuses_a_run_with_no_traces(tmp_path: Path) -> None:
    """A run made without LangSmith has nothing to queue: exit 2, and nothing added."""
    store = FakeLabelStore()
    results_path = _write_results(tmp_path / "results.json", _results([_trial("t-b1", 1, None)]))

    code = main(["queue", "--results", str(results_path)], store=store)

    assert code == 2
    assert store.added == {}


def test_langsmith_store_reuses_an_existing_queue() -> None:
    """A queue that already exists is reused, so each run does not make a duplicate."""
    client = MagicMock()
    client.list_annotation_queues.return_value = [SimpleNamespace(id="uuid-1")]
    store = LangSmithLabelStore(cast(langsmith.Client, client), "proj")

    queue_id = store.ensure_queue("grounded")

    assert queue_id == "uuid-1"
    client.list_annotation_queues.assert_called_once_with(name="proj-grounded", limit=1)
    client.create_annotation_queue.assert_not_called()


def test_langsmith_store_creates_a_queue_with_the_judge_question() -> None:
    """A new queue asks the person the same question the judge was asked."""
    client = MagicMock()
    client.list_annotation_queues.return_value = []
    client.create_annotation_queue.return_value = SimpleNamespace(id="uuid-2")
    store = LangSmithLabelStore(cast(langsmith.Client, client), "proj")

    queue_id = store.ensure_queue("grounded")

    assert queue_id == "uuid-2"
    kwargs = client.create_annotation_queue.call_args.kwargs
    assert kwargs["rubric_instructions"] == QUESTIONS["grounded"]


def test_langsmith_store_keeps_the_newest_label_per_run() -> None:
    """A relabel adds a second feedback item. The newest one wins, and blank items are skipped."""
    client = MagicMock()
    client.list_feedback.return_value = [
        SimpleNamespace(
            run_id="r1",
            score=1,
            value=None,
            comment="changed my mind",
            modified_at=datetime(2026, 1, 2),
            created_at=None,
        ),
        SimpleNamespace(
            run_id="r1",
            score=0,
            value=None,
            comment="",
            modified_at=datetime(2026, 1, 1),
            created_at=None,
        ),
        SimpleNamespace(
            run_id="r2", score=None, value=None, comment="", modified_at=None, created_at=None
        ),
        SimpleNamespace(
            run_id="r3",
            score=None,
            value="No",
            comment="",
            modified_at=None,
            created_at=datetime(2026, 1, 1),
        ),
    ]
    store = LangSmithLabelStore(cast(langsmith.Client, client), "proj")

    labels = store.labels("grounded", ["r1", "r2", "r3"])

    assert set(labels) == {"r1", "r3"}
    assert labels["r1"].yes is True
    assert labels["r1"].comment == "changed my mind"
    assert labels["r3"].yes is False
