"""The results file and the markdown, and that `evals.labels` reads the file unchanged."""

import json
from pathlib import Path

from evals.labels import LabelTask, label_tasks, queue
from evals.security.gates import Verdict
from evals.security.graders import Grade
from evals.security.metrics import GradedTrial, all_lines
from evals.security.report import (
    Cost,
    JudgeRecord,
    RunReport,
    ToolCallRecord,
    ToolResultRecord,
    TranscriptRecord,
    TrialRecord,
    render_markdown,
    to_json,
    write_json,
)
from evals.security.tasks import Task
from tests.evals.security.helpers import make_task
from tests.evals.test_labels import FakeLabelStore


def _record(
    task: Task,
    trial: int = 1,
    *,
    outcome: str = "passed",
    trace_id: str | None = "trace-1",
    judge: JudgeRecord | None = None,
    error: str | None = None,
    answer: str = "an answer",
) -> TrialRecord:
    return TrialRecord(
        task_id=task.id,
        trial=trial,
        outcome=outcome,  # type: ignore[arg-type]
        trace_id=trace_id,
        question=task.input,
        transcript=TranscriptRecord(
            answer=answer,
            tool_calls=(ToolCallRecord("fetch_web_pages", '{"urls": ["https://p.test/a"]}'),),
            tool_results=(ToolResultRecord("fetch_web_pages", "page text"),),
            summary_text=None,
            error=error,
        ),
        judges=(judge,) if judge else (),
        attack_succeeded=False,
        utility_ok=True,
        failure_reasons=(),
        tokens_in=4200,
        tokens_out=300,
        latency_ms=12.5,
    )


def _report(
    tasks: list[Task],
    records: list[TrialRecord],
    *,
    verdict: Verdict | None = None,
    stopped_early: str | None = None,
    not_started: int = 0,
) -> RunReport:
    by_id = {task.id: task for task in tasks}
    graded = [
        GradedTrial(by_id[r.task_id], r.trial, Grade(r.outcome, r.attack_succeeded, r.utility_ok))
        for r in records
    ]
    return RunReport(
        run_id="run-1",
        mode="release",
        model="m",
        judge_model="j",
        prompt_version="pv",
        dataset_version="dv",
        tasks=tuple(tasks),
        trials=tuple(records),
        lines=tuple(all_lines(graded)),
        verdict=verdict or Verdict(False, ("S1: 0 of 1 attack trials failed",)),
        cost=Cost(
            tokens_in=4200, tokens_out=300, judge_tokens_in=0, judge_tokens_out=0, usd=0.0006
        ),
        stopped_early=stopped_early,
        not_started=not_started,
    )


S1 = make_task(task_id="s1-a", line="S1")
S6_TWIN = make_task(
    task_id="s6-t",
    line="S6",
    twin=True,
    expect={"judge": "over_refusal", "system_prompt_overlap_max": 0.5},
)
S9 = make_task(
    task_id="s9-a",
    line="S9",
    expect={"judge": "harmful_compliance", "should_refuse": True},
    goal="harm",
    attack={"shape": "pair", "channel": "user", "source": "t"},
)
S11 = make_task(task_id="s11-t", line="S11", twin=True, expect={"judge": "over_refusal"})


def _all_judged_report() -> RunReport:
    return _report(
        [S1, S6_TWIN, S9, S11],
        [
            _record(S1, trace_id="t-s1"),
            _record(S6_TWIN, trace_id="t-s6", judge=JudgeRecord("over_refusal", False, "helped")),
            _record(
                S9, trace_id="t-s9", judge=JudgeRecord("harmful_compliance", True, "gave steps")
            ),
            _record(S11, trace_id="t-s11", judge=JudgeRecord("over_refusal", None, "no verdict")),
        ],
    )


# --- the results file -----------------------------------------------------------


def test_results_file_has_the_documented_top_level_keys() -> None:
    data = to_json(_all_judged_report())

    assert sorted(data) == [
        "cost", "dataset_version", "judge_model", "judge_status", "lines", "mode", "model",
        "not_started", "prompt_version", "run_id", "stopped_early", "tasks", "trials", "verdict",
    ]  # fmt: skip


def test_task_entries_carry_the_label_behavior() -> None:
    data = to_json(_all_judged_report())

    assert {t["task_id"]: (t["behavior"], t["line"], t["twin"]) for t in data["tasks"]} == {
        "s1-a": ("s1", "S1", False),
        "s6-t": ("s11", "S6", True),
        "s9-a": ("s9", "S9", False),
        "s11-t": ("s11", "S11", True),
    }


def test_trial_entry_has_the_documented_fields() -> None:
    [trial, *_] = to_json(_all_judged_report())["trials"]

    assert sorted(trial) == [
        "attack_succeeded", "failure_reasons", "judges", "latency_ms", "outcome", "question",
        "task_id", "tokens_in", "tokens_out", "trace_id", "transcript", "trial", "utility_ok",
    ]  # fmt: skip
    assert sorted(trial["transcript"]) == [
        "answer", "error", "summary_text", "tool_calls", "tool_results",
    ]  # fmt: skip
    assert trial["transcript"]["tool_calls"] == [
        {"name": "fetch_web_pages", "arguments": '{"urls": ["https://p.test/a"]}'}
    ]


def test_task_counts_follow_the_trials() -> None:
    report = _report(
        [S1], [_record(S1, 1), _record(S1, 2, outcome="failed"), _record(S1, 3, outcome="errored")]
    )

    [task] = to_json(report)["tasks"]

    assert (task["passed"], task["failed"], task["errored"]) == (1, 1, 1)


def test_file_is_valid_json_on_disk(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "security.json"

    write_json(_all_judged_report(), out)

    assert json.loads(out.read_text(encoding="utf-8"))["run_id"] == "run-1"


# --- evals.labels reads it unchanged --------------------------------------------


def test_label_tasks_picks_up_s9_and_s11_trials(tmp_path: Path) -> None:
    out = tmp_path / "security.json"
    write_json(_all_judged_report(), out)

    found = label_tasks(json.loads(out.read_text(encoding="utf-8")))

    assert found == [
        LabelTask("t-s6", "over_refusal", "s6-t", 1, "passed", "question", "an answer",
                  "[fetch_web_pages]\npage text", False, "helped"),
        LabelTask("t-s9", "harmful_compliance", "s9-a", 1, "passed", "question", "an answer",
                  "[fetch_web_pages]\npage text", True, "gave steps"),
        LabelTask("t-s11", "over_refusal", "s11-t", 1, "passed", "question", "an answer",
                  "[fetch_web_pages]\npage text", None, "no verdict"),
    ]  # fmt: skip


def test_label_queues_get_the_security_judges(tmp_path: Path) -> None:
    out = tmp_path / "security.json"
    write_json(_all_judged_report(), out)
    store = FakeLabelStore()

    added = queue(json.loads(out.read_text(encoding="utf-8")), store, limit=10)

    assert added == {"harmful_compliance": 1, "over_refusal": 2}
    assert store.added == {"q-harmful_compliance": ["t-s9"], "q-over_refusal": ["t-s6", "t-s11"]}


def test_errored_and_untraced_trials_are_not_queued() -> None:
    report = _report(
        [S9, S11],
        [
            _record(S9, outcome="errored", error="boom", trace_id="t-1"),
            _record(S11, trace_id=None),
        ],
    )

    assert label_tasks(to_json(report)) == []


# --- the markdown ---------------------------------------------------------------


def test_markdown_has_one_row_per_line_with_counts() -> None:
    task = make_task(task_id="s1-a", line="S1", expect={"must_contain": ["x"]})
    twin = make_task(task_id="s1-t", line="S1", twin=True, expect={"must_contain": ["x"]})
    records = [
        _record(task, 1),
        _record(task, 2),
        _record(task, 3),
        _record(task, 4),
        _record(twin),
    ]

    text = render_markdown(_report([task, twin], records))

    assert (
        "| S1 | 4 | 0 of 4 | 0 | 0.0% | 52.7% | 100.0% (1 of 1) | 100.0% (4 of 4) | 0.0% (0 of 1) |"
        in text
    )


def test_markdown_lists_blocking_reasons_notes_and_early_stop() -> None:
    report = _report(
        [S1],
        [_record(S1, outcome="failed")],
        verdict=Verdict(True, ("S1: 1 of 1 attack trials failed (s1-a)",), ("S9 not validated",)),
        stopped_early="s1-a trial 1 failed on S1",
        not_started=300,
    )

    text = render_markdown(report)

    assert "## Verdict: BLOCKED" in text
    assert "- S1: 1 of 1 attack trials failed (s1-a)" in text
    assert "- note: S9 not validated" in text
    assert "- stopped early: s1-a trial 1 failed on S1. 300 trials were not started" in text


def test_markdown_shows_the_split_by_shape() -> None:
    text = render_markdown(_report([S1], [_record(S1)]))

    assert "| S1 | shape | override | 0 of 1 |" in text


def test_markdown_lists_errored_trials() -> None:
    report = _report(
        [S1, S9],
        [
            _record(S1, outcome="failed", error="rate limited"),
            _record(S9, outcome="errored", error="timeout"),
        ],
    )

    text = render_markdown(report)

    assert "- s1-a trial 1 (counted as failed): rate limited" in text
    assert "- s9-a trial 1 (left out of the rates): timeout" in text


def test_markdown_says_when_nothing_errored() -> None:
    assert "## Errored trials\n\nNone." in render_markdown(_report([S1], [_record(S1)]))
