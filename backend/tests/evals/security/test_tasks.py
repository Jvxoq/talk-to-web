"""The task files load whole, and a bad line is caught before the first paid call."""

import json
from collections import Counter
from pathlib import Path

import pytest

from evals.security.tasks import (
    TASKS_DIR,
    Task,
    TaskLoadError,
    behavior_tag,
    dataset_version,
    load_tasks,
)
from tests.evals.security.helpers import make_task

_GOOD = {
    "id": "x-1",
    "line": "S1",
    "input": "q",
    "attack": {"shape": "override", "channel": "page", "source": "t"},
    "expect": {},
    "trials": 3,
    "tags": ["attack"],
}


def _write(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> Path:
    (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return tmp_path


def test_every_task_loads(all_tasks: list[Task]) -> None:
    assert len(all_tasks) == 180


def test_tasks_per_line_match_the_build(all_tasks: list[Task]) -> None:
    counts = Counter(task.line for task in all_tasks)

    assert counts == {"S1": 30, "S2": 30, "S3": 20, "S6": 30, "S9": 20, "S11": 20, "S13": 30}


def test_trial_total_is_420(all_tasks: list[Task]) -> None:
    assert sum(task.trials for task in all_tasks) == 420


def test_attack_tasks_get_three_trials_and_twins_one(all_tasks: list[Task]) -> None:
    trials = {(task.twin, task.trials) for task in all_tasks}

    assert trials == {(False, 3), (True, 1)}


def test_twin_counts_per_line(all_tasks: list[Task]) -> None:
    twins = Counter(task.line for task in all_tasks if task.twin)

    assert twins == {"S1": 10, "S2": 10, "S6": 10, "S11": 20, "S13": 10}


def test_load_does_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert len(load_tasks()) == 180
    assert TASKS_DIR.is_absolute()


def test_ids_are_unique_across_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.jsonl", [_GOOD])
    _write(tmp_path, "b.jsonl", [_GOOD])

    with pytest.raises(TaskLoadError, match=r"duplicate task id 'x-1' \(2 times\)"):
        load_tasks(tmp_path)


def test_attack_task_without_attack_is_rejected(tmp_path: Path) -> None:
    bad = {key: value for key, value in _GOOD.items() if key != "attack"}
    _write(tmp_path, "a.jsonl", [bad])

    with pytest.raises(TaskLoadError, match=r"a\.jsonl:1: .*needs an `attack`"):
        load_tasks(tmp_path)


def test_twin_needs_no_attack() -> None:
    assert make_task(twin=True).attack is None


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (
            [{"role": "user", "content": "hi"}],
            "history must end with an assistant turn",
        ),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "tool_fetch", "url": "https://p.test/a", "html": "<p>x</p>"},
            ],
            "history must end with an assistant turn",
        ),
        (
            [{"role": "assistant", "content": "hi"}],
            "history starts with user",
        ),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            "two user turns in a row",
        ),
    ],
    ids=["ends on user", "ends on fetch", "starts on assistant", "two users"],
)
def test_history_must_end_on_the_agent(
    tmp_path: Path, history: list[dict[str, str]], message: str
) -> None:
    _write(tmp_path, "a.jsonl", [{**_GOOD, "history": history}])

    with pytest.raises(TaskLoadError, match=message):
        load_tasks(tmp_path)


def test_every_bad_line_is_listed(tmp_path: Path) -> None:
    rows = [
        {**_GOOD, "id": "ok"},
        {**_GOOD, "id": "b1", "trials": 0},
        {**_GOOD, "id": "b2", "extra": 1},
    ]
    _write(tmp_path, "a.jsonl", rows)

    with pytest.raises(TaskLoadError) as error:
        load_tasks(tmp_path)

    lines = str(error.value).splitlines()
    assert [line.split(":")[0:2] for line in lines] == [["a.jsonl", "2"], ["a.jsonl", "3"]]


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "a.jsonl", [{**_GOOD, "surprise": True}])

    with pytest.raises(TaskLoadError, match="surprise"):
        load_tasks(tmp_path)


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError, match="No task files"):
        load_tasks(tmp_path)


def test_padding_needs_a_history(tmp_path: Path) -> None:
    _write(tmp_path, "a.jsonl", [{**_GOOD, "pad_history_to_tokens": 100}])

    with pytest.raises(TaskLoadError, match="needs a history to pad"):
        load_tasks(tmp_path)


def test_dataset_version_follows_the_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.jsonl", [_GOOD])
    before = dataset_version(tmp_path)

    _write(tmp_path, "a.jsonl", [{**_GOOD, "input": "changed"}])

    assert dataset_version(tmp_path) != before


@pytest.mark.parametrize(
    ("fields", "behavior"),
    [
        ({"line": "S9", "expect": {"judge": "harmful_compliance"}}, "s9"),
        ({"line": "S11", "twin": True, "expect": {"judge": "over_refusal"}}, "s11"),
        ({"line": "S6", "twin": True, "expect": {"judge": "over_refusal"}}, "s11"),
        ({"line": "S13"}, "s13"),
        ({"line": "S1"}, "s1"),
    ],
    ids=["jailbreak", "s11 twin", "s6 twin", "s13", "s1"],
)
def test_behavior_tag(fields: dict[str, object], behavior: str) -> None:
    assert behavior_tag(make_task(**fields)) == behavior  # type: ignore[arg-type]
