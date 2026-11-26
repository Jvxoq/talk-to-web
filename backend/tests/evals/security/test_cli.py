"""The command line: the estimate, the refusal to start, the early stop, the exit codes."""

import argparse
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.application.chat.models import ModelChunk
from app.settings import Settings
from evals import harness as harness_module
from evals.security import __main__ as cli
from evals.security.adapter import TrialResult
from evals.security.cost import (
    CONDENSER_PRICE_IN,
    JUDGE_PRICE_IN,
    JUDGE_PRICE_OUT,
    PRICE_IN,
    PRICE_OUT,
    cheapest_first,
    estimate_plan,
    trial_usd,
)
from evals.security.gates import JudgeStatus, PlannedTrial, Verdict, plan_trials
from evals.security.graders import Grade
from evals.security.report import RunReport, TranscriptRecord, TrialRecord
from evals.security.tasks import Task, TaskLoadError
from tests.application import test_system_layer as layer
from tests.evals.security.helpers import make_task
from tests.evals.security.rig import SecurityRig, call, say


@pytest.fixture
def no_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that reaches the harness would spend money. These tests must never do that."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the harness was built")

    monkeypatch.setattr(cli, "build_harness", boom)


# --- the estimate ---------------------------------------------------------------


def _agent(trials: int, *, padded: int = 0, judged: int = 0) -> float:
    usd = trials * (4_200 * PRICE_IN + 300 * PRICE_OUT)
    usd += padded * (7_000 * PRICE_IN + 13_000 * CONDENSER_PRICE_IN)
    usd += judged * (1_000 * JUDGE_PRICE_IN + 80 * JUDGE_PRICE_OUT)
    return usd / 1_000_000


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("gate", "mode gate: 60 tasks, 60 trials (0 padded, 0 judged), estimated cost $0.0389"),
        (
            "release",
            "mode release: 180 tasks, 420 trials (70 padded, 90 judged), estimated cost $0.4846",
        ),
        (
            "smoke",
            "mode smoke: 180 tasks, 180 trials (30 padded, 50 judged), estimated cost $0.2098",
        ),
    ],
)
def test_estimate_prints_counts_and_cost_then_stops(
    mode: str, expected: str, capsys: pytest.CaptureFixture[str], no_harness: None
) -> None:
    code = cli.main(["--mode", mode, "--estimate"])

    captured = capsys.readouterr()
    assert (code, captured.out.strip(), captured.err) == (0, expected, "")


@pytest.mark.parametrize(
    ("mode", "trials", "padded", "judged"),
    [("gate", 60, 0, 0), ("release", 420, 70, 90), ("smoke", 180, 30, 50)],
)
def test_estimate_follows_the_cost_model(
    all_tasks: list[Task], mode: Any, trials: int, padded: int, judged: int
) -> None:
    plan = plan_trials(all_tasks, mode)

    estimate = estimate_plan(plan, judges=mode != "gate")

    assert estimate.usd == pytest.approx(_agent(trials, padded=padded, judged=judged))
    assert (estimate.trials, estimate.padded_trials, estimate.judged_trials) == (
        trials,
        padded,
        judged,
    )


def test_padded_judged_trial_pays_for_all_three_models() -> None:
    task = make_task(
        line="S13",
        history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        pad_history_to_tokens=13_000,
        expect={"judge": "over_refusal"},
    )

    assert trial_usd(task, judged=True) == pytest.approx(_agent(1, padded=1, judged=1))
    assert trial_usd(task, judged=False) == pytest.approx(_agent(1, padded=1))


def test_release_runs_the_cheapest_line_first(all_tasks: list[Task]) -> None:
    plan = plan_trials(all_tasks, "release")
    by_line = estimate_plan(plan, judges=True).usd_by_line

    order = cheapest_first(plan, judges=True)
    built = cli.build_plan(all_tasks, "release", lines=[], limit=None)

    assert order == sorted(by_line, key=lambda line: by_line[line])
    assert order[-1] == "S13"
    assert list(dict.fromkeys(p.task.line for p in built)) == order


def test_other_modes_run_lines_in_natural_order(all_tasks: list[Task]) -> None:
    built = cli.build_plan(all_tasks, "smoke", lines=[], limit=None)

    assert list(dict.fromkeys(p.task.line for p in built)) == [
        "S1",
        "S2",
        "S3",
        "S6",
        "S9",
        "S11",
        "S13",
    ]


# --- refusing to start ----------------------------------------------------------


def test_estimate_above_the_budget_refuses_with_exit_two(
    capsys: pytest.CaptureFixture[str], no_harness: None
) -> None:
    code = cli.main(["--mode", "release", "--max-usd", "0.10"])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "mode release: 180 tasks, 420 trials" in captured.err


def test_paid_mode_prints_the_estimate_to_stderr_before_it_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[str] = []

    async def fake_run(*args: object) -> int:
        started.append("run")
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)

    code = cli.main(["--mode", "gate", "--max-usd", "1.0"])

    captured = capsys.readouterr()
    assert (code, started) == (0, ["run"])
    assert (
        captured.err.strip()
        == "mode gate: 60 tasks, 60 trials (0 padded, 0 judged), estimated cost $0.0389"
    )
    assert captured.out == ""


def test_gate_with_a_line_outside_it_selects_nothing(no_harness: None) -> None:
    assert cli.main(["--mode", "gate", "--line", "S6"]) == 2


def test_unreadable_task_files_stop_the_run(
    monkeypatch: pytest.MonkeyPatch, no_harness: None
) -> None:
    def broken() -> list[Task]:
        raise TaskLoadError("a.jsonl:1: bad")

    monkeypatch.setattr(cli, "load_tasks", broken)

    assert cli.main(["--mode", "gate"]) == 2


def test_mode_is_required() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([])

    assert error.value.code == 2


@pytest.mark.parametrize("flag", [["--limit", "0"], ["--concurrency", "0"]])
def test_non_positive_limits_are_refused(flag: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--mode", "gate", *flag])


# --- the early stop -------------------------------------------------------------


def _result(planned: PlannedTrial, outcome: str = "passed") -> TrialResult:
    grade = Grade(outcome, outcome == "failed", None)  # type: ignore[arg-type]
    record = TrialRecord(
        task_id=planned.task.id,
        trial=planned.trial,
        outcome=grade.outcome,
        trace_id=None,
        question="q",
        transcript=TranscriptRecord(answer="a"),
    )
    return TrialResult(planned.task, planned.trial, grade, record)


def _plan(n: int, line: str = "S1") -> list[PlannedTrial]:
    return [PlannedTrial(make_task(task_id=f"t-{i}", line=line), 1) for i in range(n)]


def _run_one(
    failing: set[str], started: list[str]
) -> Callable[[PlannedTrial], Awaitable[TrialResult]]:
    async def run_one(planned: PlannedTrial) -> TrialResult:
        started.append(planned.task.id)
        return _result(planned, "failed" if planned.task.id in failing else "passed")

    return run_one


def _stops(result: TrialResult) -> bool:
    return result.grade.outcome == "failed"


async def test_no_new_trial_starts_after_the_first_blocking_failure() -> None:
    started: list[str] = []

    results, not_started, why = await cli.run_plan(
        _plan(6), _run_one({"t-2"}, started), concurrency=1, stops_the_run=_stops
    )

    assert started == ["t-0", "t-1", "t-2"]
    assert [r.task.id for r in results] == ["t-0", "t-1", "t-2"]
    assert (not_started, why) == (3, "t-2 trial 1 failed on S1")


async def test_keep_going_runs_every_trial() -> None:
    started: list[str] = []

    results, not_started, why = await cli.run_plan(
        _plan(6), _run_one({"t-2"}, started), concurrency=1, stops_the_run=None
    )

    assert (len(results), not_started, why) == (6, 0, None)


async def test_trials_already_running_finish_when_the_run_stops() -> None:
    started: list[str] = []

    results, not_started, _ = await cli.run_plan(
        _plan(10), _run_one({"t-0"}, started), concurrency=3, stops_the_run=_stops
    )

    assert not_started == 10 - len(results)
    assert 1 <= len(results) <= 3
    assert "t-0" in [r.task.id for r in results]


async def test_clean_run_reports_no_stop() -> None:
    results, not_started, why = await cli.run_plan(
        _plan(4), _run_one(set(), []), concurrency=2, stops_the_run=_stops
    )

    assert (len(results), not_started, why) == (4, 0, None)


# --- the report and the exit code -----------------------------------------------


def _assemble(results: list[TrialResult], mode: Any = "gate", **kwargs: Any) -> RunReport:
    tasks = list({r.task.id: r.task for r in results}.values())
    return cli.assemble_report(
        run_id="r",
        mode=mode,
        model="m",
        judge_model="j",
        prompt_version="pv",
        tasks=tasks,
        results=results,
        lines=[],
        judge_status={},
        stopped_early=None,
        not_started=0,
        **kwargs,
    )


def test_gate_report_uses_the_gate_verdict() -> None:
    results = [_result(p) for p in _plan(2, "S1")] + [_result(p, "failed") for p in _plan(1, "S2")]

    report = _assemble(results)

    assert report.verdict.failed is True
    assert report.verdict.reasons[0] == "S2: 1 of 1 attack trials failed (t-0)"


def test_report_cost_is_priced_from_reported_tokens() -> None:
    [planned] = _plan(1)
    result = _result(planned)
    record = replace(result.record, tokens_in=1_000_000, tokens_out=1_000_000)

    report = _assemble(
        [TrialResult(result.task, 1, result.grade, record)], judge_tokens=(1_000_000, 1_000_000)
    )

    assert report.cost is not None
    assert report.cost.usd == pytest.approx(PRICE_IN + PRICE_OUT + JUDGE_PRICE_IN + JUDGE_PRICE_OUT)


def _report_with(verdict_failed: bool, errored: bool) -> RunReport:
    [planned] = _plan(1)
    base = _result(planned, "errored" if errored else "passed")
    transcript = TranscriptRecord(answer="", error="boom" if errored else None)
    record = replace(base.record, transcript=transcript)
    report = _assemble([TrialResult(base.task, 1, base.grade, record)], mode="smoke")
    return replace(report, verdict=Verdict(verdict_failed, ("x",)))


@pytest.mark.parametrize(
    ("failed", "errored", "code"),
    [(False, False, 0), (True, False, 1), (False, True, 2), (True, True, 1)],
    ids=["clean", "bar missed", "errored", "bar missed and errored"],
)
def test_exit_code(failed: bool, errored: bool, code: int) -> None:
    assert cli.exit_code(_report_with(failed, errored)) == code


def test_smoke_report_never_blocks() -> None:
    results = [_result(p, "failed") for p in _plan(3)]

    assert _assemble(results, mode="smoke").verdict.failed is False


def test_defaults() -> None:
    args = cli._parse_args(["--mode", "release"])

    assert isinstance(args, argparse.Namespace)
    assert (args.concurrency, args.max_usd, str(args.out), args.keep_going, args.line) == (
        4,
        1.0,
        "evals/results/security-latest.json",
        False,
        [],
    )


def test_release_judge_status_is_passed_to_the_verdict() -> None:
    results = [_result(p, "failed") for p in _plan(1, "S9")]
    tasks = list({r.task.id: r.task for r in results}.values())

    unvalidated = cli.assemble_report(
        run_id="r",
        mode="release",
        model="m",
        judge_model="j",
        prompt_version="pv",
        tasks=tasks,
        results=results,
        lines=["S9"],
        judge_status={},
        stopped_early=None,
        not_started=0,
    )
    validated = cli.assemble_report(
        run_id="r",
        mode="release",
        model="m",
        judge_model="j",
        prompt_version="pv",
        tasks=tasks,
        results=results,
        lines=["S9"],
        judge_status={"harmful_compliance": JudgeStatus(validated=True)},
        stopped_early=None,
        not_started=0,
    )

    assert (unvalidated.verdict.failed, validated.verdict.failed) == (False, True)


# --- when the run stops ---------------------------------------------------------


def _failed_result(line: str) -> TrialResult:
    return _result(_plan(1, line)[0], "failed")


@pytest.mark.parametrize("keep_going", [False, True])
@pytest.mark.parametrize("mode", ["gate", "smoke"])
def test_only_a_release_run_can_stop_early(mode: Any, keep_going: bool) -> None:
    assert cli.stop_rule(mode, keep_going, {}) is None


def test_keep_going_never_stops_a_release_run() -> None:
    assert cli.stop_rule("release", True, {}) is None


def test_release_stops_on_a_blocking_failure() -> None:
    stops = cli.stop_rule("release", False, {})

    assert stops is not None
    assert (stops(_failed_result("S1")), stops(_result(_plan(1)[0]))) == (True, False)


def test_release_does_not_stop_on_an_unvalidated_judge_line() -> None:
    stops = cli.stop_rule("release", False, {})
    validated = cli.stop_rule("release", False, {"harmful_compliance": JudgeStatus(validated=True)})

    assert stops is not None
    assert validated is not None
    assert (stops(_failed_result("S9")), validated(_failed_result("S9"))) == (False, True)


# --- the whole command, with the harness swapped for a scripted one --------------


@pytest.fixture
def scripted(
    monkeypatch: pytest.MonkeyPatch, all_tasks: list[Task], settings: Settings
) -> Callable[..., SecurityRig]:
    """Make `build_harness` hand over a rig whose agent follows a script."""

    def install(turns: list[list[ModelChunk]], **kwargs: Any) -> SecurityRig:
        rig = SecurityRig(settings, all_tasks, turns, **kwargs)

        @asynccontextmanager
        async def build(**_: object) -> AsyncIterator[Any]:
            yield rig.harness

        monkeypatch.setattr(cli, "build_harness", build)
        return rig

    return install


def test_gate_run_writes_the_results_and_prints_the_report(
    scripted: Callable[..., SecurityRig],
    all_tasks: list[Task],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = next(t for t in all_tasks if t.line == "S1" and not t.twin)
    assert task.page is not None
    scripted(
        [call("fetch_web_pages", urls=[task.page.url]), say(" ".join(task.expect.must_contain))]
    )
    out = tmp_path / "gate.json"

    code = cli.main(
        ["--mode", "gate", "--line", "S1", "--limit", "1", "--concurrency", "1", "--out", str(out)]
    )

    captured = capsys.readouterr()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert "## Verdict: OK" in captured.out
    assert (data["mode"], data["model"], len(data["trials"]), data["verdict"]["failed"]) == (
        "gate",
        "test-model",
        1,
        False,
    )
    assert data["trials"][0]["outcome"] == "passed"


def test_release_stops_after_the_first_blocking_failure(
    scripted: Callable[..., SecurityRig], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted([say(layer.SYSTEM_PROMPT)])
    out = tmp_path / "release.json"

    code = cli.main(
        [
            "--mode",
            "release",
            "--line",
            "S6",
            "--limit",
            "2",
            "--concurrency",
            "1",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1
    assert (len(data["trials"]), data["not_started"]) == (1, 7)
    assert data["stopped_early"] == "s6-001-direct trial 1 failed on S6"
    assert (
        "- stopped early: s6-001-direct trial 1 failed on S6. 7 trials were not started"
        in capsys.readouterr().out
    )


def test_keep_going_runs_every_planned_trial(
    scripted: Callable[..., SecurityRig], tmp_path: Path
) -> None:
    scripted([say(layer.SYSTEM_PROMPT)])
    out = tmp_path / "baseline.json"

    code = cli.main(
        [
            "--mode",
            "release",
            "--line",
            "S6",
            "--limit",
            "2",
            "--concurrency",
            "1",
            "--keep-going",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1
    assert (len(data["trials"]), data["not_started"], data["stopped_early"]) == (8, 0, None)


def test_errored_trial_on_smoke_gives_exit_two(
    scripted: Callable[..., SecurityRig], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness_module, "_RETRY_BACKOFF_SECONDS", 0.0)
    rig = scripted([])
    rig.model.fail_with = RuntimeError("provider down")

    code = cli.main(
        ["--mode", "smoke", "--line", "S9", "--limit", "1", "--out", str(tmp_path / "smoke.json")]
    )

    assert code == 2
