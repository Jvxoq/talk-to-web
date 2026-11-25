"""The eval driver's scoring, checked without spending anything.

`python -m evals` costs money per sample, so the parts of it that decide what a
number means - which runs count, what a regression is - are pure functions
over a `CaseRun`, and this is where they are proved. Nothing here builds a
harness, opens a client or reads an API key.

What these tests really guard is a three-way split. A model that got it wrong
every time is a regression. A model that got it wrong sometimes is flaky. A
provider that fell over is an error, and its empty answer would fail every
check here for a reason that has nothing to do with the model. Collapsing any
two of those sends someone to edit a prompt that was fine.
"""

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.application.chat.dto import (
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
    ReplyUsage,
)
from app.application.chat.models import Source
from app.application.chat.tools.base import ToolSpec
from evals import __main__ as driver
from evals import cases as cases_module
from evals.__main__ import _select_cases, _verdicts, dataset_version
from evals.cases import EvalCase, Expectation, behavior_of, expanded_history, load_cases
from evals.harness import CaseRun, EvalHarness, compute_prompt_version
from evals.judge import JudgeName, JudgeVerdict
from evals.report import (
    BehaviorRecord,
    RunReport,
    TaskRecord,
    behavior_records,
    estimate_cost,
    render_markdown,
)

_HARNESS = cast(EvalHarness, SimpleNamespace(retrieval_k=3))
"""Only `retrieval_k` is read by the functions under test. A real harness would
need four API keys and a Qdrant."""


def _case(**kwargs: object) -> EvalCase:
    defaults: dict[str, object] = {"id": "c-1", "suite": "tools", "input": "q"}
    return EvalCase.model_validate(defaults | kwargs)


def _run(case: EvalCase, events: list[ReplyEvent], sample: int = 1) -> CaseRun:
    return CaseRun(case=case, events=events, sample=sample)


def _finished(name: str, *, ok: bool = True, sources: tuple[Source, ...] = ()) -> ReplyEvent:
    return ReplyToolFinished(name=name, ok=ok, sources=sources)


def _answer(text: str) -> list[ReplyEvent]:
    return [ReplyDelta(text=text)]


def test_refusal_is_not_completion() -> None:
    """The tool node writes its start event before the routing policy runs, so
    a blocked call still announces itself. Grading on announcements scores a
    refusal as a success, which inverts the result of the two cases the gate
    exists for."""
    case = _case(expect=Expectation(tools=["retrieve_documents"]))
    run = _run(
        case,
        [
            ReplyToolStarted(name="search_web", summary="search_web(query='x')"),
            _finished("search_web", ok=False),
            _finished("retrieve_documents"),
        ],
    )

    assert _verdicts([run], _HARNESS).regressions == ()


def test_order_is_graded() -> None:
    """A search that ran before the retrieval answered has the same tool *set*
    as a run that got it right. Only the order tells them apart."""
    case = _case(
        expect=Expectation(
            tools=["retrieve_documents", "search_web"],
            tool_order=["retrieve_documents", "search_web"],
        )
    )
    backwards = _run(case, [_finished("search_web"), _finished("retrieve_documents")])

    assert "expected order" in _verdicts([backwards], _HARNESS).regressions[0]


def test_no_tool_task_fails_when_the_agent_calls_a_tool() -> None:
    """A task that wrote `"tools": []` means "call no tool". An empty list
    used to read as unchecked, so a search_web call passed the task."""
    case = _case(expect=Expectation.model_validate({"tools": []}))
    run = _run(
        case,
        [
            ReplyToolStarted(name="search_web", summary="search_web(query='12 times 12')"),
            _finished("search_web"),
            ReplyDelta(text="144"),
        ],
    )

    assert driver._sample_failures(run, _HARNESS)


def test_no_tool_task_passes_when_the_agent_calls_no_tool() -> None:
    case = _case(expect=Expectation.model_validate({"tools": []}))
    run = _run(case, _answer("144"))

    assert driver._sample_failures(run, _HARNESS) == []


def test_checks_tools_is_set_only_when_the_task_wrote_tools() -> None:
    """`[]` written in the JSON is a real check. An omitted field is not."""
    assert not Expectation.model_validate({"must_contain": ["144"]}).checks_tools
    assert Expectation.model_validate({"tools": []}).checks_tools


def test_broken_run_is_not_a_regression() -> None:
    """A rate limit produces an empty answer and no tool calls. It belongs in
    `errors`, and counting it here would read as the model calling nothing."""
    case = _case(expect=Expectation(tools=["retrieve_documents"], must_contain=["2019"]))
    run = CaseRun(case=case, failed=ReplyFailed(detail="busy right now"))

    assert _verdicts([run], _HARNESS) == _verdicts([], _HARNESS)


def test_forbidden_in_tool_args() -> None:
    """A forbidden string sent in a search query reaches a third party even
    when the answer is clean, which an answer-only check calls a pass."""
    case = _case(expect=Expectation(must_not_contain=["Project Halcyon"]))
    run = _run(
        case,
        [
            ReplyToolStarted(name="search_web", summary="search_web(query='Project Halcyon')"),
            _finished("search_web"),
            ReplyDelta(text="The VR-4 carries 1200 kg."),
        ],
    )

    assert "forbidden in tool args" in _verdicts([run], _HARNESS).regressions[0]


def test_retrieval_ignores_web_sources() -> None:
    """Five web results finishing first push the fixture the case is about out
    of the top-k window. Pooling them measures the ranking of two unrelated
    result sets glued together."""
    case = _case(suite="rag", expect=Expectation(expected_sources=["aurora_robotics.txt"]))
    run = _run(
        case,
        [
            ReplyToolFinished(
                name="search_web",
                sources=tuple(Source(label=f"r{i}", url=f"https://e/{i}") for i in range(5)),
            ),
            ReplyToolFinished(
                name="retrieve_documents",
                sources=(Source(label="aurora_robotics.txt", url=None),),
            ),
        ],
    )

    assert _verdicts([run], _HARNESS).regressions == ()


def test_every_sample_failing_is_a_regression() -> None:
    """Behaviour that repeats is behaviour. Re-running will not change it."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [_run(case, _answer("no idea"), sample) for sample in (1, 2, 3)]

    verdicts = _verdicts(runs, _HARNESS)

    assert len(verdicts.regressions) == 1
    assert verdicts.flaky == ()


def test_some_samples_failing_is_flaky() -> None:
    """A case that fails one sample in three has no stable verdict, so calling
    it a regression would send someone to fix a prompt that answers correctly
    two thirds of the time."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [
        _run(case, _answer("founded in 2019"), 1),
        _run(case, _answer("no idea"), 2),
        _run(case, _answer("founded in 2019"), 3),
    ]

    verdicts = _verdicts(runs, _HARNESS)

    assert verdicts.regressions == ()
    assert "failed 1/3 samples" in verdicts.flaky[0]


def test_broken_sample_does_not_make_a_case_flaky() -> None:
    """A rate limit on one sample of three is not instability in the model. It
    is one fewer sample, and the two that answered agree."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [
        _run(case, _answer("founded in 2019"), 1),
        CaseRun(case=case, sample=2, failed=ReplyFailed(detail="busy right now")),
        _run(case, _answer("founded in 2019"), 3),
    ]

    verdicts = _verdicts(runs, _HARNESS)

    assert verdicts.regressions == ()
    assert verdicts.flaky == ()


def test_graded_counts_samples() -> None:
    """The header line reports work done, not cases listed. Under `--repeat` a
    run of 20 cases generates 60 replies, and the rates below it are means over
    those."""
    report = RunReport(
        run_id="r",
        model="m",
        tags=(),
        suites=(),
        case_ids=("c-1", "c-2"),
        samples=6,
        regressions=(),
        errors=("c-1#2: busy right now",),
    )

    assert report.graded == 5


def test_tag_selection_keeps_suite_order() -> None:
    """The report reads top to bottom in `SUITES` order, and a run that ordered
    its cases by file-read order would shuffle that for no reason."""
    selected = _select_cases("all", None, ["ci"])
    suites = [case.suite for case in selected]

    assert suites == sorted(suites, key=["tools", "rag"].index)
    assert all("ci" in case.tags for case in selected)


# --- the two run modes, driven end to end with a scripted harness -----------------


class _ScriptedHarness:
    """Stands in for `EvalHarness`: each task's trials answer from a script.

    `script["a"] = ["fail", "pass", "pass"]` means the first call to
    `run_case` for task `a` gets a wrong answer, and the next two get the right
    one. A task with no script entry passes. Records every call, so a test can
    say how many trials a task was given.
    """

    model = "fake-model"
    retrieval_k = 3
    prompt_version = "pv-test"
    web_mode = "recorded"

    def __init__(self, script: dict[str, list[str]]) -> None:
        self._script = {key: list(value) for key, value in script.items()}
        self.calls: list[str] = []
        self.judge = _CountingJudge()

    async def index_fixtures(self) -> int:
        return 0

    async def run_case(self, case: EvalCase) -> CaseRun:
        self.calls.append(case.id)
        steps = self._script.get(case.id)
        step = steps.pop(0) if steps else "pass"
        if step == "error":
            return CaseRun(case=case, failed=ReplyFailed(detail="provider down"))
        events: list[ReplyEvent] = [
            ReplyToolStarted(name="search_web", summary="search_web(query='q')"),
            _finished("search_web"),
            ReplyDelta(text="ok" if step == "pass" else "nope"),
            ReplyUsage(prompt_tokens=1000, completion_tokens=100, elapsed_ms=5, model="m"),
        ]
        run = CaseRun(case=case, events=events, latency_ms=5.0)
        run.usage = cast(ReplyUsage, events[-1])
        return run


class _CountingJudge:
    model_name = "fake-judge"
    prompt_tokens = 0
    completion_tokens = 0

    def __init__(self) -> None:
        self.calls: list[JudgeName] = []

    async def verdict(self, judge: JudgeName, **_: object) -> JudgeVerdict | None:
        self.calls.append(judge)
        self.prompt_tokens += 200
        self.completion_tokens += 20
        return JudgeVerdict(judge=judge, verdict=True, reason="fake")


@pytest.fixture
def drive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    """Run the CLI against a scripted harness. Returns `(exit_code, results, harness)`."""

    def _drive(
        argv: list[str],
        script: dict[str, list[str]],
        *,
        suite: str = "tools",
        tasks: int = 2,
        tags: tuple[str, ...] = (),
    ) -> tuple[int, dict[str, Any], _ScriptedHarness]:
        harness = _ScriptedHarness(script)
        cases = [
            _case(id=f"t{n}", suite=suite, tags=list(tags), expect=Expectation(must_contain=["ok"]))
            for n in range(1, tasks + 1)
        ]

        @asynccontextmanager
        async def fake_build(**_: object) -> AsyncIterator[_ScriptedHarness]:
            yield harness

        monkeypatch.setattr(driver, "build_harness", fake_build)
        monkeypatch.setattr(driver, "_select_cases", lambda *_: cases)
        out = tmp_path / "results.json"
        code = driver.main([*argv, "--out", str(out)])
        return code, json.loads(out.read_text()), harness

    yield _drive


def test_gate_runs_each_task_once(drive: Any) -> None:
    code, results, harness = drive(["--mode", "gate"], {})

    assert code == 0
    assert sorted(harness.calls) == ["t1", "t2"]
    assert results["totals"] == {"passed": 2, "failed": 0, "errored": 0}


def test_gate_fail_then_pass_exits_zero(drive: Any) -> None:
    code, results, harness = drive(["--mode", "gate"], {"t1": ["fail", "pass", "pass"]})

    assert code == 0
    assert harness.calls.count("t1") == 3
    assert harness.calls.count("t2") == 1
    assert results["tasks"][0] == {
        "task_id": "t1",
        "suite": "tools",
        "passed": 2,
        "failed": 1,
        "errored": 0,
        "behavior": "untagged",
    }


def test_gate_fail_then_fail_exits_one(drive: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, harness = drive(["--mode", "gate"], {"t1": ["fail", "pass", "fail"]})

    assert code == 1
    assert harness.calls.count("t1") == 3
    assert "t1 failed its first trial and then failed again on trial 3" in capsys.readouterr().out


def test_gate_does_not_rerun_errored(drive: Any) -> None:
    code, results, harness = drive(["--mode", "gate"], {"t1": ["error"]})

    assert code == 2
    assert harness.calls.count("t1") == 1
    assert results["totals"] == {"passed": 1, "failed": 0, "errored": 1}


def test_gate_makes_no_judge_calls(drive: Any) -> None:
    _, results, harness = drive(["--mode", "gate"], {}, suite="rag", tags=("b1",))

    assert harness.judge.calls == []
    assert results["cost"]["judge_tokens_in"] == 0


def test_release_runs_ten_trials_with_judge(drive: Any) -> None:
    code, results, harness = drive(["--mode", "release"], {}, suite="rag", tags=("b1",))

    assert code == 0
    assert harness.calls.count("t1") == 10
    # b1 reads two judges, so 20 trials make 40 calls.
    assert sorted(set(harness.judge.calls)) == ["eligible", "grounded"]
    assert len(harness.judge.calls) == 40
    first = results["trials"][0]
    assert {j["judge"] for j in first["judges"]} == {"grounded", "eligible"}
    assert {(j["judge"], j["judged"], j["yes"]) for j in results["judges"]} == {
        ("grounded", 20, 20),
        ("eligible", 20, 20),
    }
    assert results["totals"]["passed"] == 20


def test_release_exits_one_on_three_failures_in_a_task(drive: Any) -> None:
    code, results, _ = drive(["--mode", "release"], {"t1": ["fail"] * 3})

    assert code == 1
    assert results["quality_failed"] is True
    assert results["tasks"][0]["failed"] == 3


def test_gate_errored_extra_trial_exits_two(drive: Any) -> None:
    """The first trial failed and an extra trial errored. Nothing confirmed the
    failure and nothing cleared it, so the run cannot be trusted: exit 2."""
    code, results, harness = drive(["--mode", "gate"], {"t1": ["fail", "error", "pass"]})

    assert code == 2
    assert harness.calls.count("t1") == 3
    assert results["quality_failed"] is False
    assert results["tasks"][0]["errored"] == 1


def test_release_allows_eight_failures_of_280(drive: Any) -> None:
    """28 tasks x 10 trials. Four tasks fail twice: 8 failures, no task has 3."""
    script = {f"t{n}": ["fail", "fail"] for n in range(1, 5)}

    code, results, _ = drive(["--mode", "release"], script, tasks=28)

    assert results["totals"] == {"passed": 272, "failed": 8, "errored": 0}
    assert code == 0
    assert results["quality_failed"] is False


def test_release_blocks_nine_failures_of_280(drive: Any) -> None:
    script = {f"t{n}": ["fail", "fail"] for n in range(1, 5)} | {"t5": ["fail"]}

    code, results, _ = drive(["--mode", "release"], script, tasks=28)

    assert results["totals"] == {"passed": 271, "failed": 9, "errored": 0}
    assert code == 1
    assert results["quality_failed"] is True


@pytest.mark.parametrize("mode", ["gate", "release"])
def test_mode_with_no_matching_tag_exits_two(mode: str, tmp_path: Path) -> None:
    """A gate that selected nothing measured nothing. Exit 0 would read as a pass."""
    out = tmp_path / "r.json"

    code = driver.main(["--mode", mode, "--tag", "no-such-tag", "--out", str(out)])

    assert code == 2
    assert not out.exists()


def test_plain_run_with_no_matching_tag_exits_zero(tmp_path: Path) -> None:
    code = driver.main(["--tag", "no-such-tag", "--out", str(tmp_path / "r.json")])

    assert code == 0


def test_release_errored_trials_are_excluded(drive: Any) -> None:
    """Three errors are not three failures: the task is still within its limit."""
    code, results, _ = drive(["--mode", "release"], {"t1": ["error"] * 3})

    assert code == 2
    assert results["quality_failed"] is False
    assert results["tasks"][0]["errored"] == 3
    assert len(results["errors"]) == 3


def test_plain_run_keeps_repeat(drive: Any) -> None:
    code, _, harness = drive(["--repeat", "3"], {"t1": ["fail", "fail", "fail"]})

    assert code == 1
    assert harness.calls.count("t1") == 3


def test_mode_rejects_repeat(drive: Any) -> None:
    with pytest.raises(SystemExit):
        drive(["--mode", "gate", "--repeat", "3"], {})


def test_json_keeps_transcript(drive: Any) -> None:
    _, results, _ = drive(["--mode", "gate"], {"t1": ["fail", "pass", "pass"]})

    first = results["trials"][0]
    assert first["task_id"] == "t1"
    assert first["trial"] == 1
    assert first["outcome"] == "failed"
    assert first["failure_reasons"] == ["missing ['ok']"]
    assert first["transcript"] == {
        "answer": "nope",
        "tool_calls": [{"name": "search_web", "arguments": "search_web(query='q')"}],
        "error": None,
        "seeded_turns": 0,
        "summarized": False,
        "tool_results": [],
    }
    assert first["question"] == "q"
    assert first["trace_id"] is None
    assert first["judges"] == []
    assert (first["tokens_in"], first["tokens_out"]) == (1000, 100)


def test_json_keeps_error_text(drive: Any) -> None:
    _, results, _ = drive(["--mode", "gate"], {"t1": ["error"]})

    errored = results["trials"][0]
    assert errored["outcome"] == "errored"
    assert errored["failure_reasons"] == []
    assert errored["transcript"]["error"] == "provider down"


def test_json_has_run_versions_and_cost(drive: Any) -> None:
    _, results, _ = drive(
        [
            "--mode",
            "release",
            "--price-in",
            "1",
            "--price-out",
            "2",
            "--judge-price-in",
            "3",
            "--judge-price-out",
            "4",
        ],
        {},
        suite="rag",
        tasks=1,
        tags=("b6",),
    )

    assert results["model"] == "fake-model"
    assert results["prompt_version"] == "pv-test"
    assert len(results["dataset_version"]) == 12
    # 10 trials x (1000 in, 100 out) plus one b6 judge call each, 10 x (200 in, 20 out).
    assert results["cost"]["tokens_in"] == 10_000
    assert results["cost"]["judge_tokens_in"] == 2_000
    assert results["judge_model"] == "fake-judge"
    # Each model's tokens at its own rates.
    expected = 10_000 * 1 + 1_000 * 2 + 2_000 * 3 + 200 * 4
    assert results["cost"]["usd"] == pytest.approx(expected / 1_000_000)


def test_markdown_has_task_line(drive: Any, capsys: pytest.CaptureFixture[str]) -> None:
    drive(["--mode", "gate"], {"t1": ["fail", "pass", "pass"]})

    assert "- t1: 2 passed, 1 failed, 0 errored" in capsys.readouterr().out


def test_cost_math() -> None:
    cost = estimate_cost(
        tokens_in=1_000_000,
        tokens_out=500_000,
        judge_tokens_in=100_000,
        judge_tokens_out=0,
        price_in_per_million=0.30,
        price_out_per_million=1.20,
    )

    assert cost.usd == pytest.approx(1.1 * 0.30 + 0.5 * 1.20)


def test_dataset_version_tracks_content(tmp_path: Path) -> None:
    datasets, fixtures = tmp_path / "datasets", tmp_path / "fixtures"
    datasets.mkdir()
    fixtures.mkdir()
    (datasets / "tools.jsonl").write_text("{}\n")
    (fixtures / "a.txt").write_text("one")
    before = dataset_version(datasets, fixtures, ["tools"])

    assert dataset_version(datasets, fixtures, ["tools"]) == before
    (fixtures / "a.txt").write_text("two")
    assert dataset_version(datasets, fixtures, ["tools"]) != before


def _prompt_version(
    prompt: str,
    specs: list[ToolSpec],
    *,
    model: str = "cheap-model",
    prompts: tuple[str, ...] = ("summary", "tool", "document"),
) -> str:
    return compute_prompt_version(prompt, specs, condenser_model=model, condenser_prompts=prompts)


def test_prompt_version_tracks_prompt_and_tools() -> None:
    spec = ToolSpec(name="a", description="finds files", parameters={"type": "object"})
    other = ToolSpec(name="b", description="searches", parameters={"type": "object"})
    base = _prompt_version("be helpful", [spec, other])

    assert _prompt_version("be helpful", [other, spec]) == base
    assert _prompt_version("be kind", [spec, other]) != base
    reworded = spec.model_copy(update={"description": "finds docs"})
    assert _prompt_version("be helpful", [reworded, other]) != base


def test_prompt_version_tracks_condenser() -> None:
    """The summary text and the condenser model decide what the model reads on a
    long chat, so changing either is a different prompt."""
    spec = ToolSpec(name="a", description="finds files", parameters={"type": "object"})
    base = _prompt_version("be helpful", [spec])

    assert _prompt_version("be helpful", [spec], model="other-model") != base
    assert (
        _prompt_version("be helpful", [spec], prompts=("new summary", "tool", "document")) != base
    )
    assert (
        _prompt_version("be helpful", [spec], prompts=("summary", "new tool", "document")) != base
    )
    assert _prompt_version("be helpful", [spec], prompts=("summary", "tool", "new doc")) != base


# --- results grouped by behavior ---------------------------------------------------


def _behavior_cases() -> list[EvalCase]:
    return [
        _case(id="t1", tags=["b1"], expect=Expectation(must_contain=["ok"])),
        _case(id="t2", tags=["ci", "b1"], expect=Expectation(must_contain=["ok"])),
        _case(id="t3", tags=["b3"], expect=Expectation(must_contain=["ok"])),
        _case(id="t4", expect=Expectation(must_contain=["ok"])),
    ]


def test_behavior_of_reads_the_one_tag() -> None:
    assert [behavior_of(case) for case in _behavior_cases()] == ["b1", "b1", "b3", None]


def test_behavior_sums_trials_per_behavior() -> None:
    tasks = (
        TaskRecord("t1", "tools", passed=2, failed=1, errored=0, behavior="b1"),
        TaskRecord("t2", "tools", passed=1, failed=0, errored=2, behavior="b1"),
        TaskRecord("t3", "rag", passed=3, failed=0, errored=0, behavior="b3"),
        TaskRecord("t4", "rag", passed=0, failed=4, errored=0),
    )

    assert behavior_records(tasks) == (
        BehaviorRecord("b1", tasks=2, passed=3, failed=1, errored=2),
        BehaviorRecord("b3", tasks=1, passed=3, failed=0, errored=0),
        BehaviorRecord("untagged", tasks=1, passed=0, failed=4, errored=0),
    )


def test_json_groups_results_by_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cases = _behavior_cases()
    harness = _ScriptedHarness({"t1": ["fail", "fail", "fail"], "t3": ["error"]})

    @asynccontextmanager
    async def fake_build(**_: object) -> AsyncIterator[_ScriptedHarness]:
        yield harness

    monkeypatch.setattr(driver, "build_harness", fake_build)
    monkeypatch.setattr(driver, "_select_cases", lambda *_: cases)
    out = tmp_path / "behaviors.json"
    driver.main(["--mode", "gate", "--out", str(out)])
    results = json.loads(out.read_text())

    assert results["behaviors"] == [
        {"behavior": "b1", "tasks": 2, "passed": 1, "failed": 3, "errored": 0},
        {"behavior": "b3", "tasks": 1, "passed": 0, "failed": 0, "errored": 1},
        {"behavior": "untagged", "tasks": 1, "passed": 1, "failed": 0, "errored": 0},
    ]
    assert [(t["task_id"], t["behavior"]) for t in results["tasks"]] == [
        ("t1", "b1"),
        ("t2", "b1"),
        ("t3", "b3"),
        ("t4", "untagged"),
    ]


def test_markdown_has_behavior_table() -> None:
    report = RunReport(
        run_id="r",
        model="m",
        tags=(),
        suites=(),
        case_ids=(),
        samples=0,
        regressions=(),
        behaviors=(BehaviorRecord("b2", tasks=3, passed=7, failed=1, errored=2),),
    )

    text = render_markdown(report)

    assert "| behavior | tasks | passed | failed | errored |" in text
    assert "| b2 | 3 | 7 | 1 | 2 |" in text


def test_tag_selects_one_behavior() -> None:
    """`--tag b1` is the whole selector: the existing `--tag` option, no new flag."""
    selected = _select_cases("all", None, ["b3"])

    assert [case.id for case in selected] == [
        "tools-012-empty-account-cannot-retrieve",
        "tools-013-empty-account-needs-no-tool",
        "rag-014-empty-account-has-nothing-to-cite",
    ]


@pytest.fixture
def built_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Run the CLI once and return the keyword arguments `build_harness` got."""

    def _run(argv: list[str]) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        @asynccontextmanager
        async def fake_build(**kwargs: Any) -> AsyncIterator[_ScriptedHarness]:
            seen.update(kwargs)
            harness = _ScriptedHarness({})
            harness.web_mode = kwargs["web_mode"]
            yield harness

        monkeypatch.setattr(driver, "build_harness", fake_build)
        monkeypatch.setattr(
            driver, "_select_cases", lambda *_: [_case(expect=Expectation(must_contain=["ok"]))]
        )
        driver.main([*argv, "--out", str(tmp_path / "r.json")])
        seen["results"] = json.loads((tmp_path / "r.json").read_text())
        return seen

    return _run


def test_web_tools_are_recorded_unless_live_is_asked_for(built_with: Any) -> None:
    seen = built_with(["--mode", "gate"])

    assert seen["web_mode"] == "recorded"
    assert seen["results"]["web_mode"] == "recorded"


def test_web_live_reaches_the_harness_and_the_results_file(built_with: Any) -> None:
    seen = built_with(["--mode", "gate", "--web", "live"])

    assert seen["web_mode"] == "live"
    assert seen["results"]["web_mode"] == "live"


def test_web_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        driver._parse_args(["--web", "cached"])


def test_markdown_names_the_web_mode() -> None:
    report = RunReport(
        run_id="r",
        model="m",
        tags=(),
        suites=(),
        case_ids=(),
        samples=0,
        regressions=(),
        web_mode="live",
    )

    assert "- **web tools**: live" in render_markdown(report)


def test_dataset_version_tracks_recorded_web_pages(tmp_path: Path) -> None:
    datasets, fixtures = tmp_path / "datasets", tmp_path / "fixtures"
    (fixtures / "web" / "fetch").mkdir(parents=True)
    datasets.mkdir()
    (datasets / "tools.jsonl").write_text("{}\n")
    page = fixtures / "web" / "fetch" / "p.txt"
    page.write_text("one")
    before = dataset_version(datasets, fixtures, ["tools"])

    page.write_text("two")

    assert dataset_version(datasets, fixtures, ["tools"]) != before


def test_dataset_version_moves_when_history_filler_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone edits the filler sentences. Every padded trial now sees different
    text, but the version hash read only .jsonl and fixture files, so two
    non-comparable runs carried the same dataset version."""
    evals_dir = Path(__file__).parents[2] / "evals"
    [task] = [
        c for c in load_cases(Path("tools.jsonl")) if c.id == "tools-018-recall-fact-after-summary"
    ]
    before_text = expanded_history(task.history)
    before = dataset_version(evals_dir / "datasets", evals_dir / "fixtures", ["tools"])

    monkeypatch.setattr(cases_module, "_FILLER_SENTENCES", ("A different sentence.",))

    assert expanded_history(task.history) != before_text
    assert dataset_version(evals_dir / "datasets", evals_dir / "fixtures", ["tools"]) != before
