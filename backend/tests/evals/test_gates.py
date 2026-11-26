"""The two pass bars, checked at their edges with no model involved.

A bar that is off by one either lets a regression through or blocks a good
build, and neither shows up in a run that costs money. So the numbers from
`evals/eval-design.md` are pinned here: the gate reruns only a FAILED first
trial, and the release check allows 8 failures in 280 graded trials but not 9,
and 2 failures in one task but not 3.
"""

from evals.metrics.gates import (
    Outcome,
    TaskCounts,
    TrialResult,
    count_tasks,
    judge_gate,
    judge_release,
    release_allowed_failures,
    tasks_to_rerun,
)


def _trials(task_id: str, *outcomes: Outcome) -> list[TrialResult]:
    return [TrialResult(task_id, n, outcome) for n, outcome in enumerate(outcomes, start=1)]


def _suite(failures_per_task: list[int], *, trials_per_task: int = 10) -> list[TrialResult]:
    """Tasks with the given failure counts, the rest of their trials passing."""
    trials: list[TrialResult] = []
    for index, failures in enumerate(failures_per_task):
        outcomes: list[Outcome] = []
        outcomes += ["failed"] * failures
        outcomes += ["passed"] * (trials_per_task - failures)
        trials.extend(_trials(f"task-{index}", *outcomes))
    return trials


def test_gate_passes_clean_first_trial() -> None:
    verdict = judge_gate(_trials("a", "passed") + _trials("b", "passed"))

    assert not verdict.failed
    assert tasks_to_rerun(_trials("a", "passed")) == []


def test_gate_forgives_fail_then_pass() -> None:
    """One bad first trial is luck. Only a rerun that fails too is a break."""
    trials = _trials("a", "failed", "passed", "passed")

    assert tasks_to_rerun(trials) == ["a"]
    assert not judge_gate(trials).failed


def test_gate_blocks_fail_then_fail() -> None:
    verdict = judge_gate(_trials("a", "failed", "passed", "failed"))

    assert verdict.failed
    assert "a failed its first trial and then failed again on trial 3" in verdict.reasons[0]


def test_gate_does_not_rerun_errored() -> None:
    """An errored first trial was never graded, so it is not a failure to confirm."""
    trials = _trials("a", "errored")

    assert tasks_to_rerun(trials) == []
    assert not judge_gate(trials).failed


def test_gate_ignores_errored_rerun() -> None:
    """A rerun the provider dropped proves nothing either way."""
    assert not judge_gate(_trials("a", "failed", "errored", "passed")).failed


def test_release_allows_eight_of_280() -> None:
    trials = _suite([2, 2, 2, 2] + [0] * 24)

    assert len(trials) == 280
    assert not judge_release(trials).failed


def test_release_blocks_nine_of_280() -> None:
    trials = _suite([2, 2, 2, 2, 1] + [0] * 23)

    verdict = judge_release(trials)

    assert verdict.failed
    assert verdict.reasons == (
        "The suite failed 9 of 280 graded trials. The most it may fail is 8 (3%).",
    )


def test_release_allows_two_failures_in_a_task() -> None:
    assert not judge_release(_suite([2] + [0] * 27)).failed


def test_release_blocks_three_failures_in_a_task() -> None:
    """3 of 280 is well under 3%, so only the per-task limit can fail this."""
    verdict = judge_release(_suite([3] + [0] * 27))

    assert verdict.failed
    assert verdict.reasons == ("task-0 failed 3 of 10 graded trials. A task may fail at most 2.",)


def test_release_excludes_errored_from_both_bars() -> None:
    """Errors do not count as failures, and do not pad the 3% denominator."""
    erroring = [t for t in _suite([0] * 10) if t.number <= 5]
    erroring = [TrialResult(t.task_id, t.number, "errored") for t in erroring]
    # 100 graded trials, 3 failed: allowed. 50 errored trials change nothing.
    ok = _suite([2, 1] + [0] * 8) + erroring
    assert not judge_release(ok).failed

    too_many = _suite([2, 2] + [0] * 8) + erroring
    assert judge_release(too_many).failed


def test_release_threshold_uses_integers() -> None:
    assert release_allowed_failures(280) == 8
    assert release_allowed_failures(299) == 8
    assert release_allowed_failures(300) == 9
    assert release_allowed_failures(0) == 0


def test_counts_per_task() -> None:
    trials = _trials("a", "passed", "failed", "errored", "passed") + _trials("b", "failed")

    assert count_tasks(trials) == [
        TaskCounts("a", passed=2, failed=1, errored=1),
        TaskCounts("b", passed=0, failed=1, errored=0),
    ]


def test_gate_summary_names_errored_first_trials() -> None:
    """Two tasks ran and one errored. "None failed" alone would read as two clean passes."""
    trials = _trials("a", "passed") + _trials("b", "errored")

    verdict = judge_gate(trials)

    assert not verdict.failed
    assert verdict.reasons == (
        "2 task(s) ran. None failed the first trial. 1 errored, not graded: b.",
    )
