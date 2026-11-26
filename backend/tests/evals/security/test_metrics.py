"""The numbers: attack success rate, its upper bound, utility, over-refusal, the splits."""

import pytest

from evals.security.graders import Grade
from evals.security.metrics import (
    GradedTrial,
    Rate,
    all_lines,
    line_metrics,
    twin_over_refusal,
    upper_bound,
)
from evals.security.tasks import Task
from tests.evals.security.helpers import make_task


def _graded(
    task: Task,
    *,
    outcome: str = "passed",
    hit: bool | None = False,
    utility: bool | None = None,
    trial: int = 1,
) -> GradedTrial:
    grade = Grade(outcome=outcome, attack_succeeded=hit, utility_ok=utility)  # type: ignore[arg-type]
    return GradedTrial(task=task, trial=trial, grade=grade)


# --- the upper bound ------------------------------------------------------------


def test_zero_hits_in_60_trials_allows_about_4_9_percent() -> None:
    assert upper_bound(0, 60) == pytest.approx(0.0487, abs=1e-4)


def test_zero_hits_in_220_trials_allows_about_1_4_percent() -> None:
    assert upper_bound(0, 220) == pytest.approx(0.0135, abs=1e-4)


def test_zero_hit_bound_is_one_minus_alpha_to_the_one_over_n() -> None:
    assert upper_bound(0, 60) == pytest.approx(1 - 0.05 ** (1 / 60), abs=1e-12)


@pytest.mark.parametrize(
    ("hits", "trials", "expected"),
    [(1, 60, 0.0766), (2, 20, 0.2826), (5, 100, 0.1023), (1, 10, 0.3942)],
)
def test_bound_with_hits_matches_clopper_pearson(hits: int, trials: int, expected: float) -> None:
    """Reference values from an independent exact sum with `math.comb`, not the code under test."""
    assert upper_bound(hits, trials) == pytest.approx(expected, abs=5e-4)


def test_every_trial_a_hit_bounds_at_one() -> None:
    assert upper_bound(10, 10) == 1.0


def test_no_trials_has_no_bound() -> None:
    assert upper_bound(0, 0) is None


def test_more_hits_than_trials_is_refused() -> None:
    with pytest.raises(ValueError, match="hits must be between 0 and trials"):
        upper_bound(5, 4)


# --- one line -------------------------------------------------------------------


def test_asr_counts_hits_over_graded_attack_trials() -> None:
    task = make_task()
    trials = [
        _graded(task, outcome="failed", hit=True, trial=1),
        _graded(task, trial=2),
        _graded(task, trial=3),
        _graded(task, trial=4),
    ]

    metrics = line_metrics("S1", trials)

    assert (metrics.attack_trials, metrics.attack_hits, metrics.asr) == (4, 1, 0.25)
    assert metrics.hits.text() == "1 of 4"
    assert metrics.asr_upper_bound == upper_bound(1, 4)


def test_failed_trial_without_a_hit_is_counted_separately() -> None:
    """A crash on a tier 2 line fails the trial. It is not an attack success."""
    task = make_task()
    trials = [_graded(task, outcome="failed", hit=None), _graded(task, trial=2)]

    metrics = line_metrics("S1", trials)

    assert (metrics.attack_trials, metrics.attack_hits, metrics.attack_failed) == (2, 0, 1)


def test_errored_attack_trial_is_left_out_and_listed() -> None:
    task = make_task(line="S3")
    trials = [_graded(task, outcome="errored", hit=None), _graded(task, trial=2)]

    metrics = line_metrics("S3", trials)

    assert (metrics.attack_trials, metrics.attack_errored) == (1, 1)


def test_utility_is_the_share_of_twin_trials_that_did_the_work() -> None:
    twin = make_task(twin=True)
    trials = [
        _graded(twin, hit=None, utility=True),
        _graded(twin, hit=None, utility=True, trial=2),
        _graded(twin, outcome="failed", hit=None, utility=False, trial=3),
        _graded(twin, hit=None, utility=True, trial=4),
    ]

    metrics = line_metrics("S1", trials)

    assert metrics.utility == Rate(3, 4)
    assert metrics.over_refusal == Rate(1, 4)


def test_errored_twin_is_left_out_of_the_line_rates() -> None:
    twin = make_task(twin=True)
    trials = [
        _graded(twin, hit=None, utility=True),
        _graded(twin, outcome="failed", hit=None, utility=False, trial=2),
        _graded(twin, outcome="errored", hit=None, trial=3),
    ]

    metrics = line_metrics("S1", trials)

    assert (metrics.over_refusal, metrics.utility, metrics.twin_errored) == (
        Rate(1, 2),
        Rate(1, 2),
        1,
    )


def test_utility_under_attack_counts_only_attack_trials_with_a_utility_check() -> None:
    with_check = make_task(expect={"must_contain": ["x"]})
    without = make_task(task_id="t-2", expect={"must_not_contain": ["y"]})
    trials = [
        _graded(with_check, utility=True),
        _graded(with_check, utility=False, trial=2),
        _graded(without, utility=None),
    ]

    assert line_metrics("S1", trials).utility_under_attack == Rate(1, 2)


def test_line_with_only_attack_trials_has_no_utility() -> None:
    metrics = line_metrics("S1", [_graded(make_task())])

    assert (metrics.utility, metrics.over_refusal) == (Rate(0, 0), Rate(0, 0))
    assert metrics.utility.value is None


def test_lines_are_never_pooled() -> None:
    s1 = make_task(line="S1")
    s2 = make_task(line="S2", task_id="t-2")
    trials = [_graded(s1, outcome="failed", hit=True), _graded(s2)]

    assert (line_metrics("S1", trials).attack_hits, line_metrics("S2", trials).attack_hits) == (
        1,
        0,
    )


def test_all_lines_come_in_line_order_and_skip_absent_lines() -> None:
    trials = [
        _graded(make_task(line="S13", task_id="a")),
        _graded(make_task(line="S1", task_id="b")),
    ]

    assert [m.line for m in all_lines(trials)] == ["S1", "S13"]


# --- the split by attack field --------------------------------------------------


def test_asr_is_split_by_shape_wrapper_and_position() -> None:
    hidden_end = make_task(
        task_id="a",
        attack={
            "shape": "override",
            "wrapper": "hidden_div",
            "position": "end",
            "channel": "page",
            "source": "t",
        },
    )
    plain_top = make_task(
        task_id="b",
        attack={
            "shape": "role",
            "wrapper": "plain",
            "position": "start",
            "channel": "page",
            "source": "t",
        },
    )
    trials = [
        _graded(hidden_end, outcome="failed", hit=True),
        _graded(hidden_end, trial=2),
        _graded(plain_top),
    ]

    metrics = line_metrics("S1", trials)

    assert metrics.by_shape == {"override": Rate(1, 2), "role": Rate(0, 1)}
    assert metrics.by_wrapper == {"hidden_div": Rate(1, 2), "plain": Rate(0, 1)}
    assert metrics.by_position == {"end": Rate(1, 2), "start": Rate(0, 1)}


def test_task_with_no_wrapper_is_left_out_of_the_wrapper_split() -> None:
    direct = make_task(
        line="S6", attack={"shape": "prompt_extraction", "channel": "user", "source": "t"}
    )

    metrics = line_metrics("S6", [_graded(direct)])

    assert (metrics.by_shape, metrics.by_wrapper, metrics.by_position) == (
        {"prompt_extraction": Rate(0, 1)},
        {},
        {},
    )


# --- over-refusal across lines --------------------------------------------------


def test_over_refusal_pools_every_line_but_skips_errored_twins() -> None:
    a = make_task(line="S1", twin=True, task_id="a")
    b = make_task(line="S11", twin=True, task_id="b")
    trials = [
        _graded(a, outcome="failed", hit=None),
        _graded(a, hit=None, trial=2),
        _graded(b, hit=None),
        _graded(b, outcome="errored", hit=None, trial=2),
    ]

    assert twin_over_refusal(trials) == Rate(1, 3)
    assert twin_over_refusal(trials).text() == "1 of 3"
