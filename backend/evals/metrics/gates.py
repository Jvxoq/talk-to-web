"""The two pass bars: the CI gate and the release check.

Both read the same thing, a list of trials with an outcome each, and neither
touches a model. That keeps the bars checkable with plain pytest, which matters
because a bar that is off by one lets a regression through or blocks a good
build, and neither shows up in a run that costs money.

Errored trials count for nothing. A rate limit or a provider outage is neither
a pass nor a failure, so an errored trial is left out of both bars and listed
separately. Counting it as a failure would block a build for the provider's
mistake. Counting it as a pass would hide the fact that nothing was measured.

The numbers come from `evals/eval-design.md`, "Redesign" step 6.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Outcome = Literal["passed", "failed", "errored"]

GATE_EXTRA_TRIALS = 2
"""How many more trials a task gets in the gate after its first one fails."""

RELEASE_TRIALS = 10
"""Trials per task in the release check."""

RELEASE_MAX_FAILURE_PERCENT = 3
"""The suite may fail at most this share of its graded trials (97% must pass)."""

RELEASE_TASK_FAILURE_LIMIT = 3
"""A task that fails this many trials, or more, blocks the release by itself."""


@dataclass(frozen=True, slots=True)
class TrialResult:
    """The outcome of one trial of one task. `number` starts at 1."""

    task_id: str
    number: int
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class TaskCounts:
    """How one task's trials ended."""

    task_id: str
    passed: int
    failed: int
    errored: int


@dataclass(frozen=True, slots=True)
class BarVerdict:
    """Whether a bar was met, and why, in words a person can read."""

    failed: bool
    reasons: tuple[str, ...]


def count_tasks(trials: Sequence[TrialResult]) -> list[TaskCounts]:
    """One record per task, in the order each task first appears."""
    counts: dict[str, list[int]] = {}
    for trial in trials:
        row = counts.setdefault(trial.task_id, [0, 0, 0])
        row[("passed", "failed", "errored").index(trial.outcome)] += 1
    return [TaskCounts(task_id, *row) for task_id, row in counts.items()]


def tasks_to_rerun(trials: Sequence[TrialResult]) -> list[str]:
    """Tasks whose first trial FAILED. An errored first trial is not rerun here."""
    return [t.task_id for t in trials if t.number == 1 and t.outcome == "failed"]


def judge_gate(trials: Sequence[TrialResult]) -> BarVerdict:
    """The CI gate: a task fails only if one of its extra trials fails.

    A task gets one trial. If that trial fails, it gets `GATE_EXTRA_TRIALS`
    more, and the build is blocked when any of those fails too. A healthy agent
    that misses a few answers in 100 fails a single trial now and then, and
    that alone must not block a merge.
    """
    first = [t for t in trials if t.number == 1]
    rerun_ids = set(tasks_to_rerun(trials))
    failed_again = [t for t in trials if t.number > 1 and t.outcome == "failed"]

    if failed_again:
        reasons = tuple(
            f"{t.task_id} failed its first trial and then failed again on trial {t.number}."
            for t in failed_again
        )
        return BarVerdict(failed=True, reasons=reasons)

    cleared = len(rerun_ids)
    summary = f"{len(first)} task(s) ran. "
    if cleared:
        summary += f"{cleared} failed the first trial and passed every rerun: "
        summary += ", ".join(sorted(rerun_ids)) + "."
    else:
        summary += "None failed the first trial."
    errored = sorted({t.task_id for t in first if t.outcome == "errored"})
    if errored:
        summary += f" {len(errored)} errored, not graded: " + ", ".join(errored) + "."
    return BarVerdict(failed=False, reasons=(summary,))


def release_allowed_failures(graded: int) -> int:
    """floor(3% of graded trials). 8 of 280.

    Integer arithmetic on purpose. `0.03 * 280` is 8.4 and floors to 8, but a
    float bar is one rounding error from letting a ninth failure through.
    """
    return graded * RELEASE_MAX_FAILURE_PERCENT // 100


def judge_release(trials: Sequence[TrialResult]) -> BarVerdict:
    """The release check: the suite stays under 3% failures, no task has 3."""
    counts = count_tasks(trials)
    graded = sum(c.passed + c.failed for c in counts)
    failures = sum(c.failed for c in counts)
    allowed = release_allowed_failures(graded)

    reasons: list[str] = []
    if failures > allowed:
        reasons.append(
            f"The suite failed {failures} of {graded} graded trials. "
            f"The most it may fail is {allowed} ({RELEASE_MAX_FAILURE_PERCENT}%)."
        )
    reasons.extend(
        f"{c.task_id} failed {c.failed} of {c.passed + c.failed} graded trials. "
        f"A task may fail at most {RELEASE_TASK_FAILURE_LIMIT - 1}."
        for c in counts
        if c.failed >= RELEASE_TASK_FAILURE_LIMIT
    )
    if reasons:
        return BarVerdict(failed=True, reasons=tuple(reasons))
    return BarVerdict(
        failed=False,
        reasons=(
            f"The suite failed {failures} of {graded} graded trials, "
            f"within the limit of {allowed}. No task failed "
            f"{RELEASE_TASK_FAILURE_LIMIT} or more trials.",
        ),
    )
