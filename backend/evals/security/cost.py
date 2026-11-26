"""What a security run is expected to cost, worked out before the first paid call.

The token counts below were measured on the main suite (4,200 input and 300
output tokens for a normal trial) and on a padded B5 task (13,000 more
condenser tokens). They have not been measured on the security suite. Recheck
them after the first run, with the tokens the results file reports.

Prices are the defaults of `evals.__main__`: the agent model, and the cheaper
model that writes summaries and judges.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from evals.security.gates import PlannedTrial
from evals.security.tasks import Task

TRIAL_TOKENS_IN = 4_200
"""Agent input tokens of a normal trial. Measured on the main suite."""

TRIAL_TOKENS_OUT = 300
"""Agent output tokens of a normal trial. Measured on the main suite."""

PADDED_CONDENSER_TOKENS_IN = 13_000
"""Condenser input tokens a padded trial adds, for the summary it writes."""

PADDED_AGENT_TOKENS_IN = 7_000
"""Agent input tokens a padded trial adds on top of a normal trial."""

JUDGED_TOKENS_IN = 1_000
"""Judge input tokens of one judged trial."""

JUDGED_TOKENS_OUT = 80
"""Judge output tokens of one judged trial."""

PRICE_IN = 0.12
"""USD per million agent input tokens."""

PRICE_OUT = 0.48
"""USD per million agent output tokens."""

CONDENSER_PRICE_IN = 0.15
"""USD per million condenser input tokens."""

JUDGE_PRICE_IN = 0.15
"""USD per million judge input tokens."""

JUDGE_PRICE_OUT = 0.50
"""USD per million judge output tokens."""


@dataclass(frozen=True, slots=True)
class Prices:
    """USD per million tokens."""

    agent_in: float = PRICE_IN
    agent_out: float = PRICE_OUT
    condenser_in: float = CONDENSER_PRICE_IN
    judge_in: float = JUDGE_PRICE_IN
    judge_out: float = JUDGE_PRICE_OUT


@dataclass(frozen=True, slots=True)
class Estimate:
    tasks: int
    trials: int
    judged_trials: int
    padded_trials: int
    usd: float
    usd_by_line: dict[str, float]

    def text(self) -> str:
        return (
            f"{self.tasks} tasks, {self.trials} trials "
            f"({self.padded_trials} padded, {self.judged_trials} judged), "
            f"estimated cost ${self.usd:.4f}"
        )


def trial_usd(task: Task, *, judged: bool, prices: Prices | None = None) -> float:
    """The expected cost of one trial of `task`."""
    prices = prices or Prices()
    tokens_in = TRIAL_TOKENS_IN
    usd = TRIAL_TOKENS_OUT * prices.agent_out
    if task.pad_history_to_tokens:
        tokens_in += PADDED_AGENT_TOKENS_IN
        usd += PADDED_CONDENSER_TOKENS_IN * prices.condenser_in
    usd += tokens_in * prices.agent_in
    if judged and task.expect.judge is not None:
        usd += JUDGED_TOKENS_IN * prices.judge_in + JUDGED_TOKENS_OUT * prices.judge_out
    return usd / 1_000_000


def estimate_plan(
    plan: Sequence[PlannedTrial], *, judges: bool, prices: Prices | None = None
) -> Estimate:
    """The expected cost of a planned run. Builds no harness and calls no API."""
    prices = prices or Prices()
    by_line: dict[str, float] = defaultdict(float)
    for planned in plan:
        by_line[planned.task.line] += trial_usd(planned.task, judged=judges, prices=prices)
    return Estimate(
        tasks=len({planned.task.id for planned in plan}),
        trials=len(plan),
        judged_trials=sum(1 for p in plan if judges and p.task.expect.judge is not None),
        padded_trials=sum(1 for p in plan if p.task.pad_history_to_tokens),
        usd=sum(by_line.values()),
        usd_by_line=dict(by_line),
    )


def cheapest_first(plan: Sequence[PlannedTrial], *, judges: bool) -> list[str]:
    """The lines of `plan`, cheapest line first, so a release run can stop early cheaply."""
    by_line = estimate_plan(plan, judges=judges).usd_by_line
    return sorted(by_line, key=lambda line: (by_line[line], line))
