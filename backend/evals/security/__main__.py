"""CLI entry point for the security evaluation suite.

    uv run python -m evals.security --mode gate
    uv run python -m evals.security --mode release --keep-going --out evals/results/baseline.json
    uv run python -m evals.security --mode smoke --line S6 --limit 3
    uv run python -m evals.security --mode release --estimate

This spends real money. Every paid mode prints its estimate first and refuses to
start, with exit code 2, when the estimate is above `--max-usd`. `--estimate`
prints the estimate and stops: no harness is built and no API is called.

Three modes, see `evals.security.gates`:

    gate     S1, S2 and S3 attack tasks, 1 trial each, no judge. Blocks on any failure.
    release  Every task with its own trial count. Blocks unless the bars hold.
             Lines run cheapest first, and no new trial starts after the first
             blocking failure, unless `--keep-going`. Use `--keep-going` for a baseline.
    smoke    1 trial per task, judges on, never blocks. For reading transcripts.

Exit codes, as in `evals.__main__`: 0 clean, 1 a bar was missed, 2 the run cannot
be trusted: a trial errored, the estimate was over budget, or nothing was selected.
"""

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from uuid import uuid4

from loguru import logger

from evals.harness import build_harness
from evals.security.adapter import SecurityAdapter, TrialResult, page_wrapper
from evals.security.cost import Estimate, Prices, cheapest_first, estimate_plan
from evals.security.gates import (
    MODES,
    JudgeStatus,
    JudgeStatusError,
    Mode,
    PlannedTrial,
    Verdict,
    blocks_release,
    judge_gate,
    judge_release,
    judge_smoke,
    load_judge_status,
    plan_trials,
    uses_judges,
)
from evals.security.metrics import GradedTrial, all_lines
from evals.security.report import Cost, RunReport, TrialRecord, render_markdown, write_json
from evals.security.tasks import LINES, Task, TaskLoadError, dataset_version, load_tasks

DEFAULT_OUT = Path("evals/results/security-latest.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.security",
        description="Run the chat agent's security evaluation suite against real APIs.",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--line",
        action="append",
        default=[],
        choices=LINES,
        help="Only run this line. May repeat. Default: every line of the mode.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run at most this many tasks per line."
    )
    parser.add_argument("--concurrency", type=int, default=4, help="How many trials at once.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where the JSON goes.")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print the task count, trial count and estimated cost, then stop. Calls no API.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Release only: keep starting trials after a blocking failure. For a baseline.",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=1.0,
        help="Refuse to start when the estimate is above this many US dollars.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def build_plan(
    tasks: Sequence[Task], mode: Mode, *, lines: Sequence[str], limit: int | None
) -> list[PlannedTrial]:
    """The trials of one run. Release runs the cheapest line first."""
    plan = plan_trials(tasks, mode, lines=lines, limit=limit)
    if mode != "release":
        return plan
    order = cheapest_first(plan, judges=uses_judges(mode))
    return plan_trials(tasks, mode, lines=lines, limit=limit, line_order=order)


async def run_plan(
    plan: Sequence[PlannedTrial],
    run_one: Callable[[PlannedTrial], Awaitable[TrialResult]],
    *,
    concurrency: int,
    stops_the_run: Callable[[TrialResult], bool] | None = None,
) -> tuple[list[TrialResult], int, str | None]:
    """Run the plan, `concurrency` trials at once, in plan order.

    When `stops_the_run` says yes to a finished trial, no new trial starts. Trials
    already running finish. Returns the finished trials, how many never started,
    and why the run stopped (or `None`).
    """
    gate = asyncio.Semaphore(concurrency)
    stopped: list[str] = []

    async def one(planned: PlannedTrial) -> "TrialResult | None":
        async with gate:
            if stopped:
                return None
            result = await run_one(planned)
            if stops_the_run is not None and stops_the_run(result) and not stopped:
                stopped.append(
                    f"{result.task.id} trial {result.trial} failed on {result.task.line}"
                )
            return result

    outcomes = await asyncio.gather(*(one(planned) for planned in plan))
    finished = [outcome for outcome in outcomes if outcome is not None]
    return finished, len(plan) - len(finished), stopped[0] if stopped else None


def stop_rule(
    mode: Mode, keep_going: bool, judge_status: dict[str, JudgeStatus]
) -> Callable[[TrialResult], bool] | None:
    """When to stop starting trials: only a release run, and only without `--keep-going`.

    A trial that already blocks the release makes the rest of the run a waste of
    money, because the verdict cannot go back to clean.
    """
    if mode != "release" or keep_going:
        return None

    def stops(result: TrialResult) -> bool:
        return blocks_release(result.task, result.grade, judge_status)

    return stops


def assemble_report(
    *,
    run_id: str,
    mode: Mode,
    model: str,
    judge_model: str,
    prompt_version: str,
    tasks: Sequence[Task],
    results: Sequence[TrialResult],
    lines: Sequence[str],
    judge_status: dict[str, JudgeStatus],
    stopped_early: str | None,
    not_started: int,
    judge_tokens: tuple[int, int] = (0, 0),
    prices: Prices | None = None,
) -> RunReport:
    """The report of a finished run: metrics, the mode's verdict, the cost."""
    prices = prices or Prices()
    graded = [GradedTrial(r.task, r.trial, r.grade) for r in results]
    verdict: Verdict
    if mode == "gate":
        verdict = judge_gate(graded, lines=lines)
    elif mode == "release":
        verdict = judge_release(graded, judge_status, lines=lines)
    else:
        verdict = judge_smoke(graded)

    records: tuple[TrialRecord, ...] = tuple(r.record for r in results)
    tokens_in = sum(r.tokens_in for r in records)
    tokens_out = sum(r.tokens_out for r in records)
    usd = (
        tokens_in * prices.agent_in
        + tokens_out * prices.agent_out
        + judge_tokens[0] * prices.judge_in
        + judge_tokens[1] * prices.judge_out
    ) / 1_000_000
    return RunReport(
        run_id=run_id,
        mode=mode,
        model=model,
        judge_model=judge_model,
        prompt_version=prompt_version,
        dataset_version=dataset_version(),
        tasks=tuple(tasks),
        trials=records,
        lines=tuple(all_lines(graded)),
        verdict=verdict,
        cost=Cost(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            judge_tokens_in=judge_tokens[0],
            judge_tokens_out=judge_tokens[1],
            usd=usd,
        ),
        stopped_early=stopped_early,
        not_started=not_started,
        judge_status=judge_status,
    )


def exit_code(report: RunReport) -> int:
    """0 clean, 1 a bar was missed, 2 the run cannot be trusted.

    A 1 is a prompt, a tool or a guard to fix. A 2 is a provider that fell over,
    and running it again is the whole response.
    """
    if report.verdict.failed:
        return 1
    if report.errored:
        return 2
    return 0


def _plain_estimate(estimate: Estimate, mode: Mode) -> str:
    return f"mode {mode}: {estimate.text()}"


async def _run(
    args: argparse.Namespace,
    tasks: list[Task],
    plan: list[PlannedTrial],
    judge_status: dict[str, JudgeStatus],
) -> int:
    mode: Mode = args.mode
    run_id = uuid4().hex[:12]
    use_judges = uses_judges(mode)

    async with build_harness(
        run_id=run_id, web_mode="recorded", wrap_fetcher=page_wrapper(tasks)
    ) as harness:
        adapter = SecurityAdapter(harness)
        await adapter.prepare_store()

        async def run_one(planned: PlannedTrial) -> TrialResult:
            return await adapter.run_trial(planned.task, planned.trial, judges=use_judges)

        results, not_started, stopped = await run_plan(
            plan,
            run_one,
            concurrency=args.concurrency,
            stops_the_run=stop_rule(mode, args.keep_going, judge_status),
        )
        report = assemble_report(
            run_id=run_id,
            mode=mode,
            model=harness.model,
            judge_model=harness.judge.model_name,
            prompt_version=harness.prompt_version,
            tasks=list({p.task.id: p.task for p in plan}.values()),
            results=results,
            lines=args.line,
            judge_status=judge_status,
            stopped_early=stopped,
            not_started=not_started,
            judge_tokens=(harness.judge.prompt_tokens, harness.judge.completion_tokens),
        )

    write_json(report, args.out)
    # The markdown is the deliverable of this CLI, so it is a print to stdout and
    # not a log line: a person or a CI job summary reads it as it is.
    print(render_markdown(report))  # noqa: T201
    return exit_code(report)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode: Mode = args.mode
    try:
        tasks = load_tasks()
        judge_status = load_judge_status()
    except (TaskLoadError, JudgeStatusError) as error:
        logger.error(f"Cannot start:\n{error}")
        return 2

    plan = build_plan(tasks, mode, lines=args.line, limit=args.limit)
    if not plan:
        logger.error(f"--mode {mode} selected no tasks, so nothing would be measured.")
        return 2
    estimate = estimate_plan(plan, judges=uses_judges(mode))

    if args.estimate:
        print(_plain_estimate(estimate, mode))  # noqa: T201
        return 0

    # On stderr: stdout holds only the markdown report, which CI pipes to a summary.
    print(_plain_estimate(estimate, mode), file=sys.stderr)  # noqa: T201
    if estimate.usd > args.max_usd:
        logger.error(
            f"The estimate ${estimate.usd:.4f} is above --max-usd {args.max_usd}. Not starting."
        )
        return 2
    return asyncio.run(_run(args, tasks, plan, judge_status))


if __name__ == "__main__":
    sys.exit(main())
