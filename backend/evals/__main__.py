"""CLI entry point for the eval suites.

    uv run python -m evals --suite tools --limit 10 --concurrency 4 \\
        --out evals/results/latest.json

This runs against real APIs and spends real money - see `evals.harness` for
what it builds and `evals.judge` for the one extra model call each `rag` case
makes on top of the reply itself.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from loguru import logger

from evals.cases import EvalCase, load_cases
from evals.harness import CaseRun, EvalHarness, build_harness
from evals.judge import JudgeScore
from evals.metrics.budgets import summarize_budgets
from evals.metrics.generation import check_substrings, summarize_generation
from evals.metrics.retrieval import hit_rate_at_k, mrr_at_k, sources_returned, summarize_retrieval
from evals.metrics.tools import score_tool_call, summarize_tool_scores, tools_called
from evals.report import RunReport, SuiteReport, render_markdown, write_json

_RETRIEVAL_K = 3
"""How many of the top returned sources count toward hit-rate and MRR. Matches
`Settings.retrieval_limit`'s default, so the metric grades the same window the
agent's own retrieval call actually returns."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run the chat agent's eval suites against real APIs.",
    )
    parser.add_argument("--suite", choices=["tools", "rag", "all"], default="all")
    parser.add_argument(
        "--limit", type=int, default=None, help="Run at most this many cases per suite."
    )
    parser.add_argument("--concurrency", type=int, default=1, help="How many cases to run at once.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evals/results/latest.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--tag", action="append", default=[], help="Only run cases carrying this tag. May repeat."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the answering model. Defaults to the first configured LLM model.",
    )
    return parser.parse_args(argv)


def _select_cases(suite: str, limit: int | None, tags: list[str]) -> list[EvalCase]:
    suite_names = ["tools", "rag"] if suite == "all" else [suite]
    selected: list[EvalCase] = []
    for name in suite_names:
        cases = [case for case in load_cases(Path(f"{name}.jsonl")) if case.suite == name]
        if tags:
            cases = [case for case in cases if set(tags) & set(case.tags)]
        if limit is not None:
            cases = cases[:limit]
        selected.extend(cases)
    return selected


async def _run_all(cases: list[EvalCase], harness: EvalHarness, concurrency: int) -> list[CaseRun]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(case: EvalCase) -> CaseRun:
        async with semaphore:
            return await harness.run_case(case)

    return list(await asyncio.gather(*(_one(case) for case in cases)))


async def _score_suite(suite_name: str, runs: list[CaseRun], harness: EvalHarness) -> SuiteReport:
    latencies = [run.latency_ms for run in runs]
    usages = [run.usage for run in runs]
    budgets = summarize_budgets(latencies, usages)

    tools_report = None
    if suite_name == "tools":
        scores = [score_tool_call(tools_called(run.events), run.case.expect.tools) for run in runs]
        tools_report = summarize_tool_scores(scores)

    retrieval_report = None
    generation_report = None
    if suite_name == "rag":
        hit_rates: list[float] = []
        reciprocal_ranks: list[float] = []
        judge_scores: list[JudgeScore | None] = []
        substring_checks = []
        for run in runs:
            returned = sources_returned(run.events)
            expected = run.case.expect.expected_sources
            hit_rates.append(hit_rate_at_k(returned, expected, _RETRIEVAL_K))
            reciprocal_ranks.append(mrr_at_k(returned, expected, _RETRIEVAL_K))
            substring_checks.append(
                check_substrings(
                    run.answer, run.case.expect.must_contain, run.case.expect.must_not_contain
                )
            )
            # Reference text stands in for "context" when there is one; failing
            # that, the sources the reply actually cited are what is left to
            # judge groundedness against. Neither `ReplyToolFinished` nor any
            # other reply event carries the retrieved passage text itself, so
            # this is the best grounding context available without adding new
            # instrumentation to the reply path.
            context = run.case.expect.reference or "\n".join(returned)
            judge_scores.append(
                await harness.judge.score(
                    question=run.case.input, answer=run.answer, context=context
                )
            )
        retrieval_report = summarize_retrieval(hit_rates, reciprocal_ranks)
        generation_report = summarize_generation(judge_scores, substring_checks)

    return SuiteReport(
        suite=suite_name,
        n=len(runs),
        tools=tools_report,
        retrieval=retrieval_report,
        generation=generation_report,
        budgets=budgets,
    )


def _failures(runs: list[CaseRun]) -> list[str]:
    failures: list[str] = []
    for run in runs:
        if run.error is not None:
            failures.append(f"{run.case.id}: {run.error}")
        elif run.failed is not None:
            failures.append(f"{run.case.id}: {run.failed.detail}")
    return failures


def _routing_failures(suite_name: str, runs: list[CaseRun]) -> list[str]:
    """Per-case tool-selection and retrieval misses, worth failing the run over.

    `_score_suite` only ever produces aggregate rates, which read fine in a
    markdown table but hide exactly which case regressed - and an aggregate
    staying "close enough" is how a routing bug like tools calling
    `search_web` for a document-only question ships unnoticed. This names the
    offending case directly, the same way `_failures` already does for a
    crash.
    """
    failures: list[str] = []
    if suite_name == "tools":
        for run in runs:
            score = score_tool_call(tools_called(run.events), run.case.expect.tools)
            if not score.exact_match:
                failures.append(
                    f"{run.case.id}: expected tools {list(score.expected)}, "
                    f"got {list(score.called)}"
                )
    elif suite_name == "rag":
        for run in runs:
            expected = run.case.expect.expected_sources
            if not expected:
                continue
            returned = sources_returned(run.events)
            if hit_rate_at_k(returned, expected, _RETRIEVAL_K) == 0.0:
                failures.append(
                    f"{run.case.id}: expected one of {expected} in top-{_RETRIEVAL_K}, "
                    f"got {returned[:_RETRIEVAL_K]}"
                )
    return failures


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = uuid4().hex[:12]

    cases = _select_cases(args.suite, args.limit, args.tag)
    if not cases:
        logger.warning("No cases matched the given --suite/--tag/--limit; nothing to run.")
        return 0

    async with build_harness(run_id=run_id, model=args.model) as harness:
        if any(case.suite == "rag" for case in cases):
            await harness.index_fixtures()

        runs = await _run_all(cases, harness, args.concurrency)

        by_suite: dict[str, list[CaseRun]] = {}
        for run in runs:
            by_suite.setdefault(run.case.suite, []).append(run)

        suite_reports = [
            await _score_suite(suite_name, suite_runs, harness)
            for suite_name, suite_runs in by_suite.items()
        ]
        routing_failures = [
            failure
            for suite_name, suite_runs in by_suite.items()
            for failure in _routing_failures(suite_name, suite_runs)
        ]
        report = RunReport(
            run_id=run_id,
            model=harness.model,
            tags=tuple(args.tag),
            suites=tuple(suite_reports),
            case_ids=tuple(case.id for case in cases),
            failures=tuple(_failures(runs) + routing_failures),
        )

    write_json(report, args.out)
    # The markdown table is the deliverable this CLI exists to produce - see
    # the module docstring in `evals/report.py` - so this is a print, not a
    # log line: it belongs on stdout for a human (or a PR description) to
    # read, not wrapped in loguru's structured format.
    print(render_markdown(report))  # noqa: T201
    return 1 if report.failures else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    sys.exit(main())
