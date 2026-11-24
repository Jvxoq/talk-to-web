"""CLI entry point for the eval suites.

    uv run python -m evals --suite tools --limit 10 --concurrency 4 \\
        --out evals/results/latest.json

This runs against real APIs and spends real money - see `evals.harness` for
what it builds and `evals.judge` for the one extra model call each `rag` case
makes on top of the reply itself.

Two modes set how many trials each task gets and which bar the run must clear.
The bars themselves live in `evals.metrics.gates`.

    --mode gate     1 trial per task, no judge calls. A task whose trial
                    failed gets 2 more. Exit 1 if either extra trial fails.
    --mode release  10 trials per task, judges on. Exit 1 unless the suite
                    fails at most 3% of graded trials and no task fails 3.

With no `--mode`, `--repeat N` runs N trials per task and the old rule holds:
a task that fails every trial is a regression. That is the mode for trying out
a new task by hand.

An errored trial (a rate limit, a dropped connection) is neither a pass nor a
failure. It is left out of both bars and listed, and it turns the exit code
into 2 when nothing failed.

Four rules shape this module, and each one exists because breaking it made a
run report something other than what it measured.

One sample is not a verdict. A model at temperature 0 still varies between
calls, so a single run of a case cannot separate a regression from a coin
flip. `--repeat` runs each case several times: failing every sample is a
regression, failing some is flakiness, and those are facts about different
problems. Keeping them in one list breaks the suite in both directions - a
flaky case called red teaches everyone to re-run the build, and a flaky case
called green reports nothing at all.

A broken case is never scored. A rate limit or a dropped connection produces an
empty answer and no tool calls, which every metric here reads as "the model
called nothing and said nothing" - a perfect way to turn a flaky afternoon into
a routing regression. Such a run goes into `RunReport.errors` and is dropped
from every rate above it.

Fixtures are indexed on every run, not only for `rag`. Most of
the `tools` suite asks a question about an upload, and an unindexed corpus
makes `retrieve_documents` come back empty, which is exactly the state that
releases `search_web` - so the suite that exists to prove the routing gate
works was measuring a gate with nothing behind it.

Whatever a case declares gets checked. `tool_order`, `owner`, `refused` and
`must_not_contain`-in-tool-arguments are all things `evals.cases` lets a case
say and `evals.metrics` knows how to score. A driver that quietly ignores half
of them is worse than one that never offered them: the dataset reads as if the
claim is being tested.
"""

import argparse
import asyncio
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from loguru import logger

from app.application.chat.dto import ReplyToolStarted
from evals.cases import SUITES, EvalCase, behavior_of, load_all_cases, load_cases
from evals.harness import CaseRun, EvalHarness, build_harness, fixture_text
from evals.judge import JudgeScore
from evals.metrics.budgets import summarize_budgets
from evals.metrics.gates import (
    GATE_EXTRA_TRIALS,
    RELEASE_TRIALS,
    BarVerdict,
    Outcome,
    TrialResult,
    count_tasks,
    judge_gate,
    judge_release,
    tasks_to_rerun,
)
from evals.metrics.generation import SubstringCheck, check_substrings, summarize_generation
from evals.metrics.retrieval import hit_rate_at_k, mrr_at_k, sources_returned, summarize_retrieval
from evals.metrics.tools import (
    ToolCallScore,
    score_tool_call,
    summarize_tool_scores,
    tool_arguments,
    tools_completed,
    tools_refused,
)
from evals.report import (
    UNTAGGED,
    RunReport,
    SuiteReport,
    TaskRecord,
    ToolCallRecord,
    Totals,
    Transcript,
    TrialRecord,
    behavior_records,
    estimate_cost,
    render_markdown,
    write_json,
)

_DOCUMENT_TOOL = "retrieve_documents"
"""The only tool whose sources a retrieval metric grades.

Named here rather than imported from `app.application.chat.tools`: this is the
label that appears in a `ReplyToolFinished` event and in a dataset's
`expect.tools`, and both of those are wire-level strings. A rename that broke
them would break the datasets too, and should fail loudly rather than be
followed silently by the scorer alone.
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run the chat agent's eval suites against real APIs.",
    )
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument(
        "--limit", type=int, default=None, help="Run at most this many cases per suite."
    )
    parser.add_argument("--concurrency", type=int, default=1, help="How many cases to run at once.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help=(
            "Run every case this many times (default 1). A case that fails every "
            "trial is a regression; one that fails some is reported as flaky. "
            "Not allowed with --mode, which sets the trial count itself."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["gate", "release"],
        default=None,
        help=(
            "gate: 1 trial per task, no judge, 2 more for a failed task. "
            "release: 10 trials per task, judges on. Omit for a plain run."
        ),
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=0.30,
        help="USD per million input tokens, for the cost estimate (Together DeepSeek V4.1 Flash).",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=1.20,
        help="USD per million output tokens, for the cost estimate.",
    )
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
    args = parser.parse_args(argv)
    if args.mode is not None and args.repeat is not None:
        parser.error("--repeat cannot be used with --mode; the mode sets the trial count.")
    if args.repeat is None:
        args.repeat = 1
    return args


def _select_cases(suite: str, limit: int | None, tags: list[str]) -> list[EvalCase]:
    """The cases one invocation runs, in `SUITES` order.

    A full run goes through `load_all_cases`, which rejects an id reused across
    two files. Loading each file separately would not: the report is keyed by
    case id, and two rows it cannot tell apart is a silent way to lose one.
    """
    all_cases = load_all_cases() if suite == "all" else load_cases(Path(f"{suite}.jsonl"))
    selected: list[EvalCase] = []
    for name in SUITES:
        cases = [case for case in all_cases if case.suite == name]
        if tags:
            cases = [case for case in cases if set(tags) & set(case.tags)]
        if limit is not None:
            cases = cases[:limit]
        selected.extend(cases)
    return selected


async def _run_trials(
    cases: list[EvalCase], harness: EvalHarness, concurrency: int, trials: Sequence[int]
) -> list[CaseRun]:
    """Run every case once for each trial number in `trials`, up to `concurrency` at once.

    Trials of one task are scheduled like any other work rather than run back
    to back. They are independent by construction - `run_case` gives each its
    own thread - and spreading them out keeps a slow task from holding the
    whole run open at the end.

    `trials` is a list of numbers, not a count, so the gate can run trials 2
    and 3 of a failed task without renumbering the first one.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(case: EvalCase, trial: int) -> CaseRun:
        async with semaphore:
            run = await harness.run_case(case)
            run.sample = trial
            return run

    schedule = [(case, trial) for trial in trials for case in cases]
    return list(await asyncio.gather(*(_one(case, trial) for case, trial in schedule)))


def _substring_check(run: CaseRun) -> SubstringCheck:
    return check_substrings(
        run.answer,
        run.case.expect.must_contain,
        run.case.expect.must_not_contain,
        # Tool arguments as well as the answer: a forbidden string sent in a
        # `search_web` query reaches a third party even if the answer is clean.
        tool_arguments=tool_arguments(run.events),
    )


def _tool_score(run: CaseRun) -> ToolCallScore:
    return score_tool_call(
        tools_completed(run.events),
        run.case.expect.tools,
        refused=tools_refused(run.events),
        required_order=run.case.expect.tool_order,
    )


def _judge_context(case: EvalCase, returned: list[str]) -> str:
    """The source text an answer was supposed to be drawn from.

    The fixtures the case names, read off disk - never the gold answer. Those
    are different questions: grading groundedness against the gold answer means
    an invented fact that happens to be right scores a perfect 1.0, which is
    the failure the metric exists to catch. `expect.reference` goes to the
    judge separately, as the reference.

    Falls back to the labels the reply actually cited when a case names no
    fixture, which is all a negative case ("this is not in the corpus") has and
    all it needs.
    """
    texts = [text for name in case.fixtures if (text := fixture_text(name)) is not None]
    missing = [name for name in case.fixtures if fixture_text(name) is None]
    if missing:
        logger.warning(f"Case {case.id!r} names fixtures that do not exist: {missing}")
    if texts:
        return "\n\n".join(texts)
    return "\n".join(returned)


async def _score_suite(
    suite_name: str, runs: list[CaseRun], harness: EvalHarness, *, use_judge: bool = True
) -> SuiteReport:
    """Score the cases that produced an answer. Broken runs are counted, not graded.

    `n` is every case the suite ran, so a suite that half fell over cannot be
    mistaken for a smaller suite that passed. The rates underneath are computed
    over `graded` only.

    `use_judge=False` is the gate: no judge call is made, so the quality
    numbers come from the code graders alone.
    """
    graded = [run for run in runs if not run.broke]

    budgets = summarize_budgets([run.latency_ms for run in graded], [run.usage for run in graded])

    tools_report = None
    if any(run.case.expect.tools or run.case.expect.tool_order for run in graded):
        tools_report = summarize_tool_scores([_tool_score(run) for run in graded])

    substring_checks = [_substring_check(run) for run in graded]

    retrieval_report = None
    judge_scores: list[JudgeScore | None] = []
    if suite_name == "rag":
        hit_rates: list[float | None] = []
        reciprocal_ranks: list[float | None] = []
        for run in graded:
            # Restricted to the document tool. Pooling in a concurrent
            # `search_web`'s results pushes the fixture the case is about out of
            # the top-k window and reads as a retrieval miss.
            returned = sources_returned(run.events, tools=[_DOCUMENT_TOOL])
            expected = run.case.expect.expected_sources
            hit_rates.append(hit_rate_at_k(returned, expected, harness.retrieval_k))
            reciprocal_ranks.append(mrr_at_k(returned, expected, harness.retrieval_k))
            if use_judge:
                judge_scores.append(
                    await harness.judge.score(
                        question=run.case.input,
                        answer=run.answer,
                        context=_judge_context(run.case, returned),
                        reference=run.case.expect.reference,
                    )
                )
        retrieval_report = summarize_retrieval(hit_rates, reciprocal_ranks)

    generation_report = None
    if substring_checks or judge_scores:
        generation_report = summarize_generation(substring_checks, judge_scores)

    return SuiteReport(
        suite=suite_name,
        n=len(runs),
        tools=tools_report,
        retrieval=retrieval_report,
        generation=generation_report,
        budgets=budgets,
    )


def _errors(runs: list[CaseRun]) -> list[str]:
    """Samples the run itself broke on - not graded, and not the model's fault."""
    errors: list[str] = []
    for run in runs:
        if run.error is not None:
            errors.append(f"{_label(run)}: raised after {run.attempts} attempt(s): {run.error}")
        elif run.failed is not None:
            errors.append(
                f"{_label(run)}: failed after {run.attempts} attempt(s): {run.failed.detail}"
            )
    return errors


def _label(run: CaseRun) -> str:
    """`case-id` on a single-sample run, `case-id#2` once a case is repeated."""
    return run.case.id if run.sample == 1 else f"{run.case.id}#{run.sample}"


def _sample_failures(run: CaseRun, harness: EvalHarness) -> list[str]:
    """What one sample got wrong, or nothing.

    Every check is deterministic - no judge, no second model call - which is
    what makes this half of the suite the half worth gating a merge on.
    """
    case = run.case
    failures: list[str] = []

    if case.expect.tools or case.expect.tool_order:
        score = _tool_score(run)
        if not score.exact_match:
            failures.append(
                f"expected tools {list(score.expected)}, got {list(score.completed)}"
                + (f" (refused: {list(score.refused)})" if score.refused else "")
            )
        if not score.order_ok:
            failures.append(f"expected order {case.expect.tool_order}, got {list(score.completed)}")

    if case.expect.expected_sources:
        returned = sources_returned(run.events, tools=[_DOCUMENT_TOOL])
        if hit_rate_at_k(returned, case.expect.expected_sources, harness.retrieval_k) == 0.0:
            failures.append(
                f"expected one of {case.expect.expected_sources} in "
                f"top-{harness.retrieval_k}, got {returned[: harness.retrieval_k]}"
            )

    check = _substring_check(run)
    if check.missing_required:
        failures.append(f"missing {list(check.missing_required)}")
    if check.forbidden_in_answer:
        failures.append(f"forbidden in answer {list(check.forbidden_in_answer)}")
    if check.forbidden_in_tool_args:
        failures.append(f"forbidden in tool args {list(check.forbidden_in_tool_args)}")

    return failures


@dataclass(frozen=True, slots=True)
class Verdicts:
    """A run's per-case punch lists, split by whether the failure repeats."""

    regressions: tuple[str, ...]
    flaky: tuple[str, ...]


def _verdicts(runs: list[CaseRun], harness: EvalHarness) -> Verdicts:
    """Judge each case across all of its samples.

    A case that failed every sample is a regression: the behaviour is what it
    is, and re-running will not change it. A case that failed some samples is
    flaky, and that is a different fact about a different problem - the answer
    is not stable, so neither is any verdict drawn from one sample of it.

    Folding the two together is what makes an eval suite untrustworthy in both
    directions. Call a flaky case red and people re-run the build until it goes
    green, which is how a real regression gets waved through. Call it green and
    a case that fails half the time reports nothing at all.
    """
    regressions: list[str] = []
    flaky: list[str] = []
    for case_id, samples in _by_case(runs).items():
        # A broken sample has an empty answer, which fails every check above
        # for a reason `errors` already named. Dropped, not counted as a
        # failing sample, so a rate limit cannot make a case look unstable.
        graded = [run for run in samples if not run.broke]
        if not graded:
            continue
        failed = [(run, reasons) for run in graded if (reasons := _sample_failures(run, harness))]
        if not failed:
            continue
        detail = "; ".join(failed[0][1])
        if len(failed) == len(graded):
            regressions.append(f"{case_id}: {detail}")
        else:
            flaky.append(f"{case_id}: failed {len(failed)}/{len(graded)} samples: {detail}")
    return Verdicts(regressions=tuple(regressions), flaky=tuple(flaky))


def _by_case(runs: list[CaseRun]) -> dict[str, list[CaseRun]]:
    """Every sample of each case, keyed by case id, in first-seen order."""
    grouped: dict[str, list[CaseRun]] = {}
    for run in runs:
        grouped.setdefault(run.case.id, []).append(run)
    return grouped


def _outcome(run: CaseRun, failures: list[str]) -> Outcome:
    if run.broke:
        return "errored"
    return "failed" if failures else "passed"


def _transcript(run: CaseRun) -> Transcript:
    error = run.error
    if error is None and run.failed is not None:
        error = run.failed.detail
    return Transcript(
        answer=run.answer,
        tool_calls=tuple(
            ToolCallRecord(name=event.name, arguments=event.summary)
            for event in run.events
            if isinstance(event, ReplyToolStarted)
        ),
        error=error,
    )


def _trial_record(run: CaseRun, harness: EvalHarness) -> TrialRecord:
    """One trial as the results file keeps it: outcome, reasons and transcript.

    A broken run is never graded, so it gets no failure reasons. Its empty
    answer would fail every check for a reason the error text already gives.
    """
    failures = [] if run.broke else _sample_failures(run, harness)
    return TrialRecord(
        task_id=run.case.id,
        trial=run.sample,
        outcome=_outcome(run, failures),
        failure_reasons=tuple(failures),
        transcript=_transcript(run),
        tokens_in=run.usage.prompt_tokens if run.usage else 0,
        tokens_out=run.usage.completion_tokens if run.usage else 0,
        latency_ms=run.latency_ms,
    )


def _trial_results(trials: Sequence[TrialRecord]) -> list[TrialResult]:
    return [TrialResult(t.task_id, t.trial, t.outcome) for t in trials]


def dataset_version(datasets_dir: Path, fixtures_dir: Path, suites: Sequence[str]) -> str:
    """A 12-character hash of the dataset files and fixtures a run used.

    Hashes each suite's `.jsonl` and every fixture file, by name and bytes, in
    sorted order. Fixtures are all included because `index_fixtures` indexes
    the whole directory for every suite, so any of them can change an answer.
    """
    digest = hashlib.sha256()
    files = [datasets_dir / f"{suite}.jsonl" for suite in sorted(suites)]
    if fixtures_dir.is_dir():
        files.extend(sorted(path for path in fixtures_dir.iterdir() if path.is_file()))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes() if path.is_file() else b"")
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def _plain_verdict(report_regressions: tuple[str, ...], flaky: tuple[str, ...]) -> BarVerdict:
    """The old rule for a run with no `--mode`: failing every trial is red."""
    if report_regressions:
        return BarVerdict(
            failed=True,
            reasons=tuple(f"Failed every graded trial: {entry}" for entry in report_regressions),
        )
    note = f" {len(flaky)} task(s) failed some trials." if flaky else ""
    return BarVerdict(failed=False, reasons=(f"No task failed every graded trial.{note}",))


def _judge_bar(mode: str | None, trials: Sequence[TrialRecord], verdicts: Verdicts) -> BarVerdict:
    results = _trial_results(trials)
    if mode == "gate":
        return judge_gate(results)
    if mode == "release":
        return judge_release(results)
    return _plain_verdict(verdicts.regressions, verdicts.flaky)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = uuid4().hex[:12]

    cases = _select_cases(args.suite, args.limit, args.tag)
    if not cases:
        logger.warning("No cases matched the given --suite/--tag/--limit; nothing to run.")
        return 0

    async with build_harness(run_id=run_id, model=args.model) as harness:
        # Always, for every suite. Most `tools` cases ask about an upload, and
        # an unindexed corpus turns the routing gate this suite grades into a
        # gate with nothing behind it - see the module docstring.
        await harness.index_fixtures()

        use_judge = args.mode != "gate"
        if args.mode == "release":
            first_wave = list(range(1, RELEASE_TRIALS + 1))
        elif args.mode == "gate":
            first_wave = [1]
        else:
            first_wave = list(range(1, max(1, args.repeat) + 1))

        runs = await _run_trials(cases, harness, args.concurrency, first_wave)
        if args.mode == "gate":
            # Only a task whose first trial FAILED is rerun. An errored one
            # was never graded, so there is nothing to confirm.
            first = [_trial_record(run, harness) for run in runs]
            failed_ids = set(tasks_to_rerun(_trial_results(first)))
            if failed_ids:
                again = [case for case in cases if case.id in failed_ids]
                extra = range(2, 2 + GATE_EXTRA_TRIALS)
                runs += await _run_trials(again, harness, args.concurrency, list(extra))

        by_suite: dict[str, list[CaseRun]] = {}
        for run in runs:
            by_suite.setdefault(run.case.suite, []).append(run)

        suite_reports = [
            await _score_suite(name, by_suite[name], harness, use_judge=use_judge)
            for name in SUITES
            if name in by_suite
        ]
        verdicts = _verdicts(runs, harness)
        trials = tuple(_trial_record(run, harness) for run in runs)
        bar = _judge_bar(args.mode, trials, verdicts)
        tasks = _task_records(cases, trials)
        report = RunReport(
            run_id=run_id,
            model=harness.model,
            tags=tuple(args.tag),
            suites=tuple(suite_reports),
            case_ids=tuple(case.id for case in cases),
            samples=len(runs),
            regressions=verdicts.regressions,
            flaky=verdicts.flaky,
            errors=tuple(_errors(runs)),
            mode=args.mode,
            prompt_version=harness.prompt_version,
            dataset_version=dataset_version(
                Path(__file__).parent / "datasets",
                Path(__file__).parent / "fixtures",
                sorted({case.suite for case in cases}),
            ),
            trials=trials,
            tasks=tasks,
            behaviors=behavior_records(tasks),
            totals=_totals(trials),
            cost=estimate_cost(
                tokens_in=sum(t.tokens_in for t in trials),
                tokens_out=sum(t.tokens_out for t in trials),
                judge_tokens_in=harness.judge.prompt_tokens,
                judge_tokens_out=harness.judge.completion_tokens,
                price_in_per_million=args.price_in,
                price_out_per_million=args.price_out,
            ),
            quality_failed=bar.failed,
            verdict=bar.reasons,
        )

    write_json(report, args.out)
    # The markdown table is the deliverable this CLI exists to produce - see
    # the module docstring in `evals/report.py` - so this is a print, not a
    # log line: it belongs on stdout for a human (or a PR description) to
    # read, not wrapped in loguru's structured format.
    print(render_markdown(report))  # noqa: T201
    return _exit_code(report)


def _task_records(cases: list[EvalCase], trials: Sequence[TrialRecord]) -> tuple[TaskRecord, ...]:
    """One record per task, in dataset order, with its trial counts."""
    counts = {c.task_id: c for c in count_tasks(_trial_results(trials))}
    records: list[TaskRecord] = []
    for case in cases:
        row = counts.get(case.id)
        records.append(
            TaskRecord(
                task_id=case.id,
                suite=case.suite,
                passed=row.passed if row else 0,
                failed=row.failed if row else 0,
                errored=row.errored if row else 0,
                behavior=behavior_of(case) or UNTAGGED,
            )
        )
    return tuple(records)


def _totals(trials: Sequence[TrialRecord]) -> Totals:
    return Totals(
        passed=sum(1 for t in trials if t.outcome == "passed"),
        failed=sum(1 for t in trials if t.outcome == "failed"),
        errored=sum(1 for t in trials if t.outcome == "errored"),
    )


def _exit_code(report: RunReport) -> int:
    """0 clean, 1 the quality bar was missed, 2 the run could not be trusted.

    Three codes rather than two, because the two mean opposite things to
    whoever is looking. A 1 is a prompt or a tool description to fix. A 2 is a
    provider that fell over, and re-running is the whole response - failing a
    merge on it trains everyone to re-run red builds, including the ones a 1
    produced.
    """
    if report.quality_failed:
        return 1
    if report.errors:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    sys.exit(main())
