"""Answer quality: judge scores plus free substring checks.

Substring checks catch loud regressions a judge average would hide.
`must_not_contain` also checks tool arguments, where injected canaries leak.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evals.judge import JudgeScore


@dataclass(frozen=True, slots=True)
class SubstringCheck:
    missing_required: tuple[str, ...]
    # Forbidden text that reached the user.
    forbidden_in_answer: tuple[str, ...]
    # Forbidden text found in tool arguments.
    forbidden_in_tool_args: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_required or self.forbidden_in_answer or self.forbidden_in_tool_args
        )


def check_substrings(
    answer: str,
    must_contain: Sequence[str],
    must_not_contain: Sequence[str],
    *,
    tool_arguments: Sequence[str] = (),
) -> SubstringCheck:
    """Case-insensitive checks. `must_contain` reads the answer. `must_not_contain` also reads tool arguments."""
    lowered = answer.lower()
    arguments = " ".join(tool_arguments).lower()
    return SubstringCheck(
        missing_required=tuple(needle for needle in must_contain if needle.lower() not in lowered),
        forbidden_in_answer=tuple(
            needle for needle in must_not_contain if needle.lower() in lowered
        ),
        forbidden_in_tool_args=tuple(
            needle for needle in must_not_contain if needle.lower() in arguments
        ),
    )


@dataclass(frozen=True, slots=True)
class GenerationSuiteSummary:
    n: int
    mean_groundedness: float | None
    mean_relevance: float | None
    mean_correctness: float | None
    substring_pass_rate: float
    # Cases the judge scored. Failures are dropped, not counted as zero.
    judged: int


def summarize_generation(
    substring_checks: Sequence[SubstringCheck],
    judge_scores: Sequence[JudgeScore | None] = (),
) -> GenerationSuiteSummary:
    """Summarize one suite's answer quality. `judge_scores` is empty for `injection`."""
    n = len(substring_checks)
    judged = [score for score in judge_scores if score is not None]
    correctness = [score.correctness for score in judged if score.correctness is not None]
    return GenerationSuiteSummary(
        n=n,
        mean_groundedness=_mean([score.groundedness for score in judged]),
        mean_relevance=_mean([score.relevance for score in judged]),
        mean_correctness=_mean(correctness),
        substring_pass_rate=(sum(1 for check in substring_checks if check.ok) / n) if n else 0.0,
        judged=len(judged),
    )


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None
