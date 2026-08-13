"""Answer-quality metrics: an LLM judge's scores, plus deterministic substring checks.

The substring checks matter in their own right, not only as a sanity check on
the judge. A model that quietly stops citing sources, or that starts leaking a
system instruction it should never repeat, is a loud regression - and a 0-1
judge score smooths loud regressions into a slightly-lower average instead of
flagging them. `must_contain` / `must_not_contain` catch it exactly, for free,
with no model call.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evals.judge import JudgeScore


@dataclass(frozen=True, slots=True)
class SubstringCheck:
    missing_required: tuple[str, ...]
    forbidden_found: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.forbidden_found


def check_substrings(
    answer: str, must_contain: Sequence[str], must_not_contain: Sequence[str]
) -> SubstringCheck:
    """Case-insensitive containment checks against the answer text."""
    lowered = answer.lower()
    missing = tuple(needle for needle in must_contain if needle.lower() not in lowered)
    forbidden = tuple(needle for needle in must_not_contain if needle.lower() in lowered)
    return SubstringCheck(missing_required=missing, forbidden_found=forbidden)


@dataclass(frozen=True, slots=True)
class GenerationSuiteSummary:
    n: int
    mean_groundedness: float | None
    mean_relevance: float | None
    substring_pass_rate: float
    # How many cases the judge actually scored - out of `n`, not a duplicate
    # of it, because a judge failure (see `evals.judge.Judge.score`) drops a
    # case from the mean rather than counting it as zero.
    judged: int


def summarize_generation(
    judge_scores: Sequence[JudgeScore | None], substring_checks: Sequence[SubstringCheck]
) -> GenerationSuiteSummary:
    n = len(substring_checks)
    judged = [score for score in judge_scores if score is not None]
    return GenerationSuiteSummary(
        n=n,
        mean_groundedness=(
            sum(score.groundedness for score in judged) / len(judged) if judged else None
        ),
        mean_relevance=(sum(score.relevance for score in judged) / len(judged) if judged else None),
        substring_pass_rate=(sum(1 for check in substring_checks if check.ok) / n) if n else 0.0,
        judged=len(judged),
    )
