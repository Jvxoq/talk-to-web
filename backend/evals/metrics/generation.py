"""Answer-quality metrics from deterministic substring checks.

The judges' yes/no verdicts are counted per trial in `evals.report`, not here.
A model that quietly stops citing sources, or that starts leaking a system
instruction it should never repeat, is a loud regression. `must_contain` /
`must_not_contain` catch it exactly, for free, with no model call.

`must_not_contain` is checked against the tool arguments as well as the answer,
because a forbidden string sent in a `search_web` query reaches a third party
even when the answer itself is clean.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubstringCheck:
    missing_required: tuple[str, ...]
    # Forbidden text that reached the user.
    forbidden_in_answer: tuple[str, ...]
    # Forbidden text that reached a tool's arguments - an exfiltration attempt
    # that got as far as the wire, whether or not it also reached the answer.
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
    """Case-insensitive containment checks.

    `must_contain` is graded against the answer only: a required fact is
    required in what the user reads, and finding it in a search query proves
    nothing. `must_not_contain` is graded against the answer and every tool
    call's arguments, for the reason in the module docstring.
    """
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
    substring_pass_rate: float


def summarize_generation(substring_checks: Sequence[SubstringCheck]) -> GenerationSuiteSummary:
    """Summarize one suite's substring checks."""
    n = len(substring_checks)
    return GenerationSuiteSummary(
        n=n,
        substring_pass_rate=(sum(1 for check in substring_checks if check.ok) / n) if n else 0.0,
    )
