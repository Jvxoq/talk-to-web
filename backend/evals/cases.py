"""Eval case definitions, loaded eagerly.

Pydantic validates every case at load time, not mid-run — a typo in a dataset
file must surface before the first paid API call, not after the twentieth.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_DATASETS_DIR = Path(__file__).parent / "datasets"


class Expectation(BaseModel):
    """What a case's answer is graded against. Every field is optional — a
    case may check only what it cares about."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The exact set of tools the agent should call, for the `tools` suite.
    tools: list[str] = Field(default_factory=list)
    # Fixture filenames (or URLs) the answer should be able to cite, for the
    # `rag` suite's retrieval metrics.
    expected_sources: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    # A gold answer, handed to the judge as grounding context when a case has
    # no better source of "what the answer should look like".
    reference: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: Literal["tools", "rag", "injection"]
    input: str
    # Fixture filenames this case depends on. Informational rather than
    # load-bearing: `EvalHarness.index_fixtures` indexes the whole
    # `evals/fixtures/` directory once per run rather than per case, so this
    # field documents intent and lets a report explain a miss, but nothing
    # reads it to decide what to index.
    fixtures: list[str] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)


class CasesLoadError(ValueError):
    """One or more lines of a dataset file failed to parse or validate.

    Raised eagerly, naming every bad line found — not just the first — so
    fixing a dataset does not mean discovering its typos one `uv run` at a
    time.
    """


def load_cases(path: Path) -> list[EvalCase]:
    """Parse every line of a JSONL dataset into an `EvalCase`.

    `path` resolves against `evals/datasets/` when relative, never the
    current working directory — the CLI must behave the same whether it is
    run from `backend/` or from the repo root.
    """
    resolved = path if path.is_absolute() else _DATASETS_DIR / path
    if not resolved.is_file():
        raise CasesLoadError(f"No such dataset file: {resolved}")

    cases: list[EvalCase] = []
    errors: list[str] = []
    for lineno, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            cases.append(EvalCase.model_validate_json(stripped))
        except ValidationError as error:
            errors.append(f"{resolved.name}:{lineno}: {error}")

    if errors:
        raise CasesLoadError("\n".join(errors))

    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise CasesLoadError(f"{resolved.name}: duplicate case ids: {duplicates}")

    return cases
