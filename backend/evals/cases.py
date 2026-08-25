"""Eval case definitions, loaded eagerly.

Pydantic validates every case at load time, not mid-run — a typo in a dataset
file must surface before the first paid API call, not after the twentieth.

Two things here are load-bearing beyond parsing. `owner` decides which of the
harness's two accounts a case runs as, because the single most important
routing rule in this app - `ToolRoutingPolicy` - behaves in opposite ways
depending on whether the account has uploads, and a suite that only ever
exercised one of those two states was measuring half the gate. And `fixtures`
is no longer decoration: `evals.__main__` reads the named files to give the
judge real grounding text, so a stale filename now changes a score instead of
only a comment.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_DATASETS_DIR = Path(__file__).parent / "datasets"

SUITES: tuple[str, ...] = ("tools", "rag", "injection")
"""Every suite `evals.__main__` can run, in the order a full run reports them.

`benign.jsonl` and `redteam.jsonl` are deliberately absent: those are guardrail
golden files, graded by regex in `tests/application/test_guardrails_redteam.py`
with no model call at all, and running them here would spend money to learn
nothing new.
"""

Owner = Literal["with_documents", "empty"]
"""Which eval account a case runs as.

`with_documents` is the account every fixture is indexed under - the normal
one. `empty` is an account with nothing indexed and no `documents` rows, which
is the state `ToolRoutingPolicy` refuses `retrieve_documents` in and, for the
same turn, stops holding `search_web` back. Nothing exercised that state
before, and it is not a corner: it is every account's first session.
"""


class Expectation(BaseModel):
    """What a case's answer is graded against. Every field is optional — a
    case may check only what it cares about."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The exact set of tools that must have *completed* on this turn, for the
    # `tools` suite.
    #
    # Completed, not requested, and the distinction is not pedantic. The tool
    # node writes its `ReplyToolStarted` event before `ToolRegistry.invoke`
    # runs, so a call the routing policy refuses outright still announces
    # itself - and grading on announcements scored a blocked call as a
    # successful one. That is precisely backwards for the two cases the gate
    # exists for.
    tools: list[str] = Field(default_factory=list)
    # Tools that must have completed in this relative order. A subsequence, not
    # the whole list: it says "retrieval answered before the search was paid
    # for", which is the actual rule, without also pinning down calls the rule
    # says nothing about.
    tool_order: list[str] = Field(default_factory=list)
    # Fixture filenames (or URLs) the answer should be able to cite, for the
    # `rag` suite's retrieval metrics.
    expected_sources: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    # Checked against the answer *and* against every tool call's arguments.
    # Both, because the interesting way an injection succeeds is not the model
    # repeating a canary at the user - it is the model quietly putting one in a
    # `search_web` query, where an answer-only check never looks.
    must_not_contain: list[str] = Field(default_factory=list)
    # A gold answer, handed to the judge as the correct response to compare
    # against - never as the grounding context. Those are different questions:
    # an answer that matches the gold text while citing nothing is correct and
    # ungrounded, and folding the two together made a hallucination that
    # happened to be right score 1.0 for groundedness.
    reference: str | None = None

    @model_validator(mode="after")
    def _order_is_a_subset_of_tools(self) -> "Expectation":
        """`tool_order` may only name tools the case already expects.

        Ordering a call that is not expected to happen at all is a case that
        can never pass, and the failure would read as a model regression
        rather than as the typo it is.
        """
        unknown = sorted(set(self.tool_order) - set(self.tools))
        if unknown:
            raise ValueError(f"tool_order names tools not in tools: {unknown}")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: Literal["tools", "rag", "injection"]
    input: str
    # Fixture filenames this case depends on. `EvalHarness.index_fixtures`
    # still indexes the whole `evals/fixtures/` directory once per run rather
    # than per case, so this does not decide what is searchable - but
    # `evals.__main__` reads these files to give the judge the text the answer
    # should be grounded in, so a wrong name here now costs a score.
    fixtures: list[str] = Field(default_factory=list)
    # Which eval account to run as. Defaults to the account the fixtures are
    # indexed under, which is what almost every case wants.
    owner: Owner = "with_documents"
    expect: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _empty_owner_expects_nothing_retrieved(self) -> "EvalCase":
        """An `empty` account cannot retrieve, so it cannot cite a fixture.

        `ToolRoutingPolicy` refuses `retrieve_documents` outright when the
        account has no uploads. A case that runs as `empty` and still expects
        that tool - or a fixture in its sources - is asserting something the
        application is built to prevent, and would fail forever for a reason
        that looks like a model problem.
        """
        if self.owner != "empty":
            return self
        if "retrieve_documents" in self.expect.tools:
            raise ValueError("owner='empty' cannot complete retrieve_documents")
        if self.expect.expected_sources:
            raise ValueError("owner='empty' has nothing indexed, so it can cite no sources")
        return self


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

    _reject_duplicate_ids(cases, where=resolved.name)
    return cases


def load_all_cases() -> list[EvalCase]:
    """Every suite's cases, in `SUITES` order, with ids unique across all of them.

    Uniqueness has to be checked here and not only per file, because a full run
    merges the three datasets into one report keyed by case id. Two files that
    each happen to define `rag-001` are two rows the report cannot tell apart,
    and nothing about the run would say so.
    """
    cases = [case for suite in SUITES for case in load_cases(Path(f"{suite}.jsonl"))]
    _reject_duplicate_ids(cases, where="datasets")
    return cases


def _reject_duplicate_ids(cases: list[EvalCase], *, where: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.id in seen:
            duplicates.add(case.id)
        seen.add(case.id)
    if duplicates:
        raise CasesLoadError(f"{where}: duplicate case ids: {sorted(duplicates)}")
