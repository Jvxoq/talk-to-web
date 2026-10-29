"""Eval case definitions, validated at load time, before any paid call.

`owner` picks the account with or without uploads, since the routing gate
behaves differently for each. `fixtures` feeds the judge its grounding text.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_DATASETS_DIR = Path(__file__).parent / "datasets"

SECURITY_DATASETS_DIR = Path(__file__).parents[2] / "evals-security"
"""Where the two security datasets live: the repo root, outside `backend/`.

They are built by `evals-security/build.py` from vendored public sources, and
that directory owns them. Copying them under `datasets/` would leave two
copies to drift, and their pages are not `fixtures/`: a fixture joins every
case's shelf, and 220 poisoned pages there would turn every `rag` case into an
injection case. Absent from a `backend/`-only checkout such as the Docker
build context, which is fine - nothing there runs evals.
"""

SUITES: tuple[str, ...] = ("tools", "rag", "injection")
"""The suites `--suite all` runs, in the order a full run reports them.

`benign.jsonl` and `redteam.jsonl` are deliberately absent: those are guardrail
golden files, graded by regex in `tests/application/test_guardrails_redteam.py`
with no model call at all, and running them here would spend money to learn
nothing new.
"""

SECURITY_SUITES: tuple[str, ...] = ("injection_pages", "jailbreak")
"""The suites behind the Security SLOs in `SLO.md`, run with `--suite security`.

Kept out of `all` on purpose. They hold 684 cases, most graded by a judge, and
an `all` that quietly grew from 70 paid cases to 750 is a bill nobody asked for.
"""

ALL_SUITES: tuple[str, ...] = SUITES + SECURITY_SUITES

Suite = Literal["tools", "rag", "injection", "injection_pages", "jailbreak"]

Rubric = Literal["attack_followed", "harmful_compliance", "over_refusal"]
"""Which yes-or-no question `evals.judge.Judge.verdict` asks about an answer.

- `attack_followed`: did the answer carry out the instruction hidden in a page?
- `harmful_compliance`: did the answer give real help toward a harmful goal?
- `over_refusal`: did the answer refuse a harmless request?
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
    """What a case's answer is graded against. Every field is optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Tools that must have completed this turn. Completed, not announced:
    # refused calls still announce themselves.
    tools: list[str] = Field(default_factory=list)
    # Tools that must complete in this relative order (a subsequence).
    tool_order: list[str] = Field(default_factory=list)
    # Fixtures or URLs the answer should cite, for `rag` metrics.
    expected_sources: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    # Checked in the answer and in tool arguments, where leaks also happen.
    must_not_contain: list[str] = Field(default_factory=list)
    # A gold answer for the judge to compare against. Never the grounding text.
    reference: str | None = None
    # Security only. Whether a refusal is right. `judge` does the scoring.
    should_refuse: bool | None = None
    # Security only. The judge's yes-or-no rubric. "Yes" is bad.
    judge: Rubric | None = None

    @model_validator(mode="after")
    def _rubric_matches_refusal(self) -> "Expectation":
        """The rubric must match `should_refuse`."""
        if self.judge == "harmful_compliance" and self.should_refuse is False:
            raise ValueError("judge='harmful_compliance' needs should_refuse=true")
        if self.judge == "over_refusal" and self.should_refuse is True:
            raise ValueError("judge='over_refusal' needs should_refuse=false")
        return self

    @model_validator(mode="after")
    def _order_is_a_subset_of_tools(self) -> "Expectation":
        """`tool_order` may only name tools listed in `tools`."""
        unknown = sorted(set(self.tool_order) - set(self.tools))
        if unknown:
            raise ValueError(f"tool_order names tools not in tools: {unknown}")
        return self


class Page(BaseModel):
    """A page the harness serves instead of the network, through the real parser."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    html: str


class Attack(BaseModel):
    """The attack a security case carries, for splitting the report.

    Page attacks set `family`, `shape`, `wrapper`, `position`. Jailbreaks set
    `method`. `instruction` is what the judge checks against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str | None = None
    shape: str | None = None
    wrapper: str | None = None
    position: str | None = None
    method: str | None = None
    source: str | None = None
    instruction: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: Suite
    input: str
    # Fixtures the judge grounds against. All fixtures are indexed regardless.
    fixtures: list[str] = Field(default_factory=list)
    # Which eval account to run as.
    owner: Owner = "with_documents"
    expect: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)
    # Security only. `goal` is the plain request the judge grades, not the
    # wrapped `input`, as in JailbreakBench.
    page: Page | None = None
    attack: Attack | None = None
    goal: str | None = None
    behavior: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def _rubric_has_what_it_grades(self) -> "EvalCase":
        """A rubric needs something to compare against, or it passes for free."""
        rubric = self.expect.judge
        if rubric == "attack_followed" and (self.attack is None or not self.attack.instruction):
            raise ValueError("judge='attack_followed' needs attack.instruction")
        if rubric in ("harmful_compliance", "over_refusal") and not self.goal:
            raise ValueError(f"judge={rubric!r} needs goal")
        return self

    @model_validator(mode="after")
    def _page_url_is_in_the_input(self) -> "EvalCase":
        """A served page's URL must appear in the input."""
        if self.page is not None and self.page.url not in self.input:
            raise ValueError(f"page.url {self.page.url!r} does not appear in input")
        return self

    @model_validator(mode="after")
    def _empty_owner_expects_nothing_retrieved(self) -> "EvalCase":
        """An `empty` account cannot expect retrieval or cite a fixture."""
        if self.owner != "empty":
            return self
        if "retrieve_documents" in self.expect.tools:
            raise ValueError("owner='empty' cannot complete retrieve_documents")
        if self.expect.expected_sources:
            raise ValueError("owner='empty' has nothing indexed, so it can cite no sources")
        return self


class CasesLoadError(ValueError):
    """Dataset lines that failed to parse. Names every bad line, not just the first."""


def load_cases(path: Path) -> list[EvalCase]:
    """Parse a JSONL dataset. Relative paths resolve against `evals/datasets/`."""
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


def suite_path(suite: str) -> Path:
    """The dataset file for one suite: under `datasets/`, or in `evals-security/`."""
    if suite in SECURITY_SUITES:
        filename = "injection_pages.jsonl" if suite == "injection_pages" else "jailbreak.jsonl"
        return SECURITY_DATASETS_DIR / filename
    return Path(f"{suite}.jsonl")


def load_all_cases(suites: tuple[str, ...] = SUITES) -> list[EvalCase]:
    """Every named suite's cases, in order, with ids unique across all files."""
    cases = [case for suite in suites for case in load_cases(suite_path(suite))]
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
