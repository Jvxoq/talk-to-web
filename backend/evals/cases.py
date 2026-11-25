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

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_DATASETS_DIR = Path(__file__).parent / "datasets"

SUITES: tuple[str, ...] = ("tools", "rag")
"""Every suite `evals.__main__` can run, in the order a full run reports them.

Security cases (prompt injection, red team) live in a separate suite outside
this package.
"""

BEHAVIORS: tuple[str, ...] = ("b1", "b2", "b3", "b4", "b5", "b6")
"""The behavior tags, in the order the report lists them.

They match the table in `eval-design.md`: B1 answer about the file, B2 read a
link or search the web, B3 never search files on an empty account, B4 say "not
in your file", B5 summarize a long chat, B6 answer when the loop limit hits.
Every task carries exactly one, next to its other tags. `--tag b1` selects one
behavior through the existing `--tag` option.
"""

Owner = Literal["with_documents", "empty"]
"""Which eval account a case runs as.

`with_documents` is the account every fixture is indexed under - the normal
one. `empty` is an account with nothing indexed and no `documents` rows, which
is the state `ToolRoutingPolicy` refuses `retrieve_documents` in and, for the
same turn, stops holding `search_web` back. Nothing exercised that state
before, and it is not a corner: it is every account's first session.
"""


class HistoryTurn(BaseModel):
    """One earlier turn of a multi-turn task, seeded before the final `input`.

    `text` is the part a reader of the dataset cares about, such as the fact
    the final question depends on. `pad_chars` adds deterministic filler after
    it, so a task can pass the summarize threshold (about 12k tokens) without a
    50 KB literal in the `.jsonl` file. The filler is plain chatter that holds
    no fact, so it cannot change the answer. See `expanded_history`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    text: str
    # Upper bound is a typo guard: 200k characters is already about 50k tokens.
    pad_chars: int = Field(default=0, ge=0, le=200_000)

    @model_validator(mode="after")
    def _text_is_not_blank(self) -> "HistoryTurn":
        """A blank turn seeds nothing, and most providers reject an empty message."""
        if not self.text.strip():
            raise ValueError("history text must not be blank")
        return self


_FILLER_SENTENCES: tuple[str, ...] = (
    "The garden shed still needs a fresh coat of paint before the weather turns.",
    "A neighbour offered to lend a ladder, which would make the gutters much easier.",
    "The bookshelf in the hallway has started to lean a little toward the window.",
    "Someone left a folded newspaper on the bench outside the bakery this morning.",
    "The kettle in the shared kitchen takes forever to boil these days.",
    "A cyclist rang a bell twice at the corner and then turned toward the river.",
    "The radio played a slow song about trains while the soup simmered on the stove.",
    "Rain tapped on the skylight for most of the afternoon and then stopped quietly.",
)
"""Neutral sentences for `HistoryTurn.pad_chars`. No digits, no names, no money, no
codes, so no task needle can collide with them."""


def _filler(chars: int, seed: int) -> str:
    """About `chars` characters of filler, the same every time for the same inputs.

    Whole sentences only, so the result is at most `chars` long (one sentence
    over when `chars` is smaller than a sentence). `seed` shifts where in the
    list a turn starts, so neighbouring turns do not repeat each other.
    """
    sentences: list[str] = []
    size = 0
    index = seed * 3
    while True:
        sentence = _FILLER_SENTENCES[index % len(_FILLER_SENTENCES)]
        if sentences and size + len(sentence) + 1 > chars:
            break
        sentences.append(sentence)
        size += len(sentence) + 1
        index += 1
    return " ".join(sentences)


def filler_fingerprint() -> str:
    """The padding inputs as one string, for the dataset version.

    The sentences and the code that picks them. A change to either gives every
    padded task different text, so two runs on either side are not comparable.
    """
    return "\x00".join((*_FILLER_SENTENCES, inspect.getsource(_filler)))


def expanded_history(
    history: Sequence[HistoryTurn],
) -> list[tuple[Literal["user", "assistant"], str]]:
    """Every history turn as `(role, text)`, with its filler added."""
    return [
        (
            turn.role,
            f"{turn.text} {_filler(turn.pad_chars, index)}" if turn.pad_chars else turn.text,
        )
        for index, turn in enumerate(history)
    ]


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
    # Both, because a forbidden string can leave through a `search_web` query
    # as easily as through the answer.
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

    @property
    def checks_tools(self) -> bool:
        """True when the task wrote `tools` (even `[]`, meaning "no tool") or `tool_order`.

        An empty list alone cannot tell "not checked" from "call no tool", so
        this reads which fields the JSON actually wrote.
        """
        return "tools" in self.model_fields_set or bool(self.tool_order)


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: Literal["tools", "rag"]
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
    # Earlier turns of a multi-turn task (B5). The harness seeds them into the
    # trial's own fresh thread without calling any model, then runs `input` as
    # the one turn the model under test answers. Empty for a one-turn task.
    history: list[HistoryTurn] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _history_alternates(self) -> "EvalCase":
        """History starts with the user and ends with the assistant, turn by turn.

        It ends with the assistant because `input` is the next user turn. Two
        user turns in a row is a shape the app never stores, and a seeded thread
        shaped that way would grade a state no real chat reaches.
        """
        for index, turn in enumerate(self.history):
            expected = "user" if index % 2 == 0 else "assistant"
            if turn.role != expected:
                raise ValueError(
                    f"history turn {index} is {turn.role!r}, expected {expected!r}: "
                    "history must start with user and alternate"
                )
        if self.history and self.history[-1].role != "assistant":
            raise ValueError("history must end with an assistant turn, before the final input")
        return self

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
    merges the datasets into one report keyed by case id. Two files that
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


def behavior_of(case: EvalCase) -> str | None:
    """The one behavior tag on a task, or `None` when it carries none.

    `tests/evals/test_datasets.py` guarantees every real task has exactly one.
    A task built by hand in a test may have none, and the report files it under
    "untagged" instead of failing a run.
    """
    found = [tag for tag in case.tags if tag in BEHAVIORS]
    return found[0] if len(found) == 1 else None
