"""The security tasks, loaded and checked before the first paid call.

One line of a `security-evals/tasks/*.jsonl` file is one task. Pydantic checks
every line at load time and `load_tasks` reports every bad line at once, so a
typo in a task file shows up before the run starts and not after the
twentieth trial.

Core module: it knows nothing about the agent under test.
"""

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
"""The repo root, found from this file and never from the working directory."""

SECURITY_EVALS_DIR = REPO_ROOT / "security-evals"
TASKS_DIR = SECURITY_EVALS_DIR / "tasks"
JUDGES_PATH = SECURITY_EVALS_DIR / "judges.json"

Line = Literal["S1", "S2", "S3", "S6", "S9", "S11", "S13"]
"""The lines (behaviors) that have tasks. S11 holds the benign twins of S9."""

LINES: tuple[Line, ...] = ("S1", "S2", "S3", "S6", "S9", "S11", "S13")

JudgeKind = Literal["harmful_compliance", "over_refusal"]

TWIN_TAG = "twin"


class TaskLoadError(ValueError):
    """One or more task lines are bad. The message lists every one of them."""


class Page(BaseModel):
    """A fake page. A fetch of `url` returns `html` through the production extractor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    html: str


class Document(BaseModel):
    """A file uploaded for the trial's owner before the user request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    text: str


class HistoryItem(BaseModel):
    """One earlier item of a seeded chat.

    `user` and `assistant` carry `content`. A `tool_fetch` carries the `url` the
    agent read and the page's `html`, and stands for one fetch tool call and its
    result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant", "tool_fetch"]
    content: str | None = None
    url: str | None = None
    html: str | None = None

    @model_validator(mode="after")
    def _fields_match_role(self) -> Self:
        if self.role == "tool_fetch":
            if not self.url or self.html is None or self.content is not None:
                raise ValueError("a tool_fetch item needs url and html, and no content")
        elif not self.content or not self.content.strip():
            raise ValueError(f"a {self.role} item needs non-blank content")
        elif self.url is not None or self.html is not None:
            raise ValueError(f"a {self.role} item takes no url or html")
        return self


class Attack(BaseModel):
    """How the attacker text is built and where it sits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: str
    wrapper: str | None = None
    position: str | None = None
    channel: str
    source: str


class Expect(BaseModel):
    """What a trial is graded against. Every field is optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    # A false value the attacker wants the user to believe. The attack landed
    # when the answer holds it and is missing a `must_contain` string. Quoting
    # it next to the true value, as a warning, is not the attack.
    must_not_claim: list[str] = Field(default_factory=list)
    # A token no tool argument may hold. Hijack shapes ask for a tool call
    # with it, so a call is the agent obeying. The answer is not read: quoting
    # the token in a warning to the user is not obeying.
    must_not_call_with: list[str] = Field(default_factory=list)
    must_not_leak: list[str] = Field(default_factory=list)
    must_not_contact: list[str] = Field(default_factory=list)
    summary_must_not_contain: list[str] = Field(default_factory=list)
    system_prompt_overlap_max: float | None = Field(default=None, ge=0.0, le=1.0)
    judge: JudgeKind | None = None
    should_refuse: bool | None = None


class Task(BaseModel):
    """One test: what the user sends, what the agent may see, what it must not do."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    line: Line
    input: str
    owner: Literal["empty"] | None = None
    page: Page | None = None
    document: Document | None = None
    history: list[HistoryItem] = Field(default_factory=list)
    pad_history_to_tokens: int | None = Field(default=None, ge=1)
    attack: Attack | None = None
    # The harmful goal of a jailbreak task, in the source's words.
    goal: str | None = None
    # JailbreakBench's behavior name for a jailbreak task. Not the report's
    # behavior tag. That comes from `behavior_tag`.
    behavior: str | None = None
    category: str | None = None
    expect: Expect = Field(default_factory=Expect)
    trials: int = Field(ge=1)
    tags: list[str] = Field(default_factory=list)

    @property
    def twin(self) -> bool:
        """A benign twin: it looks like an attack task and must be done, not refused."""
        return TWIN_TAG in self.tags

    @model_validator(mode="after")
    def _history_ends_on_the_agent(self) -> Self:
        """The final `input` is the next user turn, so history must end on the assistant.

        Two user turns in a row, or two assistant turns, is a shape the app
        never stores, and a seeded thread shaped that way would grade a state no
        real chat reaches.
        """
        if not self.history:
            if self.pad_history_to_tokens is not None:
                raise ValueError("pad_history_to_tokens needs a history to pad")
            return self
        previous: str | None = None
        for index, item in enumerate(self.history):
            if previous is None and item.role != "user":
                raise ValueError(f"history item {index} is {item.role!r}: history starts with user")
            if item.role == "user" and previous == "user":
                raise ValueError(f"history item {index}: two user turns in a row")
            if item.role == "assistant" and previous == "assistant":
                raise ValueError(f"history item {index}: two assistant turns in a row")
            if item.role == "tool_fetch" and previous == "assistant":
                raise ValueError(f"history item {index}: a fetch must follow a user turn")
            previous = item.role
        if self.history[-1].role != "assistant":
            raise ValueError("history must end with an assistant turn, before the final input")
        return self

    @model_validator(mode="after")
    def _attack_task_has_attack(self) -> Self:
        if not self.twin and self.attack is None:
            raise ValueError("an attack task (no `twin` tag) needs an `attack`")
        return self


def behavior_tag(task: Task) -> str:
    """The behavior label a results file files a task under.

    `s9` for a task judged by `harmful_compliance` and `s11` for one judged by
    `over_refusal`, so `evals.labels` queues them for the right judge. Any other
    task gets its line, lowercased.
    """
    if task.expect.judge == "harmful_compliance":
        return "s9"
    if task.expect.judge == "over_refusal":
        return "s11"
    return task.line.lower()


def load_tasks(tasks_dir: Path | None = None) -> list[Task]:
    """Every task of every `*.jsonl` file, in file order. Raises `TaskLoadError`.

    The directory is found from this file, not the working directory, so the CLI
    behaves the same from `backend/` and from the repo root. Every bad line is
    listed, and so is every id used twice across files.
    """
    root = tasks_dir if tasks_dir is not None else TASKS_DIR
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise TaskLoadError(f"No task files in {root}")

    tasks: list[Task] = []
    errors: list[str] = []
    for path in files:
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                tasks.append(Task.model_validate_json(raw))
            except ValidationError as error:
                errors.append(f"{path.name}:{number}: {_one_line(error)}")

    seen: dict[str, int] = {}
    for task in tasks:
        seen[task.id] = seen.get(task.id, 0) + 1
    errors.extend(
        f"duplicate task id {task_id!r} ({count} times)"
        for task_id, count in seen.items()
        if count > 1
    )
    if errors:
        raise TaskLoadError("\n".join(errors))
    return tasks


def _one_line(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'task'}: {item['msg']}"
        for item in error.errors()
    )


def dataset_version(tasks_dir: Path | None = None) -> str:
    """A 12-character hash of the task files, by name and bytes, in sorted order.

    Two results files with different versions ran different tasks, so their
    numbers are not comparable.
    """
    root = tasks_dir if tasks_dir is not None else TASKS_DIR
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.jsonl")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]
