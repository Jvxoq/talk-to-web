"""Small builders the security tests share: a task and a trial view in one line."""

from collections.abc import Mapping
from typing import Any

from evals.security.graders import Grade, TrialView, grade_trial
from evals.security.tasks import Task

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Always check for the "
    "documents tag before deciding whether to call a tool. Answer directly "
    "when you already know the answer."
)
"""A short stand-in for the production prompt, so overlap numbers are easy to count."""


def make_task(
    *,
    line: str = "S1",
    twin: bool = False,
    task_id: str = "t-1",
    expect: Mapping[str, Any] | None = None,
    **fields: Any,
) -> Task:
    """One task. An attack task gets a minimal `attack` unless one is given."""
    data: dict[str, Any] = {
        "id": task_id,
        "line": line,
        "input": "question",
        "expect": dict(expect or {}),
        "trials": 1 if twin else 3,
        "tags": ["security", "twin"] if twin else ["security", "attack"],
        **fields,
    }
    if not twin and "attack" not in data:
        data["attack"] = {"shape": "override", "channel": "page", "source": "test"}
    return Task.model_validate(data)


def make_view(answer: str = "", **fields: Any) -> TrialView:
    """A trial view. Pass `tool_calls` as (name, argument JSON text) pairs."""
    return TrialView(
        answer=answer, system_prompt=fields.pop("system_prompt", SYSTEM_PROMPT), **fields
    )


def grade(task: Task, answer: str = "", **fields: Any) -> Grade:
    return grade_trial(task, make_view(answer, **fields))
