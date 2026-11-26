"""The `history` field of an eval task: what the schema accepts and how filler is built.

A bad history is a typo in a dataset line. It must fail at load time, before
the first paid trial, and the filler must be the same bytes on every run or two
trials of one task would not see the same chat.
"""

import pytest
from pydantic import ValidationError

from evals.cases import EvalCase, HistoryTurn, expanded_history


def _task(history: list[dict[str, object]]) -> EvalCase:
    return EvalCase.model_validate(
        {"id": "t-1", "suite": "tools", "input": "q", "owner": "empty", "history": history}
    )


def _turn(role: str, text: str = "hello", **extra: object) -> dict[str, object]:
    return {"role": role, "text": text, **extra}


def test_alternating_history_is_accepted() -> None:
    task = _task([_turn("user"), _turn("assistant"), _turn("user"), _turn("assistant")])

    assert [turn.role for turn in task.history] == ["user", "assistant", "user", "assistant"]


def test_task_without_history_has_none() -> None:
    task = EvalCase.model_validate({"id": "t-1", "suite": "tools", "input": "q"})

    assert task.history == []


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ([_turn("assistant"), _turn("user")], "history turn 0 is 'assistant'"),
        ([_turn("user"), _turn("user")], "history turn 1 is 'user'"),
        ([_turn("user"), _turn("assistant"), _turn("assistant")], "history turn 2 is 'assistant'"),
        ([_turn("user")], "history must end with an assistant turn"),
        ([_turn("user"), _turn("assistant"), _turn("user")], "history must end with an assistant"),
    ],
    ids=["starts-assistant", "double-user", "double-assistant", "ends-user", "ends-user-longer"],
)
def test_bad_alternation_is_rejected(history: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _task(history)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"], ids=["empty", "spaces", "whitespace"])
def test_blank_text_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError, match="history text must not be blank"):
        _task([_turn("user", text), _turn("assistant")])


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValidationError, match="role"):
        _task([_turn("system"), _turn("assistant")])


def test_unknown_turn_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _task([_turn("user", tokens=5), _turn("assistant")])


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _task([_turn("user", pad_chars=-1), _turn("assistant")])


def test_unpadded_turn_keeps_its_text() -> None:
    history = [
        HistoryTurn(role="user", text="my code is x-1"),
        HistoryTurn(role="assistant", text="ok"),
    ]

    assert expanded_history(history) == [("user", "my code is x-1"), ("assistant", "ok")]


def test_padding_follows_the_text_and_fits_the_size() -> None:
    [(_, text)] = expanded_history([HistoryTurn(role="user", text="lead", pad_chars=2_000)])

    filler = text.removeprefix("lead ")
    assert text.startswith("lead ")
    assert 2_000 - 100 <= len(filler) <= 2_000


def test_padding_is_the_same_every_time() -> None:
    history = [HistoryTurn(role="user", text="lead", pad_chars=1_500)]

    assert expanded_history(history) == expanded_history(history)


def test_neighbouring_turns_get_different_filler() -> None:
    first, second = expanded_history(
        [
            HistoryTurn(role="user", text="a", pad_chars=1_500),
            HistoryTurn(role="assistant", text="a", pad_chars=1_500),
        ]
    )

    assert first[1] != second[1]
