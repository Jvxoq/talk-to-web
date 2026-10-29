"""The state the agent carries between nodes.

Mutable, because LangGraph writes reducer results back into it.
"""

from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel

from app.application.chat.models import ChatMessage

# Returned first in `messages` to make the reducer replace the history.
# Matched by identity before serialization, so it is never stored.
RESET = ChatMessage(role="system", content="__RESET__")


def merge_messages(left: list[ChatMessage], right: list[ChatMessage]) -> list[ChatMessage]:
    """Reducer for `messages`: append, or replace when the first item is `RESET`."""
    if right and right[0] is RESET:
        return list(right[1:])
    return left + right


class AgentState(BaseModel):
    """The conversation so far, plus how many laps of the loop it has cost."""

    # Nodes return only new messages. The reducer merges them into history.
    messages: Annotated[list[ChatMessage], merge_messages]

    # Summary of older messages, written whole by the summarize node.
    summary: str = ""

    # Model turns in this reply. Reset per request, or it would only climb.
    iterations: int = 0


def tools_run_this_turn(messages: Sequence[ChatMessage]) -> frozenset[str]:
    """Tools that have returned a result since the last user message.

    Counts answered calls, not requested ones, since calls in one lap run
    together. Scoped to this turn, not the whole thread.
    """

    names_by_id: dict[str, str] = {}
    answered: set[str] = set()
    for message in reversed(messages):
        if message.role == "user":
            break
        for call in message.tool_calls:
            names_by_id[call.id] = call.name
        if message.role == "tool" and message.tool_call_id is not None:
            answered.add(message.tool_call_id)
    return frozenset(names_by_id[call_id] for call_id in answered if call_id in names_by_id)
