"""What the agent carries from one node to the next.

Mutable on purpose. LangGraph rebuilds the state object between nodes from the
values each node returned, so a frozen model would fail the moment a reducer
tried to write the merged result back.
"""

from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel

from app.application.chat.models import ChatMessage

# A sentinel the summarization node returns as the first element of `messages`
# to tell the reducer to *replace* the history rather than append to it. The
# identity check works because the reducer runs on the node's in-memory return
# value, before the checkpointer serialises anything - so the sentinel itself is
# never persisted.
RESET = ChatMessage(role="system", content="__RESET__")


def merge_messages(left: list[ChatMessage], right: list[ChatMessage]) -> list[ChatMessage]:
    """The reducer for `messages`: append, unless the node asked to reset.

    Summarization must drop messages, not append them, so it returns `RESET` as
    the first element and the reducer replaces the whole history with what
    follows it. Every other node returns only the messages it produced, which
    this appends to the checkpointed history.
    """
    if right and right[0] is RESET:
        return list(right[1:])
    return left + right


class AgentState(BaseModel):
    """The conversation so far, plus how many laps of the loop it has cost."""

    # `merge_messages` is the reducer: a node returns only the messages it
    # produced, and LangGraph merges them with the checkpointed history. Without
    # it a node returning one message would silently replace the whole history.
    messages: Annotated[list[ChatMessage], merge_messages]

    # A running summary of the older part of the thread, written by the
    # summarization node. Plain field (no reducer): the node writes it whole.
    summary: str = ""

    # Counts model turns within a single reply, not for the lifetime of the
    # thread - the use case resets it on every request, because a checkpointed
    # thread that kept counting would hit the ceiling and refuse to use tools
    # ever again.
    iterations: int = 0


def tools_run_this_turn(messages: Sequence[ChatMessage]) -> frozenset[str]:
    """Every tool whose result is already in the history, since the last question.

    Asked for is not the same as run, and the difference is the whole point: the
    tool node launches every call in one lap concurrently, so a tool requested
    alongside another has produced nothing the model can read yet. A name counts
    here only once a tool turn answering its call has landed - which is exactly
    when a later tool could be said to have "already tried" it.

    Scoped to the turn, not the thread: a search run three questions ago says
    nothing about whether this question has been looked up yet, and a thread
    that accumulated tool names forever would let the first lap of every later
    reply skip straight past a retrieval.
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
