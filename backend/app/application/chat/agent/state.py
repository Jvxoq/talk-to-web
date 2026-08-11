"""What the agent carries from one node to the next.

Mutable on purpose. LangGraph rebuilds the state object between nodes from the
values each node returned, so a frozen model would fail the moment a reducer
tried to write the merged result back.
"""

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
