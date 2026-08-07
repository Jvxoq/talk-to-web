"""What the agent carries from one node to the next.

Mutable on purpose. LangGraph rebuilds the state object between nodes from the
values each node returned, so a frozen model would fail the moment a reducer
tried to write the merged result back.
"""

import operator
from typing import Annotated

from pydantic import BaseModel

from app.application.chat.models import ChatMessage


class AgentState(BaseModel):
    """The conversation so far, plus how many laps of the loop it has cost."""

    # `operator.add` is the reducer: a node returns only the messages it
    # produced, and LangGraph appends them. Without it a node returning one
    # message would silently replace the whole history.
    messages: Annotated[list[ChatMessage], operator.add]

    # Counts model turns within a single reply, not for the lifetime of the
    # thread - the use case resets it on every request, because a checkpointed
    # thread that kept counting would hit the ceiling and refuse to use tools
    # ever again.
    iterations: int = 0
