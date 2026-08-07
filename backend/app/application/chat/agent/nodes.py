"""The two things the agent can do: think, and use tools.

Both are factories rather than plain functions, because a node needs its
collaborators (the model, the registry) and LangGraph only ever hands it the
state. Per-request values - which model, what temperature - arrive the other
way, through the run config, because the graph is compiled once at startup and
shared by every request in flight.
"""

import asyncio
from typing import Any, Final, Protocol

from langgraph.config import get_config, get_stream_writer
from loguru import logger

from app.application.chat.agent.state import AgentState
from app.application.chat.models import ChatMessage, ToolCall
from app.application.chat.ports import ChatModel
from app.application.chat.tools.base import ToolRegistry

# The vocabulary of the custom stream. These payloads are an internal protocol
# between this module and `use_cases.generate_reply`, which is the only reader:
# nothing outside the application layer ever sees one. Keeping them as plain
# dicts (rather than DTOs) keeps the checkpointer's serialisation trivial.
DELTA: Final = "delta"
TOOL_START: Final = "tool_start"
TOOL_END: Final = "tool_end"


class Node(Protocol):
    """
    What LangGraph will accept as a node.

    A `Protocol` rather than a `Callable` alias because LangGraph matches node
    signatures structurally, on a parameter actually named `state` - a bare
    `Callable[[AgentState], ...]` has a positional-only parameter and is
    rejected.
    """

    async def __call__(self, state: AgentState) -> dict[str, Any]: ...


# How much of a tool's arguments to show the user in the "started" event.
_SUMMARY_CHARS = 120


def _summarise(call: ToolCall) -> str:
    """A one-line, human-readable version of what the tool was asked for."""
    if not call.arguments:
        return call.name
    rendered = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    if len(rendered) > _SUMMARY_CHARS:
        rendered = f"{rendered[:_SUMMARY_CHARS]}..."
    return f"{call.name}({rendered})"


def make_agent_node(model: ChatModel, tools: ToolRegistry) -> Node:
    """Build the node that asks the model what to do next."""
    # The tool specs cannot change between requests - the registry is built at
    # startup - so they are computed once rather than per turn.
    specs = tools.specs()

    async def agent(state: AgentState) -> dict[str, Any]:
        configurable: dict[str, Any] = dict(get_config().get("configurable") or {})
        model_name = str(configurable["model"])
        temperature = float(configurable.get("temperature", 0.0))

        # The custom stream, not the return value, is what reaches the user
        # while the model is still talking. Returning the finished text would
        # make every reply arrive in one lump after the turn completed.
        writer = get_stream_writer()

        parts: list[str] = []
        requested: tuple[ToolCall, ...] = ()

        async for chunk in model.stream(
            model=model_name,
            temperature=temperature,
            messages=state.messages,
            tools=specs,
        ):
            if chunk.text:
                parts.append(chunk.text)
                writer({"type": DELTA, "text": chunk.text})
            if chunk.tool_calls:
                requested += chunk.tool_calls

        message = ChatMessage(role="assistant", content="".join(parts), tool_calls=requested)
        # `iterations` has no reducer, so the incremented value is written whole.
        return {"messages": [message], "iterations": state.iterations + 1}

    return agent


def make_tool_node(tools: ToolRegistry) -> Node:
    """Build the node that runs whatever the model asked for."""

    async def tool_node(state: AgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        calls = state.messages[-1].tool_calls if state.messages else ()
        if not calls:
            # Routing should make this unreachable; returning nothing is still
            # cheaper than raising into an open stream if it ever happens.
            logger.debug("Tool node reached with no pending calls")
            return {"messages": []}

        async def run(call: ToolCall) -> ChatMessage:
            # Announced before the call, so the spinner has a reason on it for
            # the whole of a slow fetch rather than after the fact.
            writer({"type": TOOL_START, "name": call.name, "summary": _summarise(call)})
            outcome = await tools.invoke(call)
            writer({"type": TOOL_END, "name": call.name, "ok": outcome.ok})
            # The id is what pairs this result back to its request; providers
            # reject a tool turn that has none.
            return ChatMessage(role="tool", content=outcome.content, tool_call_id=call.id)

        # Concurrent, because two independent lookups should cost the user the
        # slower one rather than their sum. `ToolRegistry.invoke` never raises,
        # so a bare gather cannot leave a sibling orphaned here.
        messages = await asyncio.gather(*(run(call) for call in calls))
        return {"messages": list(messages)}

    return tool_node
