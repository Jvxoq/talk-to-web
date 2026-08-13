"""Wiring: two nodes, one loop, one exit condition.

Deliberately the shortest file in the package. Adding a capability to the agent
should mean one `add_node` and one `add_edge` here, and nothing else - which is
only true while this stays a wiring diagram rather than a place logic accretes.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.nodes import make_agent_node, make_tool_node
from app.application.chat.agent.state import AgentState
from app.application.chat.agent.summarization import make_summarize_node
from app.application.chat.ports import ChatModel, TokenCounter, Tracer
from app.application.chat.tools.base import ToolRegistry

AgentGraph = CompiledStateGraph[AgentState, Any, Any, Any]


def build_agent_graph(
    *,
    model: ChatModel,
    tools: ToolRegistry,
    max_iterations: int,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    condenser: Condenser,
    counter: TokenCounter,
    history_token_budget: int,
    recent_token_budget: int,
    tool_output_token_budget: int,
    tracer: Tracer,
) -> AgentGraph:
    """Compile the agent once, at startup, for every request to share."""
    if max_iterations < 1:
        raise ValueError("The agent needs at least one model turn per reply")

    def route_after_agent(state: AgentState) -> str:
        """Loop back through the tools, or stop."""
        last = state.messages[-1] if state.messages else None
        if last is None or not last.tool_calls:
            return END
        if state.iterations >= max_iterations:
            # A model that keeps asking for tools is looping, and every lap is a
            # request the user is already waiting on. Stopping here leaves the
            # last assistant turn in place, so the reply ends short rather than
            # never.
            logger.warning(f"Agent hit its ceiling of {max_iterations} iterations; stopping")
            return END
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", make_agent_node(model, tools, tracer))
    builder.add_node(
        "tools",
        make_tool_node(tools, condenser, counter, tool_output_token_budget, tracer),
    )
    builder.add_node(
        "summarize",
        make_summarize_node(counter, condenser, history_token_budget, recent_token_budget, tracer),
    )

    # Summarize runs on both entry paths - the start of a reply and after every
    # tool lap - because that is where the growth actually happens: history
    # carried in from the checkpointer, and a tool result landing mid-reply.
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "summarize")

    return builder.compile(checkpointer=checkpointer)
