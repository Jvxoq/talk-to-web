"""Wiring for the agent graph. Keep logic out of here."""

import json
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.nodes import make_agent_node, make_tool_node
from app.application.chat.agent.state import AgentState
from app.application.chat.agent.summarization import make_summarize_node
from app.application.chat.models import ChatMessage
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
    max_request_tokens: int,
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
            # Safety net. The last lap has no tools, but a model may call one anyway.
            logger.warning(f"Agent hit its ceiling of {max_iterations} iterations; stopping")
            return END
        return "tools"

    # Tool schemas are sent every request but are not in `state.messages`.
    # Measure them once and subtract from the request limit.
    tool_schema_tokens = counter.count(
        [
            ChatMessage(
                role="system",
                content=json.dumps(
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                ),
            )
            for spec in tools.specs()
        ]
    )
    request_ceiling = max(max_request_tokens - tool_schema_tokens, 0)

    builder = StateGraph(AgentState)
    builder.add_node("agent", make_agent_node(model, tools, max_iterations, tracer))
    builder.add_node(
        "tools",
        make_tool_node(tools, condenser, counter, tool_output_token_budget, tracer),
    )
    builder.add_node(
        "summarize",
        make_summarize_node(
            counter,
            condenser,
            history_token_budget,
            recent_token_budget,
            request_ceiling,
            tracer,
        ),
    )

    # Summarize at the start of a reply and after every tool lap.

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "summarize")

    return builder.compile(checkpointer=checkpointer)
