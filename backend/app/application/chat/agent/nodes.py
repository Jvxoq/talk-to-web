"""The agent's two nodes: think, and use tools.

Factories, because LangGraph passes nodes only the state. Per-request values
come through the run config, since one compiled graph serves every request.
"""

import asyncio
import time
from typing import Any, Final, Protocol

from langgraph.config import get_config, get_stream_writer
from loguru import logger

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.state import AgentState, tools_run_this_turn
from app.application.chat.agent.usage import emit_usage
from app.application.chat.models import ChatMessage, ToolCall
from app.application.chat.ports import ChatModel, TokenCounter, Tracer
from app.application.chat.tools.base import ToolContext, ToolRegistry

# Custom-stream event types, read only by `use_cases.generate_reply`.
DELTA: Final = "delta"
TOOL_START: Final = "tool_start"
TOOL_END: Final = "tool_end"


class Node(Protocol):
    """A LangGraph node. A `Protocol`, because LangGraph needs a parameter named `state`."""

    async def __call__(self, state: AgentState) -> dict[str, Any]: ...


# How much of a tool's arguments to show the user in the "started" event.
_SUMMARY_CHARS = 120


def _optional_int(value: Any) -> int | None:
    """Read an optional config int. `None` fails closed: no thread owns no documents."""
    return None if value is None else int(value)


def _summarise(call: ToolCall) -> str:
    """A one-line, human-readable version of what the tool was asked for."""
    if not call.arguments:
        return call.name
    rendered = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    if len(rendered) > _SUMMARY_CHARS:
        rendered = f"{rendered[:_SUMMARY_CHARS]}..."
    return f"{call.name}({rendered})"


def make_agent_node(
    model: ChatModel, tools: ToolRegistry, max_iterations: int, tracer: Tracer
) -> Node:
    """Build the node that asks the model what to do next."""
    # The registry is built at startup, so the specs are computed once.
    specs = tools.specs()

    async def agent(state: AgentState) -> dict[str, Any]:
        configurable: dict[str, Any] = dict(get_config().get("configurable") or {})
        model_name = str(configurable["model"])
        temperature = float(configurable.get("temperature", 0.0))

        # No tools on the last lap, which forces a text answer instead of an empty reply.
        final_lap = state.iterations >= max_iterations - 1
        turn_specs = () if final_lap else specs

        # The custom stream reaches the user while the model is still talking.
        writer = get_stream_writer()

        parts: list[str] = []
        requested: tuple[ToolCall, ...] = ()

        async with tracer.span(
            "llm.agent", kind="generation", model=model_name, temperature=temperature
        ) as span:
            start = time.perf_counter()
            ttft_ms: float | None = None
            try:
                async for chunk in model.stream(
                    model=model_name,
                    temperature=temperature,
                    messages=state.messages,
                    tools=turn_specs,
                ):
                    if chunk.text:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        parts.append(chunk.text)
                        writer({"type": DELTA, "text": chunk.text})
                    if chunk.tool_calls:
                        requested += chunk.tool_calls
                    if chunk.usage is not None:
                        emit_usage(model=model_name, usage=chunk.usage)
                        span.set(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
            except Exception as error:
                span.record_error(error)
                raise
            span.set(tool_calls=[call.name for call in requested], ttft_ms=ttft_ms)

        message = ChatMessage(role="assistant", content="".join(parts), tool_calls=requested)
        # `iterations` has no reducer, so the incremented value is written whole.
        return {"messages": [message], "iterations": state.iterations + 1}

    return agent


def make_tool_node(
    tools: ToolRegistry,
    condenser: Condenser,
    counter: TokenCounter,
    tool_output_token_budget: int,
    tracer: Tracer,
) -> Node:
    """Build the node that runs whatever the model asked for."""

    async def tool_node(state: AgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        configurable: dict[str, Any] = dict(get_config().get("configurable") or {})
        # `owner_id` and `document_scoped` are required. A default would fail
        # silently in the dangerous direction.
        # `prior_tools` is recomputed each lap. Summarizing keeps a suffix, so
        # this is safe. Calls in one lap run together, so a search sent with a
        # retrieval is held back until the retrieval answers.
        messages = state.messages
        context = ToolContext(
            owner_id=int(configurable["owner_id"]),
            # From the request, never from the model.
            conversation_id=_optional_int(configurable.get("conversation_id")),
            document_scoped=bool(configurable["document_scoped"]),
            # `.get` is safe here: a missing value costs a wasted call, not a leak.
            has_documents=bool(configurable.get("has_documents", True)),
            prior_tools=tools_run_this_turn(messages),
        )
        calls = messages[-1].tool_calls if messages else ()
        if not calls:
            # Should be unreachable. Returning nothing beats raising into an open stream.
            logger.debug("Tool node reached with no pending calls")
            return {"messages": []}

        async def run(call: ToolCall) -> ChatMessage:
            # Announced first, so the spinner explains a slow call while it runs.
            writer({"type": TOOL_START, "name": call.name, "summary": _summarise(call)})
            # One span per call, to show which concurrent call was slow.
            async with tracer.span(
                f"tool.{call.name}", kind="span", arguments=call.arguments
            ) as span:
                start = time.perf_counter()
                try:
                    outcome = await tools.invoke(call, context)
                    content = await _compress(
                        outcome.content, call, condenser, counter, tool_output_token_budget
                    )
                except Exception as error:
                    span.record_error(error)
                    raise
                span.set(
                    ok=outcome.ok,
                    sources=len(outcome.sources),
                    output_chars=len(content),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            writer(
                {
                    "type": TOOL_END,
                    "name": call.name,
                    "ok": outcome.ok,
                    # Plain dicts, because the custom stream carries JSON only.
                    "sources": [source.model_dump() for source in outcome.sources],
                }
            )
            # Providers reject a tool result without the call id.
            return ChatMessage(role="tool", content=content, tool_call_id=call.id)

        # Concurrent, so the user waits for the slowest call, not the sum.
        # `invoke` never raises, so a bare gather is safe.
        messages = await asyncio.gather(*(run(call) for call in calls))
        return {"messages": list(messages)}

    return tool_node


async def _compress(
    content: str,
    call: ToolCall,
    condenser: Condenser,
    counter: TokenCounter,
    budget: int,
) -> str:
    """Shorten a tool result that is over budget.

    Small results pass untouched. Large ones are condensed against the call's
    arguments. If condensing fails, the result is truncated.
    """

    if counter.count([ChatMessage(role="tool", content=content)]) <= budget:
        return content

    focus = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    condensed = await condenser.condense(content, focus=focus)
    if condensed is not None:
        return condensed

    logger.warning("Tool output condensation failed; truncating result for {}", call.name)
    return content[: condenser.max_chars]
