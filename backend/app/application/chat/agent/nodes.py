"""The two things the agent can do: think, and use tools.

Both are factories rather than plain functions, because a node needs its
collaborators (the model, the registry) and LangGraph only ever hands it the
state. Per-request values - which model, what temperature - arrive the other
way, through the run config, because the graph is compiled once at startup and
shared by every request in flight.
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

# The vocabulary of the custom stream: an internal protocol between this
# module and `use_cases.generate_reply`, the only reader. Plain dicts, so
# the checkpointer's serialisation stays trivial.
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


def _optional_int(value: Any) -> int | None:
    """Read a config value that is allowed to be absent.

    A turn with no conversation is a real shape - a client that never opened
    one - so a missing thread id is `None` rather than an error. It is not a
    permissive default: no thread owns no documents, so this fails closed.
    """
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

        # The last lap gets no tools bound. A model still asking for tools at the
        # ceiling is looping, and `route_after_agent` would discard that request
        # and end the reply on whatever was streamed alongside it - usually
        # nothing. Taking tools away forces a real text answer instead.
        final_lap = state.iterations >= max_iterations - 1
        turn_specs = () if final_lap else specs

        # The custom stream, not the return value, is what reaches the user while
        # the model is still talking.
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
        # Both subscripted, not `.get(...)`: a run config missing either is a
        # miswiring, and a default would fail quietly in the worst direction. A
        # default owner searches somebody else's documents; a default
        # `document_scoped` sends a private question to the web with no log line.
        #
        # `document_scoped` comes from the config rather than being recomputed
        # from `state.messages`: the summarize node can replace that history, and
        # a run that compressed the current turn away would match an older
        # question. It is settled once, in `GenerateReply`.
        #
        # `prior_tools` does have to be recomputed every lap, because it grows as
        # the turn does. Safe to read from the history because the summarize node
        # keeps a suffix, so dropping the current turn drops every earlier one too.
        #
        # Two tools requested in the same lap run concurrently, so neither has
        # answered and a search alongside a retrieval is held back. That is the
        # intent: the retrieval's answer should exist before a search is paid for.
        messages = state.messages
        context = ToolContext(
            owner_id=int(configurable["owner_id"]),
            # Next to `owner_id`, for the same reason: which thread is asking is a
            # fact about the request, never something the model may name.
            conversation_id=_optional_int(configurable.get("conversation_id")),
            document_scoped=bool(configurable["document_scoped"]),
            # `.get`, unlike the two above: this one's default is the behaviour that
            # existed before the flag did, so a missing value costs a wasted call
            # rather than a leak.
            has_documents=bool(configurable.get("has_documents", True)),
            prior_tools=tools_run_this_turn(messages),
        )
        calls = messages[-1].tool_calls if messages else ()
        if not calls:
            # Routing should make this unreachable; returning nothing is still
            # cheaper than raising into an open stream if it ever happens.
            logger.debug("Tool node reached with no pending calls")
            return {"messages": []}

        async def run(call: ToolCall) -> ChatMessage:
            # Announced before the call, so the spinner has a reason on it for
            # the whole of a slow fetch rather than after the fact.
            writer({"type": TOOL_START, "name": call.name, "summary": _summarise(call)})
            # One span per call: the node runs them concurrently, and one span would
            # hide which was slow.
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
                    # Dumped to plain dicts, not passed as `Source` instances:
                    # the custom stream is a JSON-only channel between this
                    # module and `use_cases.generate_reply`, and a pydantic
                    # model surviving that trip is an accident, not a contract.
                    "sources": [source.model_dump() for source in outcome.sources],
                }
            )
            # The id is what pairs this result back to its request; providers
            # reject a tool turn that has none.
            return ChatMessage(role="tool", content=content, tool_call_id=call.id)

        # Concurrent, because two independent lookups should cost the user the
        # slower one rather than their sum. `ToolRegistry.invoke` never raises,
        # so a bare gather cannot leave a sibling orphaned here.
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
    """Shorten a tool result that would otherwise blow the request budget.

    A small result is returned untouched - a three-line retrieval must not cost
    an extra model call. A large one is condensed against the call's arguments,
    so the condenser keeps what answers the question rather than the first N
    characters. If the condenser fails, the result is truncated so the message
    is still bounded.
    """
    if counter.count([ChatMessage(role="tool", content=content)]) <= budget:
        return content

    focus = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    condensed = await condenser.condense(content, focus=focus)
    if condensed is not None:
        return condensed

    logger.warning("Tool output condensation failed; truncating result for {}", call.name)
    return content[: condenser.max_chars]
