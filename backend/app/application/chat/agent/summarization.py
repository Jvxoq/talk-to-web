"""The node that keeps a long thread within its token budget.

Runs at the start of a reply and after every tool lap. Over budget, it
summarizes the older messages into one and keeps the recent ones verbatim.
"""

from typing import Any

from loguru import logger

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.nodes import Node
from app.application.chat.agent.progress import emit_summarizing
from app.application.chat.agent.state import RESET, AgentState
from app.application.chat.models import ChatMessage
from app.application.chat.ports import TokenCounter, Tracer

# Marks the summary so the model can tell it from the live thread.
_SUMMARY_PREFIX = "Summary of the earlier conversation:\n\n"


def make_summarize_node(
    counter: TokenCounter,
    condenser: Condenser,
    history_token_budget: int,
    recent_token_budget: int,
    max_request_tokens: int,
    tracer: Tracer,
) -> Node:
    """Build the node that shortens the thread when it outgrows its budget."""

    # Whichever is tighter: the quality budget or the provider's limit.
    trigger_budget = min(history_token_budget, max_request_tokens)

    async def summarize(state: AgentState) -> dict[str, Any]:
        tokens_before = counter.count(state.messages)
        async with tracer.span("summarize", kind="span") as span:
            if tokens_before <= trigger_budget:
                # Under budget: the common case, so no model call at all.
                span.set(tokens_before=tokens_before, tokens_after=tokens_before, summarized=False)
                return {}

            # Announced first, because the stream goes quiet during the condenser call.
            emit_summarizing(status="start", tokens_before=tokens_before)

            system = (
                state.messages[0] if state.messages and state.messages[0].role == "system" else None
            )
            rest = state.messages[1:] if system is not None else state.messages
            head, tail = _split(rest, counter, recent_token_budget)

            # The tail alone can exceed the request limit after several tool
            # calls. Shrink older tool results first, keeping the newest intact.
            tail = await _shrink_tail(tail, counter, condenser, max_request_tokens)

            summary_text = await condenser.summarize(head)

            new_messages: list[ChatMessage] = [RESET]
            if system is not None:
                new_messages.append(system)
            if summary_text is not None:
                new_messages.append(
                    ChatMessage(role="system", content=_SUMMARY_PREFIX + summary_text)
                )
            new_messages.extend(tail)

            tokens_after = counter.count(new_messages[1:])
            span.set(tokens_before=tokens_before, tokens_after=tokens_after, summarized=True)
            # Sent even if the condenser failed, so the notice never hangs on "start".
            emit_summarizing(status="done", tokens_before=tokens_before, tokens_after=tokens_after)

            if summary_text is None:
                # Condenser failed. Drop the head: losing context beats losing the reply.
                logger.warning("History summarization failed; dropping the older messages")
                return {"messages": new_messages, "summary": state.summary}

            return {"messages": new_messages, "summary": summary_text}

    return summarize


def _split(
    messages: list[ChatMessage],
    counter: TokenCounter,
    recent_token_budget: int,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Split into a head to summarize and a tail of about `recent_token_budget` tokens.

    The cut moves forward until it does not separate a tool call from its
    replies, which providers reject.
    """
    count = 0
    cut = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if count >= recent_token_budget:
            break
        count += counter.count([messages[index]])
        cut = index

    while cut < len(messages) and not _is_safe_boundary(messages[cut]):
        cut += 1

    return messages[:cut], messages[cut:]


def _is_safe_boundary(message: ChatMessage) -> bool:
    """A message the tail may start on without orphaning a tool reply."""
    return message.role == "user" or (message.role == "assistant" and not message.tool_calls)


async def _shrink_tail(
    tail: list[ChatMessage],
    counter: TokenCounter,
    condenser: Condenser,
    max_request_tokens: int,
) -> list[ChatMessage]:
    """Condense tool replies in `tail` until it fits `max_request_tokens`.

    Oldest first. The newest reply is never condensed.
    """

    if counter.count(tail) <= max_request_tokens:
        return tail

    shrunk = list(tail)
    tool_indexes = [index for index, message in enumerate(shrunk) if message.role == "tool"][:-1]
    for index in tool_indexes:
        if counter.count(shrunk) <= max_request_tokens:
            break
        message = shrunk[index]
        condensed = await condenser.condense(message.content, focus="the current question")
        fallback = message.content[: condenser.max_chars]
        shrunk[index] = message.model_copy(update={"content": condensed or fallback})

    return shrunk
