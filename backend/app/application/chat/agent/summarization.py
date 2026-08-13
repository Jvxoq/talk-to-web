"""The node that keeps a long thread from growing without bound.

Runs on both entry paths - at the start of a reply and after every tool lap -
because that is where the growth actually happens: history carried in from the
checkpointer, and a tool result landing mid-reply. When the thread passes a
token budget it summarises the older part into one message and keeps the recent
part verbatim, so the model still answers about something said earlier without
re-sending the whole conversation on every lap.
"""

from typing import Any

from loguru import logger

from app.application.chat.agent.condenser import Condenser
from app.application.chat.agent.nodes import Node
from app.application.chat.agent.state import RESET, AgentState
from app.application.chat.models import ChatMessage
from app.application.chat.ports import TokenCounter, Tracer

# The prefix on the summary message, so the model can tell a summary of the past
# from the live thread it is answering.
_SUMMARY_PREFIX = "Summary of the earlier conversation:\n\n"


def make_summarize_node(
    counter: TokenCounter,
    condenser: Condenser,
    history_token_budget: int,
    recent_token_budget: int,
    tracer: Tracer,
) -> Node:
    """Build the node that shortens the thread when it outgrows its budget."""

    async def summarize(state: AgentState) -> dict[str, Any]:
        tokens_before = counter.count(state.messages)
        async with tracer.span("summarize", kind="span") as span:
            if tokens_before <= history_token_budget:
                # Under budget: no state change, no model call. This is the hot
                # path for most replies, so it must cost nothing - not even a
                # generation span, which is why the condenser is never reached
                # here.
                span.set(tokens_before=tokens_before, tokens_after=tokens_before, summarized=False)
                return {}

            system = (
                state.messages[0] if state.messages and state.messages[0].role == "system" else None
            )
            rest = state.messages[1:] if system is not None else state.messages
            head, tail = _split(rest, counter, recent_token_budget)

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

            if summary_text is None:
                # The condenser failed. Dropping the head outright still bounds
                # the thread and keeps the reply alive - losing grounding is
                # acceptable, losing the reply is not.
                logger.warning("History summarization failed; dropping the older messages")
                return {"messages": new_messages, "summary": state.summary}

            return {"messages": new_messages, "summary": summary_text}

    return summarize


def _split(
    messages: list[ChatMessage],
    counter: TokenCounter,
    recent_token_budget: int,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Cut `messages` into a summarisable head and a verbatim tail.

    The tail keeps roughly `recent_token_budget` tokens, but the cut is never
    allowed to orphan a tool message: an assistant turn that asked for tools
    must stay with the tool replies that answer it, or the provider rejects the
    whole request. So the cut is walked forward until the first tail message is
    a user turn or an assistant turn with no pending tool calls.
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
