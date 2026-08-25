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
from app.application.chat.agent.progress import emit_summarizing
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
    max_request_tokens: int,
    tracer: Tracer,
) -> Node:
    """Build the node that shortens the thread when it outgrows its budget."""

    # The tighter of the two: the history budget is tuned for conversation
    # quality, the request ceiling for what the provider will accept. Either
    # one crossing is a reason to summarize.
    trigger_budget = min(history_token_budget, max_request_tokens)

    async def summarize(state: AgentState) -> dict[str, Any]:
        tokens_before = counter.count(state.messages)
        async with tracer.span("summarize", kind="span") as span:
            if tokens_before <= trigger_budget:
                # Under budget: no state change, no model call. This is the hot
                # path for most replies, so it must cost nothing - not even a
                # generation span, which is why the condenser is never reached
                # here.
                span.set(tokens_before=tokens_before, tokens_after=tokens_before, summarized=False)
                return {}

            # Announced before the condenser runs, not after, for the same
            # reason a tool call is: this is the point where the user's stream
            # goes quiet for a whole model call, and the notice is only useful
            # while the wait is still happening.
            emit_summarizing(status="start", tokens_before=tokens_before)

            system = (
                state.messages[0] if state.messages and state.messages[0].role == "system" else None
            )
            rest = state.messages[1:] if system is not None else state.messages
            head, tail = _split(rest, counter, recent_token_budget)

            # `_split` refuses to cut between a tool-call turn and its replies,
            # so a lap with several concurrent tool calls can hand back a tail
            # that alone busts the request ceiling - the one budget that is not
            # optional. Shrink that tail through the same condenser tool
            # results already go through, oldest tool result first, so the
            # most recent one - what the model actually needs to answer this
            # turn - survives intact whenever a partial shrink is enough.
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
            # Reported on both outcomes below. A condenser that failed still
            # shortened the thread, and the wait the user sat through was real
            # either way - leaving the notice hanging on "start" would be the
            # one visible sign of a failure they do not need to know about.
            emit_summarizing(status="done", tokens_before=tokens_before, tokens_after=tokens_after)

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


async def _shrink_tail(
    tail: list[ChatMessage],
    counter: TokenCounter,
    condenser: Condenser,
    max_request_tokens: int,
) -> list[ChatMessage]:
    """Condense tool replies in `tail` until it fits the request ceiling.

    `_split` guarantees the tail never orphans a tool reply, not that it fits
    any budget - an assistant turn that fired several tool calls can still hand
    back a tail alone bigger than `max_request_tokens`. Oldest tool reply
    first, because the newest is what the model needs to answer the turn it is
    in; a summary of last week's search is a worse trade than one of this
    turn's third concurrent lookup.
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
