"""A cheap token estimate, good enough to decide whether to condense.

Satisfies `app.application.chat.ports.TokenCounter`. The estimate is deliberately
approximate: it only decides *whether* to condense, and being 10% out moves the
threshold a little rather than breaking anything. What it must not do is cost a
model call, which is why it is a local character count rather than a request to
the provider.
"""

from collections.abc import Sequence

from langchain_core.messages.utils import count_tokens_approximately

from app.adapters.llm.langchain_chat_model import _to_langchain_message
from app.application.chat.models import ChatMessage


class ApproximateTokenCounter:
    """Counts a conversation in the same shape the provider will bill it."""

    def count(self, messages: Sequence[ChatMessage]) -> int:
        # `count_tokens_approximately` understands the message shape directly -
        # role, tool calls and tool call ids all count - so the estimate tracks
        # what the provider actually charges rather than a bare character count.
        return count_tokens_approximately([_to_langchain_message(message) for message in messages])
