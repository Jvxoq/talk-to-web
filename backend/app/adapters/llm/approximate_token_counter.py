"""A cheap local token estimate. Implements `TokenCounter`.

Only decides whether to condense, so approximate is fine. Never calls a model.
"""

from collections.abc import Sequence

from langchain_core.messages.utils import count_tokens_approximately

from app.adapters.llm.langchain_chat_model import _to_langchain_message
from app.application.chat.models import ChatMessage


class ApproximateTokenCounter:
    """Counts a conversation in the same shape the provider will bill it."""

    def count(self, messages: Sequence[ChatMessage]) -> int:
        # Counts roles and tool calls too, not just characters.

        return count_tokens_approximately([_to_langchain_message(message) for message in messages])
