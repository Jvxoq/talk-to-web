"""Chat entities: a conversation and the exchanges recorded against it."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Message:
    """One completed prompt/response exchange."""

    prompt_content: str
    response_content: str
    id: int | None = None
    conversation_id: int | None = None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    is_success: bool = True
    status_code: int = 200
    created_at: datetime | None = None

    def token_total(self) -> int | None:
        """Total tokens, derived from the halves when the provider omitted it."""
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is None and self.response_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.response_tokens or 0)


@dataclass(slots=True)
class Conversation:
    """A titled thread of exchanges against one model, belonging to one person."""

    title: str
    model_type: str
    # Required, not `int | None`. A conversation with no owner is exactly the
    # state this field exists to make impossible: it would be readable and
    # deletable by anyone, which is what the ownerless version of this app was.
    owner_id: int
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[Message] = field(default_factory=list)

    def record(self, message: Message) -> Message:
        """Attach an exchange to this conversation."""
        message.conversation_id = self.id
        self.messages.append(message)
        return message

    def is_persisted(self) -> bool:
        return self.id is not None
