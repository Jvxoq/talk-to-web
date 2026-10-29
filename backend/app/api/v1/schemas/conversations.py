"""Wire shapes for conversations and their messages.

`Out` models read from domain objects, so routes never return entities.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.chat.entities import Conversation, Message

# Allows the `model_type` field name without Pydantic's warning.
_WIRE = ConfigDict(protected_namespaces=())
_FROM_DOMAIN = ConfigDict(from_attributes=True, protected_namespaces=())


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    title: str = Field(min_length=1, max_length=200)
    model_type: str = Field(min_length=1, max_length=100)


class ConversationOut(BaseModel):
    model_config = _FROM_DOMAIN

    id: int
    title: str
    model_type: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationOut":
        return cls.model_validate(conversation)


class MessageCreate(BaseModel):
    # Extra keys allowed: the frontend echoes whole message objects back.
    model_config = _WIRE

    prompt_content: str = Field(min_length=1, max_length=32_000)
    response_content: str = Field(max_length=1_000_000)
    prompt_tokens: int | None = Field(default=None, ge=0)
    response_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    is_success: bool = True
    status_code: int = 200


class MessageOut(BaseModel):
    model_config = _FROM_DOMAIN

    id: int | None = None
    prompt_content: str
    response_content: str
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    is_success: bool
    status_code: int
    created_at: datetime | None = None

    @classmethod
    def from_domain(cls, message: Message) -> "MessageOut":
        out = cls.model_validate(message)
        # Derive the total when the provider left it out.

        out.total_tokens = message.token_total()
        return out


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationDetailOut":
        return cls(
            id=conversation.id,
            title=conversation.title,
            model_type=conversation.model_type,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[MessageOut.from_domain(message) for message in conversation.messages],
        )
