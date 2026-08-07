"""Wire shapes for conversations and the exchanges recorded against them.

The `Out` models validate straight off the domain dataclasses via
`from_attributes`, so a route never returns an entity and the storage shape is
free to drift from the wire shape.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.chat.entities import Conversation, Message

# `model_type` trips Pydantic's reserved `model_` namespace. The field name is
# part of the frozen frontend contract, so the namespace guard goes instead.
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
    # Deliberately not `extra="forbid"`: the frontend echoes back message objects
    # it already holds, so an unknown key is not worth a 422.
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
        # The entity knows how to derive a total the provider omitted; the raw
        # attribute would report null where the halves are both known.
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
