"""Translation between ORM rows and domain entities.

The domain never sees a `Mapped` attribute and the ORM never sees a dataclass;
this module is the only place that knows both shapes.
"""

from app.adapters.persistence.models import ConversationModel, MessageModel
from app.domain.chat.entities import Conversation, Message


def message_to_domain(row: MessageModel) -> Message:
    """Read one `messages` row into a domain `Message`."""
    return Message(
        prompt_content=row.prompt_content,
        response_content=row.response_content,
        id=row.id,
        conversation_id=row.conversation_id,
        prompt_tokens=row.prompt_tokens,
        response_tokens=row.response_tokens,
        total_tokens=row.total_tokens,
        is_success=bool(row.is_success) if row.is_success is not None else True,
        status_code=row.status_code if row.status_code is not None else 200,
        created_at=row.created_at,
    )


def message_to_model(entity: Message) -> MessageModel:
    """Build an unsaved `messages` row from a domain `Message`."""
    model = MessageModel(
        conversation_id=entity.conversation_id,
        prompt_content=entity.prompt_content,
        response_content=entity.response_content,
        prompt_tokens=entity.prompt_tokens,
        response_tokens=entity.response_tokens,
        total_tokens=entity.token_total(),
        is_success=entity.is_success,
        status_code=entity.status_code,
    )
    if entity.id is not None:
        model.id = entity.id
    return model


def conversation_to_domain(row: ConversationModel) -> Conversation:
    """
    Read one `conversations` row, and its messages, into a domain `Conversation`.

    `row.messages` is only safe to touch because the repository eager-loads it
    with `selectinload`. Mappers must never trigger a lazy load: under asyncio
    that raises `MissingGreenlet` rather than quietly issuing a query, so the
    responsibility for loading relationships stays with the repository.
    """
    return Conversation(
        title=row.title,
        model_type=row.model_type,
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        messages=[message_to_domain(message) for message in row.messages],
    )


def conversation_to_model(entity: Conversation) -> ConversationModel:
    """Build an unsaved `conversations` row, with its messages, from the entity."""
    model = ConversationModel(
        title=entity.title,
        model_type=entity.model_type,
        messages=[message_to_model(message) for message in entity.messages],
    )
    if entity.id is not None:
        model.id = entity.id
    return model
