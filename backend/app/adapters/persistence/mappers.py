"""Translation between ORM rows and domain entities. The only place that knows both."""

from app.adapters.persistence.models import (
    ConversationModel,
    DocumentModel,
    MessageModel,
    RefreshTokenModel,
    UserModel,
)
from app.domain.chat.entities import Conversation, Message
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.value_objects import Email
from app.domain.ingestion.entities import UploadedDocument


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
    """Read a `conversations` row and its messages into a `Conversation`.

    The repository must eager-load `row.messages`. A lazy load raises under asyncio.
    """
    return Conversation(
        title=row.title,
        model_type=row.model_type,
        owner_id=row.owner_id,
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        messages=[message_to_domain(message) for message in row.messages],
    )


def conversation_summary_to_domain(row: ConversationModel) -> Conversation:
    """Read a `conversations` row into a `Conversation`, without touching messages."""
    return Conversation(
        title=row.title,
        model_type=row.model_type,
        owner_id=row.owner_id,
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def conversation_to_model(entity: Conversation) -> ConversationModel:
    """Build an unsaved `conversations` row, with its messages, from the entity."""
    model = ConversationModel(
        title=entity.title,
        model_type=entity.model_type,
        owner_id=entity.owner_id,
        messages=[message_to_model(message) for message in entity.messages],
    )
    if entity.id is not None:
        model.id = entity.id
    return model


def user_to_domain(row: UserModel) -> User:
    """Read one `users` row into a domain `User`."""
    return User(
        # Not re-sanitized, so a stricter rule never locks out existing users.
        email=Email(row.email),
        password_hash=row.password_hash,
        id=row.id,
        is_active=row.is_active,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def user_to_model(entity: User) -> UserModel:
    """Build an unsaved `users` row from a domain `User`."""
    model = UserModel(
        email=entity.email.value,
        password_hash=entity.password_hash,
        is_active=entity.is_active,
    )
    if entity.id is not None:
        model.id = entity.id
    return model


def document_to_domain(row: DocumentModel) -> UploadedDocument:
    """Read one `documents` row into a domain `UploadedDocument`."""
    return UploadedDocument(
        name=row.name,
        reference=row.reference,
        owner_id=row.owner_id,
        conversation_id=row.conversation_id,
        id=row.id,
        chunks_indexed=row.chunks_indexed,
        summary=row.summary,
        created_at=row.created_at,
    )


def document_to_model(entity: UploadedDocument) -> DocumentModel:
    """Build an unsaved `documents` row from a domain `UploadedDocument`."""
    model = DocumentModel(
        name=entity.name,
        reference=entity.reference,
        owner_id=entity.owner_id,
        conversation_id=entity.conversation_id,
        chunks_indexed=entity.chunks_indexed,
        summary=entity.summary,
    )
    if entity.id is not None:
        model.id = entity.id
    return model


def refresh_token_to_domain(row: RefreshTokenModel) -> RefreshToken:
    """Read one `refresh_tokens` row into a domain `RefreshToken`."""
    return RefreshToken(
        user_id=row.user_id,
        fingerprint=row.fingerprint,
        expires_at=row.expires_at,
        id=row.id,
        revoked_at=row.revoked_at,
        created_at=row.created_at,
    )


def refresh_token_to_model(entity: RefreshToken) -> RefreshTokenModel:
    """Build an unsaved `refresh_tokens` row from a domain `RefreshToken`."""
    model = RefreshTokenModel(
        user_id=entity.user_id,
        fingerprint=entity.fingerprint,
        expires_at=entity.expires_at,
        revoked_at=entity.revoked_at,
    )
    if entity.id is not None:
        model.id = entity.id
    return model
