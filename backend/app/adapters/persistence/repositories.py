"""SQLAlchemy implementations of the chat repository ports."""

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.persistence.mappers import (
    conversation_to_domain,
    conversation_to_model,
    message_to_domain,
    message_to_model,
)
from app.adapters.persistence.models import ConversationModel
from app.domain.chat.entities import Conversation, Message


class SqlAlchemyConversationRepository:
    """
    Persists conversations with SQLAlchemy.

    Structurally satisfies `app.application.chat.ports.ConversationRepository`;
    it deliberately does not import or inherit from it, so the dependency arrow
    keeps pointing inward.

    Nothing here commits. The session is owned by the unit of work, which is the
    only thing that knows where the transaction boundary is; a repository that
    committed would make a multi-step use case impossible to roll back.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, conversation_id: int) -> Conversation | None:
        """Load a conversation with its messages, or `None` if there is no such row."""
        statement = (
            select(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            # Eager load: the mapper reads `row.messages`, and a lazy load from
            # async code raises instead of quietly issuing a second query.
            .options(selectinload(ConversationModel.messages))
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is None:
            logger.debug(f"Conversation {conversation_id} not found")
            return None
        return conversation_to_domain(row)

    async def add(self, conversation: Conversation) -> Conversation:
        """Insert a conversation and return it with its database-assigned id."""
        row = conversation_to_model(conversation)
        self._session.add(row)
        # Flush, not commit: the INSERT runs so the id and column defaults are
        # populated, but the transaction stays open for the unit of work to close.
        await self._session.flush()
        logger.debug(f"Inserted conversation {row.id}")
        # `row.messages` was assigned by the mapper, so the collection is already
        # loaded here - reading it does not trigger a lazy load.
        return conversation_to_domain(row)

    async def add_message(self, conversation_id: int, message: Message) -> Message:
        """Insert one exchange against a conversation and return it with its id."""
        message.conversation_id = conversation_id
        row = message_to_model(message)
        row.conversation_id = conversation_id
        self._session.add(row)
        await self._session.flush()
        logger.debug(f"Inserted message {row.id} on conversation {conversation_id}")
        return message_to_domain(row)

    async def delete(self, conversation_id: int) -> None:
        """Remove a conversation. Its messages go with it via the FK cascade."""
        await self._session.execute(
            delete(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        logger.debug(f"Deleted conversation {conversation_id}")
