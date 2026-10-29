"""Read one conversation back."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationNotFound


class GetConversation:
    """Fetch a thread by id. Raises `ConversationNotFound` if missing.

    Someone else's thread is "not found" too, so ids reveal nothing.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, conversation_id: int, owner_id: int) -> Conversation:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(conversation_id, owner_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            return conversation
