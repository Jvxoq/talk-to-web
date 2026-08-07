"""Read one conversation back."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationNotFound


class GetConversation:
    """
    Fetch a thread by id.

    Absence is a business outcome, not an empty result: the caller is told with
    a domain error so every route reports a missing thread the same way.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, conversation_id: int) -> Conversation:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(conversation_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            return conversation
