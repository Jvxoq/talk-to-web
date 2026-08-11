"""Read one conversation back."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationNotFound


class GetConversation:
    """
    Fetch a thread by id.

    Absence is a business outcome, not an empty result: the caller is told with
    a domain error so every route reports a missing thread the same way.

    Someone else's thread is reported as missing too, rather than as forbidden.
    A 403 would confirm that the id exists and belongs to somebody, which is one
    request per integer away from a map of who has how many conversations.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, conversation_id: int, owner_id: int) -> Conversation:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(conversation_id, owner_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            return conversation
