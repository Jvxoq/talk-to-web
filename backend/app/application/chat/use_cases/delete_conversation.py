"""Remove a conversation and everything recorded against it."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.errors import ConversationNotFound


class DeleteConversation:
    """
    Delete a thread, failing loudly when there was nothing to delete.

    A silent no-op would let a client believe it removed someone else's
    conversation, so the missing case is reported rather than swallowed.

    The owner is passed all the way down to the DELETE rather than checked here
    and dropped: a scoped read followed by an unscoped write is still a way to
    delete a stranger's thread the moment the two drift apart.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, conversation_id: int, owner_id: int) -> None:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(conversation_id, owner_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            await uow.conversations.delete(conversation_id, owner_id)
            await uow.commit()
