"""Remove a conversation and everything recorded against it."""

from loguru import logger

from app.application.chat.ports import DocumentRemover
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

    Attachments go first, one at a time and completely. The `documents` row
    would fall to the foreign key cascade on its own, but a cascade reaches
    only as far as the database: the passages in the vector store and the file
    on disk are outside it, and deleting the row first would leave both
    orphaned with nothing left naming them.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory, remove_document: DocumentRemover) -> None:
        self._uow_factory = uow_factory
        self._remove_document = remove_document

    async def __call__(self, conversation_id: int, owner_id: int) -> None:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(conversation_id, owner_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            attached = await uow.documents.list_by_conversation(owner_id, conversation_id)

        for document in attached:
            if document.id is None:
                continue
            try:
                await self._remove_document(document.id, owner_id)
            except Exception as exc:
                # Logged, not raised. A vector store that will not answer must
                # not leave the user unable to delete their own conversation;
                # the cascade still removes the row behind it.
                logger.warning(
                    f"Could not remove document {document.id} "
                    f"while deleting conversation {conversation_id}: {exc}"
                )

        async with self._uow_factory() as uow:
            await uow.conversations.delete(conversation_id, owner_id)
            await uow.commit()
