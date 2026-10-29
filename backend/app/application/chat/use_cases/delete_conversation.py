"""Remove a conversation and everything recorded against it."""

from loguru import logger

from app.application.chat.ports import DocumentRemover
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.errors import ConversationNotFound


class DeleteConversation:
    """Delete a thread. Raises `ConversationNotFound` if there is none.

    The owner scopes the DELETE too, not just the read. Attachments go first,
    because the database cascade cannot reach vectors or files.
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
            except Exception as exc:  # noqa: BLE001
                # Logged, not raised, so an outage cannot block deletion.

                logger.warning(
                    f"Could not remove document {document.id} "
                    f"while deleting conversation {conversation_id}: {exc}"
                )

        async with self._uow_factory() as uow:
            await uow.conversations.delete(conversation_id, owner_id)
            await uow.commit()
