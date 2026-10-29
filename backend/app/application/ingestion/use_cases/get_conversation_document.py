"""Read the one document a conversation currently has attached."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import UnknownConversation


class GetConversationDocument:
    """The thread's one attachment, or `None`, for the composer chip.

    The client reads it on opening a thread, since the file outlives the tab.
    A stranger's thread raises `UnknownConversation`.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, conversation_id: int, owner_id: int) -> UploadedDocument | None:
        async with self._uow_factory() as uow:
            if await uow.conversations.get(conversation_id, owner_id) is None:
                raise UnknownConversation(conversation_id)
            documents = await uow.documents.list_by_conversation(owner_id, conversation_id)
        return documents[0] if documents else None
