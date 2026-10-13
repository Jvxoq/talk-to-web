"""Read the one document a conversation currently has attached."""

from dataclasses import replace
from datetime import timedelta

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory
from app.domain.ingestion.entities import UploadedDocument
from app.domain.ingestion.errors import UnknownConversation


class GetConversationDocument:
    """Answer what a thread's composer chip should show.

    A conversation holds one attachment at a time, so this is a single
    document or none - there is no list to render. The client asks for it when
    it opens a thread, because the attachment outlives the tab: a user who
    uploaded a file yesterday, closed the thread and came back must see the
    chip again, or the model would answer out of a file the interface claims
    does not exist - and there would be nothing to press to remove it.

    The thread is checked, not trusted, exactly as an upload names its thread
    and is checked: a conversation id is a number the client sent, and only
    the owner column makes it mean something. A stranger's thread id raises
    `UnknownConversation` rather than reading as an empty attachment.

    The status is what the client polls while a new upload is indexed. An
    indexing run past `indexing_timeout` is reported as failed, because the
    background task that would have finished it may have died with its process.
    """

    def __init__(
        self, uow_factory: UnitOfWorkFactory, clock: Clock, indexing_timeout: timedelta
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._indexing_timeout = indexing_timeout

    async def __call__(self, conversation_id: int, owner_id: int) -> UploadedDocument | None:
        async with self._uow_factory() as uow:
            if await uow.conversations.get(conversation_id, owner_id) is None:
                raise UnknownConversation(conversation_id)
            documents = await uow.documents.list_by_conversation(owner_id, conversation_id)
        if not documents:
            return None
        document = documents[0]
        now = await self._clock.now()
        return replace(document, status=document.status_at(now, self._indexing_timeout))
