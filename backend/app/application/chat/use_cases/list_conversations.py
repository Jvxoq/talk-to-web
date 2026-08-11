"""List one owner's conversations for the sidebar."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation


class ListConversations:
    """Fetch every conversation this owner has, newest activity first."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, owner_id: int) -> list[Conversation]:
        async with self._uow_factory() as uow:
            return await uow.conversations.list_by_owner(owner_id)
