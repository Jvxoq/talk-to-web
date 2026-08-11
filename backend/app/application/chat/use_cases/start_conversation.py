"""Open a new conversation thread."""

from app.application.chat.dto import StartConversationInput
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation


class StartConversation:
    """
    Create the thread that later exchanges hang off.

    Returns the persisted entity rather than the one built here, because only
    the repository can supply the identity the caller needs to keep talking.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, data: StartConversationInput) -> Conversation:
        conversation = Conversation(
            title=data.title, model_type=data.model_type, owner_id=data.owner_id
        )
        async with self._uow_factory() as uow:
            stored = await uow.conversations.add(conversation)
            await uow.commit()
            return stored
