"""Open a new conversation thread."""

from app.application.chat.dto import StartConversationInput
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationLimitReached


class StartConversation:
    """Create a thread and return it with its id.

    Capped per account. Over the cap it refuses, never deletes an old thread.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory, max_per_owner: int) -> None:
        self._uow_factory = uow_factory
        self._max_per_owner = max_per_owner

    async def __call__(self, data: StartConversationInput) -> Conversation:
        conversation = Conversation(
            title=data.title, model_type=data.model_type, owner_id=data.owner_id
        )
        async with self._uow_factory() as uow:
            # Counted in the insert's transaction, against racing requests.

            existing = await uow.conversations.count_by_owner(data.owner_id)
            if existing >= self._max_per_owner:
                raise ConversationLimitReached(self._max_per_owner)

            stored = await uow.conversations.add(conversation)
            await uow.commit()
            return stored
