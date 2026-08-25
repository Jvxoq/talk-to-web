"""Open a new conversation thread."""

from app.application.chat.dto import StartConversationInput
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationLimitReached


class StartConversation:
    """
    Create the thread that later exchanges hang off.

    Returns the persisted entity rather than the one built here, because only
    the repository can supply the identity the caller needs to keep talking.

    An account may hold only so many threads at once. The cap is not a storage
    concern - a conversation row is tiny - it is what keeps the uploads and the
    history a person is carrying small enough to reason about, now that each
    thread owns its own documents. It refuses rather than evicting: making room
    by deleting someone's oldest thread destroys work they never offered up.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory, max_per_owner: int) -> None:
        self._uow_factory = uow_factory
        self._max_per_owner = max_per_owner

    async def __call__(self, data: StartConversationInput) -> Conversation:
        conversation = Conversation(
            title=data.title, model_type=data.model_type, owner_id=data.owner_id
        )
        async with self._uow_factory() as uow:
            # Counted inside the same transaction as the insert, so two
            # requests racing to open a third thread cannot both read two.
            existing = await uow.conversations.count_by_owner(data.owner_id)
            if existing >= self._max_per_owner:
                raise ConversationLimitReached(self._max_per_owner)

            stored = await uow.conversations.add(conversation)
            await uow.commit()
            return stored
