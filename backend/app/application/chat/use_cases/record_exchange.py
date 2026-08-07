"""Persist one completed prompt/response exchange."""

from app.application.chat.dto import RecordExchangeInput
from app.application.common.uow import UnitOfWorkFactory
from app.domain.chat.entities import Message
from app.domain.chat.errors import ConversationNotFound


class RecordExchange:
    """
    Store what the model answered against the thread that asked.

    The conversation is loaded first so an exchange can never be written
    against an id that no longer exists.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, data: RecordExchangeInput) -> Message:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get(data.conversation_id)
            if conversation is None:
                raise ConversationNotFound(data.conversation_id)

            message = conversation.record(
                Message(
                    prompt_content=data.prompt_content,
                    response_content=data.response_content,
                    prompt_tokens=data.prompt_tokens,
                    response_tokens=data.response_tokens,
                    total_tokens=data.total_tokens,
                    is_success=data.is_success,
                    status_code=data.status_code,
                )
            )
            stored = await uow.conversations.add_message(data.conversation_id, message)
            await uow.commit()
            return stored
