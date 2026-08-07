"""Conversation use cases against fakes — no database, no network.

Reply generation is no longer tested here: it is now a LangGraph agent, and its
tests live beside it in `tests/application/test_agent_graph.py`.
"""

import pytest

from app.application.chat.dto import RecordExchangeInput, StartConversationInput
from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationNotFound
from tests.fakes import UnitOfWorkSpy


class TestConversationUseCases:
    async def test_start_persists_and_commits(self) -> None:
        factory = UnitOfWorkSpy()
        conversation = await StartConversation(factory)(
            StartConversationInput(title="T", model_type="m")
        )

        assert conversation.id == 1
        assert factory.issued[0].committed

    async def test_get_returns_the_stored_conversation(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", id=3)

        assert (await GetConversation(factory)(3)).title == "T"

    async def test_get_missing_raises_a_domain_error(self) -> None:
        with pytest.raises(ConversationNotFound):
            await GetConversation(UnitOfWorkSpy())(404)

    async def test_record_exchange_attaches_to_its_conversation(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", id=3)

        message = await RecordExchange(factory)(
            RecordExchangeInput(conversation_id=3, prompt_content="p", response_content="r")
        )

        assert message.conversation_id == 3
        assert factory.issued[0].committed

    async def test_record_exchange_on_missing_conversation_does_not_commit(self) -> None:
        factory = UnitOfWorkSpy()

        with pytest.raises(ConversationNotFound):
            await RecordExchange(factory)(
                RecordExchangeInput(conversation_id=9, prompt_content="p", response_content="r")
            )

        assert not factory.issued[0].committed
        assert factory.issued[0].rolled_back

    async def test_delete_removes_and_commits(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", id=3)

        await DeleteConversation(factory)(3)

        assert factory.repository.rows == {}
        assert factory.issued[0].committed

    async def test_delete_missing_is_reported_not_swallowed(self) -> None:
        with pytest.raises(ConversationNotFound):
            await DeleteConversation(UnitOfWorkSpy())(404)
