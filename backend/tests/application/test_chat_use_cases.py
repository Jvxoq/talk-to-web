"""Conversation use cases against fakes — no database, no network.

Reply generation is no longer tested here: it is now a LangGraph agent, and its
tests live beside it in `tests/application/test_agent_graph.py`.
"""

import pytest

from app.application.chat.dto import RecordExchangeInput, StartConversationInput
from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.list_conversations import ListConversations
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.domain.chat.entities import Conversation
from app.domain.chat.errors import ConversationNotFound
from tests.fakes import UnitOfWorkSpy

OWNER = 1
STRANGER = 2


class TestConversationUseCases:
    async def test_start_persists_and_commits(self) -> None:
        factory = UnitOfWorkSpy()
        conversation = await StartConversation(factory)(
            StartConversationInput(title="T", model_type="m", owner_id=OWNER)
        )

        assert conversation.id == 1
        assert conversation.owner_id == OWNER
        assert factory.issued[0].committed

    async def test_get_returns_the_stored_conversation(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)

        assert (await GetConversation(factory)(3, OWNER)).title == "T"

    async def test_get_missing_raises_a_domain_error(self) -> None:
        with pytest.raises(ConversationNotFound):
            await GetConversation(UnitOfWorkSpy())(404, OWNER)

    async def test_record_exchange_attaches_to_its_conversation(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)

        message = await RecordExchange(factory)(
            RecordExchangeInput(
                conversation_id=3, owner_id=OWNER, prompt_content="p", response_content="r"
            )
        )

        assert message.conversation_id == 3
        assert factory.issued[0].committed

    async def test_record_exchange_on_missing_conversation_does_not_commit(self) -> None:
        factory = UnitOfWorkSpy()

        with pytest.raises(ConversationNotFound):
            await RecordExchange(factory)(
                RecordExchangeInput(
                    conversation_id=9, owner_id=OWNER, prompt_content="p", response_content="r"
                )
            )

        assert not factory.issued[0].committed
        assert factory.issued[0].rolled_back

    async def test_list_returns_only_this_owners_conversations(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[1] = Conversation(title="A", model_type="m", owner_id=OWNER, id=1)
        factory.repository.rows[2] = Conversation(
            title="B", model_type="m", owner_id=STRANGER, id=2
        )

        result = await ListConversations(factory)(OWNER)

        assert [c.id for c in result] == [1]

    async def test_delete_removes_and_commits(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)

        await DeleteConversation(factory)(3, OWNER)

        assert factory.repository.rows == {}
        assert factory.issued[0].committed

    async def test_delete_missing_is_reported_not_swallowed(self) -> None:
        with pytest.raises(ConversationNotFound):
            await DeleteConversation(UnitOfWorkSpy())(404, OWNER)


class TestOwnershipIsolation:
    """Someone else's conversation must be indistinguishable from a missing one.

    These are the tests the whole ownership change exists for. Each one asserts
    both halves: the stranger is refused, *and* the refusal looks exactly like a
    404 rather than a 403 - a distinguishable "forbidden" would confirm the id
    exists and belongs to somebody.
    """

    def _existing(self) -> UnitOfWorkSpy:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(
            title="Private", model_type="m", owner_id=OWNER, id=3
        )
        return factory

    async def test_a_stranger_cannot_read_it(self) -> None:
        with pytest.raises(ConversationNotFound):
            await GetConversation(self._existing())(3, STRANGER)

    async def test_a_stranger_cannot_delete_it(self) -> None:
        factory = self._existing()

        with pytest.raises(ConversationNotFound):
            await DeleteConversation(factory)(3, STRANGER)

        # The refusal is not enough on its own: the row has to still be there.
        assert 3 in factory.repository.rows
        assert not factory.issued[0].committed

    async def test_a_stranger_cannot_record_against_it(self) -> None:
        factory = self._existing()

        with pytest.raises(ConversationNotFound):
            await RecordExchange(factory)(
                RecordExchangeInput(
                    conversation_id=3,
                    owner_id=STRANGER,
                    prompt_content="p",
                    response_content="r",
                )
            )

        assert not factory.issued[0].committed
