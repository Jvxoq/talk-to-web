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
from app.domain.chat.errors import ConversationLimitReached, ConversationNotFound
from app.domain.ingestion.entities import UploadedDocument
from tests.fakes import FakeDocumentRemover, UnitOfWorkSpy

OWNER = 1
STRANGER = 2


class TestConversationUseCases:
    async def test_start_persists_and_commits(self) -> None:
        factory = UnitOfWorkSpy()
        conversation = await StartConversation(factory, max_per_owner=2)(
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

        await DeleteConversation(factory, remove_document=FakeDocumentRemover())(3, OWNER)

        assert factory.repository.rows == {}
        # The last unit of work, not the first: the read that finds the thread
        # and its attachments is separate from the write that removes it,
        # because removing a document opens a unit of work of its own.
        assert factory.issued[-1].committed

    async def test_delete_missing_is_reported_not_swallowed(self) -> None:
        with pytest.raises(ConversationNotFound):
            await DeleteConversation(UnitOfWorkSpy(), remove_document=FakeDocumentRemover())(
                404, OWNER
            )


class TestConversationLimit:
    """An account may hold only so many threads at once."""

    async def test_it_refuses_past_the_cap(self) -> None:
        factory = UnitOfWorkSpy()
        start = StartConversation(factory, max_per_owner=2)
        request = StartConversationInput(title="T", model_type="m", owner_id=OWNER)

        await start(request)
        await start(request)
        with pytest.raises(ConversationLimitReached):
            await start(request)

        assert len(factory.repository.rows) == 2

    async def test_the_cap_is_per_account(self) -> None:
        factory = UnitOfWorkSpy()
        start = StartConversation(factory, max_per_owner=1)

        await start(StartConversationInput(title="T", model_type="m", owner_id=OWNER))
        await start(StartConversationInput(title="T", model_type="m", owner_id=STRANGER))

        assert len(factory.repository.rows) == 2

    async def test_deleting_one_makes_room(self) -> None:
        factory = UnitOfWorkSpy()
        start = StartConversation(factory, max_per_owner=1)
        first = await start(StartConversationInput(title="T", model_type="m", owner_id=OWNER))
        assert first.id is not None

        await DeleteConversation(factory, remove_document=FakeDocumentRemover())(first.id, OWNER)
        await start(StartConversationInput(title="T", model_type="m", owner_id=OWNER))

        assert len(factory.repository.rows) == 1


class TestDeletingAThreadTakesItsAttachments:
    """The database cascade removes the rows. It cannot reach the vector store
    or the disk, so the documents are removed one at a time first."""

    async def test_it_removes_every_attached_document(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)
        attached = await factory.documents.add(
            UploadedDocument(
                name="a.pdf", reference="uploads/a.pdf", owner_id=OWNER, conversation_id=3
            )
        )
        remover = FakeDocumentRemover()

        await DeleteConversation(factory, remove_document=remover)(3, OWNER)

        assert remover.removed == [(attached.id, OWNER)]

    async def test_a_document_that_will_not_delete_still_lets_the_thread_go(self) -> None:
        # A vector store that will not answer must not leave someone unable to
        # delete their own conversation.
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)
        await factory.documents.add(
            UploadedDocument(
                name="a.pdf", reference="uploads/a.pdf", owner_id=OWNER, conversation_id=3
            )
        )

        await DeleteConversation(
            factory, remove_document=FakeDocumentRemover(fail_with=RuntimeError("qdrant is down"))
        )(3, OWNER)

        assert factory.repository.rows == {}

    async def test_it_leaves_another_threads_documents_alone(self) -> None:
        factory = UnitOfWorkSpy()
        factory.repository.rows[3] = Conversation(title="T", model_type="m", owner_id=OWNER, id=3)
        await factory.documents.add(
            UploadedDocument(
                name="other.pdf", reference="uploads/other.pdf", owner_id=OWNER, conversation_id=4
            )
        )
        remover = FakeDocumentRemover()

        await DeleteConversation(factory, remove_document=remover)(3, OWNER)

        assert remover.removed == []


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
            await DeleteConversation(factory, remove_document=FakeDocumentRemover())(3, STRANGER)

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
