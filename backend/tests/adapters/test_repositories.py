"""The SQLAlchemy repositories and unit of work against a real database.

Covers what fakes cannot: owner filters in SQL, flushed ids, rollback vs commit.
See `conftest.py` for what SQLite can and cannot stand in for.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.adapters.persistence.models import (
    ConversationModel,
    DocumentModel,
    MessageModel,
    UserModel,
)
from app.adapters.persistence.repositories import (
    SqlAlchemyConversationRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyRefreshTokenRepository,
    SqlAlchemyUserRepository,
)
from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.domain.chat.entities import Conversation, Message
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.value_objects import Email
from app.domain.ingestion.entities import UploadedDocument
from tests.adapters.conftest import make_user

NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def a_conversation(owner_id: int, title: str = "A thread") -> Conversation:
    return Conversation(title=title, model_type="llama-3.3-70b", owner_id=owner_id)


class FailingCommitSession(AsyncSession):
    """A real session whose `commit` fails, injected via the session factory."""

    async def commit(self) -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))


def a_document(owner_id: int, name: str = "handbook.pdf") -> UploadedDocument:
    return UploadedDocument(name=name, reference=f"uploads/{owner_id}/{name}", owner_id=owner_id)


def a_token(user_id: int, fingerprint: str) -> RefreshToken:
    return RefreshToken(
        user_id=user_id,
        fingerprint=fingerprint,
        expires_at=NOW + timedelta(days=30),
    )


class TestSqlAlchemyConversationRepository:
    async def test_add_returns_the_conversation_with_its_database_assigned_id(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyConversationRepository(session)

        stored = await repository.add(a_conversation(owner.id))

        assert stored.id is not None
        assert stored.is_persisted()
        assert stored.owner_id == owner.id

    async def test_add_flushes_without_committing(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # Flush sets the id but leaves the transaction open. Proved by rollback,
        # since a second in-memory session would see uncommitted rows.
        stored = await SqlAlchemyConversationRepository(session).add(a_conversation(owner.id))
        assert stored.id is not None

        await session.rollback()

        assert await session.scalar(select(func.count()).select_from(ConversationModel)) == 0

    async def test_get_loads_the_messages_with_the_conversation(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None
        await repository.add_message(stored.id, Message("Ping?", "Pong."))
        # Force `get` to run real SQL.
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)

        assert loaded is not None
        assert [message.prompt_content for message in loaded.messages] == ["Ping?"]

    async def test_get_refuses_another_owners_conversation(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The key isolation check: the owner filter is in the query.
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None
        session.expunge_all()

        assert await repository.get(stored.id, stranger.id) is None

    async def test_get_returns_none_for_a_conversation_that_does_not_exist(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        assert await SqlAlchemyConversationRepository(session).get(404, owner.id) is None

    async def test_add_message_attaches_the_exchange_to_the_conversation(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None

        recorded = await repository.add_message(stored.id, Message("Q", "A"))

        assert recorded.id is not None
        assert recorded.conversation_id == stored.id

    async def test_add_message_derives_the_token_total_from_the_halves(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The total is derived when only the halves were reported.
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None

        recorded = await repository.add_message(
            stored.id, Message("Q", "A", prompt_tokens=11, response_tokens=31)
        )

        assert recorded.total_tokens == 42

    async def test_list_by_owner_returns_summaries_newest_activity_first(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyConversationRepository(session)
        older = await repository.add(a_conversation(owner.id, "Older"))
        newer = await repository.add(a_conversation(owner.id, "Newer"))
        # Set timestamps by hand, so ordering is not a coin toss.
        await session.execute(
            update(ConversationModel)
            .where(ConversationModel.id == older.id)
            .values(updated_at=NOW - timedelta(days=1))
        )
        await session.execute(
            update(ConversationModel).where(ConversationModel.id == newer.id).values(updated_at=NOW)
        )
        session.expunge_all()

        listed = await repository.list_by_owner(owner.id)

        assert [conversation.title for conversation in listed] == ["Newer", "Older"]

    async def test_list_by_owner_omits_the_messages(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # Not eager-loaded, so touching `row.messages` raises `MissingGreenlet`.
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None
        await repository.add_message(stored.id, Message("Q", "A"))
        session.expunge_all()

        listed = await repository.list_by_owner(owner.id)

        assert [conversation.messages for conversation in listed] == [[]]

    async def test_list_by_owner_shows_nothing_belonging_to_anyone_else(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyConversationRepository(session)
        await repository.add(a_conversation(owner.id, "Mine"))
        await repository.add(a_conversation(stranger.id, "Theirs"))
        session.expunge_all()

        assert [
            conversation.title for conversation in await repository.list_by_owner(owner.id)
        ] == ["Mine"]

    async def test_delete_takes_the_messages_with_it(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The FK cascade removes messages. Needs a real database.
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None
        await repository.add_message(stored.id, Message("Q", "A"))

        await repository.delete(stored.id, owner.id)
        await session.flush()

        assert await session.scalar(select(func.count()).select_from(MessageModel)) == 0

    async def test_delete_leaves_another_owners_conversation_alone(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The owner filter must be in the DELETE itself.
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None

        await repository.delete(stored.id, stranger.id)
        session.expunge_all()

        assert await repository.get(stored.id, owner.id) is not None


class TestTheMappersPreserveAnIdTheEntityAlreadyHas:
    """Every `*_to_model` copies the id, or a saved entity would insert a duplicate."""

    async def test_a_conversation_and_its_messages(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        conversation = a_conversation(owner.id)
        conversation.id = 500
        conversation.messages.append(Message("Q", "A", id=900))

        stored = await SqlAlchemyConversationRepository(session).add(conversation)

        assert stored.id == 500
        assert [message.id for message in stored.messages] == [900]

    async def test_a_user(self, session: AsyncSession) -> None:
        user = User(email=Email.sanitize("fixed@example.com"), password_hash="h", id=501)

        assert (await SqlAlchemyUserRepository(session).add(user)).id == 501

    async def test_a_refresh_token(self, session: AsyncSession, owner: UserModel) -> None:
        token = a_token(owner.id, "7" * 64)
        token.id = 502

        assert (await SqlAlchemyRefreshTokenRepository(session).add(token)).id == 502

    async def test_an_uploaded_document(self, session: AsyncSession, owner: UserModel) -> None:
        document = a_document(owner.id)
        document.id = 503

        assert (await SqlAlchemyDocumentRepository(session).add(document)).id == 503


class TestSqlAlchemyDocumentRepository:
    async def test_add_returns_the_document_with_its_id(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)

        stored = await repository.add(a_document(owner.id))

        assert stored.id is not None
        assert stored.chunks_indexed == 0

    async def test_get_returns_the_stored_document(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)

        assert loaded is not None
        assert loaded.name == "handbook.pdf"
        # The reference is the only handle on the stored file.
        assert loaded.reference == stored.reference

    async def test_get_refuses_another_owners_document(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None
        session.expunge_all()

        assert await repository.get(stored.id, stranger.id) is None

    async def test_get_returns_none_for_a_document_that_does_not_exist(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        assert await SqlAlchemyDocumentRepository(session).get(404, owner.id) is None

    async def test_list_by_owner_returns_the_newest_upload_first(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        older = await repository.add(a_document(owner.id, "older.pdf"))
        newer = await repository.add(a_document(owner.id, "newer.pdf"))
        # Set timestamps by hand, so ordering is not a coin toss.
        await session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == older.id)
            .values(created_at=NOW - timedelta(days=1))
        )
        await session.execute(
            update(DocumentModel).where(DocumentModel.id == newer.id).values(created_at=NOW)
        )
        session.expunge_all()

        assert [document.name for document in await repository.list_by_owner(owner.id)] == [
            "newer.pdf",
            "older.pdf",
        ]

    async def test_list_by_owner_shows_nothing_belonging_to_anyone_else(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyDocumentRepository(session)
        await repository.add(a_document(owner.id, "mine.pdf"))
        await repository.add(a_document(stranger.id, "theirs.pdf"))
        session.expunge_all()

        assert [document.name for document in await repository.list_by_owner(owner.id)] == [
            "mine.pdf"
        ]

    async def test_set_indexed_records_how_much_was_indexed(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.set_indexed(stored.id, owner.id, 17, "what it is about")
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)
        assert loaded is not None
        assert loaded.chunks_indexed == 17
        assert loaded.summary == "what it is about"

    async def test_set_indexed_cannot_touch_another_owners_document(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.set_indexed(stored.id, stranger.id, 99, "not theirs to write")
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)
        assert loaded is not None
        assert loaded.chunks_indexed == 0
        assert loaded.summary == "", "a stranger must not be able to write a summary either"

    async def test_delete_removes_the_document(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.delete(stored.id, owner.id)
        session.expunge_all()

        assert await repository.get(stored.id, owner.id) is None

    async def test_delete_leaves_another_owners_document_alone(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # A stranger cannot delete a document by id.
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.delete(stored.id, stranger.id)
        session.expunge_all()

        assert await repository.get(stored.id, owner.id) is not None

    async def test_a_documents_rows_go_when_its_owner_does(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The FK cascade from users.
        repository = SqlAlchemyDocumentRepository(session)
        await repository.add(a_document(owner.id))

        await session.delete(owner)
        await session.flush()

        assert await session.scalar(select(func.count()).select_from(DocumentModel)) == 0


class TestSqlAlchemyUserRepository:
    async def test_add_returns_the_user_with_their_id(self, session: AsyncSession) -> None:
        repository = SqlAlchemyUserRepository(session)

        stored = await repository.add(
            User(email=Email.sanitize("New@Example.com"), password_hash="h")
        )

        assert stored.id is not None
        assert stored.email.value == "new@example.com"

    async def test_get_by_email_finds_the_user(self, session: AsyncSession) -> None:
        repository = SqlAlchemyUserRepository(session)
        stored = await repository.add(
            User(email=Email.sanitize("a@example.com"), password_hash="h")
        )
        session.expunge_all()

        found = await repository.get_by_email(Email.sanitize("a@example.com"))

        assert found is not None
        assert found.id == stored.id

    async def test_get_by_email_matches_regardless_of_the_case_typed(
        self, session: AsyncSession
    ) -> None:
        # Both sides are sanitized, so the lookup is exact and uses the index.
        repository = SqlAlchemyUserRepository(session)
        await repository.add(User(email=Email.sanitize("Mixed@Example.com"), password_hash="h"))
        session.expunge_all()

        assert await repository.get_by_email(Email.sanitize("MIXED@EXAMPLE.COM")) is not None

    async def test_get_by_email_returns_none_for_an_unknown_address(
        self, session: AsyncSession
    ) -> None:
        found = await SqlAlchemyUserRepository(session).get_by_email(
            Email.sanitize("nobody@example.com")
        )
        assert found is None

    async def test_get_returns_none_for_an_unknown_id(self, session: AsyncSession) -> None:
        assert await SqlAlchemyUserRepository(session).get(404) is None

    async def test_a_stored_user_round_trips_through_the_mapper(
        self, session: AsyncSession
    ) -> None:
        repository = SqlAlchemyUserRepository(session)
        stored = await repository.add(
            User(
                email=Email.sanitize("round@example.com"),
                password_hash="$argon2id$x",
                is_active=False,
            )
        )
        assert stored.id is not None
        session.expunge_all()

        loaded = await repository.get(stored.id)

        assert loaded is not None
        assert loaded.password_hash == "$argon2id$x"
        assert loaded.is_active is False


class TestSqlAlchemyRefreshTokenRepository:
    async def test_add_stores_the_fingerprint_and_returns_the_session_id(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyRefreshTokenRepository(session)

        stored = await repository.add(a_token(owner.id, "f" * 64))

        assert stored.id is not None
        assert stored.fingerprint == "f" * 64
        assert stored.revoked_at is None

    async def test_get_by_fingerprint_finds_the_session(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyRefreshTokenRepository(session)
        stored = await repository.add(a_token(owner.id, "a" * 64))
        session.expunge_all()

        found = await repository.get_by_fingerprint("a" * 64)

        assert found is not None
        assert found.id == stored.id

    async def test_get_by_fingerprint_returns_none_for_a_token_nobody_issued(
        self, session: AsyncSession
    ) -> None:
        # An unknown token is `None`, not an error.
        found = await SqlAlchemyRefreshTokenRepository(session).get_by_fingerprint("z" * 64)
        assert found is None

    async def test_a_revoked_session_still_exists_and_is_marked(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # An UPDATE, not a DELETE, since reuse detection needs the row.
        repository = SqlAlchemyRefreshTokenRepository(session)
        stored = await repository.add(a_token(owner.id, "b" * 64))
        assert stored.id is not None

        await repository.revoke(stored.id, NOW)
        session.expunge_all()

        found = await repository.get_by_fingerprint("b" * 64)
        assert found is not None
        assert found.is_revoked()

    async def test_revoking_twice_keeps_the_first_timestamp(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # A second revoke keeps the first timestamp.
        repository = SqlAlchemyRefreshTokenRepository(session)
        stored = await repository.add(a_token(owner.id, "c" * 64))
        assert stored.id is not None

        await repository.revoke(stored.id, NOW)
        await repository.revoke(stored.id, NOW + timedelta(hours=5))
        session.expunge_all()

        found = await repository.get_by_fingerprint("c" * 64)
        assert found is not None
        assert found.revoked_at is not None
        assert found.revoked_at.replace(tzinfo=UTC) == NOW

    async def test_revoke_all_for_user_ends_every_live_session(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyRefreshTokenRepository(session)
        for fingerprint in ("d" * 64, "e" * 64):
            await repository.add(a_token(owner.id, fingerprint))

        await repository.revoke_all_for_user(owner.id, NOW)
        session.expunge_all()

        for fingerprint in ("d" * 64, "e" * 64):
            found = await repository.get_by_fingerprint(fingerprint)
            assert found is not None and found.is_revoked()

    async def test_revoke_all_for_user_leaves_other_users_signed_in(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # Revokes every session of one user only.
        other = await make_user(session, "other@example.com")
        repository = SqlAlchemyRefreshTokenRepository(session)
        await repository.add(a_token(owner.id, "1" * 64))
        await repository.add(a_token(other.id, "2" * 64))

        await repository.revoke_all_for_user(owner.id, NOW)
        session.expunge_all()

        survivor = await repository.get_by_fingerprint("2" * 64)
        assert survivor is not None and not survivor.is_revoked()

    async def test_revoke_all_for_user_does_not_move_an_earlier_revocation(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyRefreshTokenRepository(session)
        stored = await repository.add(a_token(owner.id, "3" * 64))
        assert stored.id is not None
        await repository.revoke(stored.id, NOW)

        await repository.revoke_all_for_user(owner.id, NOW + timedelta(hours=9))
        session.expunge_all()

        found = await repository.get_by_fingerprint("3" * 64)
        assert found is not None and found.revoked_at is not None
        assert found.revoked_at.replace(tzinfo=UTC) == NOW


class TestSqlAlchemyUnitOfWork:
    async def test_it_opens_every_repository_it_declares(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Every port member must be bound on entry.
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            assert isinstance(uow.conversations, SqlAlchemyConversationRepository)
            assert isinstance(uow.users, SqlAlchemyUserRepository)
            assert isinstance(uow.refresh_tokens, SqlAlchemyRefreshTokenRepository)
            assert isinstance(uow.documents, SqlAlchemyDocumentRepository)

    async def test_one_block_can_write_across_more_than_one_repository(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Registration writes the user and session together.
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            user = await uow.users.add(
                User(email=Email.sanitize("both@example.com"), password_hash="h")
            )
            assert user.id is not None
            await uow.refresh_tokens.add(a_token(user.id, "8" * 64))
            await uow.commit()

        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            assert await uow.refresh_tokens.get_by_fingerprint("8" * 64) is not None

    async def test_a_committed_write_is_visible_to_everyone_else(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.users.add(User(email=Email.sanitize("c@example.com"), password_hash="h"))
            await uow.commit()

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 1

    async def test_leaving_the_block_without_committing_rolls_back(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Forgetting to commit is enough to roll back.
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.users.add(User(email=Email.sanitize("r@example.com"), password_hash="h"))

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 0

    async def test_an_exception_halfway_through_leaves_nothing_behind(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # A failure halfway must not commit half of it.
        with pytest.raises(RuntimeError, match="halfway"):
            async with SqlAlchemyUnitOfWork(session_factory) as uow:
                user = await uow.users.add(
                    User(email=Email.sanitize("half@example.com"), password_hash="h")
                )
                assert user.id is not None
                await uow.refresh_tokens.add(a_token(user.id, "9" * 64))
                raise RuntimeError("halfway")

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 0

    async def test_an_explicit_rollback_discards_the_writes(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.users.add(User(email=Email.sanitize("d@example.com"), password_hash="h"))
            await uow.rollback()
            await uow.commit()

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 0

    async def test_it_refuses_to_be_entered_twice(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # `AsyncSession` is not task-safe, so re-entry must fail loudly.
        uow = SqlAlchemyUnitOfWork(session_factory)
        async with uow:
            with pytest.raises(RuntimeError, match="already open"):
                await uow.__aenter__()

    async def test_a_failed_commit_rolls_back_before_the_caller_sees_it(
        self, engine: AsyncEngine
    ) -> None:
        # Re-raised after a rollback, so the session stays usable.
        factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine, class_=FailingCommitSession, expire_on_commit=False
        )

        with pytest.raises(OperationalError):
            async with SqlAlchemyUnitOfWork(factory) as uow:
                await uow.users.add(
                    User(email=Email.sanitize("boom@example.com"), password_hash="h")
                )
                await uow.commit()

        async with async_sessionmaker(engine, expire_on_commit=False)() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 0

    async def test_exiting_one_that_was_never_opened_does_nothing(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Exit without enter must not raise `AttributeError`.
        await SqlAlchemyUnitOfWork(session_factory).__aexit__()

    async def test_committing_before_it_is_open_is_an_error(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        with pytest.raises(RuntimeError, match="not open"):
            await SqlAlchemyUnitOfWork(session_factory).commit()

    async def test_it_can_be_used_again_after_it_closes(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Reuse gets fresh repositories, not ones on a closed session.
        uow = SqlAlchemyUnitOfWork(session_factory)
        async with uow:
            await uow.users.add(User(email=Email.sanitize("one@example.com"), password_hash="h"))
            await uow.commit()
        async with uow:
            await uow.users.add(User(email=Email.sanitize("two@example.com"), password_hash="h"))
            await uow.commit()

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 2

    async def test_a_second_block_does_not_inherit_the_first_ones_commit(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # `_committed` resets on entry, so reuse still rolls back.
        uow = SqlAlchemyUnitOfWork(session_factory)
        async with uow:
            await uow.users.add(User(email=Email.sanitize("first@example.com"), password_hash="h"))
            await uow.commit()
        async with uow:
            await uow.users.add(User(email=Email.sanitize("second@example.com"), password_hash="h"))

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 1
