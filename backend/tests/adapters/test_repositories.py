"""The SQLAlchemy repositories and unit of work against a real database.

Fakes cannot make these claims. `tests/fakes.py` filters in Python, so it would
pass whether or not the owner predicate is in the SQL; it holds a dict, so it
would pass whether or not `flush()` populates an id; and it has no transaction,
so it cannot tell a rollback from a commit. Everything below is exactly the part
a fake has to assume.

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
    """A real session whose `commit` fails, to exercise the recovery path.

    A failed commit is not hypothetical - a serialization failure or a lost
    connection produces one - and the interesting part is what the unit of work
    does next. Injected through the session factory, which is the seam the unit
    of work already takes, so nothing is patched.
    """

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
        # The whole reason the repository flushes rather than commits: the id is
        # populated, but the transaction is still the caller's to close. Proved
        # by rolling back rather than by reading from a second session - the
        # in-memory database shares one connection, so a second session would
        # see uncommitted rows and the assertion would prove nothing.
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
        # Expire everything, so `get` has to issue real SQL instead of handing
        # back the objects still sitting in the identity map.
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)

        assert loaded is not None
        assert [message.prompt_content for message in loaded.messages] == ["Ping?"]

    async def test_get_refuses_another_owners_conversation(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # The isolation claim, and the one that matters most in this file: the
        # owner is a predicate in the query, so there is no result to forget to
        # check.
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
        # `message_to_model` calls `token_total()` rather than copying the field,
        # so a provider that reported only the halves still gets a usable total
        # in the column.
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
        # `updated_at` defaults from the clock, and two inserts in the same
        # microsecond would make the ordering assertion a coin toss.
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
        # The relationship is deliberately not eager-loaded here. If the summary
        # mapper ever reaches for `row.messages`, the expunge below turns that
        # into the `MissingGreenlet` this test exists to catch - not a silently
        # slower sidebar.
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
        # The FK cascade, not application code, is what stops orphaned messages
        # accumulating. Proving it needs a database that enforces the constraint.
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
        # A DELETE has already happened by the time a use case could check
        # ownership, which is why the predicate lives in the statement.
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        assert stored.id is not None

        await repository.delete(stored.id, stranger.id)
        session.expunge_all()

        assert await repository.get(stored.id, owner.id) is not None


class TestTheMappersPreserveAnIdTheEntityAlreadyHas:
    """Every `*_to_model` copies an id across when the entity carries one.

    Without it, saving an entity that has already been persisted would insert a
    second row rather than address the existing one - and the caller, holding an
    entity with an id, has no reason to expect that.
    """

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
        # The storage reference round-trips intact: it is the only handle the
        # extractor and the delete path have on the actual file.
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
        # Both rows default `created_at` from the clock, and two inserts in the
        # same microsecond would make the ordering a coin toss.
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

    async def test_set_chunks_indexed_records_how_much_was_indexed(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.set_chunks_indexed(stored.id, owner.id, 17)
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)
        assert loaded is not None and loaded.chunks_indexed == 17

    async def test_set_chunks_indexed_cannot_touch_another_owners_document(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        stranger = await make_user(session, "stranger@example.com")
        repository = SqlAlchemyDocumentRepository(session)
        stored = await repository.add(a_document(owner.id))
        assert stored.id is not None

        await repository.set_chunks_indexed(stored.id, stranger.id, 99)
        session.expunge_all()

        loaded = await repository.get(stored.id, owner.id)
        assert loaded is not None and loaded.chunks_indexed == 0

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
        # A document a stranger can name by id must not be a document a stranger
        # can delete - and the DELETE has already run by the time any use case
        # could check.
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
        # The FK cascade. Without it, deleting an account would leave rows
        # pointing at a user that no longer exists.
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
        # The column holds the sanitized address and the lookup sanitizes too,
        # so the match is exact - which is what lets the query use the unique
        # index instead of a function over the column.
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
        # Reuse detection depends on this being `None` rather than an error:
        # a presented token that was never issued is a fact, not a failure.
        found = await SqlAlchemyRefreshTokenRepository(session).get_by_fingerprint("z" * 64)
        assert found is None

    async def test_a_revoked_session_still_exists_and_is_marked(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        # An UPDATE, not a DELETE. A deleted row is indistinguishable from one
        # that never existed, and that difference is the whole of reuse detection.
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
        # `WHERE revoked_at IS NULL` in the statement. Without it the second
        # revoke would move the timestamp and misreport when the session ended.
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
        # Reuse detection is a blunt instrument by design; it must still be
        # pointed at exactly one account.
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
        # A repository cannot exist without the session it writes through, so
        # they are bound on entry rather than in `__init__`. A member added to
        # the port and forgotten here would only surface when a use case
        # reached for it.
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            assert isinstance(uow.conversations, SqlAlchemyConversationRepository)
            assert isinstance(uow.users, SqlAlchemyUserRepository)
            assert isinstance(uow.refresh_tokens, SqlAlchemyRefreshTokenRepository)
            assert isinstance(uow.documents, SqlAlchemyDocumentRepository)

    async def test_one_block_can_write_across_more_than_one_repository(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # The reason they share a unit of work at all: registration writes a
        # person and their first session together, and a commit that landed one
        # without the other leaves an account nobody can sign in to.
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
        # The rule the whole unit of work exists for. Nothing here raises: just
        # forgetting to commit is enough, which is the shape the mistake
        # actually takes.
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.users.add(User(email=Email.sanitize("r@example.com"), password_hash="h"))

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 0

    async def test_an_exception_halfway_through_leaves_nothing_behind(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Registration writes a user and their first session together. A partial
        # commit here would leave an account nobody could sign in to.
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
        # An `AsyncSession` is not task-safe, so a shared unit of work is a
        # concurrency bug that presents as corrupted results much later. Failing
        # loudly at the point of misuse is the only useful moment.
        uow = SqlAlchemyUnitOfWork(session_factory)
        async with uow:
            with pytest.raises(RuntimeError, match="already open"):
                await uow.__aenter__()

    async def test_a_failed_commit_rolls_back_before_the_caller_sees_it(
        self, engine: AsyncEngine
    ) -> None:
        # Re-raised, so the caller still fails - but the session is left usable
        # and the data consistent first. A failed commit that is simply
        # propagated strands the transaction, and the next statement on that
        # session fails for a reason that has nothing to do with what went
        # wrong.
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
        # `__aexit__` can run without `__aenter__` having succeeded - a fixture
        # that raised while building one, for instance - and rolling back a
        # session that does not exist would replace the real error with an
        # `AttributeError`.
        await SqlAlchemyUnitOfWork(session_factory).__aexit__()

    async def test_committing_before_it_is_open_is_an_error(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        with pytest.raises(RuntimeError, match="not open"):
            await SqlAlchemyUnitOfWork(session_factory).commit()

    async def test_it_can_be_used_again_after_it_closes(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # `__aexit__` clears the session, so the second block gets fresh
        # repositories rather than ones pointed at a closed session.
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
        # `_committed` is reset on entry as well as on exit. If it were not, a
        # reused unit of work would skip the rollback and silently persist a
        # block that never asked to be committed.
        uow = SqlAlchemyUnitOfWork(session_factory)
        async with uow:
            await uow.users.add(User(email=Email.sanitize("first@example.com"), password_hash="h"))
            await uow.commit()
        async with uow:
            await uow.users.add(User(email=Email.sanitize("second@example.com"), password_hash="h"))

        async with session_factory() as other:
            assert await other.scalar(select(func.count()).select_from(UserModel)) == 1
