"""The persistence adapters against the database they will actually run on.

`tests/adapters/test_repositories.py` covers what the repositories author, on
SQLite, and its own docstring lists what that cannot reach. This file is that
list: timestamptz, unique-violation semantics, real foreign-key cascades, and
two sessions on two connections rather than one shared in-memory database.

Nothing here re-tests the owner predicate or the flush-not-commit rule. Those
are dialect-independent, they are already covered, and a second copy would only
double the cost of changing them.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.persistence.models import ConversationModel, MessageModel, UserModel
from app.adapters.persistence.repositories import (
    SqlAlchemyConversationRepository,
    SqlAlchemyRefreshTokenRepository,
    SqlAlchemyUserRepository,
)
from app.adapters.persistence.uow import SqlAlchemyUnitOfWork
from app.domain.chat.entities import Conversation, Message
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.value_objects import Email

NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def a_conversation(owner_id: int, title: str = "A thread") -> Conversation:
    return Conversation(title=title, model_type="llama-3.3-70b", owner_id=owner_id)


def persisted_id(conversation: Conversation) -> int:
    """The id the database assigned, narrowed for the type checker.

    `Conversation.id` is `int | None` because an unsaved one has no id yet.
    Asserting here keeps every call site below one line instead of three.
    """
    assert conversation.id is not None
    return conversation.id


class TestTimestamps:
    """`DateTime(timezone=True)` is a real `timestamptz` here, and only here."""

    async def test_a_stored_timestamp_comes_back_aware(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        """SQLite hands back a naive datetime, so the adapter suite cannot ask this.

        A naive timestamp that reaches the API is served as if it were local
        time, and the frontend renders every message hours out.
        """
        repository = SqlAlchemyConversationRepository(session)

        stored = await repository.add(a_conversation(owner.id))
        await session.commit()
        session.expunge_all()
        reloaded = await repository.get(persisted_id(stored), owner.id)

        assert reloaded is not None
        assert reloaded.created_at is not None
        assert reloaded.created_at.tzinfo is not None
        assert reloaded.created_at.utcoffset() == timedelta(0)

    async def test_an_offset_timestamp_is_normalised_to_utc(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        """Postgres stores an instant, not an offset, and returns it in UTC.

        Worth pinning: code that compares `expires_at` against `datetime.now(UTC)`
        is only correct because of it.
        """
        repository = SqlAlchemyRefreshTokenRepository(session)
        noon_in_tokyo = datetime(2024, 6, 1, 21, 0, tzinfo=timezone(timedelta(hours=9)))

        await repository.add(
            RefreshToken(user_id=owner.id, fingerprint="a" * 64, expires_at=noon_in_tokyo)
        )
        await session.commit()
        session.expunge_all()
        found = await repository.get_by_fingerprint("a" * 64)

        assert found is not None
        assert found.expires_at == noon_in_tokyo
        assert found.expires_at.utcoffset() == timedelta(0)


class TestUniqueConstraints:
    """Enforced by the database, which is the only place worth enforcing them."""

    async def test_two_users_cannot_share_an_email(self, session: AsyncSession) -> None:
        """The race the application-level check cannot close.

        `RegisterUser` looks for an existing address first, but two concurrent
        registrations both find nothing. What actually stops the duplicate is
        this constraint, and it exists in the migration or it does not exist.
        """
        repository = SqlAlchemyUserRepository(session)
        await repository.add(User(email=Email.sanitize("dup@example.com"), password_hash="h"))
        await session.commit()

        # Raised by `add`, not by `commit`: the repository flushes, so the INSERT
        # - and the constraint - happen before the transaction is closed.
        with pytest.raises(IntegrityError):
            await repository.add(User(email=Email.sanitize("dup@example.com"), password_hash="h"))

    async def test_two_sessions_cannot_share_a_fingerprint(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        """Two rows with one fingerprint would make rotation ambiguous.

        A refresh would then revoke one session and leave the other live - a
        stolen token that survives the rotation meant to spend it.
        """
        repository = SqlAlchemyRefreshTokenRepository(session)
        await repository.add(RefreshToken(user_id=owner.id, fingerprint="b" * 64, expires_at=NOW))
        await session.commit()

        with pytest.raises(IntegrityError):
            await repository.add(
                RefreshToken(user_id=owner.id, fingerprint="b" * 64, expires_at=NOW)
            )


class TestForeignKeys:
    """Postgres enforces these without a pragma, so the cascade here is the real one."""

    async def test_deleting_a_conversation_takes_its_messages(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        repository = SqlAlchemyConversationRepository(session)
        stored = await repository.add(a_conversation(owner.id))
        await repository.add_message(
            persisted_id(stored), Message(prompt_content="q", response_content="a")
        )
        await session.commit()

        await repository.delete(persisted_id(stored), owner.id)
        await session.commit()

        remaining = await session.scalar(select(func.count()).select_from(MessageModel))
        assert remaining == 0

    async def test_deleting_a_user_takes_everything_they_own(
        self, session: AsyncSession, owner: UserModel
    ) -> None:
        """What a real account deletion has to rely on.

        Both cascades are declared `ondelete="CASCADE"` on the FK, so they are
        the database's job. If one were only `cascade="all, delete-orphan"` at
        the ORM level, a `DELETE FROM users` would fail on the constraint instead.
        """
        conversations = SqlAlchemyConversationRepository(session)
        tokens = SqlAlchemyRefreshTokenRepository(session)
        stored = await conversations.add(a_conversation(owner.id))
        await conversations.add_message(
            persisted_id(stored), Message(prompt_content="q", response_content="a")
        )
        await tokens.add(RefreshToken(user_id=owner.id, fingerprint="c" * 64, expires_at=NOW))
        await session.commit()

        await session.delete(await session.get_one(UserModel, owner.id))
        await session.commit()

        for model in (ConversationModel, MessageModel):
            assert await session.scalar(select(func.count()).select_from(model)) == 0

    async def test_a_conversation_cannot_belong_to_nobody(self, session: AsyncSession) -> None:
        """An unowned conversation is a conversation no authorization check reaches."""
        with pytest.raises(IntegrityError):
            await SqlAlchemyConversationRepository(session).add(a_conversation(owner_id=999_999))


class TestTransactionIsolation:
    """Two sessions, two connections - the claim the shared in-memory suite cannot make."""

    async def test_an_uncommitted_write_is_invisible_to_another_connection(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: UserModel,
    ) -> None:
        async with session_factory() as writer:
            await SqlAlchemyConversationRepository(writer).add(a_conversation(owner.id, "Draft"))

            async with session_factory() as reader:
                visible = await reader.scalar(select(func.count()).select_from(ConversationModel))

            assert visible == 0

    async def test_a_unit_of_work_that_raises_leaves_nothing_behind(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: UserModel,
    ) -> None:
        """The rollback, observed from outside the transaction that did it.

        On SQLite every session shares one connection, so "another connection
        sees nothing" is not a statement that suite can make. Here it is the
        actual guarantee: a use case that fails halfway never half-writes.
        """
        with pytest.raises(RuntimeError):
            async with SqlAlchemyUnitOfWork(session_factory) as uow:
                await uow.conversations.add(a_conversation(owner.id, "Doomed"))
                raise RuntimeError("halfway")

        async with session_factory() as reader:
            assert await reader.scalar(select(func.count()).select_from(ConversationModel)) == 0

    async def test_a_committed_unit_of_work_is_visible_to_everybody(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        owner: UserModel,
    ) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.conversations.add(a_conversation(owner.id, "Kept"))
            await uow.commit()

        async with session_factory() as reader:
            assert await reader.scalar(select(func.count()).select_from(ConversationModel)) == 1
