"""Identity use cases against fakes — no database, no cryptography, no clock."""

import pytest

from app.application.identity.dto import LoginInput, RefreshInput, RegisterInput
from app.application.identity.sessions import SessionMinter
from app.application.identity.use_cases.authenticate_user import AuthenticateUser
from app.application.identity.use_cases.cleanup_expired_refresh_tokens import (
    CleanupExpiredRefreshTokens,
)
from app.application.identity.use_cases.identify_request import IdentifyRequest
from app.application.identity.use_cases.refresh_session import RefreshSession
from app.application.identity.use_cases.register_user import RegisterUser
from app.application.identity.use_cases.revoke_session import RevokeSession
from app.domain.identity.errors import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    InvalidToken,
    WeakPassword,
)
from app.domain.usage.errors import RateLimited
from tests.fakes import (
    FakeAccessTokenCodec,
    FakeClock,
    FakePasswordHasher,
    FakeRateLimiter,
    FakeRefreshTokenFactory,
    UnitOfWorkSpy,
)

EMAIL = "owner@example.com"
PASSWORD = "correct horse battery staple"
REFRESH_TTL = 14 * 24 * 60 * 60
CLEANUP_RETENTION = 30 * 24 * 60 * 60


class Harness:
    """Every identity use case, wired to one shared set of fakes."""

    def __init__(self, max_attempts: int = 1_000) -> None:
        self.factory = UnitOfWorkSpy()
        self.hasher = FakePasswordHasher()
        self.access = FakeAccessTokenCodec()
        self.refresh = FakeRefreshTokenFactory()
        self.clock = FakeClock()
        self.limiter = FakeRateLimiter(max_attempts=max_attempts)
        minter = SessionMinter(
            access_tokens=self.access,
            refresh_tokens=self.refresh,
            clock=self.clock,
            refresh_ttl_seconds=REFRESH_TTL,
        )
        self.register = RegisterUser(
            self.factory, hasher=self.hasher, sessions=minter, limiter=self.limiter
        )
        self.login = AuthenticateUser(
            self.factory, hasher=self.hasher, sessions=minter, limiter=self.limiter
        )
        self.refresh_session = RefreshSession(
            self.factory,
            refresh_tokens=self.refresh,
            sessions=minter,
            clock=self.clock,
            limiter=self.limiter,
        )
        self.logout = RevokeSession(self.factory, refresh_tokens=self.refresh, clock=self.clock)
        self.identify = IdentifyRequest(self.access)
        self.cleanup = CleanupExpiredRefreshTokens(
            self.factory, clock=self.clock, retention_seconds=CLEANUP_RETENTION
        )


async def registered(harness: Harness) -> str:
    """Register the standard account and return its refresh secret."""
    tokens = await harness.register(RegisterInput(email=EMAIL, password=PASSWORD))
    return tokens.refresh_token


class TestRegisterUser:
    async def test_creates_an_account_and_signs_it_in(self) -> None:
        harness = Harness()

        tokens = await harness.register(RegisterInput(email=EMAIL, password=PASSWORD))

        assert tokens.user.user_id == 1
        assert tokens.user.email == EMAIL
        assert tokens.access_token
        assert tokens.refresh_token
        assert harness.factory.issued[0].committed

    async def test_stores_what_the_hasher_produced_not_what_was_typed(self) -> None:
        # That the stored value is unrecoverable is the *hasher's* property, and
        # `tests/adapters/test_security.py` is where Argon2 is held to it. What
        # this asserts is the use case's part: the password goes through the port
        # rather than into the column.
        harness = Harness()
        await registered(harness)

        stored = harness.factory.users.rows[1]
        assert stored.password_hash == harness.hasher.hash(PASSWORD)
        assert harness.hasher.verify(PASSWORD, stored.password_hash)

    async def test_stores_only_a_fingerprint_of_the_refresh_secret(self) -> None:
        # A database that held the secret would hand over every live session to
        # anyone who read a backup.
        harness = Harness()
        secret = await registered(harness)

        fingerprints = [t.fingerprint for t in harness.factory.refresh_tokens.rows.values()]
        assert secret not in fingerprints
        assert fingerprints == [harness.refresh.fingerprint(secret)]

    async def test_normalises_the_address_before_storing_it(self) -> None:
        harness = Harness()
        tokens = await harness.register(
            RegisterInput(email="  Owner@Example.COM ", password=PASSWORD)
        )
        assert tokens.user.email == EMAIL

    async def test_a_taken_address_is_a_conflict_however_it_is_typed(self) -> None:
        harness = Harness()
        await registered(harness)

        with pytest.raises(EmailAlreadyRegistered):
            await harness.register(RegisterInput(email="OWNER@EXAMPLE.COM", password=PASSWORD))

    async def test_a_weak_password_is_refused_before_anything_is_written(self) -> None:
        harness = Harness()

        with pytest.raises(WeakPassword):
            await harness.register(RegisterInput(email=EMAIL, password="short"))

        assert harness.factory.users.rows == {}

    async def test_registration_is_rate_limited_by_address(self) -> None:
        harness = Harness(max_attempts=1)

        await harness.register(RegisterInput(email=EMAIL, password=PASSWORD, client_ip="1.2.3.4"))
        with pytest.raises(RateLimited):
            await harness.register(
                RegisterInput(email="other@example.com", password=PASSWORD, client_ip="1.2.3.4")
            )


class TestAuthenticateUser:
    async def test_the_right_password_returns_a_session(self) -> None:
        harness = Harness()
        await registered(harness)

        tokens = await harness.login(LoginInput(email=EMAIL, password=PASSWORD))

        assert tokens.user.user_id == 1
        assert tokens.refresh_token

    async def test_the_wrong_password_is_refused(self) -> None:
        harness = Harness()
        await registered(harness)

        with pytest.raises(InvalidCredentials):
            await harness.login(LoginInput(email=EMAIL, password="wrong password entirely"))

    @pytest.mark.parametrize("email", ["nobody@example.com", "not-an-email-at-all"])
    async def test_an_unknown_or_malformed_address_looks_identical_to_a_wrong_password(
        self, email: str
    ) -> None:
        # Same exception for all three cases: a distinguishable answer would turn
        # the login form into a way to test whether an address has an account.
        harness = Harness()
        await registered(harness)

        with pytest.raises(InvalidCredentials):
            await harness.login(LoginInput(email=email, password=PASSWORD))

    async def test_a_deactivated_account_cannot_sign_in(self) -> None:
        harness = Harness()
        await registered(harness)
        harness.factory.users.rows[1].is_active = False

        with pytest.raises(InvalidCredentials):
            await harness.login(LoginInput(email=EMAIL, password=PASSWORD))

    async def test_guesses_are_counted_against_the_address(self) -> None:
        harness = Harness(max_attempts=2)
        await registered(harness)

        for _ in range(2):
            with pytest.raises(InvalidCredentials):
                await harness.login(LoginInput(email=EMAIL, password="wrong password entirely"))

        with pytest.raises(RateLimited):
            await harness.login(LoginInput(email=EMAIL, password=PASSWORD))

    async def test_a_successful_sign_in_clears_the_count(self) -> None:
        harness = Harness(max_attempts=2)
        await registered(harness)

        with pytest.raises(InvalidCredentials):
            await harness.login(LoginInput(email=EMAIL, password="wrong password entirely"))
        await harness.login(LoginInput(email=EMAIL, password=PASSWORD))

        # Without the reset, one earlier typo would count against the next
        # window and lock a legitimate user out of their own account.
        assert harness.limiter.hits.get(f"login:{EMAIL}") is None


class TestRefreshSession:
    async def test_rotation_returns_a_different_pair(self) -> None:
        harness = Harness()
        first = await registered(harness)

        rotated = await harness.refresh_session(RefreshInput(refresh_token=first))

        assert rotated.refresh_token != first
        assert rotated.user.user_id == 1

    async def test_the_old_token_stops_working(self) -> None:
        harness = Harness()
        first = await registered(harness)
        await harness.refresh_session(RefreshInput(refresh_token=first))

        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=first))

    async def test_reusing_a_spent_token_kills_every_session(self) -> None:
        """Reuse means two parties hold one token, and only one of them should.

        There is no way to tell which is the real client, so neither keeps the
        session. That is deliberately harsher than ignoring it: the alternative
        lets whoever stole the token ride along indefinitely.
        """
        harness = Harness()
        first = await registered(harness)
        second = await harness.refresh_session(RefreshInput(refresh_token=first))

        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=first))

        # The token the honest client was holding is dead too.
        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=second.refresh_token))
        assert all(t.is_revoked() for t in harness.factory.refresh_tokens.rows.values())

    async def test_an_expired_token_is_refused(self) -> None:
        harness = Harness()
        secret = await registered(harness)
        harness.clock.advance(REFRESH_TTL + 1)

        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=secret))

    async def test_an_unknown_token_is_refused(self) -> None:
        harness = Harness()
        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token="never issued"))

    async def test_a_deactivated_account_cannot_refresh(self) -> None:
        harness = Harness()
        secret = await registered(harness)
        harness.factory.users.rows[1].is_active = False

        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=secret))


class TestRevokeSession:
    async def test_sign_out_ends_the_session(self) -> None:
        harness = Harness()
        secret = await registered(harness)

        await harness.logout(secret)

        with pytest.raises(InvalidToken):
            await harness.refresh_session(RefreshInput(refresh_token=secret))

    @pytest.mark.parametrize("token", ["never issued", ""])
    async def test_signing_out_an_unknown_session_is_not_an_error(self, token: str) -> None:
        # A client that has been signed out must never be left believing it is
        # still signed in because its token had already expired.
        await Harness().logout(token)

    async def test_signing_out_twice_is_not_an_error(self) -> None:
        harness = Harness()
        secret = await registered(harness)

        await harness.logout(secret)
        await harness.logout(secret)


class TestIdentifyRequest:
    async def test_a_valid_token_names_its_owner(self) -> None:
        harness = Harness()
        tokens = await harness.register(RegisterInput(email=EMAIL, password=PASSWORD))

        user = await harness.identify(tokens.access_token)

        assert user.user_id == 1
        assert user.email == EMAIL

    @pytest.mark.parametrize("token", ["", "   ", "not-a-token"])
    async def test_anything_else_is_refused(self, token: str) -> None:
        with pytest.raises(InvalidToken):
            await Harness().identify(token)


class TestCleanupExpiredRefreshTokens:
    async def test_deletes_rows_past_the_retention_window(self) -> None:
        harness = Harness()
        await registered(harness)
        # Expired well outside the retention window - old enough that reuse
        # detection has nothing left to gain from keeping the row.
        harness.clock.advance(REFRESH_TTL + CLEANUP_RETENTION + 1)

        deleted = await harness.cleanup()

        assert deleted == 1
        assert harness.factory.refresh_tokens.rows == {}

    async def test_leaves_rows_still_inside_the_retention_window(self) -> None:
        harness = Harness()
        await registered(harness)
        # Expired, but not long enough ago for the grace period to have lapsed:
        # a replay of this token would still trip reuse detection.
        harness.clock.advance(REFRESH_TTL + 1)

        deleted = await harness.cleanup()

        assert deleted == 0
        assert len(harness.factory.refresh_tokens.rows) == 1

    async def test_leaves_a_still_valid_session_alone(self) -> None:
        harness = Harness()
        await registered(harness)

        deleted = await harness.cleanup()

        assert deleted == 0
        assert len(harness.factory.refresh_tokens.rows) == 1
