"""The security adapters against the real thing.

Everything here talks to actual Argon2 and actual PyJWT. These are the claims a
fake cannot make on their behalf, and they are exactly the claims that would be
catastrophic to get wrong.
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.adapters.security.argon2_hasher import Argon2PasswordHasher
from app.adapters.security.jwt_codec import JwtAccessTokenCodec
from app.adapters.security.memory_rate_limiter import SlidingWindowRateLimiter
from app.adapters.security.refresh_tokens import Sha256RefreshTokenFactory
from app.domain.identity.errors import InvalidToken, TokenExpired
from app.domain.usage.errors import RateLimited

# Real time, not a fixed date: PyJWT checks `exp` against the actual clock, so a
# hardcoded issue time silently starts producing expired tokens the day it goes
# past. Only the codec tests need this; nothing here asserts on the value.
NOW = datetime.now(UTC)
SECRET = "a test signing secret that is comfortably over thirty-two bytes long"
PASSWORD = "correct horse battery staple"


class TestArgon2PasswordHasher:
    def test_the_password_is_not_recoverable_from_the_hash(self) -> None:
        hasher = Argon2PasswordHasher()
        hashed = hasher.hash(PASSWORD)

        assert PASSWORD not in hashed
        assert hashed.startswith("$argon2id$")

    def test_the_right_password_verifies(self) -> None:
        hasher = Argon2PasswordHasher()
        assert hasher.verify(PASSWORD, hasher.hash(PASSWORD))

    def test_the_wrong_password_does_not(self) -> None:
        hasher = Argon2PasswordHasher()
        assert not hasher.verify("something else entirely", hasher.hash(PASSWORD))

    def test_the_same_password_hashes_differently_every_time(self) -> None:
        # Per-password salt. Without it, identical passwords are visibly
        # identical in the table, and one cracked hash unlocks every account
        # that shared it.
        hasher = Argon2PasswordHasher()
        assert hasher.hash(PASSWORD) != hasher.hash(PASSWORD)

    def test_a_long_passphrase_is_not_truncated(self) -> None:
        # bcrypt would stop at 72 bytes, making these two passwords equivalent.
        # This is the reason the adapter is Argon2.
        hasher = Argon2PasswordHasher()
        base = "x" * 72
        assert not hasher.verify(base + "different", hasher.hash(base + "original"))

    @pytest.mark.parametrize("stored", ["", "not-a-hash", "$argon2id$garbage"])
    def test_a_corrupt_stored_hash_fails_the_check_rather_than_raising(self, stored: str) -> None:
        # A 500 here would be distinguishable from a wrong password, and the
        # caller could not fix it either way.
        assert not Argon2PasswordHasher().verify(PASSWORD, stored)

    def test_the_dummy_hash_is_a_real_one(self) -> None:
        # It exists to make a login for an unknown address cost the same as one
        # for a known address, which only works if verifying against it does the
        # same work.
        hasher = Argon2PasswordHasher()
        dummy = hasher.dummy_hash()

        assert dummy.startswith("$argon2id$")
        assert not hasher.verify(PASSWORD, dummy)


class TestJwtAccessTokenCodec:
    def codec(self, ttl_seconds: int = 900) -> JwtAccessTokenCodec:
        return JwtAccessTokenCodec(secret=SECRET, algorithm="HS256", ttl_seconds=ttl_seconds)

    def test_a_token_round_trips(self) -> None:
        codec = self.codec()
        issued = codec.issue(7, "owner@example.com", NOW)

        claims = codec.decode(issued.token)

        assert claims.user_id == 7
        assert claims.email == "owner@example.com"
        assert issued.expires_at == NOW + timedelta(seconds=900)

    def test_an_expired_token_is_reported_as_expired(self) -> None:
        codec = self.codec(ttl_seconds=-1)
        issued = codec.issue(7, "owner@example.com", NOW)

        with pytest.raises(TokenExpired):
            codec.decode(issued.token)

    def test_a_token_signed_with_another_secret_is_refused(self) -> None:
        other = JwtAccessTokenCodec(
            secret="an entirely different secret, also comfortably long enough",
            algorithm="HS256",
            ttl_seconds=900,
        )
        issued = other.issue(7, "owner@example.com", NOW)

        with pytest.raises(InvalidToken):
            self.codec().decode(issued.token)

    def test_an_unsigned_token_is_refused(self) -> None:
        """The classic JWT hole: `alg: none`.

        A decoder that trusted the token's own header would accept this, and
        anyone could mint a token for any account by writing one out by hand.
        """
        forged = jwt.encode(
            {"sub": "1", "email": "attacker@example.com", "exp": 9_999_999_999},
            key="",
            algorithm="none",
        )

        with pytest.raises(InvalidToken):
            self.codec().decode(forged)

    def test_a_token_missing_its_expiry_is_refused(self) -> None:
        # A token with no `exp` never expires. It must not be accepted just
        # because the signature checks out.
        forged = jwt.encode({"sub": "1", "email": "a@b.co"}, SECRET, algorithm="HS256")

        with pytest.raises(InvalidToken):
            self.codec().decode(forged)

    @pytest.mark.parametrize("token", ["", "garbage", "a.b.c"])
    def test_rubbish_is_refused(self, token: str) -> None:
        with pytest.raises(InvalidToken):
            self.codec().decode(token)


class TestSha256RefreshTokenFactory:
    def test_every_secret_is_different(self) -> None:
        factory = Sha256RefreshTokenFactory()
        secrets = {factory.new_secret() for _ in range(100)}
        assert len(secrets) == 100

    def test_the_fingerprint_is_stable_and_hides_the_secret(self) -> None:
        factory = Sha256RefreshTokenFactory()
        secret = factory.new_secret()
        fingerprint = factory.fingerprint(secret)

        assert factory.fingerprint(secret) == fingerprint
        assert secret not in fingerprint
        assert len(fingerprint) == 64

    def test_different_secrets_fingerprint_differently(self) -> None:
        factory = Sha256RefreshTokenFactory()
        assert factory.fingerprint("a") != factory.fingerprint("b")


class TestSlidingWindowRateLimiter:
    async def test_allows_up_to_the_budget(self) -> None:
        limiter = SlidingWindowRateLimiter(max_attempts=3, window_seconds=60)
        for _ in range(3):
            await limiter.hit("key")

    async def test_refuses_past_the_budget(self) -> None:
        limiter = SlidingWindowRateLimiter(max_attempts=2, window_seconds=60)
        await limiter.hit("key")
        await limiter.hit("key")

        with pytest.raises(RateLimited) as caught:
            await limiter.hit("key")

        assert 0 < caught.value.retry_after_seconds <= 61

    async def test_keys_have_separate_budgets(self) -> None:
        limiter = SlidingWindowRateLimiter(max_attempts=1, window_seconds=60)
        await limiter.hit("one")
        await limiter.hit("two")

    async def test_reset_clears_a_key(self) -> None:
        limiter = SlidingWindowRateLimiter(max_attempts=1, window_seconds=60)
        await limiter.hit("key")
        await limiter.reset("key")
        await limiter.hit("key")

    async def test_a_blocked_key_does_not_push_its_own_unblock_further_away(self) -> None:
        # Attempts are only recorded on the allowed path. Counting refused ones
        # would let a client hammering the endpoint extend its own lockout
        # indefinitely.
        limiter = SlidingWindowRateLimiter(max_attempts=1, window_seconds=60)
        await limiter.hit("key")

        first = None
        for _ in range(5):
            with pytest.raises(RateLimited) as caught:
                await limiter.hit("key")
            first = first or caught.value.retry_after_seconds

        assert caught.value.retry_after_seconds <= (first or 0)

    async def test_expired_keys_are_swept(self) -> None:
        # The map grows one entry per address ever tried, and an attacker picks
        # the addresses. Without the sweep that is a memory leak they control.
        limiter = SlidingWindowRateLimiter(max_attempts=1_000, window_seconds=0)
        for i in range(600):
            await limiter.hit(f"key-{i}")

        assert len(limiter._attempts) < 600

    def test_a_budget_of_zero_is_a_configuration_error(self) -> None:
        with pytest.raises(ValueError, match="closed door"):
            SlidingWindowRateLimiter(max_attempts=0, window_seconds=60)
