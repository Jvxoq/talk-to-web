"""Identity domain rules. No fixtures, no async, no I/O."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.errors import InvalidEmail, WeakPassword
from app.domain.identity.value_objects import (
    MIN_PASSWORD_LENGTH,
    Email,
    RawPassword,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class TestEmail:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("user@example.com", "user@example.com"),
            ("  User@Example.COM  ", "user@example.com"),
            ("a.b+tag@sub.example.co.uk", "a.b+tag@sub.example.co.uk"),
        ],
    )
    def test_normalises_case_and_whitespace(self, raw: str, expected: str) -> None:
        assert Email.sanitize(raw).value == expected

    def test_case_folding_makes_one_person_one_account(self) -> None:
        # The uniqueness check compares `Email` values, so this equality is what
        # stops "User@x.com" registering alongside "user@x.com".
        assert Email.sanitize("User@x.com") == Email.sanitize("user@x.com")

    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "no-at-sign", "@example.com", "user@", "user@host", "a b@example.com"],
    )
    def test_refuses_what_cannot_be_delivered_to(self, raw: str | None) -> None:
        with pytest.raises(InvalidEmail):
            Email.sanitize(raw)

    def test_refuses_an_address_longer_than_smtp_carries(self) -> None:
        with pytest.raises(InvalidEmail):
            Email.sanitize("a" * 250 + "@example.com")


class TestRawPassword:
    def test_accepts_a_long_enough_password(self) -> None:
        assert RawPassword("a" * MIN_PASSWORD_LENGTH).value == "a" * MIN_PASSWORD_LENGTH

    def test_refuses_a_short_one(self) -> None:
        with pytest.raises(WeakPassword):
            RawPassword("a" * (MIN_PASSWORD_LENGTH - 1))

    def test_refuses_one_long_enough_to_be_a_denial_of_service(self) -> None:
        with pytest.raises(WeakPassword):
            RawPassword("a" * 100_000)

    @pytest.mark.parametrize("value", [" correct horse battery", "correct horse battery "])
    def test_refuses_surrounding_whitespace(self, value: str) -> None:
        # A pasted password with a stray space locks the account out the first
        # time it is typed by hand.
        with pytest.raises(WeakPassword):
            RawPassword(value)


class TestUser:
    def test_is_not_persisted_until_it_has_an_id(self) -> None:
        user = User(email=Email.sanitize("u@example.com"), password_hash="h")
        assert not user.is_persisted()
        user.id = 1
        assert user.is_persisted()


class TestRefreshToken:
    def _token(self, **overrides: object) -> RefreshToken:
        defaults: dict[str, object] = {
            "user_id": 1,
            "fingerprint": "fp",
            "expires_at": NOW + timedelta(days=14),
        }
        return RefreshToken(**(defaults | overrides))  # type: ignore[arg-type]

    def test_a_fresh_token_is_usable(self) -> None:
        assert self._token().is_usable(NOW)

    def test_an_expired_token_is_not(self) -> None:
        assert not self._token().is_usable(NOW + timedelta(days=15))

    def test_expiry_is_exclusive_at_the_boundary(self) -> None:
        token = self._token(expires_at=NOW)
        assert not token.is_usable(NOW)

    def test_a_revoked_token_is_not_usable_even_before_expiry(self) -> None:
        token = self._token()
        token.revoke(NOW)
        assert token.is_revoked()
        assert not token.is_usable(NOW)

    def test_revoking_twice_keeps_the_first_timestamp(self) -> None:
        # Rotation and sign-out can both land on the same row. Moving the
        # timestamp would misreport when the session actually ended.
        token = self._token()
        token.revoke(NOW)
        token.revoke(NOW + timedelta(hours=1))
        assert token.revoked_at == NOW
