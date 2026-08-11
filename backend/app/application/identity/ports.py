"""What the identity use cases need from the outside world.

Structural `Protocol`s: an adapter satisfies one by having the methods, without
importing or inheriting from anything here. Cryptography, token formats and
attempt counting are all infrastructure — the use cases below know only that
passwords can be hashed and compared, that a token can be issued and read back,
and that some attempts are too many.
"""

from datetime import datetime
from typing import Protocol

from app.application.identity.dto import IssuedAccessToken, TokenClaims
from app.domain.identity.entities import RefreshToken, User
from app.domain.identity.value_objects import Email


class PasswordHasher(Protocol):
    """Turns a password into something safe to store, and checks it later."""

    def hash(self, plain: str) -> str: ...

    def verify(self, plain: str, hashed: str) -> bool: ...

    def dummy_hash(self) -> str:
        """A real hash of nothing in particular.

        Exists so a login for an address with no account can still pay the cost
        of a verification. Skipping the work is measurable from outside, and a
        login that answers faster for unknown addresses is an account
        enumeration oracle.
        """
        ...


class AccessTokenCodec(Protocol):
    """Signs short-lived access tokens, and reads them back."""

    def issue(self, user_id: int, email: str, now: datetime) -> IssuedAccessToken: ...

    def decode(self, token: str) -> TokenClaims:
        """Verify and read a token, or raise `InvalidToken` / `TokenExpired`."""
        ...


class RefreshTokenFactory(Protocol):
    """Mints refresh secrets and reduces them to storable fingerprints."""

    def new_secret(self) -> str: ...

    def fingerprint(self, secret: str) -> str: ...


class UserRepository(Protocol):
    async def get(self, user_id: int) -> User | None: ...

    async def get_by_email(self, email: Email) -> User | None: ...

    async def add(self, user: User) -> User: ...


class RefreshTokenRepository(Protocol):
    async def add(self, token: RefreshToken) -> RefreshToken: ...

    async def get_by_fingerprint(self, fingerprint: str) -> RefreshToken | None: ...

    async def revoke(self, token_id: int, at: datetime) -> None: ...

    async def revoke_all_for_user(self, user_id: int, at: datetime) -> None: ...

    async def delete_expired_before(self, cutoff: datetime) -> int:
        """Delete every row whose `expires_at` is older than `cutoff`.

        A real DELETE, unlike `revoke` - these rows are old enough that reuse
        detection has nothing left to gain from keeping them. Returns the count
        removed, for the cleanup job to log.
        """
        ...


class RateLimiter(Protocol):
    """Counts attempts against a key and refuses the ones over budget."""

    async def hit(self, key: str) -> None:
        """Record one attempt, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None:
        """Forget a key's attempts. Called on success, so one good login clears the count."""
        ...
