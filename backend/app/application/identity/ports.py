"""Ports the identity use cases need. Crypto and token formats stay in adapters."""

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
        """A real hash to verify against when the account does not exist.

        Same timing either way, so login cannot reveal which emails exist.
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
        """Delete rows that expired before `cutoff`. Returns the count removed."""

        ...


class RateLimiter(Protocol):
    """Counts attempts against a key and refuses the ones over budget."""

    async def hit(self, key: str) -> None:
        """Record one attempt, raising `RateLimited` if the budget is spent."""
        ...

    async def reset(self, key: str) -> None:
        """Forget a key's attempts. Called on success, so one good login clears the count."""
        ...
