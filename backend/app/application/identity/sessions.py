"""Minting a session — the one piece of work register, login and refresh share.

Not a use case: nothing calls it from outside, and it does not own a
transaction. It is given the caller's unit of work precisely so the refresh
token row lands in the same transaction as whatever caused it, and rolls back
with it.
"""

from datetime import timedelta

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWork
from app.application.identity.dto import SessionTokens, UserContext
from app.application.identity.ports import AccessTokenCodec, RefreshTokenFactory
from app.domain.identity.entities import RefreshToken, User


class SessionMinter:
    """Issues one access/refresh pair and records the refresh half."""

    def __init__(
        self,
        access_tokens: AccessTokenCodec,
        refresh_tokens: RefreshTokenFactory,
        clock: Clock,
        refresh_ttl_seconds: int,
    ) -> None:
        self._access_tokens = access_tokens
        self._refresh_tokens = refresh_tokens
        self._clock = clock
        self._refresh_ttl = timedelta(seconds=refresh_ttl_seconds)

    async def mint(self, uow: UnitOfWork, user: User) -> SessionTokens:
        """Issue a pair for `user`, writing the refresh fingerprint through `uow`."""
        if user.id is None:
            # Unreachable through the use cases, which all mint after the insert.
            # Named anyway, because a session pointing at no user would only be
            # discovered as a foreign key violation at commit.
            raise ValueError("Cannot mint a session for an unpersisted user")

        now = await self._clock.now()
        access = self._access_tokens.issue(user.id, user.email.value, now)

        secret = self._refresh_tokens.new_secret()
        expires_at = now + self._refresh_ttl
        await uow.refresh_tokens.add(
            RefreshToken(
                user_id=user.id,
                # The secret itself never reaches the repository.
                fingerprint=self._refresh_tokens.fingerprint(secret),
                expires_at=expires_at,
            )
        )

        return SessionTokens(
            access_token=access.token,
            expires_in=int((access.expires_at - now).total_seconds()),
            refresh_token=secret,
            refresh_expires_in=int(self._refresh_ttl.total_seconds()),
            user=UserContext(user_id=user.id, email=user.email.value),
        )
