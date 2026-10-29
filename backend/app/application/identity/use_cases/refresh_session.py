"""Trade a refresh token for a new pair, and notice when one has been stolen."""

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.dto import RefreshInput, SessionTokens
from app.application.identity.ports import RateLimiter, RefreshTokenFactory
from app.application.identity.sessions import SessionMinter
from app.domain.identity.errors import InvalidToken


class RefreshSession:
    """Rotate a session: spend the presented token and issue a new one.

    A rotated token shown again means someone kept a copy. Every session for
    that user is then revoked, and both sides must sign in again.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        refresh_tokens: RefreshTokenFactory,
        sessions: SessionMinter,
        clock: Clock,
        limiter: RateLimiter,
    ) -> None:
        self._uow_factory = uow_factory
        self._refresh_tokens = refresh_tokens
        self._sessions = sessions
        self._clock = clock
        self._limiter = limiter

    async def __call__(self, data: RefreshInput) -> SessionTokens:
        if data.client_ip is not None:
            await self._limiter.hit(f"refresh:{data.client_ip}")

        fingerprint = self._refresh_tokens.fingerprint(data.refresh_token)
        now = await self._clock.now()

        async with self._uow_factory() as uow:
            stored = await uow.refresh_tokens.get_by_fingerprint(fingerprint)
            if stored is None:
                raise InvalidToken("no such session")

            if stored.is_revoked():
                # We cannot tell who is real, so nobody keeps the session.
                logger.warning(f"Refresh token reuse detected for user {stored.user_id}")
                await uow.refresh_tokens.revoke_all_for_user(stored.user_id, now)
                await uow.commit()
                raise InvalidToken("session was already used")

            if not stored.is_usable(now):
                raise InvalidToken("session has expired")

            user = await uow.users.get(stored.user_id)
            if user is None or not user.is_active:
                raise InvalidToken("account is unavailable")

            # Revoke first, so a crash signs the user out instead of leaving two sessions.

            if stored.id is not None:
                await uow.refresh_tokens.revoke(stored.id, now)

            tokens = await self._sessions.mint(uow, user)
            await uow.commit()

        logger.debug(f"Rotated session for user {user.id}")
        return tokens
