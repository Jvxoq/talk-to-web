"""Trade a refresh token for a new pair, and notice when one has been stolen."""

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.dto import RefreshInput, SessionTokens
from app.application.identity.ports import RateLimiter, RefreshTokenFactory
from app.application.identity.sessions import SessionMinter
from app.domain.identity.errors import InvalidToken


class RefreshSession:
    """
    Rotate a session: the presented token is spent, and a new one replaces it.

    Rotation is what makes a leaked refresh token a bounded problem, and reuse
    detection is what turns it into a detectable one. A token that has already
    been rotated can only be presented by someone who kept a copy - the real
    client moved on to the new one - so the whole family is revoked and both
    parties are forced to sign in again. That is deliberately harsher than
    ignoring it: one interrupted session beats an attacker riding along
    indefinitely.
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
                # Two people hold this token and only one of them should. There
                # is no way to tell which, so neither keeps the session.
                logger.warning(f"Refresh token reuse detected for user {stored.user_id}")
                await uow.refresh_tokens.revoke_all_for_user(stored.user_id, now)
                await uow.commit()
                raise InvalidToken("session was already used")

            if not stored.is_usable(now):
                raise InvalidToken("session has expired")

            user = await uow.users.get(stored.user_id)
            if user is None or not user.is_active:
                raise InvalidToken("account is unavailable")

            # Spend the presented token before minting its replacement, so a
            # crash between the two leaves the client signed out rather than
            # holding two live sessions.
            if stored.id is not None:
                await uow.refresh_tokens.revoke(stored.id, now)

            tokens = await self._sessions.mint(uow, user)
            await uow.commit()

        logger.debug(f"Rotated session for user {user.id}")
        return tokens
