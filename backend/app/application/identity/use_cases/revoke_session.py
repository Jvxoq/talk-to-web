"""Sign out: end one session, on purpose this time."""

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.ports import RefreshTokenFactory


class RevokeSession:
    """
    Revoke the presented refresh token, if it is still live.

    Never raises, and never reports whether there was anything to revoke.
    Logging out is not an operation a caller can fail at, and a signed-out
    client must not be left thinking it is still signed in because its token had
    already expired. The access token it still holds outlives this by its own
    (short) lifetime - that is the price of stateless verification, and the
    reason the access TTL is minutes rather than days.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        refresh_tokens: RefreshTokenFactory,
        clock: Clock,
    ) -> None:
        self._uow_factory = uow_factory
        self._refresh_tokens = refresh_tokens
        self._clock = clock

    async def __call__(self, refresh_token: str) -> None:
        fingerprint = self._refresh_tokens.fingerprint(refresh_token)
        now = await self._clock.now()

        async with self._uow_factory() as uow:
            stored = await uow.refresh_tokens.get_by_fingerprint(fingerprint)
            if stored is None or stored.id is None:
                logger.debug("Sign-out presented an unknown session")
                return

            await uow.refresh_tokens.revoke(stored.id, now)
            await uow.commit()
            logger.debug(f"Signed out session {stored.id}")
