"""Sign out: end one session, on purpose this time."""

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.ports import RefreshTokenFactory


class RevokeSession:
    """Revoke the presented refresh token. Never raises, never says if it existed.

    The access token stays valid for its short remaining lifetime.
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
