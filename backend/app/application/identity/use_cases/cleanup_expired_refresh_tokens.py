"""Housekeeping: delete refresh_tokens rows long past expiry.

Not in the API. Run on a schedule via `app/cleanup_expired_refresh_tokens.py`.
"""

from datetime import timedelta

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory


class CleanupExpiredRefreshTokens:
    """Delete refresh tokens that expired before the retention window.

    Not zero, because reuse detection in `RefreshSession` needs the old rows.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        clock: Clock,
        retention_seconds: int,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._retention = timedelta(seconds=retention_seconds)

    async def __call__(self) -> int:
        now = await self._clock.now()
        cutoff = now - self._retention

        async with self._uow_factory() as uow:
            deleted = await uow.refresh_tokens.delete_expired_before(cutoff)
            await uow.commit()

        logger.info(f"Cleanup deleted {deleted} expired refresh token row(s)")
        return deleted
