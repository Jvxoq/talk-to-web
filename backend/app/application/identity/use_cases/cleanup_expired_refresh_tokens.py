"""Housekeeping: purge refresh_tokens rows long past their expiry.

Not reachable through the API - nothing about "delete old session rows" is a
user-facing action - so nothing in `app/api/` calls this. It exists to be run
on a schedule; see `app/cleanup_expired_refresh_tokens.py` for the entry point.
"""

from datetime import timedelta

from loguru import logger

from app.application.common.clock import Clock
from app.application.common.uow import UnitOfWorkFactory


class CleanupExpiredRefreshTokens:
    """
    Delete every refresh token whose `expires_at` is older than the retention
    window.

    The window is not zero: `RefreshSession` treats a *revoked* row turning up
    again as reuse and revokes the whole family, and that detection only works
    while the row still exists. Retention keeps that signal alive for a while
    past ordinary expiry, rather than deleting a row the moment it stops being
    usable.
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
