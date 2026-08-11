"""The real clock."""

from datetime import UTC, datetime


class SystemClock:
    """
    Reads the wall clock, satisfying `app.application.common.clock.Clock`.

    Always timezone-aware and always UTC. A naive datetime compared against an
    aware one raises, and the `expires_at` columns are `timestamptz`, so a naive
    value here would fail at the first expiry check rather than at the boundary
    that produced it.
    """

    async def now(self) -> datetime:
        return datetime.now(UTC)
