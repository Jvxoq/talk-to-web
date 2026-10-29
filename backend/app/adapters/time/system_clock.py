"""The real clock."""

from datetime import UTC, datetime


class SystemClock:
    """Reads the wall clock in UTC, timezone-aware. Implements `Clock`.

    Naive datetimes would fail against the `timestamptz` columns.
    """

    async def now(self) -> datetime:
        return datetime.now(UTC)
