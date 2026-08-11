"""Reading the current time, expressed as a port.

`datetime.now()` called from inside a use case is untestable I/O wearing a
stdlib costume: every assertion about expiry then either sleeps or reaches for a
patch. Asking for the time through a port makes "this token expired an hour ago"
a value a test supplies rather than a condition it has to wait for.
"""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    async def now(self) -> datetime: ...
