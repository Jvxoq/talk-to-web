"""The current time, as a port, so tests can supply it."""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    async def now(self) -> datetime: ...
