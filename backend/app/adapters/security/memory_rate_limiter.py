"""In-process attempt counting, as a sliding window."""

import time
from collections import deque

from app.domain.usage.errors import RateLimited

_SWEEP_EVERY = 256
"""How many `hit` calls pass between sweeps of the expired keys."""


class SlidingWindowRateLimiter:
    """Counts recent attempts per key. Fits every context's `RateLimiter` port.

    In-process, so N replicas allow N times the budget. Fine for one container.
    Uses a monotonic clock, so a wall-clock jump cannot grant free attempts.
    """

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        if max_attempts < 1:
            raise ValueError("A rate limiter that allows nothing is a closed door")
        self._max_attempts = max_attempts
        self._window = float(window_seconds)
        self._attempts: dict[str, deque[float]] = {}
        self._since_sweep = 0

    async def hit(self, key: str) -> None:
        """Record one attempt, raising `RateLimited` once the budget is spent."""
        now = time.monotonic()
        self._maybe_sweep(now)

        window = self._attempts.setdefault(key, deque())
        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self._max_attempts:
            # The oldest counted attempt is the next to expire.
            retry_after = int(window[0] + self._window - now) + 1
            raise RateLimited(retry_after)

        # Only allowed attempts count, so retries while blocked do not extend the wait.
        window.append(now)

    async def reset(self, key: str) -> None:
        """Forget a key. One good login clears the failures that preceded it."""
        self._attempts.pop(key, None)

    def _maybe_sweep(self, now: float) -> None:
        """Drop expired keys every few hundred calls, so the map cannot grow forever."""

        self._since_sweep += 1
        if self._since_sweep < _SWEEP_EVERY:
            return
        self._since_sweep = 0

        cutoff = now - self._window
        stale = [
            key for key, window in self._attempts.items() if not window or window[-1] <= cutoff
        ]
        for key in stale:
            del self._attempts[key]
