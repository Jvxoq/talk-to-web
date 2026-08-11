"""In-process attempt counting, as a sliding window."""

import time
from collections import deque

from app.domain.usage.errors import RateLimited

_SWEEP_EVERY = 256
"""How many `hit` calls pass between sweeps of the expired keys."""


class SlidingWindowRateLimiter:
    """
    Counts recent attempts per key.

    Satisfies the `RateLimiter` port as each context declares it -
    `application.identity.ports`, `application.chat.ports` and
    `application.ingestion.ports` all describe the same two methods, and
    structural typing means one instance answers all three without importing
    any of them.

    Deliberately in-process, with the consequence stated plainly: each replica
    keeps its own count, so N replicas allow N times the budget. That is a real
    weakening and it is the right trade here - the alternative is running Redis
    for a counter, and this deployment has one backend container. Swapping in a
    shared adapter later changes this file and the composition root, and nothing
    else, because the use cases only ever see the port.

    A monotonic clock, not the `Clock` port: this measures elapsed time rather
    than reading a business timestamp, and a wall clock stepping backwards over
    NTP would hand out free attempts.
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
            # From the oldest attempt still counted: that is the one whose
            # expiry frees a slot.
            retry_after = int(window[0] + self._window - now) + 1
            raise RateLimited(retry_after)

        # Appended only on the allowed path, so a caller that keeps hammering a
        # blocked key does not push its own unblock time further away.
        window.append(now)

    async def reset(self, key: str) -> None:
        """Forget a key. One good login clears the failures that preceded it."""
        self._attempts.pop(key, None)

    def _maybe_sweep(self, now: float) -> None:
        """Drop keys whose windows have emptied.

        Without this the dictionary grows one entry per address ever tried,
        which is a memory leak an attacker controls the size of. Sweeping every
        few hundred calls keeps it bounded without walking the whole map on a
        hot path.
        """
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
