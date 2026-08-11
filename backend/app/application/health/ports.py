"""What the readiness check needs from the outside world."""

from typing import Protocol


class ReadinessProbe(Protocol):
    """One dependency, and a cheap way of asking whether it is answering.

    `check` reports by raising: any exception means "not ready". A boolean
    return would force every adapter to swallow its own driver errors, and the
    one thing worth logging - what actually went wrong - would be gone by the
    time the use case saw the result.
    """

    @property
    def name(self) -> str:
        """What this dependency is called in the response. Stable; operators alert on it."""
        ...

    async def check(self) -> None:
        """Return if the dependency is answering, raise if it is not."""
        ...
