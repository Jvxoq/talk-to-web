"""What the readiness check needs from the outside world."""

from typing import Protocol


class ReadinessProbe(Protocol):
    """One dependency's cheap health check. `check` raises when not ready."""

    @property
    def name(self) -> str:
        """The dependency's name in the response. Keep it stable."""

        ...

    async def check(self) -> None:
        """Return if the dependency is answering, raise if it is not."""
        ...
