"""Ask every dependency whether it is answering."""

import asyncio
from collections.abc import Sequence

from loguru import logger

from app.application.health.dto import ComponentReadiness, ReadinessReport
from app.application.health.ports import ReadinessProbe


class CheckReadiness:
    """Probe every required dependency.

    Probes run concurrently, each capped by `timeout_seconds`, so the endpoint
    never hangs. A hung readiness check is worse than "not ready".
    """

    def __init__(self, probes: Sequence[ReadinessProbe], timeout_seconds: float) -> None:
        self._probes = tuple(probes)
        self._timeout_seconds = timeout_seconds

    async def __call__(self) -> ReadinessReport:
        results = await asyncio.gather(*(self._probe(probe) for probe in self._probes))
        return ReadinessReport(components=tuple(results))

    async def _probe(self, probe: ReadinessProbe) -> ComponentReadiness:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await probe.check()
        except Exception as error:  # noqa: BLE001
            # Broad, since drivers raise anything. Catches `TimeoutError` too.
            # An outside `CancelledError` still passes through.

            logger.warning(f"Readiness probe {probe.name!r} failed: {error}")
            return ComponentReadiness(name=probe.name, ready=False)
        return ComponentReadiness(name=probe.name, ready=True)
