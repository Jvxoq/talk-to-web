"""Ask every dependency whether it is answering."""

import asyncio
from collections.abc import Sequence

from loguru import logger

from app.application.health.dto import ComponentReadiness, ReadinessReport
from app.application.health.ports import ReadinessProbe


class CheckReadiness:
    """
    Probe every dependency this process cannot serve a request without.

    Two properties matter more than they look:

    *Concurrent.* Probes run together, so the endpoint costs the slowest one
    rather than their sum. Sequentially, a Postgres probe already sitting on its
    timeout would make the Qdrant answer arrive after the orchestrator gave up.

    *Bounded.* Every probe is capped by `timeout_seconds`. A readiness endpoint
    that can hang is worse than one that reports "not ready": the orchestrator
    is left waiting on an answer instead of taking the process out of the pool,
    which is the exact failure the probe exists to catch.
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
        except Exception as error:
            # Broad on purpose: a probe fails through whichever exception its
            # driver happens to raise, and a readiness check that only handles
            # the ones we predicted is a 500 waiting for an unfamiliar outage.
            #
            # It catches the timeout too - `asyncio.timeout` converts its own
            # cancellation into `TimeoutError` on the way out - while a
            # cancellation from outside stays a `CancelledError`, which is a
            # `BaseException` and passes straight through. That distinction is
            # what keeps a worker being shut down from reporting itself ready.
            logger.warning(f"Readiness probe {probe.name!r} failed: {error}")
            return ComponentReadiness(name=probe.name, ready=False)
        return ComponentReadiness(name=probe.name, ready=True)
