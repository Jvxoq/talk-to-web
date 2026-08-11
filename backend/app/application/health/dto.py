"""What a readiness check reports.

Deliberately says nothing about status codes or JSON: the use case decides
whether this process can serve traffic, and `app/api/v1/routers/health.py`
decides what that looks like on the wire.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    """One dependency's verdict.

    No exception, message or traceback: a probe failure carries connection
    strings and credentials, and `/ready` is the one endpoint that is reachable
    without a token. The detail goes to the log; the caller gets a name and a
    boolean, which is all a rollout gate can act on anyway.
    """

    name: str
    ready: bool


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    components: tuple[ComponentReadiness, ...]

    @property
    def ready(self) -> bool:
        """Every dependency answered. A single failure fails the whole check.

        There is no partial readiness here on purpose: a request that needs
        Qdrant fails just as hard as one that needs Postgres, so a process
        missing either should not be sent traffic.
        """
        return all(component.ready for component in self.components)
