"""Wire shapes for the probe endpoints."""

from pydantic import BaseModel

from app.application.health.dto import ReadinessReport


class ReadinessResponse(BaseModel):
    """What an orchestrator or a deploy gate reads.

    `status` is the field to alert on; `checks` is there so a failed rollout
    says *which* dependency was missing without anyone having to open the logs.
    Each value is `"ok"` or `"down"` and nothing more - the endpoint needs no
    credentials, so it must not describe the failure it saw.
    """

    status: str
    checks: dict[str, str]

    @classmethod
    def of(cls, report: ReadinessReport) -> "ReadinessResponse":
        return cls(
            status="ready" if report.ready else "degraded",
            checks={
                component.name: "ok" if component.ready else "down"
                for component in report.components
            },
        )
