"""Wire shapes for the probe endpoints."""

from pydantic import BaseModel

from app.application.health.dto import ReadinessReport


class ReadinessResponse(BaseModel):
    """What a deploy gate reads. `checks` names the failing dependency.

    Values are only "ok" or "down", because the endpoint is public.
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
