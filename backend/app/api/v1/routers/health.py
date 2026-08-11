"""Probe endpoints."""

from fastapi import APIRouter, Response, status

from app.api.dependencies import CheckReadinessDep
from app.api.v1.schemas.health import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    # Deliberately static, with no dependency checks: a liveness probe that
    # touches the database turns a two-second DB blip into every pod being
    # killed and restarted at once, which is how a blip becomes an outage.
    return {"status": "ok"}


@router.get(
    "/ready",
    # Declared so the 503 is in the schema rather than a surprise. The status
    # code itself is set below, because the body is the same either way.
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(check_readiness: CheckReadinessDep, response: Response) -> ReadinessResponse:
    """Report whether this process can actually serve a request.

    Unlike `/health` this one does touch its dependencies, and the difference is
    the whole point: liveness asks "is this process wedged?", readiness asks "is
    it worth sending traffic to?". Only the second may fail on a transient
    outage, because its remedy is to route around this instance rather than to
    restart it.

    The status code is the answer. A gate that has to parse the body to find out
    whether the deployment succeeded is a gate that eventually forgets to.
    """
    report = await check_readiness()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse.of(report)
