"""Probe endpoints."""

from fastapi import APIRouter, Response, status

from app.api.dependencies import CheckReadinessDep
from app.api.v1.schemas.health import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    # Static on purpose. Checking the database here would restart every pod on a blip.
    return {"status": "ok"}


@router.get(
    "/ready",
    # Documents the 503. The body is the same either way.
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(check_readiness: CheckReadinessDep, response: Response) -> ReadinessResponse:
    """Whether this process can serve traffic. Checks dependencies, unlike `/health`.

    The status code is the answer: 200 or 503.
    """

    report = await check_readiness()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse.of(report)
