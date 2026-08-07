"""Probe endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    # Deliberately static, with no dependency checks: a liveness probe that
    # touches the database turns a two-second DB blip into every pod being
    # killed and restarted at once, which is how a blip becomes an outage.
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}
