"""`GET /api/health` — versions, stages and model availability."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..constants import PIPELINE_VERSION, SCHEMA_VERSION, STAGES
from ..health import model_availability

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    registry = getattr(request.app.state, "registry", None)
    tracked = len(registry.jobs) if registry is not None else 0
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "analyses_tracked_in_memory": tracked,
        "stages": list(STAGES),
        "models": model_availability(),
    }


__all__ = ["STAGES", "router"]
