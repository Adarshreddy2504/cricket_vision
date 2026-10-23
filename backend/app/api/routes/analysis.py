"""Upload, status and result routes from the six-route frontend contract."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..jobs import JobRegistry
from ..upload import (
    UploadRejected,
    place_source,
    remove_upload_dir,
    save_validated_upload,
)

router = APIRouter()


def _registry(request: Request) -> JobRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="The analysis service is not ready.")
    return registry


def _run_dir(request: Request, analysis_id: str) -> Path:
    return request.app.state.settings.runs_dir / analysis_id


async def _accept_upload(request: Request, upload: UploadFile) -> dict:
    settings = request.app.state.settings
    registry = _registry(request)
    temporary: Path | None = None
    analysis_id: str | None = None
    try:
        temporary, size, source_name, extension, _probe = await save_validated_upload(
            upload, settings
        )
        analysis_id = uuid.uuid4().hex
        job = registry.create(
            analysis_id=analysis_id,
            source_name=source_name,
            size=size,
        )
        source_path = place_source(
            analysis_id=analysis_id,
            temporary=temporary,
            extension=extension,
            settings=settings,
        )
        temporary = None
        registry.start(job, source_path)
        return {
            "analysis_id": analysis_id,
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage,
            "bytes": size,
            "status_url": f"/api/analysis/{analysis_id}/status",
            "result_url": f"/api/analysis/{analysis_id}/result",
        }
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        if analysis_id is not None:
            registry.discard(analysis_id)
            shutil.rmtree(_run_dir(request, analysis_id), ignore_errors=True)
            remove_upload_dir(analysis_id, settings)
        raise HTTPException(
            status_code=500,
            detail=f"The analysis could not be started: {exc}",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@router.post("/analyze", status_code=202)
async def analyze(
    request: Request,
    video: Annotated[UploadFile, File()],
) -> dict:
    return await _accept_upload(request, video)


@router.post("/upload", status_code=202)
async def upload(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> dict:
    return await _accept_upload(request, file)


@router.get("/analysis/{analysis_id}/status")
def analysis_status(request: Request, analysis_id: str) -> dict:
    job = _registry(request).get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return job.as_dict()


@router.post("/analysis/{analysis_id}/cancel", status_code=202)
def cancel_analysis(request: Request, analysis_id: str) -> dict:
    """Request cooperative cancellation and return the current status.

    The worker marks the job failed with an explicit "cancelled" reason at its
    next progress check. A completed job is returned unchanged.
    """
    job = _registry(request).cancel(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return job.as_dict()


@router.get("/analysis/{analysis_id}/result")
def analysis_result(request: Request, analysis_id: str) -> JSONResponse:
    registry = _registry(request)
    job = registry.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if job.status in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="The analysis is still running; poll its status document.",
        )
    if job.status == "failed":
        raise HTTPException(
            status_code=409,
            detail="The analysis failed; read its status document for the reason.",
        )

    result_path = _run_dir(request, analysis_id) / "result.json"
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="The result document is unavailable.",
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The result document is unreadable.",
        ) from exc

    if not isinstance(document, dict) or document.get("analysis_id") != analysis_id:
        raise HTTPException(
            status_code=500,
            detail="The result document does not match this analysis.",
        )
    return JSONResponse(content=document)


__all__ = ["cancel_analysis", "router"]
