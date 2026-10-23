"""Delivery clip bytes, including HTTP byte-range support for <video>."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()

_CLIP_NAME = re.compile(r"^delivery_\d{3,}\.(?:webm|mp4)$")
_MEDIA_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
}


def _clip_path(request: Request, analysis_id: str, name: str) -> Path:
    if not _CLIP_NAME.fullmatch(name) or Path(name).name != name:
        raise HTTPException(status_code=404, detail="Clip not found.")
    registry = getattr(request.app.state, "registry", None)
    job = registry.get(analysis_id) if registry is not None else None
    if job is None or job.status != "completed":
        raise HTTPException(status_code=404, detail="Clip not found.")
    path = request.app.state.settings.runs_dir / analysis_id / "clips" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return path


@router.get("/analysis/{analysis_id}/clips/{name}")
def clip(request: Request, analysis_id: str, name: str) -> FileResponse:
    path = _clip_path(request, analysis_id, name)
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES[path.suffix.lower()],
        headers={"Cache-Control": "private, max-age=3600"},
    )


__all__ = ["router"]
