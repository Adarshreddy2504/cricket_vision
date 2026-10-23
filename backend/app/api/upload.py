"""Upload validation and safe placement of the temporary source video.

A rejected upload must leave no analysis directory and no half-written file.
The file is streamed to `data/tmp` first, checked with OpenCV, and only then
given an analysis id. That ordering is what makes the cleanup guarantee true.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from ..pipeline.video import VideoProbe, VideoProbeError, probe_video
from .config import Settings

ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadRejected(Exception):
    """A client-correctable upload failure with an explicit HTTP status."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def sanitise_name(filename: str | None) -> str:
    name = Path(filename or "upload").name
    return name or "upload"


async def save_validated_upload(
    upload: UploadFile,
    settings: Settings,
) -> tuple[Path, int, str, str, VideoProbe]:
    """
    Stream, size-check and probe an upload.

    Returns `(temporary_path, size, source_name, extension, probe)`. The caller
    owns the temporary file and must move or delete it. Every failure removes
    the partial file before raising.
    """
    source_name = sanitise_name(upload.filename)
    extension = Path(source_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            "Unsupported video format. Use mp4, avi, mov, mkv or webm."
        )

    temporary = settings.temp_dir / f"{uuid.uuid4().hex}{extension}"
    size = 0
    try:
        with temporary.open("wb") as handle:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    limit_gib = settings.max_upload_bytes / (1024 ** 3)
                    raise UploadRejected(
                        "The uploaded video is larger than the configured "
                        f"limit ({limit_gib:.2f} GiB).",
                        status_code=413,
                    )
                handle.write(chunk)
        if size == 0:
            raise UploadRejected("The uploaded video is empty.")
        try:
            probe = probe_video(temporary)
        except VideoProbeError as exc:
            raise UploadRejected(str(exc)) from exc
        return temporary, size, source_name, extension, probe
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def place_source(
    *,
    analysis_id: str,
    temporary: Path,
    extension: str,
    settings: Settings,
) -> Path:
    upload_dir = settings.uploads_dir / analysis_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    destination = upload_dir / f"source{extension}"
    shutil.move(str(temporary), str(destination))
    return destination


def remove_upload_dir(analysis_id: str, settings: Settings) -> None:
    directory = settings.uploads_dir / analysis_id
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


__all__ = [
    "ALLOWED_EXTENSIONS",
    "UPLOAD_CHUNK_BYTES",
    "UploadRejected",
    "VideoProbe",
    "place_source",
    "probe_video",
    "remove_upload_dir",
    "sanitise_name",
    "save_validated_upload",
]
