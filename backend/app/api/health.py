"""Health metadata that never loads a model or touches the network."""

from __future__ import annotations

from pathlib import Path

from ..core.config import resolve_ball_model_path
from ..pitch.config import (
    resolve_pitch_box_model_path,
    resolve_pitch_segmentation_model_path,
)
from ..shots.config import ShotConfig


def _status(resolver, unavailable: str) -> dict:
    try:
        path = Path(resolver())
    except FileNotFoundError as exc:
        return {"available": False, "reason": str(exc)}
    if not path.is_file():
        return {"available": False, "reason": unavailable}
    return {"available": True, "reason": None}


def model_availability() -> dict:
    """What the server can actually load, without importing torch.

    The response deliberately omits absolute paths: a health check does not
    need to disclose the host's filesystem layout. The reason string describes
    the missing filename or the resolver's message.
    """
    shot = ShotConfig.from_env().as_metadata()
    return {
        "ball_detector": _status(
            resolve_ball_model_path,
            "ball detector weights not found",
        ),
        "pitch_segmentation": _status(
            resolve_pitch_segmentation_model_path,
            "pitch segmentation weights not found",
        ),
        "pitch_box": _status(
            resolve_pitch_box_model_path,
            "pitch box weights not found",
        ),
        "shot_classifier": {
            "available": bool(shot.get("model_available")),
            "reason": shot.get("model_error"),
            "md5": shot.get("expected_md5") or shot.get("documented_md5"),
        },
    }


__all__ = ["model_availability"]
