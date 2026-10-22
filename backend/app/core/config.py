"""
config.py — Runtime configuration and filesystem layout for the tracking module.

Everything is derived from one root so the module never depends on where it
was started from, and every tunable can be overridden from the environment
without editing code.

THE MODEL WEIGHTS ARE EXTERNAL
------------------------------
The detector's YOLO checkpoint is NOT tracked in git (see `.gitignore` and
`docs/TRACKING.md`). Two environment variables point at it:

    CRICKET_MODELS_DIR   the directory holding the weights
    CRICKET_BALL_MODEL   one weights file, overriding the directory

`resolve_ball_model_path()` is the single place that turns those into a path,
and it names every path it tried when the weights are missing — a missing
checkpoint must fail loudly at startup rather than surfacing much later as
"the detector found nothing".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_MAX_DIAG_PX,
    DETECTOR_MIN_DIAG_PX,
    DETECTOR_RESIZE_H,
    DETECTOR_RESIZE_W,
    GATE_RADIUS,
    MAX_MISSED,
    MAX_TRAIL,
    MIN_TRACK_DETECTIONS,
)

# backend/app/core/config.py -> backend/app/core -> backend/app -> backend
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv() -> None:
    """
    Read `<project>/.env` if it exists, never overriding the real environment.

    `override=False` means a shell-exported variable always wins, which is
    what lets CI and the tests set these values directly. Import failures are
    swallowed: `python-dotenv` is a convenience, and a missing package must
    not stop the module from importing.
    """
    env_file = PROJECT_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        load_dotenv(env_file, override=False)
    except Exception:
        return


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw is not None and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    """
    Resolve a directory from the environment, falling back to the repo default.

    A path may be pointed elsewhere, but a relative path is always resolved
    against the project directory so the module started from a different
    working directory still finds the same files.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def _env_file(name: str, default: Path) -> Path:
    """Same idea as `_env_path`, for an individual model file."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


MODELS_DIR = _env_path("CRICKET_MODELS_DIR", BACKEND_DIR / "models")

# The checkpoint's name inside `MODELS_DIR`. The file itself is not tracked in
# git — see docs/TRACKING.md for how to supply it.
BALL_MODEL_FILENAME = "ball_detector.pt"

BALL_MODEL_PATH = _env_file(
    "CRICKET_BALL_MODEL", MODELS_DIR / BALL_MODEL_FILENAME
)


def resolve_ball_model_path(models_dir: Path | None = None) -> Path:
    """
    Locate the ball detector's weights.

    Raises FileNotFoundError naming every path tried, rather than letting the
    detector fail later with a generic "found nothing" that looks like an
    empty frame.
    """
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    candidates = [base / BALL_MODEL_FILENAME, Path(BALL_MODEL_PATH)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Ball detector weights not found. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nSet CRICKET_BALL_MODEL to the weights file, or CRICKET_MODELS_DIR "
        "to the directory holding it. See docs/TRACKING.md."
    )


@dataclass(frozen=True)
class TrackingConfig:
    """
    Settings for the ball-tracking module.

    Environment-resolved ONCE at import (via `from_env` below), exactly like
    the rest of this module. Frozen, so a caller can build a modified copy
    with `dataclasses.replace` without inventing a new configuration object.

    `model_path` is the detector's YOLO checkpoint. It is NOT bundled with the
    source; see the module docstring.
    """

    model_path: Path = field(default_factory=lambda: Path(BALL_MODEL_PATH))
    # Detector
    conf_threshold: float = DETECTOR_CONF_THRESHOLD
    resize_width: int = DETECTOR_RESIZE_W
    resize_height: int = DETECTOR_RESIZE_H
    min_diag_px: int = DETECTOR_MIN_DIAG_PX
    max_diag_px: int = DETECTOR_MAX_DIAG_PX
    # "auto" picks CUDA when it is available, CPU otherwise. Any other value is
    # passed to ultralytics verbatim.
    device: str = "auto"
    # Tracker
    gate_radius_px: int = GATE_RADIUS
    max_missed_frames: int = MAX_MISSED
    max_trail_points: int = MAX_TRAIL
    # Service
    min_track_detections: int = MIN_TRACK_DETECTIONS

    @classmethod
    def from_env(cls) -> TrackingConfig:
        """Build a configuration from the environment."""
        return cls(
            model_path=Path(BALL_MODEL_PATH),
            conf_threshold=_env_float(
                "CRICKET_BALL_CONF", DETECTOR_CONF_THRESHOLD
            ),
            resize_width=max(32, _env_int(
                "CRICKET_BALL_RESIZE_W", DETECTOR_RESIZE_W
            )),
            resize_height=max(32, _env_int(
                "CRICKET_BALL_RESIZE_H", DETECTOR_RESIZE_H
            )),
            min_diag_px=_env_int(
                "CRICKET_BALL_MIN_DIAG", DETECTOR_MIN_DIAG_PX
            ),
            max_diag_px=_env_int(
                "CRICKET_BALL_MAX_DIAG", DETECTOR_MAX_DIAG_PX
            ),
            device=_env_str("CRICKET_BALL_DEVICE", "auto"),
            gate_radius_px=max(1, _env_int(
                "CRICKET_BALL_GATE_RADIUS", GATE_RADIUS
            )),
            max_missed_frames=max(0, _env_int(
                "CRICKET_BALL_MAX_MISSED", MAX_MISSED
            )),
            max_trail_points=max(1, _env_int(
                "CRICKET_BALL_MAX_TRAIL", MAX_TRAIL
            )),
            min_track_detections=max(1, _env_int(
                "CRICKET_BALL_MIN_TRACK_DETECTIONS", MIN_TRACK_DETECTIONS
            )),
        )

    def as_metadata(self) -> dict:
        """Reproducibility record: configuration only, never a credential."""
        return {
            "model_path": str(self.model_path),
            "model_found": self.model_path.is_file(),
            "conf_threshold": self.conf_threshold,
            "resize_width": self.resize_width,
            "resize_height": self.resize_height,
            "min_diag_px": self.min_diag_px,
            "max_diag_px": self.max_diag_px,
            "device": self.device,
            "gate_radius_px": self.gate_radius_px,
            "max_missed_frames": self.max_missed_frames,
            "max_trail_points": self.max_trail_points,
            "min_track_detections": self.min_track_detections,
        }


TRACKING_CONFIG = TrackingConfig.from_env()


__all__ = [
    "BACKEND_DIR",
    "BALL_MODEL_FILENAME",
    "BALL_MODEL_PATH",
    "MODELS_DIR",
    "PROJECT_DIR",
    "TRACKING_CONFIG",
    "TrackingConfig",
    "resolve_ball_model_path",
]
