"""
config.py — Where the pitch detector's weights live, and how they are tuned.

THE WEIGHTS ARE EXTERNAL
------------------------
The pitch checkpoints are NOT tracked in git (see `.gitignore` and
`docs/PITCH.md`). Three environment variables point at them:

    CRICKET_MODELS_DIR       the directory holding the weights
    CRICKET_PITCH_BOX_MODEL  the pitch-BOX detector's weights file
    CRICKET_PITCH_SEG_MODEL  the pitch-SEGMENTATION detector's weights file

Both default to a file inside `backend/models/`, so dropping the weights
there is enough; the environment variables exist for the case where they
live somewhere else.

A bare file name is always resolved against the project directory, so the
module finds the same files however it was started.

MISSING WEIGHTS ARE NAMED, NOT GUESSED AT
------------------------------------------
`resolve_pitch_box_model_path()` and `resolve_pitch_segmentation_model_path()`
raise `FileNotFoundError` naming every path they tried. The detectors
themselves never raise for a missing file: they answer with an explicit
"unavailable" state, because a caller that is told "no pitch found" would
quietly record an empty analysis instead of a missing model.

The module never downloads a checkpoint. `ultralytics.YOLO("some-name.pt")`
WILL fetch the file from the internet when it is not present, so the
detectors check that the weights exist as a file BEFORE they build a
model. See `app/pitch/detector.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_IMGSZ,
    MASK_MAX_AREA_FRACTION,
    MASK_MAX_ASPECT_RATIO,
    MASK_MIN_AREA_FRACTION,
    MASK_MIN_ASPECT_RATIO,
)

# backend/app/pitch/config.py -> backend/app/pitch -> backend/app -> backend
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv() -> None:
    """
    Read `<project>/.env` if it exists, never overriding the real environment.

    `override=False` means a shell-exported variable always wins, which is
    what lets CI and the tests set these values directly. Import failures
    are swallowed: `python-dotenv` is a convenience, and a missing package
    must not stop the module from importing.
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
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _env_file(name: str, default: Path) -> Path:
    """Same idea as `_env_path`, for an individual model file."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


MODELS_DIR = _env_path("CRICKET_MODELS_DIR", BACKEND_DIR / "models")

# The checkpoints' names inside `MODELS_DIR`. The files themselves are not
# tracked in git — see docs/PITCH.md for how to supply them.
PITCH_BOX_MODEL_FILENAME = "pitch_detect.pt"
PITCH_SEG_MODEL_FILENAME = "pitch_seg_yolo11s.pt"

PITCH_BOX_MODEL_PATH = _env_file(
    "CRICKET_PITCH_BOX_MODEL", MODELS_DIR / PITCH_BOX_MODEL_FILENAME
)
PITCH_SEG_MODEL_PATH = _env_file(
    "CRICKET_PITCH_SEG_MODEL", MODELS_DIR / PITCH_SEG_MODEL_FILENAME
)


def resolve_pitch_box_model_path(models_dir: Path | None = None) -> Path:
    """
    Locate the pitch-BOX detector's weights.

    Raises `FileNotFoundError` naming every path tried, rather than letting
    the detector fail later with a generic "found nothing" that looks like
    an empty frame.
    """
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    candidates = [base / PITCH_BOX_MODEL_FILENAME, Path(PITCH_BOX_MODEL_PATH)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Pitch-box detector weights not found. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nSet CRICKET_PITCH_BOX_MODEL to the weights file, or "
        "CRICKET_MODELS_DIR to the directory holding it. See docs/PITCH.md."
    )


def resolve_pitch_segmentation_model_path(models_dir: Path | None = None) -> Path:
    """
    Locate the pitch-SEGMENTATION detector's weights.

    Raises `FileNotFoundError` naming every path tried, for the same reason
    as the box detector's resolver.
    """
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    candidates = [base / PITCH_SEG_MODEL_FILENAME, Path(PITCH_SEG_MODEL_PATH)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Pitch-segmentation detector weights not found. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nSet CRICKET_PITCH_SEG_MODEL to the weights file, or "
        "CRICKET_MODELS_DIR to the directory holding it. See docs/PITCH.md."
    )


def describe_weights(path: Path) -> str:
    """The startup line for one weights file: present and how big, or not."""
    path = Path(path)
    if not path.is_file():
        return f"NOT FOUND at {path}"
    size = path.stat().st_size
    return f"{path} ({size / 1048576:.1f} MiB)"


@dataclass(frozen=True)
class PitchConfig:
    """
    Settings for the pitch detectors.

    Resolved ONCE at import (via `from_env` below), exactly like the rest
    of the module. Frozen, so a caller can build a modified copy with
    `dataclasses.replace` without inventing a new configuration object.

    The two `model_path`s are the YOLO checkpoints. They are NOT bundled
    with the source; see the module docstring.
    """

    box_model_path: Path = Path(PITCH_BOX_MODEL_PATH)
    segmentation_model_path: Path = Path(PITCH_SEG_MODEL_PATH)
    conf_threshold: float = DETECTOR_CONF_THRESHOLD
    imgsz: int = DETECTOR_IMGSZ
    # "auto" picks CUDA when it is available, CPU otherwise. Any other value
    # is passed to ultralytics verbatim.
    device: str = "auto"
    # Shape-plausibility bounds for a detected pitch mask.
    min_area_fraction: float = MASK_MIN_AREA_FRACTION
    max_area_fraction: float = MASK_MAX_AREA_FRACTION
    min_aspect_ratio: float = MASK_MIN_ASPECT_RATIO
    max_aspect_ratio: float = MASK_MAX_ASPECT_RATIO

    @classmethod
    def from_env(cls) -> PitchConfig:
        """Build a configuration from the environment."""
        return cls(
            box_model_path=Path(PITCH_BOX_MODEL_PATH),
            segmentation_model_path=Path(PITCH_SEG_MODEL_PATH),
            conf_threshold=_env_float(
                "CRICKET_PITCH_CONF", DETECTOR_CONF_THRESHOLD
            ),
            imgsz=max(32, _env_int("CRICKET_PITCH_IMGSZ", DETECTOR_IMGSZ)),
            device=_env_str("CRICKET_PITCH_DEVICE", "auto"),
            min_area_fraction=_env_float(
                "CRICKET_PITCH_MIN_AREA_FRACTION", MASK_MIN_AREA_FRACTION
            ),
            max_area_fraction=_env_float(
                "CRICKET_PITCH_MAX_AREA_FRACTION", MASK_MAX_AREA_FRACTION
            ),
            min_aspect_ratio=_env_float(
                "CRICKET_PITCH_MIN_ASPECT_RATIO", MASK_MIN_ASPECT_RATIO
            ),
            max_aspect_ratio=_env_float(
                "CRICKET_PITCH_MAX_ASPECT_RATIO", MASK_MAX_ASPECT_RATIO
            ),
        )

    def as_metadata(self) -> dict:
        """Reproducibility record: configuration only, never a credential."""
        return {
            "box_model_path": str(self.box_model_path),
            "box_model_found": self.box_model_path.is_file(),
            "segmentation_model_path": str(self.segmentation_model_path),
            "segmentation_model_found": self.segmentation_model_path.is_file(),
            "conf_threshold": self.conf_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "min_area_fraction": self.min_area_fraction,
            "max_area_fraction": self.max_area_fraction,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
        }


PITCH_CONFIG = PitchConfig.from_env()


__all__ = [
    "BACKEND_DIR",
    "MODELS_DIR",
    "PITCH_BOX_MODEL_FILENAME",
    "PITCH_BOX_MODEL_PATH",
    "PITCH_CONFIG",
    "PITCH_SEG_MODEL_FILENAME",
    "PITCH_SEG_MODEL_PATH",
    "PROJECT_DIR",
    "PitchConfig",
    "describe_weights",
    "resolve_pitch_box_model_path",
    "resolve_pitch_segmentation_model_path",
]
