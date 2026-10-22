"""
shots — The cricket shot classifier and the records it produces.

The package is import-light on purpose. Importing `app.shots` must not import
torch, torchvision or OpenCV, because `app.schemas.shots` and the analytics
modules need the class vocabulary and nothing else. The heavy modules are
loaded on first attribute access:

    from app.shots import ShotClassifier      # loads inference, model, torch
    from app.shots.constants import SHOT_CLASSES   # stays light

This also avoids a circular import: `app.schemas.shots` imports the vocabulary
from `app.shots.constants`, and `app.shots.inference` imports the schema.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .config import ShotConfig
    from .errors import ShotError, ShotInferenceError, ShotInputError, ShotModelUnavailable
    from .inference import LoadedShotModel, ShotClassifier, load_shot_model, predict_shot
    from .model import ImprovedSOTAModel
    from .preprocessing import frames_to_tensor, sample_clip_frames, sample_indices

_EXPORTS: dict[str, str] = {
    "ImprovedSOTAModel": ".model",
    "LoadedShotModel": ".inference",
    "ShotClassifier": ".inference",
    "ShotConfig": ".config",
    "ShotError": ".errors",
    "ShotInferenceError": ".errors",
    "ShotInputError": ".errors",
    "ShotModelUnavailable": ".errors",
    "frames_to_tensor": ".preprocessing",
    "load_shot_model": ".inference",
    "predict_shot": ".inference",
    "sample_clip_frames": ".preprocessing",
    "sample_indices": ".preprocessing",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "ImprovedSOTAModel",
    "LoadedShotModel",
    "ShotClassifier",
    "ShotConfig",
    "ShotError",
    "ShotInferenceError",
    "ShotInputError",
    "ShotModelUnavailable",
    "frames_to_tensor",
    "load_shot_model",
    "predict_shot",
    "sample_clip_frames",
    "sample_indices",
]
