"""
shots_helpers.py — Fixtures shared by the Part 3 unit tests.

The point of these helpers is to make the tests depend on VALUES rather than on
a 68 MB checkpoint or a real cricket video. A tiny stand-in model exercises the
loader and the forward pass; a tiny written clip exercises the clip path; plain
dataclasses stand in for Part 1's and Part 2's records.

Nothing here skips. If a codec is unavailable the helper raises, because a
silently skipped clip test would hide a broken decode path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from app.shots.constants import SHOT_CLASSES, SHOT_NUM_CLASSES

# Small frames: the transform resizes to 224x224 regardless, and a 48x64 frame
# keeps the tests fast. Values are deterministic so tensor assertions are stable.
FRAME_HEIGHT = 48
FRAME_WIDTH = 64


def bgr_frame(
    value: int = 0,
    height: int = 32,
    width: int = 24,
) -> np.ndarray:
    """A uniform uint8 BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def clip_frames(count: int = 30, height: int = 32, width: int = 24) -> list[np.ndarray]:
    """A deterministic list of BGR frames, each a slightly different shade."""
    return [bgr_frame((index * 7) % 256, height, width) for index in range(count)]


class StubShotModel(nn.Module):
    """
    A callable stand-in for `ImprovedSOTAModel`.

    `logits` may be a 1-D or 2-D tensor. When omitted, the stub returns zeros,
    which softmax turns into a uniform distribution — still a valid 10-way
    prediction, so the schema's checks are exercised rather than bypassed.
    """

    def __init__(self, logits: object = None, *, fail: bool = False) -> None:
        super().__init__()
        self.register_buffer("probe", torch.zeros(1))
        self._logits = logits
        self.fail = fail
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.fail:
            raise RuntimeError("stub forward failure")
        if self._logits is not None:
            logits = torch.as_tensor(self._logits)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            return logits
        return torch.zeros(x.shape[0], SHOT_NUM_CLASSES)


class TinyModel(nn.Module):
    """A tiny module whose state dict is cheap to save and strictly reloadable."""

    def __init__(
        self,
        num_classes: int = SHOT_NUM_CLASSES,
        temporal_dim: int = 8,
        n_frames: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temporal_dim = temporal_dim
        self.n_frames = n_frames
        self.proj = nn.Linear(3, num_classes)
        self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(1, 3, 4))
        return self.proj(pooled)


def tiny_factory(**kwargs: object) -> TinyModel:
    """A `model_factory` for `load_shot_model` that honours `num_classes`."""
    num_classes = int(kwargs.get("num_classes", SHOT_NUM_CLASSES))
    return TinyModel(num_classes=num_classes)


def write_tiny_checkpoint(
    path: Path,
    *,
    prefix: str = "",
    wrapper: bool = True,
    drop: tuple[str, ...] = (),
) -> Path:
    """
    Save a TinyModel state dict, optionally wrapped and/or key-prefixed.

    `class_weights` is deliberately omitted from most fixtures: production
    checkpoints carry a training copy that the loader must not restore.
    """
    source = TinyModel()
    state = {
        f"{prefix}{key}": value.detach().clone()
        for key, value in source.state_dict().items()
        if key != "class_weights" and key not in drop
    }
    payload = {"state_dict": state, "epoch": 0, "note": "fixture"} if wrapper else state
    torch.save(payload, path)
    return path


def write_non_checkpoint(path: Path) -> Path:
    """A file that is readable but is not a torch checkpoint."""
    path.write_bytes(b"this is not a checkpoint")
    return path


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_clip(
    directory: Path,
    *,
    frame_count: int = 30,
    fps: float = 25.0,
    size: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
) -> Path:
    """
    Write a real clip with a real codec and return its path.

    Tries MJPG, XVID and mp4v in turn. A helper that raised `SkipTest` here
    would let a broken clip path pass CI unnoticed, so it raises instead.
    """
    directory.mkdir(parents=True, exist_ok=True)
    width, height = size

    for codec in ("MJPG", "XVID", "mp4v"):
        suffix = ".mp4" if codec == "mp4v" else ".avi"
        candidate = directory / f"clip_{codec}{suffix}"
        writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            continue

        for index in range(frame_count):
            frame = np.full((height, width, 3), (index * 7) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        capture = cv2.VideoCapture(str(candidate))
        try:
            reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        if reported == frame_count:
            return candidate
        candidate.unlink(missing_ok=True)

    raise RuntimeError("no usable video writer codec (tried MJPG, XVID, mp4v)")


# ── Stand-ins for Part 1 and Part 2 records ──────────────────────────────────


@dataclass
class Point:
    frame: int
    x: float
    y: float
    source: str = "detected"
    confidence: float | None = 0.9


@dataclass
class Trajectory:
    points: list[Point] = field(default_factory=list)


@dataclass
class PitchPoint:
    frame: int
    across: float
    along: float
    source: str = "detected"
    units: str = "metre"
    inside_pitch: bool = True


@dataclass
class PitchTrajectory:
    points: list[PitchPoint] = field(default_factory=list)
    units: str = "metre"


def shot_ok(
    label: str = "Cover",
    score: float = 0.7,
    **overrides: object,
):
    """A valid successful `ShotClassification` with a complete distribution."""
    from app.schemas.shots import ShotClassification

    others = [name for name in SHOT_CLASSES if name != label]
    remaining = (1.0 - score) / len(others)
    probabilities = {name: remaining for name in others}
    probabilities[label] = score

    fields = {
        "status": "ok",
        "label": label,
        "score": score,
        "class_probabilities": probabilities,
        "input_source": "in_memory",
        "frames_sampled": 30,
        "unique_frames_sampled": 30,
        "frames_repeated": False,
        "sampled_clip_frames": list(range(30)),
    }
    fields.update(overrides)
    return ShotClassification(**fields)


def shot_failed(
    status: str = "model_unavailable",
    error: str = "test reason",
    **overrides: object,
):
    """A valid failed `ShotClassification`."""
    from app.schemas.shots import ShotClassification

    fields = {"status": status, "error": error}
    fields.update(overrides)
    return ShotClassification(**fields)
