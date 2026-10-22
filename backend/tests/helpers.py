"""
Shared test helpers: synthetic frames and stub models.

The stubs exist so the tracking module can be tested deterministically and
without the YOLO weights, which are deliberately NOT tracked in git (see
docs/TRACKING.md). They are not a shortcut around the real detector: the stub
model implements the exact `ultralytics.YOLO` contract the detector uses
(`predict(source=..., conf=..., verbose=..., device=...)` returning results
whose `boxes.data` is an `n x 6` array of `x1, y1, x2, y2, score, cls`), so the
selection, filtering and coordinate-scaling logic under test is the real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def make_frame(width: int = 1280, height: int = 720, value: int = 30) -> np.ndarray:
    """A flat BGR frame of the requested size."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_video(
    path: Path,
    n_frames: int = 12,
    width: int = 320,
    height: int = 180,
    fps: float = 25.0,
    background: int = 30,
) -> Path:
    """Write a synthetic video of flat frames, for the video-decode path."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    for _ in range(n_frames):
        writer.write(np.full((height, width, 3), background, dtype=np.uint8))
    writer.release()
    return path


class StubBoxes:
    """The `result.boxes` object ultralytics hands back."""

    def __init__(self, rows: Sequence[Sequence[float]]):
        self.data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)


class StubResult:
    """One element of the list `model.predict()` returns."""

    def __init__(self, rows: Optional[Sequence[Sequence[float]]]):
        self.boxes = None if rows is None else StubBoxes(rows)


class StubYoloModel:
    """
    Deterministic stand-in for `ultralytics.YOLO`.

    `frames` is a list of box-lists, one per `predict()` call. Each entry is
    either None (no boxes at all) or a list of `x1, y1, x2, y2, score, cls`
    rows in the coordinates of the frame the model was handed.
    """

    def __init__(self, frames: Optional[Iterable[Optional[Sequence[Sequence[float]]]]] = None):
        self._frames = list(frames or [])
        self.calls: list[dict] = []

    def predict(self, source=None, conf=None, verbose=None, device=None):
        self.calls.append(
            {"source": source, "conf": conf, "verbose": verbose, "device": device}
        )
        index = len(self.calls) - 1
        if index < len(self._frames):
            rows = self._frames[index]
        else:
            rows = []
        return [StubResult(rows)]


class StubDetector:
    """
    Deterministic stand-in for `BallDetector`.

    Emits a position for the frames inside each configured window and None
    elsewhere, so tracking, provenance and occlusion handling can be asserted
    exactly. `windows` are (start_call, end_call) pairs, 1-based and inclusive.
    """

    def __init__(
        self,
        windows: Optional[Sequence[tuple[int, int]]] = None,
        confidence: float = 0.9,
        model_path: str = "stub",
    ) -> None:
        self.model = object()          # the service only checks this is not None
        self.model_path = model_path
        self.device = "stub"
        self.confidence = confidence
        self.windows = list(windows or [])
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def detect(self, frame):
        cx, cy, _conf = self.detect_with_confidence(frame)
        return None if cx is None else (cx, cy)

    def detect_with_confidence(self, frame):
        self.calls += 1
        for start, end in self.windows:
            if start <= self.calls <= end:
                return (100 + self.calls, 200 + self.calls, self.confidence)
        return None, None, None


class SequenceDetector:
    """
    A detector that returns an exact, scripted sequence of positions.

    Each entry is either an `(x, y)` position or None for "no detection this
    frame". The last entry repeats once the script runs out, so a test can
    script a prefix and leave the tail empty.
    """

    def __init__(self, positions: Sequence[Optional[tuple[int, int]]], confidence: float = 0.9):
        self.model = object()
        self.model_path = "sequence"
        self.device = "stub"
        self.confidence = confidence
        self.positions = list(positions)
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def detect(self, frame):
        cx, cy, _ = self.detect_with_confidence(frame)
        return None if cx is None else (cx, cy)

    def detect_with_confidence(self, frame):
        self.calls += 1
        index = min(self.calls, len(self.positions)) - 1
        position = self.positions[index]
        if position is None:
            return None, None, None
        return position[0], position[1], self.confidence
