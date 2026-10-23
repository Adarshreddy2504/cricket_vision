"""Video probing and frame sampling shared by the API and the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoProbeError(RuntimeError):
    """The file cannot be opened and decoded as video."""


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    width: int
    height: int
    fps: float
    declared_frames: int


def probe_video(path: Path) -> VideoProbe:
    """Open the file, read one frame and report the source frame geometry."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoProbeError("The uploaded file is not a readable video.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoProbeError("The uploaded file is not a readable video.")
        height, width = frame.shape[:2]
        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if reported_width > 0 and reported_height > 0:
        width, height = reported_width, reported_height
    if width <= 0 or height <= 0:
        raise VideoProbeError("The uploaded file is not a readable video.")
    if not math.isfinite(fps) or fps < 0:
        fps = 0.0
    return VideoProbe(
        path=Path(path),
        width=int(width),
        height=int(height),
        fps=float(fps),
        declared_frames=max(0, declared),
    )


def sample_clip_frames(
    path: Path,
    offsets: list[int],
) -> dict[int, np.ndarray]:
    """Read selected 0-based positions from a clip, in one sequential pass."""
    wanted = sorted({max(0, int(offset)) for offset in offsets})
    if not wanted:
        return {}
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {}
        index = 0
        wanted_set = set(wanted)
        while wanted_set:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted_set:
                frames[index] = frame.copy()
                wanted_set.remove(index)
            index += 1
    finally:
        capture.release()
    return frames


def sample_source_frames(
    path: Path,
    frame_numbers: list[int],
) -> dict[int, np.ndarray]:
    """Read selected 1-based source frames, seeking only when it is worthwhile."""
    wanted = sorted({max(1, int(number)) for number in frame_numbers})
    if not wanted:
        return {}
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {}
        wanted_set = set(wanted)
        index = 1
        last_wanted = wanted[-1]
        while wanted_set and index <= last_wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted_set:
                frames[index] = frame.copy()
                wanted_set.remove(index)
            index += 1
    finally:
        capture.release()
    return frames


__all__ = [
    "VideoProbe",
    "VideoProbeError",
    "probe_video",
    "sample_clip_frames",
    "sample_source_frames",
]
