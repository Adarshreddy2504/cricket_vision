"""Deterministic model doubles for the real tracking/pitch/shot pipeline.

The production code is exercised in full. Only the three external model
boundaries are replaced: ball detection, pitch detection and shot inference.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.pitch.corners import PitchCorners
from app.pitch.detector import CornerMethod, DetectionStatus, PitchDetection
from app.schemas.shots import ShotClassification
from app.shots.constants import SHOT_CLASSES


class SequenceDetector:
    """A ball detector that returns a predetermined list of positions."""

    available = True
    load_error = None

    def __init__(self, positions):
        self._positions = list(positions)
        self._index = 0
        self.calls = 0

    def detect_with_confidence(self, frame):
        self.calls += 1
        if self._index >= len(self._positions):
            return None, None, None
        x, y, confidence = self._positions[self._index]
        self._index += 1
        return x, y, confidence


class NoneDetector:
    """A working detector that measures no ball on any frame."""

    available = True
    load_error = None

    def __init__(self):
        self.calls = 0

    def detect_with_confidence(self, frame):
        self.calls += 1
        return None, None, None


class FakePitchDetector:
    """A pitch detector returning one fixed outcome, labelled with a method."""

    def __init__(
        self,
        *,
        status: DetectionStatus = DetectionStatus.OK,
        method: CornerMethod | None = CornerMethod.SEGMENTATION,
        corners: PitchCorners | None = None,
        confidence: float | None = 0.93,
        reason: str | None = None,
    ):
        self.status = status
        self.method = method
        self.corners = corners if corners is not None else pitch_corners()
        self.confidence = confidence
        self.reason = reason
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def detect(self, frame, *, frame_number: int = 0) -> PitchDetection:
        self.calls += 1
        return PitchDetection(
            status=self.status,
            frame_number=frame_number,
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            method=self.method if self.status is DetectionStatus.OK else None,
            corners=self.corners if self.status is DetectionStatus.OK else None,
            confidence=self.confidence,
            reason=self.reason,
        )


class FakeShotClassifier:
    """A classifier that returns a fixed label and records what it received."""

    def __init__(self, label: str = "Cover", score: float = 0.91):
        self.label = label
        self.score = score
        self.calls = []

    def classify(
        self,
        clip_path=None,
        in_memory_frames=None,
        expected_total_frames=None,
        delivery_id=None,
        frame_offset=None,
    ) -> ShotClassification:
        self.calls.append(
            {
                "clip_path": clip_path,
                "in_memory_frames": in_memory_frames,
                "expected_total_frames": expected_total_frames,
                "delivery_id": delivery_id,
                "frame_offset": frame_offset,
            }
        )
        others = (1.0 - self.score) / max(1, len(SHOT_CLASSES) - 1)
        probabilities = {
            label: (self.score if label == self.label else others)
            for label in SHOT_CLASSES
        }
        return ShotClassification(
            status="ok",
            label=self.label,
            score=self.score,
            class_probabilities=probabilities,
            input_source="in_memory",
            frames_sampled=30,
            unique_frames_sampled=30,
            sampled_clip_frames=list(range(30)),
        )


def pitch_corners(
    *,
    left: float = 40.0,
    top: float = 5.0,
    right: float = 120.0,
    bottom: float = 115.0,
) -> PitchCorners:
    return PitchCorners(
        top_left=(left, top),
        top_right=(right, top),
        bottom_right=(right, bottom),
        bottom_left=(left, bottom),
    )


def ball_path(count: int = 40):
    """A continuous ball flight down the pitch, inside the 160x120 test frame."""
    return [
        (60 + index * 1.5, 20 + index * 2.5, 0.9)
        for index in range(count)
    ]


def as_detection_replacement(detection: PitchDetection, frame_number: int, frame) -> PitchDetection:
    """Small helper used by tests that need to re-label a detection's frame."""
    return replace(
        detection,
        frame_number=frame_number,
        image_size=(int(frame.shape[1]), int(frame.shape[0])),
    )


def blank_frame(width: int = 160, height: int = 120) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)
