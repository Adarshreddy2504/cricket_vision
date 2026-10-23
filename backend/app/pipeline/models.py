"""Process-wide lazy loading of the three optional model families.

Loading is deliberately separate from analysis. The ball detector is required,
so its failure becomes a loud `DetectorUnavailableError` on the first frame.
The pitch detectors and the shot classifier are optional: a missing or corrupt
checkpoint becomes an explicit unavailable result and a diagnostic, not a
fabricated label or an empty-looking success.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..pitch.config import (
    resolve_pitch_box_model_path,
    resolve_pitch_segmentation_model_path,
)
from ..pitch.detector import (
    DetectionStatus,
    PitchBoxDetector,
    PitchDetection,
    PitchSegmentationDetector,
)
from ..shots.inference import ShotClassifier
from ..tracking.detector import BallDetector

log = logging.getLogger(__name__)


class MissingPitchDetector:
    """A detector slot with no usable weights, so every call says why."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        self.available = False
        self.load_error = reason

    def detect(self, frame, *, frame_number: int = 0) -> PitchDetection:
        return PitchDetection(
            status=DetectionStatus.UNAVAILABLE,
            frame_number=int(frame_number),
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            reason=self.reason,
        )


class MissingBallDetector:
    """Only used if constructing the real detector raises unexpectedly."""

    def __init__(self, reason: str) -> None:
        self.available = False
        self.load_error = reason

    def detect_with_confidence(self, frame):  # pragma: no cover - defensive
        return None, None, None


class PipelineModels:
    """One lazily loaded instance of each shared model per server process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ball: Any | None = None
        self._ball_error: str | None = None
        self._shot: Any | None = None
        self._shot_error: str | None = None
        self._shot_available: bool | None = None
        self._pitch: list[tuple[str, Any]] | None = None

    # ── Ball ─────────────────────────────────────────────────────────────────
    def ball_detector(self):
        with self._lock:
            if self._ball is None:
                try:
                    self._ball = BallDetector()
                except Exception as exc:  # noqa: BLE001 - reported as unavailable
                    log.exception("could not construct the ball detector")
                    self._ball_error = str(exc) or type(exc).__name__
                    self._ball = MissingBallDetector(self._ball_error)
                else:
                    self._ball_error = getattr(self._ball, "load_error", None)
            return self._ball

    def ball_status(self) -> dict:
        detector = self.ball_detector()
        return {
            "name": "ball_detector",
            "available": bool(getattr(detector, "available", False)),
            "reason": getattr(detector, "load_error", None),
        }

    # ── Shot ─────────────────────────────────────────────────────────────────
    def shot_classifier(self):
        with self._lock:
            if self._shot is None:
                try:
                    self._shot = ShotClassifier.from_checkpoint()
                except Exception as exc:  # noqa: BLE001 - missing, corrupt, incompatible
                    log.warning("shot classifier unavailable: %s", exc)
                    self._shot_error = str(exc) or type(exc).__name__
                    self._shot = ShotClassifier.unavailable(self._shot_error)
                    self._shot_available = False
                else:
                    self._shot_error = None
                    self._shot_available = True
            return self._shot

    def shot_status(self) -> dict:
        self.shot_classifier()
        return {
            "name": "shot_classifier",
            "available": bool(self._shot_available),
            "reason": self._shot_error,
        }

    # ── Pitch ────────────────────────────────────────────────────────────────
    def pitch_detectors(self) -> list[tuple[str, Any]]:
        with self._lock:
            if self._pitch is None:
                self._pitch = [
                    self._make_pitch(
                        "pitch_segmentation",
                        resolve_pitch_segmentation_model_path,
                        PitchSegmentationDetector,
                    ),
                    self._make_pitch(
                        "pitch_box",
                        resolve_pitch_box_model_path,
                        PitchBoxDetector,
                    ),
                ]
            return list(self._pitch)

    def _make_pitch(self, name: str, resolver, detector_class) -> tuple[str, Any]:
        try:
            path = Path(resolver())
        except Exception as exc:  # noqa: BLE001 - absent weights are expected
            return name, MissingPitchDetector(name, str(exc) or type(exc).__name__)
        try:
            return name, detector_class(path)
        except Exception as exc:  # noqa: BLE001 - report, do not fail tracking
            log.warning("%s could not be constructed: %s", name, exc)
            return name, MissingPitchDetector(name, str(exc) or type(exc).__name__)

    def pitch_status(self) -> list[dict]:
        statuses = []
        for name, detector in self.pitch_detectors():
            statuses.append(
                {
                    "name": name,
                    "available": bool(getattr(detector, "available", False)),
                    "reason": getattr(detector, "load_error", None),
                }
            )
        return statuses

    def diagnostics(self) -> dict:
        return {
            "ball_detector": self.ball_status(),
            "pitch_detectors": self.pitch_status(),
            "shot_classifier": self.shot_status(),
        }


class UnavailableShotClassifier:
    """A classifier-shaped object that never invents a label."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def classify(self, *args, **kwargs):
        from ..schemas.shots import ShotClassification

        return ShotClassification(
            status="model_unavailable",
            error=self.reason,
            input_source="unavailable",
        )


def classifier_from_factory(factory) -> tuple[Any, str | None]:
    """Call an injected classifier factory and convert failure to a record."""
    try:
        classifier = factory()
    except Exception as exc:  # noqa: BLE001 - explicit unavailable state
        reason = str(exc) or type(exc).__name__
        return UnavailableShotClassifier(reason), reason
    return classifier, None


__all__ = [
    "MissingBallDetector",
    "MissingPitchDetector",
    "PipelineModels",
    "UnavailableShotClassifier",
    "classifier_from_factory",
]
