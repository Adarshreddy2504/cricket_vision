"""
tracking_service.py — The public entry point of the tracking module.

It turns a stream of frames into a trajectory with truthful provenance and
stable track identifiers, and it fails loudly on bad input rather than
returning a plausible-looking but wrong answer.

THE CONTRACT
------------
For every frame the service returns a `FrameTracking` that says:

    position    where the ball is, in the SAME pixel space as the frame that
                was handed in, or None if the ball is lost.
    source      "detected", "predicted" or None — the same vocabulary as the
                tracker, and it means exactly the same thing.
    confidence  the detector's confidence, on detected points only. A
                predicted point has no confidence, because no measurement
                was made.
    track_id    which flight this point belongs to. Stable while the ball is
                being followed — including across a short occlusion — and
                different once the ball has genuinely been lost and
                re-acquired.
    detection   the raw detector output for this frame, before the filter
                touched it. Kept so a consumer can compare the measurement
                with the smoothed position.

WHAT IS NOT IN HERE
-------------------
No pitch detection, no shot classification, no analytics, no overlay
rendering, no clip writing. This module answers one question — where the ball
went — and the rest of the system is built on top of that answer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from ..core.config import TRACKING_CONFIG, TrackingConfig
from ..core.constants import MIN_TRACK_DETECTIONS, TRACKING_SCHEMA_VERSION
from ..schemas.tracking import Trajectory, TrajectoryPoint
from .detector import BallDetector
from .errors import (
    DetectorUnavailableError,
    InvalidFrameError,
    TrackingError,
    VideoOpenError,
)
from .frames import validate_frame
from .kalman_tracker import BallTracker

log = logging.getLogger(__name__)


@dataclass
class FrameTracking:
    """
    What the service knows about one frame.

    `position` is None exactly when the ball is lost — and then `source` and
    `track_id` are None too. No point is invented to bridge a hole, because a
    hole in the trajectory is information.
    """

    frame: int
    position: tuple | None = None
    source: str | None = None
    confidence: float | None = None
    track_id: int | None = None
    detection: tuple | None = None
    gated_out: bool = False

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "position": list(self.position) if self.position is not None else None,
            "source": self.source,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "detection": list(self.detection) if self.detection is not None else None,
            "gated_out": self.gated_out,
        }


@dataclass
class Track:
    """
    One flight of the ball: the points that share a track identifier.

    A track is confirmed once it has enough DETECTED points to be a flight
    rather than a stray false positive. The threshold comes from the
    configuration; the default is `MIN_TRACK_DETECTIONS`.
    """

    track_id: int
    points: list[TrajectoryPoint] = field(default_factory=list)
    min_detections: int = MIN_TRACK_DETECTIONS

    @property
    def start_frame(self) -> int | None:
        return self.points[0].frame if self.points else None

    @property
    def end_frame(self) -> int | None:
        return self.points[-1].frame if self.points else None

    @property
    def detected_count(self) -> int:
        return sum(1 for p in self.points if p.source == "detected")

    @property
    def predicted_count(self) -> int:
        return sum(1 for p in self.points if p.source == "predicted")

    @property
    def is_confirmed(self) -> bool:
        """True when this track is backed by enough real detections to be a flight."""
        return self.detected_count >= self.min_detections

    def detected_points(self) -> list[TrajectoryPoint]:
        """The detection-confirmed points only — the valid input for a physics fit."""
        return [p for p in self.points if p.source == "detected"]

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "point_count": len(self.points),
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "is_confirmed": self.is_confirmed,
            "points": [p.as_dict() for p in self.points],
        }


@dataclass
class TrackingResult:
    """
    Everything one tracking pass produced.

    `points` is the full trajectory in frame order. `tracks` groups those
    points by flight. `trajectory` is the same points wrapped in the schema
    object that reports the gaps.
    """

    points: list[TrajectoryPoint] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    trajectory: Trajectory = field(default_factory=Trajectory)
    frames_decoded: int = 0
    detection_frames: int = 0
    predicted_frames: int = 0
    video: dict | None = None

    @property
    def confirmed_tracks(self) -> list[Track]:
        """The tracks backed by enough detections to be presented as flights."""
        return [t for t in self.tracks if t.is_confirmed]

    def as_dict(self) -> dict:
        return {
            "schema_version": TRACKING_SCHEMA_VERSION,
            "frames_decoded": self.frames_decoded,
            "detection_frames": self.detection_frames,
            "predicted_frames": self.predicted_frames,
            "point_count": len(self.points),
            "confirmed_track_count": len(self.confirmed_tracks),
            "points": [p.as_dict() for p in self.points],
            "tracks": [t.as_dict() for t in self.tracks],
            "trajectory": self.trajectory.as_dict(),
            "video": self.video,
        }


class TrackingService:
    """
    Tracks the ball through a stream of frames.

    The detector may be injected, which is how the tests drive the service
    without the YOLO weights (which are deliberately not tracked in git).
    Injected detectors must provide `detect_with_confidence(frame)` returning
    `(x, y, confidence)` — or `detect(frame)` returning `(x, y)` — and expose
    an `available` attribute.
    """

    def __init__(
        self,
        detector: Any = None,
        config: TrackingConfig | None = None,
    ) -> None:
        self.config = config if config is not None else TRACKING_CONFIG

        self._tracker = BallTracker(
            max_missed_frames=self.config.max_missed_frames,
            gate_radius_px=self.config.gate_radius_px,
            max_trail_points=self.config.max_trail_points,
        )

        self.detector = detector if detector is not None else self._build_detector()

        self._frame_counter = 0
        self._next_track_id = 1
        self._current_track_id: int | None = None

        self._points: list[TrajectoryPoint] = []
        self._tracks: dict[int, Track] = {}
        self._track_order: list[int] = []

        self._frames_decoded = 0
        self._detection_frames = 0
        self._predicted_frames = 0
        self._video: dict | None = None

    # ── Construction helpers ──────────────────────────────────────────────────
    def _build_detector(self) -> BallDetector:
        """Build the default detector from the configuration."""
        return BallDetector(
            model_path=self.config.model_path,
            conf_threshold=self.config.conf_threshold,
            resize_width=self.config.resize_width,
            resize_height=self.config.resize_height,
            min_diag_px=self.config.min_diag_px,
            max_diag_px=self.config.max_diag_px,
            device=self.config.device,
        )

    def _require_detector(self) -> None:
        """
        Refuse to run without a usable detector.

        A detector with no weights cannot find a ball in any frame. Reporting
        that as "nothing detected" would quietly record an empty analysis
        instead of a missing model, which is the single most misleading thing
        this module could do — so it raises instead.
        """
        if self.detector is None:
            raise DetectorUnavailableError(
                "No detector was supplied to the tracking service."
            )
        if not getattr(self.detector, "available", False):
            detail = getattr(self.detector, "load_error", None)
            raise DetectorUnavailableError(
                "The ball detector has no usable model"
                + (f": {detail}" if detail else ".")
                + " See docs/TRACKING.md for how to supply the weights."
            )

    # ── Per-frame tracking ────────────────────────────────────────────────────
    def process_frame(self, frame, frame_no: int | None = None) -> FrameTracking:
        """
        Track one frame.

        `frame` is a BGR image. `frame_no` is the 1-based index of the frame
        in the stream; when it is omitted the service numbers the frames
        itself, in the order they arrive.

        Raises `InvalidFrameError` for a frame that is not a usable image, and
        `DetectorUnavailableError` when there is no usable detector.
        """
        frame = validate_frame(frame)
        self._require_detector()

        self._frame_counter += 1
        if frame_no is None:
            frame_no = self._frame_counter
        else:
            frame_no = int(frame_no)
            # Keep the automatic numbering ahead of any explicit number, so
            # the two can never collide.
            self._frame_counter = max(self._frame_counter, frame_no)

        cx, cy, confidence = self._detect(frame)
        detection = None if cx is None else (int(cx), int(cy))

        result = self._tracker.update(detection, frame_no)

        track_id: int | None = None
        point_confidence: float | None = None

        if result.source == "detected":
            # A new flight begins whenever the tracker had no live track to
            # carry on with — either this is the very first detection, or the
            # previous one was lost and the ball has just been re-acquired.
            if self._current_track_id is None:
                self._current_track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[self._current_track_id] = Track(
                    track_id=self._current_track_id,
                    min_detections=self.config.min_track_detections,
                )
                self._track_order.append(self._current_track_id)
            track_id = self._current_track_id
            point_confidence = confidence
        elif result.source == "predicted":
            # Coasting: the flight continues, so the identifier does too.
            track_id = self._current_track_id
        else:
            # The ball is lost. The flight is over and the identifier is
            # retired, so the next detection starts a new one.
            self._current_track_id = None

        if result.position is not None and result.source is not None:
            point = TrajectoryPoint(
                frame=frame_no,
                x=int(result.position[0]),
                y=int(result.position[1]),
                source=result.source,
                confidence=point_confidence,
                track_id=track_id,
            )
            self._points.append(point)
            if track_id is not None:
                self._tracks[track_id].points.append(point)
            if result.source == "detected":
                self._detection_frames += 1
            else:
                self._predicted_frames += 1

        self._frames_decoded += 1

        return FrameTracking(
            frame=frame_no,
            position=result.position,
            source=result.source,
            confidence=point_confidence,
            track_id=track_id,
            detection=detection,
            gated_out=result.gated_out,
        )

    def _detect(self, frame) -> tuple:
        """Ask the detector for (x, y, confidence); None where there is nothing."""
        if hasattr(self.detector, "detect_with_confidence"):
            cx, cy, confidence = self.detector.detect_with_confidence(frame)
            return cx, cy, confidence
        position = self.detector.detect(frame)
        if position is None:
            return None, None, None
        return position[0], position[1], None

    # ── Batch tracking ────────────────────────────────────────────────────────
    def track_frames(self, frames: Iterable) -> TrackingResult:
        """Track an iterable of frames and return the result."""
        for frame in frames:
            self.process_frame(frame)
        return self.result

    def track_video(self, path) -> TrackingResult:
        """
        Decode a video and track the ball through it.

        The 1-based decode counter is the authoritative frame numbering; the
        container's advertised frame count is recorded as a diagnostic only,
        because it is frequently wrong for the formats cricket footage
        arrives in.
        """
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(f"Could not open the video: {video_path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.process_frame(frame)
        finally:
            capture.release()

        self._video = {
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frame_count": declared,
            "frames_decoded": self._frames_decoded,
        }
        return self.result

    # ── Results ───────────────────────────────────────────────────────────────
    @property
    def result(self) -> TrackingResult:
        """The result of everything tracked so far."""
        return TrackingResult(
            points=list(self._points),
            tracks=[self._tracks[tid] for tid in self._track_order],
            trajectory=Trajectory(points=list(self._points)),
            frames_decoded=self._frames_decoded,
            detection_frames=self._detection_frames,
            predicted_frames=self._predicted_frames,
            video=self._video,
        )

    def as_dict(self) -> dict:
        """The result as plain, serialisable data."""
        return self.result.as_dict()

    def reset(self) -> None:
        """Forget the current pass, so the service can be reused."""
        self._tracker.reset()
        self._frame_counter = 0
        self._next_track_id = 1
        self._current_track_id = None
        self._points = []
        self._tracks = {}
        self._track_order = []
        self._frames_decoded = 0
        self._detection_frames = 0
        self._predicted_frames = 0
        self._video = None


__all__ = [
    "DetectorUnavailableError",
    "FrameTracking",
    "InvalidFrameError",
    "Track",
    "TrackingError",
    "TrackingResult",
    "TrackingService",
    "VideoOpenError",
]
