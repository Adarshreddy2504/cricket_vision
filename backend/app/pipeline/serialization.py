"""The clean frontend envelope and the separate diagnostic document.

The two documents have different audiences. `result.json` contains only the
schema 2.1 fields the browser parses, with `null` preserved wherever a value
was not measured. `diagnostics.json` explains the decisions behind those nils:
which detector ran, which frame calibrated the pitch, which clip failed, which
model was missing, and what the analytics roll-up actually measured.

Nothing in this module fills a missing measurement with a zero. A zero would be
indistinguishable from a genuine measurement of zero, which is exactly the
failure the honesty rules exist to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..pitch.calibration import PitchCalibration
from ..pitch.detector import PitchDetection
from ..schemas.pitch import PitchTrajectory
from ..schemas.tracking import Trajectory
from ..tracking.tracking_service import Track
from .clips import ClipArtifact, ClipWindow


@dataclass
class DeliveryAnalysis:
    """Everything measured for one candidate delivery."""

    index: int
    track: Track
    window: ClipWindow
    clip: ClipArtifact
    calibration: PitchCalibration | None = None
    detection: PitchDetection | None = None
    pitch_trajectory: PitchTrajectory | None = None
    calibration_reason: str | None = None
    pitch_attempts: list[dict] = field(default_factory=list)
    shot: Any = None
    shot_model_error: str | None = None
    analytics: Any = None
    analytics_error: str | None = None


@dataclass
class AnalysisBundle:
    analysis_id: str
    source_name: str
    width: int
    height: int
    fps: float
    declared_frames: int
    tracking: Any
    windows: list[ClipWindow]
    deliveries: list[DeliveryAnalysis]
    models: dict
    frames_decoded: int
    detection_frames: int
    predicted_frames: int


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mean_confidence(track: Track) -> float | None:
    values = [
        float(point.confidence)
        for point in track.points
        if point.source == "detected" and point.confidence is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _orientation(
    calibration: PitchCalibration,
    pitch_trajectory: PitchTrajectory | None,
) -> dict | None:
    if pitch_trajectory is None or not pitch_trajectory.points:
        return None
    points = [point for point in pitch_trajectory.points if point.source == "detected"]
    if len(points) < 2:
        points = list(pitch_trajectory.points)
    if len(points) < 2:
        return None
    delta = float(points[-1].along) - float(points[0].along)
    if abs(delta) < 0.02:
        return None
    long_axis = calibration.geometry.long_axis.value
    return {
        "axis": "v" if long_axis == "vertical" else "u",
        "batting_end": 1 if delta > 0 else 0,
        "travel_delta": round(delta, 4),
        "calibration_frame": int(calibration.frame_number),
    }


def _pitch_block(delivery: DeliveryAnalysis) -> dict:
    calibration = delivery.calibration
    detection = delivery.detection
    if calibration is None:
        return {
            "state": "no_calibration",
            "detected": bool(detection.ok) if detection is not None else False,
            "calibrated": False,
            "confidence": (
                _round(detection.confidence) if detection is not None else None
            ),
            "using_previous_calibration": False,
            "keypoints": {},
            "corners": {},
            "center": None,
            "homography_available": False,
            "orientation": None,
            "source": None,
            "geometry_reason": delivery.calibration_reason or "no_calibration",
        }

    corners = calibration.corners.as_dict()
    center = [
        round(
            sum(point[0] for point in corners.values()) / len(corners),
            3,
        ),
        round(
            sum(point[1] for point in corners.values()) / len(corners),
            3,
        ),
    ]
    geometry_reason = "; ".join(calibration.reasons) or None
    if calibration.is_approximate and not geometry_reason:
        geometry_reason = "approximate_corners"
    return {
        "state": "calibrated" if calibration.is_valid else "approximate",
        "detected": True,
        "calibrated": calibration.is_valid,
        "confidence": _round(calibration.confidence),
        "using_previous_calibration": False,
        "keypoints": {},
        "corners": corners,
        "center": center,
        "homography_available": delivery.pitch_trajectory is not None,
        "orientation": _orientation(calibration, delivery.pitch_trajectory),
        "source": calibration.method.value,
        "geometry_reason": geometry_reason,
    }


def _position_by_frame(
    pitch_trajectory: PitchTrajectory | None,
) -> dict[int, dict]:
    positions: dict[int, dict] = {}
    if pitch_trajectory is None:
        return positions
    for point in pitch_trajectory.points:
        positions[int(point.frame)] = {
            "x": _round(point.across),
            "y": _round(point.along),
            "inside_pitch": bool(point.inside_pitch),
        }
    return positions


def _pitch_status(delivery: DeliveryAnalysis) -> str:
    calibration = delivery.calibration
    if calibration is not None:
        return "valid" if calibration.is_valid else "approximate"
    if delivery.detection is not None:
        return delivery.detection.status.value
    return "not_attempted"


def _flags(delivery: DeliveryAnalysis) -> list[str]:
    flags = [
        "candidate_delivery_boundary_unvalidated",
        "metric_calibration_required",
        "bounce_detection_unavailable",
    ]
    calibration = delivery.calibration
    if calibration is None:
        flags.append("pitch_calibration_unavailable")
    else:
        flags.append("pitch_calibration_single_frame")
        if calibration.is_approximate:
            flags.append("pitch_calibration_approximate")
        if _orientation(calibration, delivery.pitch_trajectory) is None:
            flags.append("pitch_orientation_ambiguous")
        if (
            delivery.pitch_trajectory is not None
            and delivery.pitch_trajectory.skipped > 0
        ):
            flags.append("pitch_projection_skipped_points")
    if not delivery.clip.available:
        flags.append("clip_unavailable")
    if delivery.clip.error == "source_decode_ended_early":
        flags.append("clip_source_decode_ended_early")
    if (
        delivery.window.truncated_start
        or delivery.window.truncated_end
        or delivery.window.limited_by_max_frames
    ):
        flags.append("clip_window_truncated")
    if delivery.track.predicted_count > 0:
        flags.append("trajectory_contains_predicted_points")
    if delivery.track.low_confidence_count > 0:
        flags.append("trajectory_contains_low_confidence_detections")
    continuity = Trajectory(points=list(delivery.track.points)).continuity
    if continuity != "continuous":
        flags.append("trajectory_gapped")
    shot_status = getattr(delivery.shot, "status", None)
    if shot_status == "model_unavailable":
        flags.append("shot_model_unavailable")
    elif shot_status not in (None, "ok"):
        flags.append("shot_classification_unavailable")
    return flags


def _delivery_document(bundle: AnalysisBundle, delivery: DeliveryAnalysis) -> dict:
    index = delivery.index
    positions = _position_by_frame(delivery.pitch_trajectory)
    points = [
        {
            "frame": int(point.frame),
            "x": int(point.x),
            "y": int(point.y),
            "source": point.source,
            "confidence": _round(point.confidence, 4),
            "pitch_position": positions.get(int(point.frame)),
        }
        for point in delivery.track.points
    ]
    clip_url = (
        f"/api/analysis/{bundle.analysis_id}/clips/{delivery.clip.file_name}"
        if delivery.clip.available and delivery.clip.file_name
        else None
    )
    shot_status = getattr(delivery.shot, "status", None)
    return {
        "id": index,
        "video": {
            "clip_url": clip_url,
            "start_frame": delivery.window.start_frame,
            "end_frame": delivery.window.end_frame,
            "fps": _round(delivery.clip.fps) if delivery.clip.fps else None,
            "width": delivery.clip.width or bundle.width or None,
            "height": delivery.clip.height or bundle.height or None,
            "duration_seconds": _round(delivery.clip.duration_seconds, 3),
        },
        "trajectory": {
            "points": points,
            "detected_points": int(delivery.track.detected_count),
            "predicted_points": int(delivery.track.predicted_count),
            "continuity": Trajectory(
                points=list(delivery.track.points)
            ).continuity,
        },
        "bounce": {
            "detected": False,
            "frame": None,
            "x": None,
            "y": None,
            "ground_x_m": None,
            "ground_y_m": None,
        },
        "bowling": {
            "speed_kmh": None,
            "line": None,
            "length": None,
            "swing": None,
            "release_angle": None,
            "bounce_angle": None,
            "calibrated": False,
        },
        "shot": {
            "classification": (
                getattr(delivery.shot, "label", None)
                if shot_status == "ok"
                else None
            ),
            "confidence": (
                _round(getattr(delivery.shot, "score", None), 4)
                if shot_status == "ok"
                else None
            ),
        },
        "quality": {
            "trajectory_valid": bool(delivery.track.is_confirmed),
            "tracking_confidence": _mean_confidence(delivery.track),
            "requires_human_review": True,
            "flags": _flags(delivery),
        },
        "pitch": _pitch_block(delivery),
    }


def _default_pitch_block() -> dict:
    return {
        "state": "no_calibration",
        "detected": False,
        "calibrated": False,
        "confidence": None,
        "using_previous_calibration": False,
        "keypoints": {},
        "corners": {},
        "center": None,
        "homography_available": False,
        "orientation": None,
        "source": None,
        "geometry_reason": "no_calibrated_delivery",
    }


def _diagnostic_delivery(bundle: AnalysisBundle, delivery: DeliveryAnalysis) -> dict:
    shot = delivery.shot
    shot_dict = shot.as_dict() if hasattr(shot, "as_dict") else None
    analytics = None
    if delivery.analytics is not None:
        analytics = (
            delivery.analytics.as_dict()
            if hasattr(delivery.analytics, "as_dict")
            else delivery.analytics
        )
    return {
        "id": delivery.index,
        "track_id": delivery.track.track_id,
        "window": delivery.window.as_dict(),
        "clip": delivery.clip.as_dict(),
        "pitch": {
            "status": _pitch_status(delivery),
            "reason": delivery.calibration_reason,
            "detection": (
                delivery.detection.as_dict()
                if delivery.detection is not None
                else None
            ),
            "calibration": (
                delivery.calibration.as_dict()
                if delivery.calibration is not None
                else None
            ),
            "attempts": delivery.pitch_attempts,
            "projected_points": (
                len(delivery.pitch_trajectory.points)
                if delivery.pitch_trajectory is not None
                else 0
            ),
            "skipped_points": (
                delivery.pitch_trajectory.skipped
                if delivery.pitch_trajectory is not None
                else 0
            ),
        },
        "shot": shot_dict,
        "analytics": analytics,
        "analytics_error": delivery.analytics_error,
        "tracking": {
            "detected_points": int(delivery.track.detected_count),
            "predicted_points": int(delivery.track.predicted_count),
            "low_confidence_points": int(delivery.track.low_confidence_count),
        },
        "flags": _flags(delivery),
    }


def build_documents(bundle: AnalysisBundle) -> tuple[dict, dict]:
    """Return `(frontend_result, diagnostics)` with no shared mutation."""
    delivery_documents = [
        _delivery_document(bundle, delivery) for delivery in bundle.deliveries
    ]
    result = {
        "analysis_id": bundle.analysis_id,
        "status": "completed",
        "pitch": (
            delivery_documents[-1]["pitch"]
            if delivery_documents
            else _default_pitch_block()
        ),
        "deliveries": delivery_documents,
    }

    diagnostics = {
        "analysis_id": bundle.analysis_id,
        "schema_version": "2.1",
        "pipeline_version": "1.1.0",
        "source_name": bundle.source_name,
        "frames_decoded": bundle.frames_decoded,
        "detection_frames": bundle.detection_frames,
        "predicted_frames": bundle.predicted_frames,
        "video": {
            "width": bundle.width,
            "height": bundle.height,
            "fps": bundle.fps,
            "declared_frames": bundle.declared_frames,
            "frames_decoded": bundle.frames_decoded,
            "detection_frames": bundle.detection_frames,
            "predicted_frames": bundle.predicted_frames,
            "timebase": {
                "authority": "1-based frame index",
                "frame_rate_source": "container metadata (OpenCV CAP_PROP_FPS)",
                "constant_frame_rate_assumed": True,
                "per_frame_timestamps_preserved": False,
                "variable_frame_rate_supported": False,
                "seconds_semantics": (
                    "seconds are derived from frame counts and the container "
                    "fps; no per-frame presentation timestamp is read"
                ),
                "missing_fps_semantics": (
                    "if the container reports no usable fps, clip duration is "
                    "null and clip encoding fails with invalid_video_fps"
                ),
            },
        },
        "confirmed_tracks": len(
            [track for track in bundle.tracking.tracks if track.is_confirmed]
        ),
        "tracking": {
            "low_confidence_points": sum(
                int(track.low_confidence_count)
                for track in bundle.tracking.tracks
            ),
            "association_rule": (
                "detections below the track-start confidence threshold may "
                "only extend a live track and must pass the Kalman gate; they "
                "never start a new track"
            ),
        },
        "candidate_windows": [window.as_dict() for window in bundle.windows],
        "candidate_boundary_validated": False,
        "boundary_definition": (
            "one confirmed Part 1 trajectory is one candidate delivery; "
            "this is not a validated delivery boundary"
        ),
        "models": bundle.models,
        "deliveries": [
            _diagnostic_delivery(bundle, delivery)
            for delivery in bundle.deliveries
        ],
    }
    return result, diagnostics


__all__ = ["AnalysisBundle", "DeliveryAnalysis", "build_documents"]
