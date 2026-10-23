"""Connect tracking, pitch calibration, shot classification and analytics.

The orchestration deliberately does not invent the missing Part 5 stages:
there is no delivery segmenter, no bounce detector, no overlay renderer and no
metric calibration. What it does instead is make each missing capability
legible in the result: a confirmed track becomes a *candidate* delivery, a
missing pitch model leaves the pitch block uncalibrated with a reason, and a
missing shot checkpoint leaves the label null with a flag. Nothing is filled
with a plausible-looking zero.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from ..analytics.summary import summarise_delivery
from ..pitch.adapter import project_trajectory
from ..pitch.calibration import PitchCalibration
from ..pitch.detector import detection_to_calibration
from ..pitch.errors import CalibrationError
from ..schemas.shots import ShotClassification
from ..shots.preprocessing import sample_indices
from ..tracking.tracking_service import TrackingService
from . import PipelineResult
from .clips import ClipArtifact, ClipExtractor, ClipWindow, candidate_windows
from .models import PipelineModels, classifier_from_factory
from .serialization import (
    AnalysisBundle,
    DeliveryAnalysis,
    build_documents,
)
from .video import (
    VideoProbeError,
    probe_video,
    sample_clip_frames,
    sample_source_frames,
)

if TYPE_CHECKING:
    from ..api.config import Settings

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """The analysis cannot produce an honest result at all."""


class AnalysisPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        models: PipelineModels | None = None,
        tracker_factory: Callable[[], Any] | None = None,
        pitch_detectors_factory: Callable[[], list[Any]] | None = None,
        shot_classifier_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.models = models or PipelineModels()
        self.tracker_factory = tracker_factory
        self.pitch_detectors_factory = pitch_detectors_factory
        self.shot_classifier_factory = shot_classifier_factory

    # ── Public entry point ───────────────────────────────────────────────────
    def run(
        self,
        *,
        analysis_id: str,
        source_path: Path,
        run_dir: Path,
        progress: Callable[..., None],
    ) -> PipelineResult:
        source_path = Path(source_path)
        run_dir = Path(run_dir)
        if not source_path.is_file():
            raise PipelineError("the uploaded source video is missing")

        progress("tracking", 0, None, "probing the video")
        try:
            probe = probe_video(source_path)
        except VideoProbeError as exc:
            raise PipelineError(str(exc)) from exc

        tracker, ball_status = self._make_tracker()
        total_frames = probe.declared_frames or None
        tracking = self._track(
            source_path,
            tracker,
            coordinates=(probe.width, probe.height, probe.fps),
            total_frames=total_frames,
            progress=progress,
        )
        if tracking.frames_decoded == 0:
            raise PipelineError("the source video decoded zero frames")

        windows = candidate_windows(
            tracking.confirmed_tracks,
            self.settings,
            total_frames=tracking.frames_decoded,
        )
        tracks_by_id = {track.track_id: track for track in tracking.confirmed_tracks}
        progress(
            "validation",
            0,
            max(1, len(windows)),
            "writing candidate delivery clips",
        )
        clips = ClipExtractor(self.settings).extract(
            source_path,
            windows,
            clips_dir=run_dir / "clips",
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            on_progress=lambda done, total: progress(
                "validation",
                done,
                total,
                "writing candidate delivery clips",
            ),
        )

        pitch_specs, pitch_factory_error = self._pitch_specs()
        shot_classifier, shot_error = self._make_shot_classifier()

        deliveries: list[DeliveryAnalysis] = []
        for position, window in enumerate(windows, start=1):
            track = tracks_by_id.get(window.track_id)
            if track is None:
                log.warning("window %s references a vanished track", window.delivery_index)
                continue
            clip = clips[window.delivery_index]
            (
                calibration,
                detection,
                pitch_trajectory,
                calibration_reason,
                pitch_attempts,
            ) = self._calibrate(
                source_path,
                window,
                clip,
                track,
                pitch_specs,
                pitch_factory_error,
            )
            progress(
                "validation",
                position,
                max(1, len(windows)),
                "calibrating and validating candidate deliveries",
            )

            shot = self._classify_shot(
                source_path,
                window,
                clip,
                shot_classifier,
                shot_error,
            )
            progress(
                "shot_classification",
                position,
                max(1, len(windows)),
                "classifying shot candidates",
            )

            analytics, analytics_error = self._analytics(
                window,
                track,
                shot,
                pitch_trajectory,
                probe.fps,
            )
            deliveries.append(
                DeliveryAnalysis(
                    index=window.delivery_index,
                    track=track,
                    window=window,
                    clip=clip,
                    calibration=calibration,
                    detection=detection,
                    pitch_trajectory=pitch_trajectory,
                    calibration_reason=calibration_reason,
                    pitch_attempts=pitch_attempts,
                    shot=shot,
                    shot_model_error=shot_error,
                    analytics=analytics,
                    analytics_error=analytics_error,
                )
            )

        models = self._model_diagnostics(
            ball_status=ball_status,
            pitch_specs=pitch_specs,
            pitch_factory_error=pitch_factory_error,
            shot_error=shot_error,
        )
        progress(
            "overlay",
            1,
            1,
            "overlays are client-rendered; no server overlay artefact is produced",
        )

        bundle = AnalysisBundle(
            analysis_id=analysis_id,
            source_name=source_path.name,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            declared_frames=probe.declared_frames,
            tracking=tracking,
            windows=windows,
            deliveries=deliveries,
            models=models,
            frames_decoded=tracking.frames_decoded,
            detection_frames=tracking.detection_frames,
            predicted_frames=tracking.predicted_frames,
        )
        result, diagnostics = build_documents(bundle)
        diagnostics["source_name"] = source_path.name
        return PipelineResult(result=result, diagnostics=diagnostics)

    # ── Tracking ─────────────────────────────────────────────────────────────
    def _make_tracker(self) -> tuple[Any, dict]:
        if self.tracker_factory is not None:
            service = self.tracker_factory()
            detector = getattr(service, "detector", None)
            return service, {
                "name": "ball_detector",
                "available": bool(getattr(detector, "available", False)),
                "reason": getattr(detector, "load_error", None),
            }
        status = self.models.ball_status()
        service = TrackingService(detector=self.models.ball_detector())
        return service, status

    def _track(
        self,
        source_path: Path,
        tracker: Any,
        *,
        coordinates: tuple[int, int, float],
        total_frames: int | None,
        progress: Callable[..., None],
    ):
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise PipelineError("the source video could not be opened for tracking")
        frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames += 1
                tracker.process_frame(frame, frame_no=frames)
                progress(
                    "tracking",
                    frames,
                    total_frames,
                    f"tracked frame {frames}"
                    + (f" of {total_frames}" if total_frames else ""),
                )
        finally:
            capture.release()

        result = tracker.result
        result.video = {
            "width": coordinates[0],
            "height": coordinates[1],
            "fps": coordinates[2],
            "frames_decoded": frames,
        }
        return result

    # ── Pitch ────────────────────────────────────────────────────────────────
    def _pitch_specs(self) -> tuple[list[tuple[str, Any]], str | None]:
        if self.pitch_detectors_factory is not None:
            try:
                raw = list(self.pitch_detectors_factory())
            except Exception as exc:  # noqa: BLE001 - explicit unavailable state
                return [], str(exc) or type(exc).__name__
            specs: list[tuple[str, Any]] = []
            for index, item in enumerate(raw, start=1):
                if isinstance(item, tuple) and len(item) == 2:
                    name, detector = item
                else:
                    detector = item
                    name = getattr(detector, "name", f"pitch_detector_{index}")
                specs.append((str(name), detector))
            return specs, None
        return self.models.pitch_detectors(), None

    def _calibration_frames(
        self,
        source_path: Path,
        window: ClipWindow,
        clip: ClipArtifact,
    ) -> list[tuple[int, Any]]:
        if clip.available and clip.path is not None and clip.frame_count > 0:
            offsets = sorted(
                {0, max(0, clip.frame_count // 2), max(0, clip.frame_count - 1)}
            )
            frames = sample_clip_frames(clip.path, offsets)
            return [
                (window.start_frame + offset, frames[offset])
                for offset in offsets
                if offset in frames
            ]

        offsets = sorted(
            {0, max(0, window.frame_count // 2), max(0, window.frame_count - 1)}
        )
        absolute = [window.start_frame + offset for offset in offsets]
        frames = sample_source_frames(source_path, absolute)
        return [(frame_no, frames[frame_no]) for frame_no in absolute if frame_no in frames]

    def _calibrate(
        self,
        source_path: Path,
        window: ClipWindow,
        clip: ClipArtifact,
        track: Any,
        specs: list[tuple[str, Any]],
        factory_error: str | None,
    ) -> tuple[
        PitchCalibration | None,
        Any | None,
        Any | None,
        str | None,
        list[dict],
    ]:
        frames = self._calibration_frames(source_path, window, clip)
        attempts: list[dict] = []
        if factory_error is not None:
            attempts.append(
                {
                    "detector": "pitch_detectors",
                    "status": "unavailable",
                    "reason": factory_error,
                }
            )
        if not frames:
            return None, None, None, "pitch_calibration_frames_unavailable", attempts
        if not specs:
            return None, None, None, factory_error or "no_pitch_detector", attempts

        approximate: tuple[PitchCalibration, Any] | None = None
        last_reason: str | None = None
        last_detection: Any | None = None
        for name, detector in specs:
            available = bool(getattr(detector, "available", False))
            if not available:
                reason = getattr(detector, "load_error", None) or "pitch weights unavailable"
                last_reason = reason
                attempts.append(
                    {"detector": name, "status": "unavailable", "reason": reason}
                )
                continue
            for frame_number, frame in frames:
                try:
                    detection = detector.detect(frame, frame_number=frame_number)
                except Exception as exc:  # noqa: BLE001 - one bad detector is not fatal
                    reason = str(exc) or type(exc).__name__
                    last_reason = reason
                    attempts.append(
                        {
                            "detector": name,
                            "frame": frame_number,
                            "status": "error",
                            "reason": reason,
                        }
                    )
                    continue
                last_detection = detection
                last_reason = detection.reason or last_reason
                attempts.append(
                    {
                        "detector": name,
                        "frame": frame_number,
                        "status": detection.status.value,
                        "reason": detection.reason,
                    }
                )
                if not detection.ok:
                    continue
                try:
                    calibration = detection_to_calibration(detection)
                except Exception as exc:  # noqa: BLE001
                    last_reason = str(exc) or type(exc).__name__
                    continue
                if calibration is None or not calibration.is_usable:
                    continue
                if calibration.is_valid:
                    return (
                        calibration,
                        detection,
                        self._project(track, calibration),
                        None,
                        attempts,
                    )
                if approximate is None:
                    approximate = (calibration, detection)

        if approximate is not None:
            calibration, detection = approximate
            return (
                calibration,
                detection,
                self._project(track, calibration),
                "; ".join(calibration.reasons) or "approximate_corners",
                attempts,
            )
        return (
            None,
            last_detection,
            None,
            last_reason or "pitch_not_detected",
            attempts,
        )

    @staticmethod
    def _project(track: Any, calibration: PitchCalibration):
        try:
            return project_trajectory(track.points, calibration)
        except CalibrationError as exc:
            log.warning("pitch projection refused: %s", exc)
            return None

    # ── Shot ─────────────────────────────────────────────────────────────────
    def _make_shot_classifier(self) -> tuple[Any, str | None]:
        if self.shot_classifier_factory is not None:
            return classifier_from_factory(self.shot_classifier_factory)
        status = self.models.shot_status()
        return self.models.shot_classifier(), status.get("reason")

    def _classify_shot(
        self,
        source_path: Path,
        window: ClipWindow,
        clip: ClipArtifact,
        classifier: Any,
        shot_error: str | None,
    ) -> Any:
        if classifier is None:
            return ShotClassification(
                status="model_unavailable",
                error=shot_error or "no shot classifier was supplied",
            )
        try:
            if clip.available and clip.path is not None:
                return classifier.classify(
                    clip_path=clip.path,
                    delivery_id=window.delivery_index,
                    frame_offset=window.start_frame - 1,
                )

            offsets = sample_indices(
                max(1, clip.frame_count or window.frame_count)
            )
            absolute = [window.start_frame + offset for offset in offsets]
            frames_by_source = sample_source_frames(source_path, absolute)
            sparse = {
                offset: frames_by_source[window.start_frame + offset]
                for offset in offsets
                if (window.start_frame + offset) in frames_by_source
            }
            return classifier.classify(
                in_memory_frames=sparse,
                expected_total_frames=max(1, clip.frame_count or window.frame_count),
                delivery_id=window.delivery_index,
                frame_offset=window.start_frame - 1,
            )
        except Exception as exc:  # noqa: BLE001 - record, never fabricate a label
            log.exception("shot classification failed for delivery %s", window.delivery_index)
            return ShotClassification(
                status="inference_failed",
                error=str(exc) or type(exc).__name__,
            )

    # ── Analytics ────────────────────────────────────────────────────────────
    @staticmethod
    def _analytics(
        window: ClipWindow,
        track: Any,
        shot: Any,
        pitch_trajectory: Any,
        fps: float,
    ) -> tuple[Any | None, str | None]:
        try:
            return (
                summarise_delivery(
                    window.delivery_index,
                    shot=shot,
                    trajectory=track.points,
                    fps=fps if fps > 0 else None,
                    pitch_trajectory=pitch_trajectory,
                ),
                None,
            )
        except Exception as exc:  # noqa: BLE001 - derived summary is not the result
            log.exception("analytics failed for delivery %s", window.delivery_index)
            return None, str(exc) or type(exc).__name__

    # ── Diagnostics ──────────────────────────────────────────────────────────
    @staticmethod
    def _model_diagnostics(
        *,
        ball_status: dict,
        pitch_specs: list[tuple[str, Any]],
        pitch_factory_error: str | None,
        shot_error: str | None,
    ) -> dict:
        pitch_status = []
        for name, detector in pitch_specs:
            pitch_status.append(
                {
                    "name": name,
                    "available": bool(getattr(detector, "available", False)),
                    "reason": getattr(detector, "load_error", None),
                }
            )
        if pitch_factory_error is not None:
            pitch_status.append(
                {
                    "name": "pitch_detectors",
                    "available": False,
                    "reason": pitch_factory_error,
                }
            )
        return {
            "ball_detector": ball_status,
            "pitch_detectors": pitch_status,
            "shot_classifier": {
                "name": "shot_classifier",
                "available": shot_error is None,
                "reason": shot_error,
            },
        }


__all__ = ["AnalysisPipeline", "PipelineError"]
