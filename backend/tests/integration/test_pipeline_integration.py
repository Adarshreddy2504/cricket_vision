"""The real tracking/pitch/shot/analytics modules behind the HTTP server."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.config import Settings
from app.pipeline import AnalysisPipeline
from app.pitch.detector import CornerMethod, DetectionStatus
from app.shots.errors import ShotModelUnavailable
from app.tracking.tracking_service import DetectorUnavailableError, TrackingService

from .conftest import upload_video, wait_for_terminal
from .pipeline_fakes import (
    FakePitchDetector,
    FakeShotClassifier,
    NoneDetector,
    SequenceDetector,
    ball_path,
)


def real_pipeline(
    settings: Settings,
    *,
    detector=None,
    pitch_detector=None,
    shot_classifier=None,
    shot_factory=None,
    tracker_factory=None,
):
    if tracker_factory is None:
        detector = detector if detector is not None else SequenceDetector(ball_path())

        def tracker_factory():
            return TrackingService(detector=detector)
    if pitch_detector is None:
        pitch_detector = FakePitchDetector()
    if shot_classifier is None and shot_factory is None:
        shot_classifier = FakeShotClassifier()
    return AnalysisPipeline(
        settings=settings,
        tracker_factory=tracker_factory,
        pitch_detectors_factory=lambda: [pitch_detector],
        shot_classifier_factory=(
            shot_factory
            if shot_factory is not None
            else lambda: shot_classifier
        ),
    )


def run_upload(tmp_path: Path, sample_video: Path, pipeline):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(settings=settings, pipeline_factory=lambda _s: pipeline)
    client = TestClient(app)
    with client:
        accepted = upload_video(client, sample_video)
        assert accepted.status_code == 202, accepted.text
        analysis_id = accepted.json()["analysis_id"]
        status = wait_for_terminal(client, analysis_id)
        result = None
        if status["status"] == "completed":
            result = client.get(f"/api/analysis/{analysis_id}/result").json()
    return settings, analysis_id, status, result


def read_diagnostics(settings: Settings, analysis_id: str) -> dict:
    return json.loads(
        (settings.runs_dir / analysis_id / "diagnostics.json").read_text(
            encoding="utf-8"
        )
    )


def test_confirmed_track_becomes_a_qualified_candidate_with_real_modules(
    tmp_path, sample_video
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(settings)
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "completed", status
    assert result is not None
    assert len(result["deliveries"]) == 1
    delivery = result["deliveries"][0]

    assert delivery["pitch"]["calibrated"] is True
    assert delivery["pitch"]["state"] == "calibrated"
    assert delivery["pitch"]["orientation"]["axis"] == "v"
    assert delivery["pitch"]["orientation"]["batting_end"] == 1
    assert delivery["pitch"]["orientation"]["travel_delta"] > 0
    assert delivery["pitch"]["orientation"]["calibration_frame"] is not None

    assert delivery["trajectory"]["detected_points"] == 40
    assert delivery["trajectory"]["predicted_points"] == 0
    assert all(
        point["pitch_position"] is not None
        for point in delivery["trajectory"]["points"]
    )

    # No bounce detector exists, and no metric calibration is justified.
    assert delivery["bounce"] == {
        "detected": False,
        "frame": None,
        "x": None,
        "y": None,
        "ground_x_m": None,
        "ground_y_m": None,
    }
    assert delivery["bowling"] == {
        "speed_kmh": None,
        "line": None,
        "length": None,
        "swing": None,
        "release_angle": None,
        "bounce_angle": None,
        "calibrated": False,
    }

    assert delivery["shot"]["classification"] == "Cover"
    assert delivery["shot"]["confidence"] == 0.91
    assert delivery["video"]["clip_url"] is not None
    assert delivery["quality"]["requires_human_review"] is True
    assert delivery["quality"]["trajectory_valid"] is True
    assert "candidate_delivery_boundary_unvalidated" in delivery["quality"]["flags"]
    assert "metric_calibration_required" in delivery["quality"]["flags"]
    assert "bounce_detection_unavailable" in delivery["quality"]["flags"]
    assert "pitch_calibration_single_frame" in delivery["quality"]["flags"]

    diagnostics = read_diagnostics(settings, analysis_id)
    assert diagnostics["confirmed_tracks"] == 1
    assert diagnostics["candidate_boundary_validated"] is False
    assert diagnostics["deliveries"][0]["analytics"]["trajectory"]["units"] == "pixel"
    assert (
        diagnostics["deliveries"][0]["analytics"]["trajectory"]["mean_speed_px_s"]
        is not None
    )
    assert diagnostics["deliveries"][0]["shot"]["status"] == "ok"
    assert diagnostics["deliveries"][0]["pitch"]["status"] == "valid"
    assert not list(settings.uploads_dir.glob(f"{analysis_id}/*"))


def test_uncalibrated_pitch_keeps_every_pitch_value_unavailable(
    tmp_path, sample_video
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(
        settings,
        pitch_detector=FakePitchDetector(
            status=DetectionStatus.NO_DETECTION,
            method=None,
            corners=None,
            confidence=None,
            reason="no pitch-shaped region",
        ),
    )
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "completed", status
    delivery = result["deliveries"][0]
    assert delivery["pitch"]["state"] == "no_calibration"
    assert delivery["pitch"]["detected"] is False
    assert delivery["pitch"]["calibrated"] is False
    assert delivery["pitch"]["orientation"] is None
    assert delivery["pitch"]["geometry_reason"] == "no pitch-shaped region"
    assert all(
        point["pitch_position"] is None
        for point in delivery["trajectory"]["points"]
    )
    assert delivery["bounce"]["detected"] is False
    assert delivery["bowling"]["speed_kmh"] is None
    assert "pitch_calibration_unavailable" in delivery["quality"]["flags"]
    assert delivery["shot"]["classification"] == "Cover"

    diagnostics = read_diagnostics(settings, analysis_id)
    assert diagnostics["deliveries"][0]["pitch"]["status"] == "no_detection"


def test_box_estimate_is_exposed_as_approximate_never_as_calibrated(
    tmp_path, sample_video
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(
        settings,
        pitch_detector=FakePitchDetector(
            method=CornerMethod.BOX_ESTIMATE,
            confidence=None,
        ),
    )
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "completed", status
    delivery = result["deliveries"][0]
    assert delivery["pitch"]["detected"] is True
    assert delivery["pitch"]["calibrated"] is False
    assert delivery["pitch"]["state"] == "approximate"
    assert "app" in delivery["pitch"]["geometry_reason"].lower()
    assert "pitch_calibration_approximate" in delivery["quality"]["flags"]
    assert delivery["bowling"]["calibrated"] is False
    assert delivery["bowling"]["speed_kmh"] is None

    diagnostics = read_diagnostics(settings, analysis_id)
    assert diagnostics["deliveries"][0]["pitch"]["status"] == "approximate"


def test_missing_shot_checkpoint_marks_labels_unavailable_without_failing(
    tmp_path, sample_video
):
    def unavailable_shot_model():
        raise ShotModelUnavailable("shot checkpoint missing at the configured path")

    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(settings, shot_factory=unavailable_shot_model)
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "completed", status
    delivery = result["deliveries"][0]
    assert delivery["shot"] == {"classification": None, "confidence": None}
    assert "shot_model_unavailable" in delivery["quality"]["flags"]
    assert delivery["pitch"]["calibrated"] is True

    diagnostics = read_diagnostics(settings, analysis_id)
    assert diagnostics["deliveries"][0]["shot"]["status"] == "model_unavailable"
    assert "checkpoint missing" in diagnostics["deliveries"][0]["shot"]["error"]
    assert "shot checkpoint missing" in diagnostics["models"]["shot_classifier"]["reason"]


def test_no_confirmed_track_is_a_zero_delivery_result_not_a_fake_delivery(
    tmp_path, sample_video
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(settings, detector=NoneDetector())
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "completed", status
    assert result["deliveries"] == []
    assert result["pitch"]["state"] == "no_calibration"
    diagnostics = read_diagnostics(settings, analysis_id)
    assert diagnostics["confirmed_tracks"] == 0
    assert diagnostics["frames_decoded"] > 0
    assert not list(settings.uploads_dir.glob(f"{analysis_id}/*"))


def test_missing_ball_detector_fails_loudly_instead_of_analysing_nothing(
    tmp_path, sample_video
):
    def missing_ball_model():
        raise DetectorUnavailableError(
            "ball detector weights not found; set CRICKET_BALL_MODEL"
        )

    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    pipeline = real_pipeline(settings, tracker_factory=missing_ball_model)
    settings, analysis_id, status, result = run_upload(
        tmp_path, sample_video, pipeline
    )

    assert status["status"] == "failed"
    assert result is None
    assert "ball detector weights not found" in status["error"]
    assert not list(settings.uploads_dir.glob(f"{analysis_id}/*"))
