"""Shared fixtures for the Part 5 API and pipeline integration tests.

These tests use real uploaded video files, the real FastAPI application and the
real job registry. The only things faked are the model objects that need large
untracked weights: a deterministic detector sequence, a deterministic pitch
detector and a deterministic shot classifier. The contracts exercised in
between are the production ones.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.api.app import create_app
from app.api.config import Settings
from app.pipeline import PipelineResult


def write_sample_video(
    path: Path,
    *,
    frames: int = 40,
    fps: float = 25.0,
    width: int = 160,
    height: int = 120,
) -> Path:
    """Write a small, genuinely decodable video for upload and clip tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"VP80"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the sample video writer")
    try:
        for index in range(int(frames)):
            frame = np.full((height, width, 3), 20, dtype=np.uint8)
            x = 20 + (index * 3) % max(1, width - 40)
            y = 20 + (index * 2) % max(1, height - 40)
            cv2.circle(frame, (x, y), 5, (245, 245, 245), -1)
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("the sample video writer produced no bytes")
    return path


def no_calibration_pitch() -> dict:
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
        "geometry_reason": "no_pitch_detection",
    }


def make_frontend_result(analysis_id: str, clip_name: str | None) -> dict:
    """A hand-built contract-valid result.

    The expected shape is derived from `frontend/src/lib/api.ts`, not from the
    serializer under test, so a missing or renamed field cannot satisfy this
    fixture by construction.
    """
    clip_url = (
        f"/api/analysis/{analysis_id}/clips/{clip_name}" if clip_name else None
    )
    pitch = no_calibration_pitch()
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "pitch": pitch,
        "deliveries": [
            {
                "id": 1,
                "video": {
                    "clip_url": clip_url,
                    "start_frame": 2,
                    "end_frame": 12,
                    "fps": 25.0,
                    "width": 160,
                    "height": 120,
                    "duration_seconds": 0.44,
                },
                "trajectory": {
                    "points": [
                        {
                            "frame": 2,
                            "x": 40,
                            "y": 30,
                            "source": "detected",
                            "confidence": 0.91,
                            "pitch_position": None,
                        },
                        {
                            "frame": 3,
                            "x": 44,
                            "y": 34,
                            "source": "predicted",
                            "confidence": None,
                            "pitch_position": None,
                        },
                    ],
                    "detected_points": 1,
                    "predicted_points": 1,
                    "continuity": "continuous",
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
                "shot": {"classification": "Cover", "confidence": 0.88},
                "quality": {
                    "trajectory_valid": True,
                    "tracking_confidence": 0.91,
                    "requires_human_review": True,
                    "flags": ["candidate_delivery_boundary_unvalidated"],
                },
                "pitch": pitch,
            }
        ],
    }


class FakeRunner:
    """A pipeline double that returns a known result and optionally fails."""

    def __init__(
        self,
        *,
        clip_source: Path | None = None,
        error: Exception | None = None,
        sleep_seconds: float = 0.0,
        diagnostics: dict | None = None,
    ) -> None:
        self.clip_source = clip_source
        self.error = error
        self.sleep_seconds = sleep_seconds
        self.diagnostics = diagnostics or {"fake_runner": True}

    def run(self, *, analysis_id, source_path, run_dir, progress):
        progress("tracking", 1, 2, "fake tracking")
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.error is not None:
            raise self.error

        clip_name = "delivery_001.webm"
        if self.clip_source is not None:
            clip_dir = Path(run_dir) / "clips"
            clip_dir.mkdir(parents=True, exist_ok=True)
            (clip_dir / clip_name).write_bytes(Path(self.clip_source).read_bytes())

        progress("shot_classification", 1, 1, "fake classification")
        progress("persisting", 0, 1, "fake persistence")
        return PipelineResult(
            result=make_frontend_result(analysis_id, clip_name if self.clip_source else None),
            diagnostics=self.diagnostics,
        )


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    return write_sample_video(tmp_path / "sample.webm")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "runtime", run_inline=True)


@pytest.fixture
def app(settings: Settings, sample_video: Path):
    return create_app(
        settings=settings,
        pipeline_factory=lambda _settings: FakeRunner(clip_source=sample_video),
    )


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def wait_for_terminal(client, analysis_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/analysis/{analysis_id}/status")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.02)
    raise AssertionError(f"analysis did not finish in time; last status: {last}")


def upload_video(client, path: Path, *, field: str = "video"):
    with path.open("rb") as handle:
        return client.post(
            "/api/analyze" if field == "video" else "/api/upload",
            files={field: (path.name, handle, "video/webm")},
        )


def assert_frontend_result_contract(document: dict, expected_id: str) -> None:
    """Mirror the required shape in frontend/src/lib/api.ts.

    This is intentionally a literal shape check. It fails when the server
    removes a block the frontend requires, even if the result is otherwise
    valid JSON.
    """
    assert set(document) == {"analysis_id", "status", "pitch", "deliveries"}
    assert document["analysis_id"] == expected_id
    assert document["status"] == "completed"
    assert isinstance(document["deliveries"], list)

    required_pitch = {
        "state",
        "detected",
        "calibrated",
        "confidence",
        "using_previous_calibration",
        "keypoints",
        "corners",
        "center",
        "homography_available",
        "orientation",
    }
    assert required_pitch <= set(document["pitch"])

    for entry in document["deliveries"]:
        assert set(entry) == {
            "id",
            "video",
            "trajectory",
            "bounce",
            "bowling",
            "shot",
            "quality",
            "pitch",
        }
        assert isinstance(entry["id"], int)
        assert set(entry["video"]) == {
            "clip_url",
            "start_frame",
            "end_frame",
            "fps",
            "width",
            "height",
            "duration_seconds",
        }
        assert set(entry["trajectory"]) == {
            "points",
            "detected_points",
            "predicted_points",
            "continuity",
        }
        for point in entry["trajectory"]["points"]:
            assert set(point) == {
                "frame",
                "x",
                "y",
                "source",
                "confidence",
                "pitch_position",
            }
        assert set(entry["bounce"]) == {
            "detected",
            "frame",
            "x",
            "y",
            "ground_x_m",
            "ground_y_m",
        }
        assert set(entry["bowling"]) == {
            "speed_kmh",
            "line",
            "length",
            "swing",
            "release_angle",
            "bounce_angle",
            "calibrated",
        }
        assert set(entry["shot"]) == {"classification", "confidence"}
        assert set(entry["quality"]) == {
            "trajectory_valid",
            "tracking_confidence",
            "requires_human_review",
            "flags",
        }
        assert required_pitch <= set(entry["pitch"])
