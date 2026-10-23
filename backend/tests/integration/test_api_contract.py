"""The browser-facing HTTP contract: routes, documents, bytes and cleanup."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.config import Settings

from .conftest import (
    FakeRunner,
    assert_frontend_result_contract,
    upload_video,
    wait_for_terminal,
)


def test_health_reports_versions_stages_and_model_availability(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["schema_version"] == "2.1"
    assert body["pipeline_version"] == "1.1.0"
    assert body["analyses_tracked_in_memory"] == 0
    assert body["stages"] == [
        "uploading",
        "tracking",
        "validation",
        "shot_classification",
        "overlay",
        "persisting",
    ]
    assert set(body["models"]) == {
        "ball_detector",
        "pitch_segmentation",
        "pitch_box",
        "shot_classifier",
    }
    assert all("available" in entry for entry in body["models"].values())


def test_rejected_upload_leaves_no_analysis_directory(client, settings):
    response = client.post(
        "/api/analyze",
        files={"video": ("notes.txt", b"this is not a video", "text/plain")},
    )
    assert response.status_code == 400
    assert "format" in response.json()["detail"].lower()
    assert not list(settings.uploads_dir.glob("*"))
    assert not list(settings.runs_dir.glob("*"))


def test_empty_upload_is_rejected_without_leaving_state(client, settings):
    response = client.post(
        "/api/analyze",
        files={"video": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert not list(settings.uploads_dir.glob("*"))
    assert not list(settings.runs_dir.glob("*"))


def test_corrupt_video_is_rejected_without_leaving_state(client, settings):
    response = client.post(
        "/api/analyze",
        files={"video": ("broken.mp4", b"not really an mp4" * 8, "video/mp4")},
    )
    assert response.status_code == 400
    assert "readable video" in response.json()["detail"].lower()
    assert not list(settings.uploads_dir.glob("*"))
    assert not list(settings.runs_dir.glob("*"))


def test_oversize_upload_is_rejected_before_creating_a_job(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "runtime",
        run_inline=True,
        max_upload_bytes=32,
    )
    app = create_app(settings=settings, pipeline_factory=lambda _s: FakeRunner())
    with TestClient(app) as limited:
        response = limited.post(
            "/api/analyze",
            files={"video": ("large.mp4", b"x" * 64, "video/mp4")},
        )
    assert response.status_code == 413
    assert "larger" in response.json()["detail"].lower()
    assert not list(settings.uploads_dir.glob("*"))
    assert not list(settings.runs_dir.glob("*"))


def test_analyze_completes_with_frontend_contract_and_serves_clip_bytes(
    client, settings, sample_video
):
    response = upload_video(client, sample_video)
    assert response.status_code == 202, response.text
    accepted = response.json()
    analysis_id = accepted["analysis_id"]
    assert analysis_id
    assert accepted["status_url"] == f"/api/analysis/{analysis_id}/status"
    assert accepted["result_url"] == f"/api/analysis/{analysis_id}/result"
    assert accepted["bytes"] == sample_video.stat().st_size

    status = wait_for_terminal(client, analysis_id)
    assert set(status) == {
        "analysis_id",
        "status",
        "stage",
        "substage",
        "progress",
        "detail",
        "error",
        "frames_done",
        "frames_total",
        "source_name",
        "created_at",
        "started_at",
        "finished_at",
    }
    assert status["status"] == "completed"
    assert status["stage"] == "completed"
    assert status["progress"] == 100
    assert status["error"] is None
    assert status["source_name"] == sample_video.name

    result_response = client.get(f"/api/analysis/{analysis_id}/result")
    assert result_response.status_code == 200, result_response.text
    document = result_response.json()
    assert_frontend_result_contract(document, analysis_id)
    assert document["deliveries"][0]["video"]["clip_url"].endswith(
        "/clips/delivery_001.webm"
    )

    clip = client.get(f"/api/analysis/{analysis_id}/clips/delivery_001.webm")
    assert clip.status_code == 200
    assert clip.headers["content-type"].startswith("video/webm")
    assert int(clip.headers["content-length"]) > 0
    assert clip.headers.get("accept-ranges") == "bytes"

    ranged = client.get(
        f"/api/analysis/{analysis_id}/clips/delivery_001.webm",
        headers={"Range": "bytes=0-9"},
    )
    assert ranged.status_code == 206
    assert len(ranged.content) == 10
    assert ranged.headers["content-range"].startswith("bytes 0-9/")

    # The uploaded source is temporary. The result and clip are the artefacts.
    assert not list(settings.uploads_dir.glob(f"{analysis_id}/*"))


def test_upload_alias_accepts_file_field(tmp_path: Path, sample_video: Path):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(clip_source=sample_video),
    )
    with TestClient(app) as alias_client:
        response = upload_video(alias_client, sample_video, field="file")
        assert response.status_code == 202, response.text
        analysis_id = response.json()["analysis_id"]
        status = wait_for_terminal(alias_client, analysis_id)
        assert status["status"] == "completed"


def test_unknown_analysis_and_unsafe_clip_names_are_not_served(client):
    assert client.get("/api/analysis/missing/status").status_code == 404
    assert client.get("/api/analysis/missing/result").status_code == 404
    assert client.get("/api/analysis/missing/clips/delivery_001.webm").status_code == 404
    assert client.get("/api/analysis/missing/clips/..%2Fstatus.json").status_code == 404


def test_result_is_unavailable_until_completion(tmp_path: Path, sample_video: Path):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=False)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(
            clip_source=sample_video, sleep_seconds=0.25
        ),
    )
    with TestClient(app) as slow_client:
        response = upload_video(slow_client, sample_video)
        analysis_id = response.json()["analysis_id"]
        early = slow_client.get(f"/api/analysis/{analysis_id}/result")
        assert early.status_code == 409
        assert "still" in early.json()["detail"].lower()
        assert wait_for_terminal(slow_client, analysis_id)["status"] == "completed"


def test_persisted_documents_exist_and_are_separate(tmp_path: Path, sample_video: Path):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(
            clip_source=sample_video, diagnostics={"fake_diagnostic": 7}
        ),
    )
    with TestClient(app) as persisted_client:
        response = upload_video(persisted_client, sample_video)
        analysis_id = response.json()["analysis_id"]
        wait_for_terminal(persisted_client, analysis_id)

    run_dir = settings.runs_dir / analysis_id
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "diagnostics.json").is_file()
    assert (run_dir / "clips" / "delivery_001.webm").is_file()

    import json

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (run_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert_frontend_result_contract(result, analysis_id)
    assert diagnostics["fake_diagnostic"] == 7
    assert "path" not in diagnostics
