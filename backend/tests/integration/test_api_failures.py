"""Explicit failure states: pipeline errors, missing models and bad documents."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.config import Settings

from .conftest import FakeRunner, upload_video, wait_for_terminal


def test_pipeline_exception_is_reported_as_a_failed_job(tmp_path: Path, sample_video: Path):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(error=RuntimeError("inference exploded")),
    )
    with TestClient(app) as failed_client:
        accepted = upload_video(failed_client, sample_video)
        assert accepted.status_code == 202
        analysis_id = accepted.json()["analysis_id"]

        status = wait_for_terminal(failed_client, analysis_id)
        assert status["status"] == "failed"
        assert status["error"] == "inference exploded"
        assert status["progress"] < 100
        assert status["finished_at"] is not None

        result = failed_client.get(f"/api/analysis/{analysis_id}/result")
        assert result.status_code == 409
        clip = failed_client.get(
            f"/api/analysis/{analysis_id}/clips/delivery_001.webm"
        )
        assert clip.status_code == 404

    assert not list(settings.uploads_dir.glob(f"{analysis_id}/*"))


def test_missing_result_document_is_an_explicit_server_error(
    tmp_path: Path, sample_video: Path
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(clip_source=sample_video),
    )
    with TestClient(app) as broken_client:
        accepted = upload_video(broken_client, sample_video)
        analysis_id = accepted.json()["analysis_id"]
        wait_for_terminal(broken_client, analysis_id)

        (settings.runs_dir / analysis_id / "result.json").unlink()
        response = broken_client.get(f"/api/analysis/{analysis_id}/result")
        assert response.status_code == 500
        assert "result" in response.json()["detail"].lower()


def test_corrupt_result_document_is_not_served_as_a_partial_success(
    tmp_path: Path, sample_video: Path
):
    settings = Settings(data_dir=tmp_path / "runtime", run_inline=True)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda _s: FakeRunner(clip_source=sample_video),
    )
    with TestClient(app) as corrupt_client:
        accepted = upload_video(corrupt_client, sample_video)
        analysis_id = accepted.json()["analysis_id"]
        wait_for_terminal(corrupt_client, analysis_id)

        (settings.runs_dir / analysis_id / "result.json").write_text(
            "{not json", encoding="utf-8"
        )
        response = corrupt_client.get(f"/api/analysis/{analysis_id}/result")
        assert response.status_code == 500
        assert "unreadable" in response.json()["detail"].lower()
