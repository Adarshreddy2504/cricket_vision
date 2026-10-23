"""Per-analysis jobs, measured progress and the persisted status document.

State is keyed by `analysis_id`, never held in one module-level "current job":
two uploads must not overwrite each other's progress. Analyses run on daemon
threads and are serialised around the model objects, which are not assumed to
be thread-safe.

`status.json` is written when a job is accepted and again when it becomes
terminal. A restart therefore leaves a truthful record instead of resurrecting
a job that can never finish.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .constants import STAGE_BANDS

log = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed"})


class JobCancelled(RuntimeError):
    """The client asked the job to stop."""


class JobDeadlineExceeded(RuntimeError):
    """The configured processing deadline expired."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class Job:
    analysis_id: str
    source_name: str
    status: str = "queued"
    stage: str = "uploading"
    substage: str | None = None
    progress: float = 0.0
    detail: str | None = None
    error: str | None = None
    frames_done: int | None = None
    frames_total: int | None = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    bytes: int = 0
    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    def as_dict(self) -> dict:
        """The exact thirteen-key status document the frontend reads."""
        return {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "stage": self.stage,
            "substage": self.substage,
            "progress": self.progress,
            "detail": self.detail,
            "error": self.error,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "source_name": self.source_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def update_from_progress(
        self,
        stage: str,
        done: int | float,
        total: int | float | None,
        detail: str | None,
    ) -> None:
        """Map measured units through the stage band, never decreasing."""
        if stage in STAGE_BANDS:
            if stage != self.stage:
                self.stage = stage
                self.substage = None
            start, end = STAGE_BANDS[stage]
            if total is not None and total > 0:
                fraction = max(0.0, min(1.0, float(done) / float(total)))
                candidate = start + (end - start) * fraction
            else:
                candidate = start
            self.progress = max(self.progress, min(100.0, candidate))
        if stage == "tracking":
            self.frames_done = int(done) if done is not None else None
            self.frames_total = int(total) if total is not None else None
        if detail is not None:
            self.detail = detail


class JobRegistry:
    def __init__(self, settings: Settings, pipeline: Any) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()

    @property
    def jobs(self) -> dict[str, Job]:
        with self._lock:
            return dict(self._jobs)

    def get(self, analysis_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(analysis_id)

    def create(self, *, analysis_id: str, source_name: str, size: int) -> Job:
        job = Job(
            analysis_id=analysis_id,
            source_name=source_name,
            bytes=int(size),
        )
        with self._lock:
            self._jobs[analysis_id] = job
        self._persist(job)
        return job

    def discard(self, analysis_id: str) -> None:
        with self._lock:
            self._jobs.pop(analysis_id, None)

    def cancel(self, analysis_id: str) -> Job | None:
        """Request cooperative cancellation; the worker owns the final state.

        The endpoint never pretends a running pipeline stopped instantly. The
        request is recorded, and the next progress check raises
        `JobCancelled`. A job between model calls may therefore take a little
        longer to reach its terminal state.
        """
        job = self.get(analysis_id)
        if job is None:
            return None
        with self._lock:
            if job.status in TERMINAL_STATUSES:
                return job
            job.cancel_event.set()
            job.detail = "cancellation requested"
        self._persist(job)
        return job

    def start(self, job: Job, source_path: Path) -> None:
        if self.settings.run_inline:
            self._run_serialised(job, Path(source_path))
            return
        thread = threading.Thread(
            target=self._run_serialised,
            args=(job, Path(source_path)),
            name=f"cricket-analysis-{job.analysis_id}",
            daemon=True,
        )
        thread.start()

    # ── Execution ────────────────────────────────────────────────────────────
    def _run_serialised(self, job: Job, source_path: Path) -> None:
        with self._analysis_lock:
            self._run(job, source_path)

    def _run(self, job: Job, source_path: Path) -> None:
        run_dir = self.settings.runs_dir / job.analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)

        job.status = "running"
        job.started_at = utc_now_iso()
        job.stage = "tracking"
        job.progress = max(job.progress, STAGE_BANDS["tracking"][0])
        job.detail = "analysis started"

        max_seconds = float(self.settings.max_job_seconds)
        deadline = (
            time.monotonic() + max_seconds if max_seconds > 0 else None
        )

        def check_abort() -> None:
            if job.cancel_event.is_set():
                raise JobCancelled("analysis cancelled by client")
            if deadline is not None and time.monotonic() > deadline:
                raise JobDeadlineExceeded(
                    f"processing deadline exceeded ({max_seconds:g}s)"
                )

        def progress(
            stage: str,
            done: int | float,
            total: int | float | None,
            detail: str | None = None,
        ) -> None:
            check_abort()
            job.update_from_progress(stage, done, total, detail)

        try:
            check_abort()
            result = self.pipeline.run(
                analysis_id=job.analysis_id,
                source_path=source_path,
                run_dir=run_dir,
                progress=progress,
            )
            check_abort()
            job.stage = "persisting"
            job.progress = max(job.progress, STAGE_BANDS["persisting"][0])
            job.substage = "writing_documents"

            _atomic_write_json(run_dir / "result.json", result.result)
            _atomic_write_json(run_dir / "diagnostics.json", result.diagnostics)

            job.status = "completed"
            job.stage = "completed"
            job.substage = None
            job.progress = 100.0
            job.detail = "analysis complete"
            job.error = None
            job.finished_at = utc_now_iso()
            self._persist(job)
        except JobCancelled as exc:
            log.info("analysis %s cancelled", job.analysis_id)
            self._fail(
                job,
                run_dir,
                error=str(exc),
                detail=f"cancelled during {job.stage}",
            )
        except JobDeadlineExceeded as exc:
            log.warning(
                "analysis %s exceeded its deadline after %ss",
                job.analysis_id,
                max_seconds,
            )
            self._fail(
                job,
                run_dir,
                error=str(exc),
                detail=f"deadline exceeded during {job.stage}",
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            log.exception("analysis %s failed", job.analysis_id)
            self._fail(
                job,
                run_dir,
                error=str(exc) or type(exc).__name__,
                detail=f"failed during {job.stage}",
            )
        finally:
            if not self.settings.keep_source:
                self._remove_source(source_path)

    def _fail(self, job: Job, run_dir: Path, *, error: str, detail: str) -> None:
        """Record a truthful terminal failure and remove partial artifacts."""
        job.status = "failed"
        job.error = error
        job.finished_at = utc_now_iso()
        job.detail = detail
        self._cleanup_incomplete(run_dir)
        self._persist(job)

    @staticmethod
    def _cleanup_incomplete(run_dir: Path) -> None:
        """Keep status.json, drop partial clips and unfinished documents."""
        shutil.rmtree(run_dir / "clips", ignore_errors=True)
        for name in ("result.json", "diagnostics.json"):
            try:
                (run_dir / name).unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove partial %s", name, exc_info=True)

    def _remove_source(self, source_path: Path) -> None:
        try:
            if source_path.is_file():
                source_path.unlink()
            parent = source_path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            log.warning("could not remove temporary source %s", source_path, exc_info=True)

    def _persist(self, job: Job) -> None:
        try:
            _atomic_write_json(
                self.settings.runs_dir / job.analysis_id / "status.json",
                job.as_dict(),
            )
        except OSError:
            log.warning("could not persist status for %s", job.analysis_id, exc_info=True)

    # ── Restart recovery ────────────────────────────────────────────────────
    def load_existing(self) -> None:
        if not self.settings.runs_dir.is_dir():
            return
        for run_dir in sorted(self.settings.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "status.json"
            if not status_path.is_file():
                continue
            try:
                document = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("ignoring unreadable status document %s", status_path)
                continue
            try:
                job = Job(
                    analysis_id=str(document["analysis_id"]),
                    source_name=str(document.get("source_name") or "unknown"),
                    status=str(document.get("status") or "failed"),
                    stage=str(document.get("stage") or "failed"),
                    substage=document.get("substage"),
                    progress=float(document.get("progress") or 0.0),
                    detail=document.get("detail"),
                    error=document.get("error"),
                    frames_done=document.get("frames_done"),
                    frames_total=document.get("frames_total"),
                    created_at=str(document.get("created_at") or utc_now_iso()),
                    started_at=document.get("started_at"),
                    finished_at=document.get("finished_at"),
                    bytes=int(document.get("bytes") or 0),
                )
            except (KeyError, TypeError, ValueError):
                log.warning("ignoring malformed status document %s", status_path)
                continue

            if job.status not in TERMINAL_STATUSES:
                job.status = "failed"
                job.stage = "failed"
                job.substage = None
                job.error = "the server restarted before this analysis finished"
                job.detail = "restart recovery"
                job.finished_at = utc_now_iso()
                self._cleanup_incomplete(run_dir)
                self._persist(job)
            with self._lock:
                self._jobs[job.analysis_id] = job


def cleanup_stale_temp(settings: Settings, *, max_age_seconds: int = 24 * 60 * 60) -> None:
    """Remove abandoned upload fragments, leaving analysis runs untouched."""
    if not settings.temp_dir.is_dir():
        return
    cutoff = time.time() - max_age_seconds
    for path in settings.temp_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            log.warning("could not clean temporary file %s", path, exc_info=True)


__all__ = [
    "Job",
    "JobCancelled",
    "JobDeadlineExceeded",
    "JobRegistry",
    "TERMINAL_STATUSES",
    "cleanup_stale_temp",
    "utc_now_iso",
]
