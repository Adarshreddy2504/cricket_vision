"""Runtime settings and filesystem layout for the HTTP server.

All generated files live under one data root. That root is deliberately outside
the source tree's tracked content and is ignored by git. The defaults match the
documented Part 5 contract; every value can be overridden from the environment.

The model weights are never downloaded or bundled. Their *availability* is a
diagnostic reported by `/api/health`, and a missing ball detector is a loud job
failure rather than an empty analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# backend/app/api/config.py -> app -> backend -> project
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_DATA_DIR = PROJECT_DIR / "data"

#: The contract's upload ceiling: 2 GiB.
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # The frontend's Vite dev server is configured for port 8080.
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


@dataclass(frozen=True)
class Settings:
    """One immutable description of where the server writes and how it runs."""

    data_dir: Path
    run_inline: bool = False
    keep_source: bool = False
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    clip_pre_roll_frames: int = 10
    clip_post_roll_frames: int = 10
    max_clip_frames: int = 600
    # A processing deadline is a safety net for a stuck job, not a target.
    # It is checked between measured progress updates, so a single blocking
    # model call can overrun it before the next check.
    max_job_seconds: float = 3600.0
    # Abandoned upload fragments older than this are removed at startup.
    temp_max_age_seconds: int = 24 * 60 * 60
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.uploads_dir, self.runs_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = _env("CRICKET_DATA_DIR")
        origins = _env("CRICKET_CORS_ORIGINS")
        if data_dir:
            configured = Path(data_dir).expanduser()
            data_dir_path = (
                configured
                if configured.is_absolute()
                else (PROJECT_DIR / configured).resolve()
            )
        else:
            data_dir_path = DEFAULT_DATA_DIR
        settings = cls(
            data_dir=data_dir_path,
            run_inline=_env_bool("CRICKET_RUN_INLINE", False),
            keep_source=_env_bool("CRICKET_KEEP_SOURCE", False),
            max_upload_bytes=_env_int(
                "CRICKET_MAX_UPLOAD_BYTES",
                DEFAULT_MAX_UPLOAD_BYTES,
                minimum=1,
            ),
            clip_pre_roll_frames=_env_int("CRICKET_CLIP_PRE_ROLL", 10, minimum=0),
            clip_post_roll_frames=_env_int("CRICKET_CLIP_POST_ROLL", 10, minimum=0),
            max_clip_frames=_env_int("CRICKET_CLIP_MAX_FRAMES", 600, minimum=1),
            max_job_seconds=_env_int(
                "CRICKET_MAX_JOB_SECONDS", 3600, minimum=1
            ),
            temp_max_age_seconds=_env_int(
                "CRICKET_TEMP_MAX_AGE_SECONDS",
                24 * 60 * 60,
                minimum=60,
            ),
            cors_origins=(
                tuple(part.strip() for part in origins.split(",") if part.strip())
                if origins
                else DEFAULT_CORS_ORIGINS
            ),
        )
        return settings

    def with_data_dir(self, data_dir: Path) -> Settings:
        """A copy rooted elsewhere, used by tests and embedding callers."""
        return replace(self, data_dir=Path(data_dir))


__all__ = [
    "BACKEND_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "PROJECT_DIR",
    "Settings",
]
