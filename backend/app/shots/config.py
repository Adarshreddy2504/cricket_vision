"""
config.py — Where the shot checkpoint lives, and how the loader is configured.

WEIGHT LOCATION
---------------
The supplied `shot_classifier.ckpt` is tracked at `backend/models/` (see
`docs/SHOTS.md`). Three environment variables describe a different copy:

    CRICKET_MODELS_DIR       the directory holding the weights
    CRICKET_SHOT_MODEL       an explicit checkpoint path (wins over the directory)
    CRICKET_SHOT_MODEL_MD5   optional expected MD5, for installing a verified copy
    CRICKET_SHOT_DEVICE      "auto" (default), "cpu", "cuda", "cuda:1", ...

The file defaults to `backend/models/shot_classifier.ckpt`, so dropping the
checkpoint there is enough. A relative path is resolved against the project
directory, not the current working directory, so the module finds the same file
however it was started.

NO SILENT DOWNLOAD, NO SILENT SUBSTITUTE
----------------------------------------
Nothing here fetches a checkpoint. `resolve_shot_model_path()` raises
`FileNotFoundError` naming every path it tried; `load_shot_model()` converts
that into `ShotModelUnavailable`. An explicitly configured path that is missing
is an error — it never falls back to a different checkpoint, because a
different checkpoint produces confident labels from different weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import DOCUMENTED_SHOT_MODEL_MD5, SHOT_MODEL_FILENAME

#: `backend/`, the import root and the parent of `models/`.
BACKEND_DIR: Path = Path(__file__).resolve().parents[2]

#: The repository root. Relative paths in the environment resolve against it.
PROJECT_DIR: Path = BACKEND_DIR.parent

DEFAULT_MODELS_DIR: Path = BACKEND_DIR / "models"

ENV_MODELS_DIR: str = "CRICKET_MODELS_DIR"
ENV_SHOT_MODEL: str = "CRICKET_SHOT_MODEL"
ENV_SHOT_MODEL_MD5: str = "CRICKET_SHOT_MODEL_MD5"
ENV_SHOT_DEVICE: str = "CRICKET_SHOT_DEVICE"

DEFAULT_DEVICE: str = "auto"


def _env(name: str) -> str | None:
    """An environment value, with empty/whitespace treated as unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _absolute(raw: str | Path) -> Path:
    """Resolve *raw* against the project directory unless it is absolute."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def resolve_models_dir(models_dir: str | Path | None = None) -> Path:
    """The directory holding model weights: argument, then env, then default."""
    if models_dir is not None:
        return _absolute(models_dir)
    env_dir = _env(ENV_MODELS_DIR)
    return _absolute(env_dir) if env_dir is not None else DEFAULT_MODELS_DIR


def resolve_shot_model_path(
    model_path: str | Path | None = None,
    *,
    models_dir: str | Path | None = None,
) -> Path:
    """
    Locate the shot checkpoint, or raise `FileNotFoundError` naming every path.

    An explicit `model_path` (or `CRICKET_SHOT_MODEL`) is authoritative: when it
    is set and absent, the default location is still reported, but it is not
    silently adopted in its place.
    """
    candidates: list[Path] = []

    if model_path is not None:
        candidates.append(_absolute(model_path))
    else:
        env_model = _env(ENV_SHOT_MODEL)
        if env_model is not None:
            candidates.append(_absolute(env_model))
        candidates.append(resolve_models_dir(models_dir) / SHOT_MODEL_FILENAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "shot classifier checkpoint not found; set CRICKET_SHOT_MODEL or place "
        f"{SHOT_MODEL_FILENAME} in the models directory. Tried:\n{tried}"
    )


def describe_weights(path: str | Path) -> str:
    """A one-line, human-readable answer to "is the checkpoint there?"."""
    path = Path(path)
    if not path.is_file():
        return f"NOT FOUND at {path}"
    return f"{path} ({path.stat().st_size / 1048576:.1f} MiB)"


@dataclass(frozen=True)
class ShotConfig:
    """Resolved configuration for loading and running the shot classifier."""

    model_path: Path | None = None
    device: str = DEFAULT_DEVICE
    expected_md5: str | None = None

    @classmethod
    def from_env(cls) -> ShotConfig:
        """Read the environment now, rather than freezing it at import time."""
        env_model = _env(ENV_SHOT_MODEL)
        return cls(
            model_path=Path(env_model) if env_model is not None else None,
            device=_env(ENV_SHOT_DEVICE) or DEFAULT_DEVICE,
            expected_md5=_env(ENV_SHOT_MODEL_MD5),
        )

    def resolve_model_path(self, *, models_dir: str | Path | None = None) -> Path:
        """The checkpoint path, or `FileNotFoundError` naming what was tried."""
        return resolve_shot_model_path(self.model_path, models_dir=models_dir)

    def as_metadata(self) -> dict:
        """
        Provenance for an analysis document.

        Never raises: a missing checkpoint is recorded as `model_available`
        False with the reason, which is exactly what a status endpoint needs.
        """
        metadata: dict = {
            "model_filename": SHOT_MODEL_FILENAME,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "device": self.device,
            "expected_md5": self.expected_md5,
            "documented_md5": DOCUMENTED_SHOT_MODEL_MD5,
            "model_available": False,
        }
        try:
            resolved = self.resolve_model_path()
        except FileNotFoundError as exc:
            metadata["model_error"] = str(exc)
            return metadata

        metadata["model_available"] = True
        metadata["model_path"] = str(resolved)
        metadata["model_size_bytes"] = resolved.stat().st_size
        return metadata


__all__ = [
    "BACKEND_DIR",
    "DEFAULT_DEVICE",
    "DEFAULT_MODELS_DIR",
    "ENV_MODELS_DIR",
    "ENV_SHOT_DEVICE",
    "ENV_SHOT_MODEL",
    "ENV_SHOT_MODEL_MD5",
    "PROJECT_DIR",
    "ShotConfig",
    "describe_weights",
    "resolve_models_dir",
    "resolve_shot_model_path",
]
