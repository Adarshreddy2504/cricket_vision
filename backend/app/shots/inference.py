"""
inference.py — Loading the shot classifier and labelling one delivery.

WHAT THIS MODULE OWNS
---------------------
Checkpoint discovery and verification, the strict load, the forward pass, output
validation, and the decision to fail ONE delivery rather than an entire
analysis. Preprocessing lives in `preprocessing.py`; the architecture lives in
`model.py`.

THREE THINGS IT REFUSES TO DO
-----------------------------
  * Download anything. The backbone is constructed with `weights=None`; the
    checkpoint is the only source of weights.
  * Load a pickle. `torch.load(..., weights_only=True)` is used, and the real
    Lightning checkpoint was verified to load under it. A checkpoint that
    requires arbitrary unpickling is rejected, not executed.
  * Invent a label. A missing checkpoint, unusable frames or a failed forward
    pass produces a `ShotClassification` whose `status` names what happened and
    whose `error` says why. `label`, `score` and `class_probabilities` are then
    absent, never filled with a default class or a made-up confidence.

INPUT SOURCES, IN ORDER
-----------------------
  1. Frames already retained by the tracking pass.
  2. The written delivery clip, decoded sequentially.

The source video is never reopened. `input_source` records which path actually
ran, so a label produced from memory is distinguishable from one produced by a
clip re-read.
"""

from __future__ import annotations

import hashlib
import logging
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..schemas.shots import ShotClassification, ShotPrediction
from .config import PROJECT_DIR, ShotConfig
from .constants import (
    SHOT_CLASSES,
    SHOT_INPUT_SHAPE,
    SHOT_N_FRAMES,
    SHOT_NUM_CLASSES,
    SHOT_TEMPORAL_DIM,
)
from .errors import (
    ShotError,
    ShotInferenceError,
    ShotInputError,
    ShotModelUnavailable,
)
from .model import ImprovedSOTAModel
from .preprocessing import (
    FrameSource,
    frames_to_tensor,
    sample_clip_frames,
    sample_indices,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedShotModel:
    """A model plus the provenance needed to explain a label later."""

    module: torch.nn.Module
    path: Path | None
    size_bytes: int | None
    md5: str | None
    device: torch.device


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Unwrap a checkpoint and remove wrapper prefixes and the training buffer."""
    if not isinstance(checkpoint, dict):
        raise ShotModelUnavailable(
            "unsupported checkpoint format; expected a dictionary, got "
            f"{type(checkpoint).__name__}"
        )

    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break

    if not isinstance(checkpoint, dict):
        raise ShotModelUnavailable(
            "checkpoint state_dict is not a dictionary, got "
            f"{type(checkpoint).__name__}"
        )

    cleaned: dict[str, Any] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ShotModelUnavailable(f"checkpoint key is not a string: {key!r}")
        new_key = key
        # DataParallel prepends `module.`, some Lightning wrappers `model.`.
        while new_key.startswith("module.") or new_key.startswith("model."):
            new_key = new_key.split(".", 1)[1]
        # The deployment architecture recreates this buffer; the checkpoint's
        # copy is a class-weighted training artefact with no inference meaning.
        if new_key == "class_weights":
            continue
        cleaned[new_key] = value

    if not cleaned:
        raise ShotModelUnavailable("checkpoint contains no state-dict entries")
    return cleaned


def resolve_device(spec: str = "auto") -> torch.device:
    """
    Turn `auto`/`cpu`/`cuda`/`cuda:N` into a usable device.

    Raises `ShotModelUnavailable` when the request cannot be honoured: a caller
    that asked for CUDA and got CPU silently would get a much slower analysis
    with no indication why.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ShotModelUnavailable(f"device must be a non-empty string, got {spec!r}")

    normalized = spec.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ShotModelUnavailable("CUDA was requested but is not available")
        return torch.device("cuda")
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1]
        if not index_text.isdigit():
            raise ShotModelUnavailable(f"invalid CUDA device index in {spec!r}")
        index = int(index_text)
        if not torch.cuda.is_available():
            raise ShotModelUnavailable(f"CUDA device {index} was requested but CUDA is unavailable")
        if index >= torch.cuda.device_count():
            raise ShotModelUnavailable(
                f"CUDA device {index} was requested but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
        return torch.device(f"cuda:{index}")

    raise ShotModelUnavailable(
        f"unsupported device {spec!r}; use 'auto', 'cpu', 'cuda' or 'cuda:N'"
    )


def load_shot_model(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    expected_md5: str | None = None,
    model_factory: Callable[..., torch.nn.Module] | None = None,
) -> LoadedShotModel:
    """
    Load the checkpoint into the architecture, strictly.

    `strict=True` is essential: it turns an architecture/checkpoint mismatch
    into a loud failure at load time instead of leaving randomly initialised
    layers to produce confident, wrong shot labels.

    Parameters
    ----------
    checkpoint_path
        An existing file. Missing, empty, corrupt and architecture-mismatched
        files all raise `ShotModelUnavailable` with the reason.
    device
        Passed to `resolve_device`.
    expected_md5
        Optional integrity check. A mismatch is fatal; no fallback checkpoint is
        tried.
    model_factory
        Injection point for tests. Defaults to `ImprovedSOTAModel` and is called
        with `num_classes`, `temporal_dim` and `n_frames` keyword arguments.

    Security
    --------
    `weights_only=True` restricts unpickling to tensors and primitive
    containers. The project's Lightning checkpoint loads under that restriction;
    an arbitrary pickle does not, and is refused rather than executed.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise ShotModelUnavailable(f"shot checkpoint not found: {path}")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ShotModelUnavailable(f"shot checkpoint is empty: {path}")

    md5 = _file_md5(path)
    if expected_md5 is not None:
        expected = expected_md5.strip().lower()
        if md5.lower() != expected:
            raise ShotModelUnavailable(
                f"shot checkpoint checksum mismatch for {path}: "
                f"expected {expected}, got {md5}"
            )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any read/unpickle failure is "unavailable"
        raise ShotModelUnavailable(f"could not read shot checkpoint {path}: {exc}") from exc

    state_dict = _extract_state_dict(checkpoint)

    factory = model_factory if model_factory is not None else ImprovedSOTAModel
    try:
        model = factory(
            num_classes=SHOT_NUM_CLASSES,
            temporal_dim=SHOT_TEMPORAL_DIM,
            n_frames=SHOT_N_FRAMES,
        )
    except Exception as exc:  # noqa: BLE001 - a factory failure means no model
        raise ShotModelUnavailable(f"could not construct the shot model: {exc}") from exc

    fresh_state = model.state_dict()
    if "class_weights" in fresh_state:
        state_dict["class_weights"] = fresh_state["class_weights"].detach().clone()

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001 - any mismatch makes the weights unusable
        raise ShotModelUnavailable(
            f"shot checkpoint does not match the architecture ({path}): {exc}"
        ) from exc

    torch_device = resolve_device(device)
    try:
        model = model.to(torch_device)
    except Exception as exc:  # noqa: BLE001 - the model cannot be placed on the device
        raise ShotModelUnavailable(
            f"could not move the shot model to {torch_device}: {exc}"
        ) from exc

    model.eval()
    log.info("Shot model loaded from %s on %s", path, torch_device)

    return LoadedShotModel(
        module=model,
        path=path.resolve(),
        size_bytes=size_bytes,
        md5=md5,
        device=torch_device,
    )


def _infer_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


@torch.inference_mode()
def predict_shot(
    module: torch.nn.Module,
    frames: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> ShotPrediction:
    """
    Run one `(1, 30, 3, 224, 224)` tensor and return the validated prediction.

    Everything the model returns is checked: shape, dtype, finiteness, and a
    probability vector that sums to 1. A model that returns a different number
    of logits cannot be silently read as the 10-class vocabulary.
    """
    if not isinstance(frames, torch.Tensor):
        raise ShotInputError(f"frames must be a torch.Tensor, got {type(frames).__name__}")
    if tuple(frames.shape) != SHOT_INPUT_SHAPE:
        raise ShotInputError(
            f"expected tensor shape {SHOT_INPUT_SHAPE}, got {tuple(frames.shape)}"
        )
    if not frames.is_floating_point():
        raise ShotInputError(f"expected a floating-point tensor, got {frames.dtype}")

    target_device = torch.device(device) if device is not None else _infer_device(module)
    moved = frames.to(target_device)

    try:
        logits = module(moved)
    except ShotError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as an inference failure
        raise ShotInferenceError(f"model forward pass failed: {exc}") from exc

    if not isinstance(logits, torch.Tensor):
        raise ShotInferenceError(
            f"model returned {type(logits).__name__}, not a torch.Tensor"
        )
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ShotInferenceError(f"unexpected model output shape {tuple(logits.shape)}")
    if logits.shape[1] != SHOT_NUM_CLASSES:
        raise ShotInferenceError(
            f"expected {SHOT_NUM_CLASSES} class logits, got {logits.shape[1]}"
        )
    if not logits.is_floating_point():
        raise ShotInferenceError(f"model logits have dtype {logits.dtype}, not floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ShotInferenceError("model returned non-finite logits")

    probabilities = F.softmax(logits, dim=1).reshape(-1)
    values = probabilities.detach().to("cpu").tolist()

    best_index = max(range(len(values)), key=values.__getitem__)
    label = SHOT_CLASSES[best_index]
    rounded = {
        name: round(float(value), 6) for name, value in zip(SHOT_CLASSES, values, strict=True)
    }
    return ShotPrediction(
        label=label,
        score=rounded[label],
        class_probabilities=rounded,
    )


class ShotClassifier:
    """
    Classifies one delivery's shot, or records exactly why it could not.

    Construct with `from_checkpoint()` when a missing model should be an error,
    or with `unavailable()` when the caller wants a classifier that produces
    `model_unavailable` records instead of raising. `classify()` itself never
    raises for delivery data.
    """

    def __init__(
        self,
        loaded: LoadedShotModel | None,
        *,
        unavailable_reason: str | None = None,
    ) -> None:
        if loaded is None and not unavailable_reason:
            raise ValueError("a classifier needs a model or an unavailable reason")
        if loaded is not None and unavailable_reason:
            raise ValueError("cannot supply both a model and an unavailable reason")
        self._loaded = loaded
        self._unavailable_reason = unavailable_reason

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | None = None,
        expected_md5: str | None = None,
        model_factory: Callable[..., torch.nn.Module] | None = None,
    ) -> ShotClassifier:
        """
        Load a checkpoint, raising `ShotModelUnavailable` on any failure.

        `None` for `checkpoint_path`, `device` and `expected_md5` means "read it
        from the environment / the documented default", not "skip the check".
        """
        config = ShotConfig.from_env()

        if checkpoint_path is None:
            try:
                path = config.resolve_model_path()
            except FileNotFoundError as exc:
                raise ShotModelUnavailable(str(exc)) from exc
        else:
            path = Path(checkpoint_path).expanduser()
            if not path.is_absolute():
                path = (PROJECT_DIR / path).resolve()

        loaded = load_shot_model(
            path,
            device=device if device is not None else config.device,
            expected_md5=expected_md5 if expected_md5 is not None else config.expected_md5,
            model_factory=model_factory,
        )
        return cls(loaded)

    @classmethod
    def unavailable(cls, reason: str) -> ShotClassifier:
        """A classifier that records `model_unavailable` for every delivery."""
        return cls(None, unavailable_reason=reason)

    def _model_metadata(self) -> dict:
        if self._loaded is None:
            return {}
        return {
            "model_path": str(self._loaded.path) if self._loaded.path is not None else None,
            "model_size_bytes": self._loaded.size_bytes,
            "model_md5": self._loaded.md5,
            "model_device": str(self._loaded.device),
        }

    def classify(
        self,
        clip_path: str | Path | None = None,
        in_memory_frames: FrameSource | None = None,
        expected_total_frames: int | None = None,
        delivery_id: int | str | None = None,
        frame_offset: int | None = None,
    ) -> ShotClassification:
        """
        Classify one delivery, preferring frames already in memory.

        Parameters
        ----------
        clip_path
            The written clip, used only as a fallback source of pixels.
        in_memory_frames
            Either a mapping keyed by 0-based clip position (a sparse subset),
            or a dense sequence indexed by clip position. A dense sequence
            without `expected_total_frames` is treated as the whole clip.
        expected_total_frames
            The clip's frame count, required for a sparse mapping.
        delivery_id
            Only used to make log lines attributable.
        frame_offset
            `clip_start_frame - 1`. When given, the sampled positions are also
            recorded as source frame numbers.

        Never raises for delivery data: failures become a status and an error on
        the returned record. A malformed `frame_offset` is a caller bug and does
        raise `ValueError`.
        """
        if frame_offset is not None:
            if (
                isinstance(frame_offset, bool)
                or not isinstance(frame_offset, numbers.Integral)
                or frame_offset < 0
            ):
                raise ValueError(
                    f"frame_offset must be a non-negative integer, got {frame_offset!r}"
                )
            frame_offset = int(frame_offset)

        if self._loaded is None:
            log.warning(
                "Shot classification unavailable for delivery %s: %s",
                delivery_id,
                self._unavailable_reason,
            )
            return ShotClassification(
                status="model_unavailable",
                error=self._unavailable_reason,
            )

        frames, input_source, positions, gather_error = self._gather(
            clip_path,
            in_memory_frames,
            expected_total_frames,
        )

        if frames is None:
            log.warning(
                "Shot inference skipped for delivery %s: %s",
                delivery_id,
                gather_error,
            )
            return ShotClassification(
                status="invalid_input",
                input_source=input_source,
                error=gather_error,
                **self._model_metadata(),
            )

        frames_sampled = len(frames)
        unique_positions = len(set(positions))
        repeated = unique_positions < frames_sampled
        original_frames = (
            [position + frame_offset for position in positions]
            if frame_offset is not None
            else None
        )

        log.debug(
            "shot input: delivery=%s source=%s clip_positions=%s original_frames=%s "
            "unique=%d repeated=%s",
            delivery_id,
            input_source,
            positions,
            original_frames,
            unique_positions,
            repeated,
        )

        common = {
            "input_source": input_source,
            "frames_sampled": frames_sampled,
            "unique_frames_sampled": unique_positions,
            "frames_repeated": repeated,
            "sampled_clip_frames": list(positions),
            "sampled_original_frames": original_frames,
            **self._model_metadata(),
        }

        try:
            tensor = frames_to_tensor(frames)
        except Exception as exc:  # noqa: BLE001 - bad pixels are an input failure
            log.error("Shot input unusable for delivery %s: %s", delivery_id, exc)
            return ShotClassification(
                status="invalid_input",
                error=str(exc) or type(exc).__name__,
                **common,
            )

        try:
            prediction = predict_shot(
                self._loaded.module,
                tensor,
                device=self._loaded.device,
            )
        except Exception as exc:  # noqa: BLE001 - one delivery must not fail the analysis
            log.error("Shot inference failed for delivery %s: %s", delivery_id, exc)
            return ShotClassification(
                status="inference_failed",
                error=str(exc) or type(exc).__name__,
                **common,
            )

        return ShotClassification(
            status="ok",
            label=prediction.label,
            score=prediction.score,
            class_probabilities=prediction.class_probabilities,
            **common,
        )

    @staticmethod
    def _positions_for(
        source: FrameSource,
        expected_total_frames: int | None,
    ) -> tuple[list[int] | None, str | None]:
        """Sampled clip positions for *source*, or a reason they cannot be known."""
        if isinstance(source, Mapping):
            if expected_total_frames is None:
                return (
                    None,
                    "in-memory frames are keyed by clip position, so "
                    "expected_total_frames is required",
                )
            total = expected_total_frames
        else:
            total = expected_total_frames if expected_total_frames is not None else len(source)

        try:
            return sample_indices(total), None
        except ShotInputError as exc:
            return None, str(exc)

    @classmethod
    def _gather(
        cls,
        clip_path: str | Path | None,
        in_memory_frames: FrameSource | None,
        expected_total_frames: int | None,
    ) -> tuple[list | None, str, list[int], str | None]:
        """
        Return `(frames, input_source, sampled_positions, reason)`.

        `frames` is None when no usable source exists. Positions are returned
        alongside the frames so the caller can audit the mapping without
        recomputing it, and so both paths provably use the same sample.
        """
        reasons: list[str] = []

        if in_memory_frames is not None:
            if isinstance(in_memory_frames, (str, bytes)):
                reasons.append("in_memory_frames is a string, not a frame sequence")
            else:
                positions, position_error = cls._positions_for(
                    in_memory_frames,
                    expected_total_frames,
                )
                if positions is None:
                    reasons.append(position_error or "could not sample in-memory frames")
                elif not positions:
                    reasons.append("in-memory frame source contains no frames")
                elif isinstance(in_memory_frames, Mapping):
                    missing = [
                        position for position in positions if position not in in_memory_frames
                    ]
                    if missing:
                        reasons.append(
                            f"{len(missing)} of {len(positions)} sampled positions absent "
                            f"(first {missing[0]})"
                        )
                    else:
                        return (
                            [in_memory_frames[position] for position in positions],
                            "in_memory",
                            positions,
                            None,
                        )
                else:
                    if max(positions) < len(in_memory_frames):
                        return (
                            [in_memory_frames[position] for position in positions],
                            "in_memory",
                            positions,
                            None,
                        )
                    reasons.append(
                        f"{len(in_memory_frames)} frames retained but clip position "
                        f"{max(positions)} was required"
                    )

        if clip_path is not None:
            path = Path(clip_path)
            if not path.is_file():
                reasons.append(f"clip not found: {path}")
            else:
                try:
                    frames, positions = sample_clip_frames(path)
                except ShotInputError as exc:
                    reasons.append(str(exc))
                else:
                    if len(frames) == SHOT_N_FRAMES:
                        return frames, "clip_file", positions, None
                    reasons.append(f"clip yielded {len(frames)} of {SHOT_N_FRAMES} frames")

        if not reasons:
            reasons.append(
                "no frame source supplied (in_memory_frames and clip_path are both empty)"
            )
        return None, "unavailable", [], "; ".join(reasons)


__all__ = [
    "LoadedShotModel",
    "ShotClassifier",
    "load_shot_model",
    "predict_shot",
    "resolve_device",
]
