"""
detector.py — Finding the pitch in a frame.

TWO DETECTORS, TWO HONESTY LEVELS
---------------------------------
This module wraps two local YOLO checkpoints, and they are NOT equivalent:

    PitchSegmentationDetector   a segmentation model over the complete
                                wicket-to-wicket pitch. The mask boundary
                                IS the pitch's outline, so the four corners
                                it yields are a MEASUREMENT of where the
                                corners are. The calibration built from them
                                can be VALID.

    PitchBoxDetector            a detection model over the pitch-shaped
                                REGION. A box says where the pitch is, not
                                where its corners are, so the four points
                                derived from it are GUESSES. The calibration
                                built from them is APPROXIMATE and can never
                                be presented as validated.

Both report which one they are through `CornerMethod`, and the
calibration is built from that label rather than from anything the caller
remembers to pass in.

THE WEIGHTS ARE NEVER DOWNLOADED
--------------------------------
`ultralytics.YOLO("name.pt")` fetches the checkpoint from the internet when
the file is not present. Every detector here therefore checks that the
weights exist as a FILE before it builds a model, and passes an absolute
path when it does. A missing checkpoint is an explicit "unavailable" state,
never a surprise download and never a crash.

THE FRAME IS NEVER RESIZED
--------------------------
The frame goes to the model at its ORIGINAL resolution and ultralytics
letterboxes it to `imgsz` itself. Pre-resizing the frame is what squashed
frames and broke the aspect ratio in an earlier version of this project.

EVERY OUTCOME IS COUNTED
-------------------------
Each call increments exactly one of `frames_seen` -> {detections,
no_detections, unavailable, errors}, and `stats()` reports the tally. No
frame is ever dropped silently.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .calibration import (
    CornerMethod,
    PitchCalibration,
    detected_calibration,
)
from .constants import (
    BOX_TOP_INSET,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_CONF_TIE_BAND,
    DETECTOR_IMGSZ,
    MASK_MAX_AREA_FRACTION,
    MASK_MAX_ASPECT_RATIO,
    MASK_MIN_AREA_FRACTION,
    MASK_MIN_ASPECT_RATIO,
    MAX_QUAD_AREA_FRACTION,
    MIN_BOX_AREA_FRACTION,
)
from .corners import ImageSize, PitchCorners, corners_from_box
from .errors import CalibrationError, InvalidFrameError
from .geometry import PitchGeometry
from .mask_geometry import mask_centroid, quad_from_mask, validate_mask

log = logging.getLogger(__name__)

# Ultralytics' own model cache, keyed by the RESOLVED weights path. Two
# detectors in one process pay the load cost once, and the weights file is
# never copied or renamed.
_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class DetectionStatus(str, Enum):
    """What one detection attempt produced."""

    #: Four corners were found.
    OK = "ok"
    #: The model ran and found no pitch-shaped region.
    NO_DETECTION = "no_detection"
    #: There are no usable weights, so nothing could be attempted.
    UNAVAILABLE = "unavailable"
    #: The attempt failed. The reason says how.
    ERROR = "error"


@dataclass(frozen=True)
class PitchDetection:
    """
    The outcome of one detection attempt.

    `status`      what happened. The first thing a caller should read.
    `corners`     the four corners in source pixels, in the canonical
                  order. None unless the status is OK.
    `method`      what produced them, which decides how far the resulting
                  calibration can be trusted.
    `confidence`  the model's own score for the detection.
    `reason`      why there are no corners, or why the attempt failed.
    `quad_method` how the four corners were derived from the detection.
    `rejections`  per-reason tally of the candidates that were refused.
    """

    status: DetectionStatus
    frame_number: int
    image_size: ImageSize
    method: CornerMethod | None = None
    corners: PitchCorners | None = None
    confidence: float | None = None
    reason: str | None = None
    quad_method: str | None = None
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Were four corners found?"""
        return self.status is DetectionStatus.OK and self.corners is not None

    @property
    def is_approximate(self) -> bool:
        """Are these corners a guess rather than a measurement?"""
        return self.method is CornerMethod.BOX_ESTIMATE

    def as_dict(self) -> dict:
        """The persisted form."""
        return {
            "status": self.status.value,
            "frame_number": self.frame_number,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "method": None if self.method is None else self.method.value,
            "corners": None if self.corners is None else self.corners.as_dict(),
            "confidence": (
                None if self.confidence is None else round(float(self.confidence), 4)
            ),
            "reason": self.reason,
            "quad_method": self.quad_method,
            "rejections": dict(self.rejections),
        }


@runtime_checkable
class PitchDetector(Protocol):
    """
    The detector interface.

    `detect` never raises for an environment problem: a missing checkpoint
    is an explicit UNAVAILABLE status. It DOES raise `InvalidFrameError`
    for a frame that is not a usable BGR image, because that is a mistake
    in the caller's code rather than a fact about the world.
    """

    @property
    def available(self) -> bool:
        """Can this detector run at all?"""
        ...

    def detect(self, frame: np.ndarray, *, frame_number: int = 0) -> PitchDetection:
        """One frame in, one outcome out."""
        ...


def validate_frame(frame: np.ndarray) -> ImageSize:
    """
    The frame's `(width, height)`, or `InvalidFrameError`.

    A frame that is not a 3-channel uint8 array is refused here rather than
    deeper inside ultralytics, where it would become a confusing failure
    about tensor shapes.
    """
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(
            f"frame must be a numpy array, got {type(frame).__name__}"
        )
    if frame.ndim != 3:
        raise InvalidFrameError(
            f"frame must be a 3-dimensional image, got {frame.ndim} dimensions"
        )
    if frame.shape[2] != 3:
        raise InvalidFrameError(
            f"frame must have 3 channels (BGR), got {frame.shape[2]}"
        )
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise InvalidFrameError(f"frame has no pixels: {frame.shape}")
    if frame.dtype != np.uint8:
        raise InvalidFrameError(f"frame must be uint8, got {frame.dtype}")
    return (int(frame.shape[1]), int(frame.shape[0]))


def resolve_device(prefer: str = "auto") -> str:
    """
    The device inference actually runs on: CUDA when available (or asked
    for), else CPU — never a crash on a machine without a GPU.

    An explicit `cpu` is always honoured. An explicit `cuda*` that cannot
    be honoured falls back to CPU with a warning, because a detector that
    refuses to run is worse than a slow one.
    """
    want = (prefer or "auto").strip().lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch missing or broken: CPU it is
        cuda = False
    if want.startswith("cuda"):
        if cuda:
            return want
        log.warning(
            "pitch detection: CUDA requested (%s) but unavailable; using CPU", want
        )
        return "cpu"
    return "cuda" if cuda else "cpu"


def detection_to_calibration(
    detection: PitchDetection,
    *,
    geometry: PitchGeometry | None = None,
) -> PitchCalibration | None:
    """
    The calibration a detection justifies, or None when it justifies none.

    The provenance and the method are taken from the detection itself, so a
    box-derived quad cannot be turned into a VALID calibration by accident:
    the label that makes it approximate travels with the corners.
    """
    if not detection.ok or detection.corners is None or detection.method is None:
        return None
    return detected_calibration(
        detection.corners,
        image_size=detection.image_size,
        method=detection.method,
        confidence=detection.confidence,
        frame_number=detection.frame_number,
        geometry=geometry,
    )


@dataclass(frozen=True)
class _Candidate:
    """One accepted mask quad, awaiting selection."""

    confidence: float
    corners: PitchCorners
    quad_method: str
    center: tuple[float, float]


class _YoloPitchDetector:
    """
    The machinery both detectors share: weights, caching, counting.

    Subclasses implement `_extract`, which turns a ultralytics result into
    a `PitchDetection`.
    """

    #: Which ultralytics task this detector runs.
    task = "detect"

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: object | None = None,
        conf: float = DETECTOR_CONF_THRESHOLD,
        imgsz: int = DETECTOR_IMGSZ,
        device: str = "auto",
        min_area_fraction: float = MASK_MIN_AREA_FRACTION,
        max_area_fraction: float = MASK_MAX_AREA_FRACTION,
        min_aspect_ratio: float = MASK_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = MASK_MAX_ASPECT_RATIO,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = resolve_device(device)
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

        # A pre-loaded model may be injected, which is how the tests exercise
        # the extraction and selection logic without the weights being present.
        # An injected model must implement the `ultralytics.YOLO` contract:
        # `predict(source=..., conf=..., imgsz=..., device=..., verbose=False)`
        # returning a sequence whose first element has `boxes` and `masks`.
        self._injected = model is not None
        self._model = model
        self._load_error: str | None = None
        self._last_center: tuple[float, float] | None = None

        self.frames_seen = 0
        self.detections = 0
        self.no_detections = 0
        self.unavailable = 0
        self.errors = 0
        self.last_error: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """Can this detector run at all? False when the weights are missing."""
        if self._injected:
            return True
        return self.model_path.is_file()

    @property
    def load_error(self) -> str | None:
        """Why the weights could not be loaded, or None when they are usable."""
        return self._load_error

    @property
    def disabled_reason(self) -> str | None:
        """Why the detector cannot run, or None when it can."""
        if self._injected:
            return None
        if self.model_path.is_file():
            return None
        return f"weights not found at {self.model_path}"

    def detect(self, frame: np.ndarray, *, frame_number: int = 0) -> PitchDetection:
        """
        One frame in, one outcome out.

        Raises `InvalidFrameError` for a frame that is not a usable BGR
        image. Every other outcome is a status, because "the model is not
        there" and "the model found nothing" are different facts and a
        caller has to be able to tell them apart.
        """
        width, height = validate_frame(frame)
        image_size = (width, height)
        self.frames_seen += 1

        if not self._injected and not self.model_path.is_file():
            self.unavailable += 1
            return PitchDetection(
                status=DetectionStatus.UNAVAILABLE,
                frame_number=frame_number,
                image_size=image_size,
                reason=f"weights not found at {self.model_path}",
            )

        model = self._model_or_none()
        if model is None:
            self.unavailable += 1
            return PitchDetection(
                status=DetectionStatus.UNAVAILABLE,
                frame_number=frame_number,
                image_size=image_size,
                reason=self._load_error or "pitch weights unavailable",
            )

        try:
            results = model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 — a soft failure, always
            self.errors += 1
            self.last_error = str(exc)[:300]
            log.warning("Pitch detection failed: %s", self.last_error)
            return PitchDetection(
                status=DetectionStatus.ERROR,
                frame_number=frame_number,
                image_size=image_size,
                reason=self.last_error,
            )

        detection = self._extract(
            results, frame_number=frame_number, image_size=image_size
        )
        if detection.ok:
            self.detections += 1
            self._last_center = mask_centroid(detection.corners.points())
        else:
            self.no_detections += 1
        return detection

    def stats(self) -> dict:
        """
        The tally. `frames_seen == detections + no_detections + unavailable
        + errors`, always.
        """
        return {
            "task": self.task,
            "model_path": str(self.model_path),
            "model_found": self.model_path.is_file(),
            "device": self.device,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "frames_seen": self.frames_seen,
            "detections": self.detections,
            "no_detections": self.no_detections,
            "unavailable": self.unavailable,
            "errors": self.errors,
            "last_error": self.last_error,
        }

    # ── Internals ────────────────────────────────────────────────────────────
    def _extract(
        self,
        results,
        *,
        frame_number: int,
        image_size: ImageSize,
    ) -> PitchDetection:
        raise NotImplementedError

    def _model_or_none(self):
        """
        The loaded model, or None when the weights could not be loaded.

        The failure is cached, so a missing or broken checkpoint costs one
        attempt rather than one per frame. The file is checked to exist
        BEFORE `ultralytics.YOLO` is constructed: that constructor will
        fetch a checkpoint over the network when the file is not there, and
        this module never downloads weights.
        """
        if self._injected:
            return self._model
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        if not self.model_path.is_file():
            self._load_error = f"weights not found at {self.model_path}"
            return None
        key = str(self.model_path.resolve())
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
        if cached is not None:
            self._model = cached
            return cached
        try:
            from ultralytics import YOLO

            log.info("Loading local pitch weights: %s", key)
            model = YOLO(key)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"pitch weights could not be loaded from {key}: {exc}"
            log.warning("Pitch weights unavailable: %s", self._load_error)
            return None
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = model
        self._model = model
        return model

    def _select(self, candidates: Sequence[_Candidate]) -> _Candidate:
        """
        The best candidate: highest confidence, ties broken by consistency
        with the last accepted estimate.

        Consistency only decides a TIE. A 0.90 mask somewhere new beats a
        0.88 mask next door, because confidence is the model's own evidence
        and position is only ours. But when two regions are equally
        believable — two pitch-shaped patches under a split screen — the
        one where the pitch has been is the honest choice.
        """
        best_confidence = max(c.confidence for c in candidates)
        contenders = [
            c for c in candidates if c.confidence >= best_confidence - DETECTOR_CONF_TIE_BAND
        ]
        if len(contenders) == 1:
            return contenders[0]
        previous = self._last_center
        if previous is not None:
            return min(
                contenders,
                key=lambda c: (c.center[0] - previous[0]) ** 2
                + (c.center[1] - previous[1]) ** 2,
            )
        return max(contenders, key=lambda c: c.confidence)


class PitchBoxDetector(_YoloPitchDetector):
    """
    The pitch-BOX detector: where the pitch-shaped REGION is.

    The box is turned into four APPROXIMATE corners by
    `corners.corners_from_box`. Those four points are guesses: they never
    become a validated calibration, and the calibration built from them
    comes back APPROXIMATE with the reason "approximate_corners".
    """

    task = "detect"

    def _extract(
        self,
        results,
        *,
        frame_number: int,
        image_size: ImageSize,
    ) -> PitchDetection:
        result = results[0] if results else None
        if result is None:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="model returned no results",
            )

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="no_pitch_box",
            )

        best_index = int(boxes.conf.argmax())
        class_id = _int_at(getattr(boxes, "cls", None), best_index)
        if class_id is not None and class_id != 0:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="wrong_class",
            )

        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_index].tolist()]
        confidence = float(boxes.conf[best_index])

        width, height = image_size
        area_fraction = ((x2 - x1) * (y2 - y1)) / float(width * height)
        if area_fraction < MIN_BOX_AREA_FRACTION:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="pitch_box_too_small",
            )
        if area_fraction > MAX_QUAD_AREA_FRACTION:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="pitch_box_too_large",
            )

        try:
            corners = corners_from_box(x1, y1, x2, y2, top_inset=BOX_TOP_INSET)
        except CalibrationError as exc:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason=str(exc),
            )

        return PitchDetection(
            status=DetectionStatus.OK,
            frame_number=frame_number,
            image_size=image_size,
            method=CornerMethod.BOX_ESTIMATE,
            corners=corners,
            confidence=confidence,
            quad_method="box_estimate",
        )


class PitchSegmentationDetector(_YoloPitchDetector):
    """
    The pitch-SEGMENTATION detector: where the pitch's OUTLINE is.

    The mask boundary is simplified to four corners by `mask_geometry`, and
    those four corners are a measurement of where the corners are — so the
    calibration built from them can be VALID.

    Every instance the model returns is either ACCEPTED or REJECTED WITH A
    REASON — too small, too large, too square, too elongated, outside the
    frame, not simplifiable to a quad, wrong class — and the tally is
    reported on the detection.
    """

    task = "segment"

    def _extract(
        self,
        results,
        *,
        frame_number: int,
        image_size: ImageSize,
    ) -> PitchDetection:
        result = results[0] if results else None
        if result is None:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="model returned no results",
            )

        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if not polygons:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason="no_pitch_mask",
            )

        boxes = getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None) if boxes is not None else None
        class_ids = getattr(boxes, "cls", None) if boxes is not None else None

        candidates: list[_Candidate] = []
        rejections: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejections[reason] = rejections.get(reason, 0) + 1

        for index, polygon in enumerate(polygons):
            confidence = _float_at(confidences, index)
            if confidence is None or confidence < self.conf:
                reject("below_confidence")
                continue
            class_id = _int_at(class_ids, index)
            if class_id is not None and class_id != 0:
                reject("wrong_class")
                continue

            points = _as_points(polygon)
            ok, reason = validate_mask(
                points,
                image_size,
                min_area_fraction=self.min_area_fraction,
                max_area_fraction=self.max_area_fraction,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
            )
            if not ok:
                reject(reason)
                continue

            quad, quad_method = quad_from_mask(points)
            if quad is None:
                reject("quad_unavailable")
                continue
            center = mask_centroid(points)
            if center is None:
                reject("quad_unavailable")
                continue

            candidates.append(
                _Candidate(
                    confidence=float(confidence),
                    corners=PitchCorners(*quad),
                    quad_method=quad_method,
                    center=center,
                )
            )

        if not candidates:
            return PitchDetection(
                status=DetectionStatus.NO_DETECTION,
                frame_number=frame_number,
                image_size=image_size,
                reason=_reason_text(rejections),
                rejections=rejections,
            )

        best = self._select(candidates)
        return PitchDetection(
            status=DetectionStatus.OK,
            frame_number=frame_number,
            image_size=image_size,
            method=CornerMethod.SEGMENTATION,
            corners=best.corners,
            confidence=best.confidence,
            quad_method=best.quad_method,
            rejections=rejections,
        )


# ── helpers ──────────────────────────────────────────────────────────────────
def _as_points(polygon) -> list[tuple[float, float]]:
    """An (N, 2) array/list of mask vertices, or a best-effort partial read."""
    try:
        array = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in array.tolist()]
    except Exception:  # noqa: BLE001 — malformed polygon is a refusal, not a crash
        try:
            return [(float(p[0]), float(p[1])) for p in polygon]
        except Exception:  # noqa: BLE001
            return []


def _float_at(values, index: int) -> float | None:
    if values is None:
        return None
    try:
        return float(values[index])
    except Exception:  # noqa: BLE001
        return None


def _int_at(values, index: int) -> int | None:
    if values is None:
        return None
    try:
        return int(values[index])
    except Exception:  # noqa: BLE001
        return None


def _reason_text(rejections: dict[str, int]) -> str:
    if not rejections:
        return "model returned no masks"
    parts = [f"{count}x {reason}" for reason, count in sorted(rejections.items())]
    return "no usable pitch mask (" + ", ".join(parts) + ")"


__all__ = [
    "DetectionStatus",
    "PitchBoxDetector",
    "PitchDetection",
    "PitchDetector",
    "PitchSegmentationDetector",
    "detection_to_calibration",
    "resolve_device",
    "validate_frame",
]
