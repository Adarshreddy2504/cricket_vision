"""
detector.py — YOLO-based cricket ball detector.

THE ONLY BALL DETECTOR IN THIS MODULE.

Responsibilities:
  - Load the YOLO11 model once, from a path that is never bundled with the
    source (see `app/core/config.py`).
  - Run inference on a resized frame, for speed.
  - Return the best ball bounding-box centre, in full-resolution pixels.
  - Reject obviously-wrong detections: the wrong class, or a bounding box
    whose size cannot be a cricket ball.
  - Return the winning box's confidence alongside the centre, so the tracker
    and any later stage can tell how much the measurement is worth.

THE COORDINATE SPACE IS THE CONTRACT
------------------------------------
Every coordinate this module returns is in the pixel space of the frame that
was handed in. The frame is downscaled for inference and the answer is scaled
back by exactly the factor it was downscaled by, so a caller never has to
think about it. A detector that returned resized coordinates would produce a
ball that appears to move at half its real speed.

THE MODEL IS EXTERNAL
---------------------
The checkpoint is not tracked in git. The loader never downloads anything:
if the weights are not on disk the detector reports that it is unavailable,
and the tracking service refuses to run. A missing file must never turn into
a network fetch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.config import BALL_MODEL_PATH
from ..core.constants import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_MAX_DIAG_PX,
    DETECTOR_MIN_DIAG_PX,
    DETECTOR_RESIZE_H,
    DETECTOR_RESIZE_W,
)
from .frames import validate_frame

log = logging.getLogger(__name__)

# The class id the ball detector uses. Every other class is refused.
BALL_CLASS_ID = 0


class BallDetector:
    """
    Wraps YOLO inference for cricket ball detection.

    The model may be injected, which is how the tests exercise the selection,
    filtering and coordinate-scaling logic without the weights being present.
    An injected model must implement the `ultralytics.YOLO` contract:
    `predict(source=..., conf=..., verbose=..., device=...)` returning a list
    whose first element has a `boxes` attribute, either None or an object
    whose `data` is an `n x 6` array of `x1, y1, x2, y2, score, cls`.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: Any = None,
        conf_threshold: float = DETECTOR_CONF_THRESHOLD,
        resize_width: int = DETECTOR_RESIZE_W,
        resize_height: int = DETECTOR_RESIZE_H,
        min_diag_px: int = DETECTOR_MIN_DIAG_PX,
        max_diag_px: int = DETECTOR_MAX_DIAG_PX,
        device: str | None = None,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.resize_width = int(resize_width)
        self.resize_height = int(resize_height)
        self.min_diag_px = int(min_diag_px)
        self.max_diag_px = int(max_diag_px)
        self._model_path = Path(model_path) if model_path is not None else Path(BALL_MODEL_PATH)

        self._model = model
        self._load_error: str | None = None
        if model is None:
            self._model, self._load_error, auto_device = self._load(self._model_path)
            self._device = auto_device if device in (None, "auto") else device
        else:
            self._device = "cpu" if device in (None, "auto") else device

    # ── Construction helpers ──────────────────────────────────────────────────
    @staticmethod
    def _load(path: Path):
        """
        Load the YOLO weights, or explain why they could not be loaded.

        Returns `(model, error, device)`. `model` is None when the weights are
        missing or the runtime is unavailable; `error` then says which.

        The file is checked BEFORE the runtime is imported, and it is never
        downloaded: a missing checkpoint has to be reported, not fetched.
        """
        if not path.is_file():
            return None, f"the model weights are not at {path}", "cpu"
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            return None, f"the detector runtime is not available: {exc}", "cpu"
        try:
            model = YOLO(str(path))
        except Exception as exc:
            return None, f"failed to load the model at {path}: {exc}", "cpu"
        device = 0 if torch.cuda.is_available() else "cpu"
        return model, None, device

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """True when a usable model is loaded."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Why the model is unavailable, or None when it loaded."""
        return self._load_error

    @property
    def model_path(self) -> Path:
        """Where the weights were looked for."""
        return self._model_path

    @property
    def model(self) -> Any:
        """The loaded model object, or None."""
        return self._model

    @property
    def device(self) -> str | int:
        """The inference device ultralytics was told to use."""
        return self._device

    def detect(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Run inference on *frame* (full-resolution BGR).

        Returns (cx, cy) in full-resolution pixel coords, or None.
        """
        cx, cy, _conf = self.detect_with_confidence(frame)
        if cx is None:
            return None
        return (cx, cy)

    def detect_with_confidence(
        self, frame: np.ndarray
    ) -> tuple[int | None, int | None, float | None]:
        """
        Same as :meth:`detect` but also returns the winning box confidence.

        Returned coordinates are ALWAYS in full-resolution pixel space, scaled
        back by (w_full / resize_width, h_full / resize_height).

        Returns (None, None, None) when the detector has no model or the
        model found nothing plausible.
        """
        frame = validate_frame(frame)

        if self._model is None:
            return None, None, None

        h_full, w_full = frame.shape[:2]
        small = cv2.resize(frame, (self.resize_width, self.resize_height))
        result = self._model.predict(
            source=small, conf=self.conf_threshold, verbose=False, device=self._device
        )[0]

        if result is None or getattr(result, "boxes", None) is None:
            return None, None, None

        sx = w_full / self.resize_width
        sy = h_full / self.resize_height

        best_center = None
        best_conf = -1.0

        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            if int(cls) != BALL_CLASS_ID:
                continue

            # Size sanity check, on the resized coordinates.
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if not (self.min_diag_px <= diag <= self.max_diag_px):
                continue

            # Prefer the highest-confidence detection.
            if score > best_conf:
                best_conf = score
                best_center = (
                    int((x1 + x2) / 2 * sx),
                    int((y1 + y2) / 2 * sy),
                )

        if best_center is None:
            return None, None, None

        return best_center[0], best_center[1], round(float(best_conf), 4)
