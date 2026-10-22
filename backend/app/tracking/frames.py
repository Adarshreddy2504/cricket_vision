"""
frames.py — Frame validation shared by the detector and the tracking service.

Both accept a frame from the outside world, and both must refuse the same
shapes for the same reasons. Keeping the rule in one place means the detector
cannot accept a frame the service would have rejected, and vice versa.
"""

from __future__ import annotations

import numpy as np

from .errors import InvalidFrameError


def validate_frame(frame) -> np.ndarray:
    """
    Check that *frame* is a usable BGR image and return it unchanged.

    Raises InvalidFrameError (a ValueError) for anything that is not an
    `H x W x 3` uint8-or-wider image with a non-zero size. The dtype is not
    constrained: OpenCV will happily read a float image, and refusing one
    would be a restriction the caller never asked for.
    """
    if frame is None:
        raise InvalidFrameError("the frame is None")
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(
            f"the frame must be a numpy array, got {type(frame).__name__}"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InvalidFrameError(
            f"the frame must be an H x W x 3 BGR image, got shape {frame.shape}"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise InvalidFrameError(f"the frame is empty: shape {frame.shape}")
    return frame
