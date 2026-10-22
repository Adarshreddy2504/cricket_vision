"""
tracking — The ball-tracking module.

The public surface of the module:

    BallDetector      finds the ball in a frame (YOLO), in source pixels.

The errors are exported here too, so a caller can catch `TrackingError` and be
sure it has caught everything this module throws.
"""

from .detector import BallDetector
from .errors import (
    DetectorUnavailableError,
    InvalidFrameError,
    TrackingError,
    VideoOpenError,
)
from .frames import validate_frame

__all__ = [
    "BallDetector",
    "DetectorUnavailableError",
    "InvalidFrameError",
    "TrackingError",
    "VideoOpenError",
    "validate_frame",
]
