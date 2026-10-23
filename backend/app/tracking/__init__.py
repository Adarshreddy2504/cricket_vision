"""
tracking — The ball-tracking module.

The public surface of the module:

    BallDetector      finds the ball in a frame (YOLO), in source pixels.
    BallTracker       smooths those findings into a trajectory (Kalman).
    TrackingService   drives the two over a stream of frames and produces the
                      documented result.

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
from .kalman_tracker import BallTracker, KalmanBallTracker, TrackResult
from .tracking_service import (
    FrameTracking,
    Track,
    TrackingResult,
    TrackingService,
)

__all__ = [
    "BallDetector",
    "BallTracker",
    "DetectorUnavailableError",
    "FrameTracking",
    "InvalidFrameError",
    "KalmanBallTracker",
    "Track",
    "TrackResult",
    "TrackingError",
    "TrackingResult",
    "TrackingService",
    "VideoOpenError",
    "validate_frame",
]
