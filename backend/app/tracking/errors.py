"""
errors.py — The failure vocabulary of the tracking module.

Every error the module raises is one of these, so a caller can catch
`TrackingError` and be sure it has caught everything this module throws.

The hierarchy is deliberately shallow:

    TrackingError
      +- VideoOpenError          the source video could not be opened
      +- DetectorUnavailableError the detector has no usable model
      +- InvalidFrameError       a frame was not a usable BGR image

`InvalidFrameError` is ALSO a `ValueError`, so code that only cares that the
argument was bad can catch `ValueError` and be correct.
"""

from __future__ import annotations


class TrackingError(Exception):
    """Base class for every error the tracking module raises."""


class VideoOpenError(TrackingError):
    """Raised when the source video cannot be opened at all."""


class DetectorUnavailableError(TrackingError):
    """
    Raised when the ball detector has no usable model.

    This is deliberately NOT the same as "no ball was found". A detector with
    no weights cannot find a ball in any frame, and a caller that is told
    "nothing detected" would quietly record an empty analysis instead of a
    missing model.
    """


class InvalidFrameError(TrackingError, ValueError):
    """Raised when a frame is not a usable BGR image."""
