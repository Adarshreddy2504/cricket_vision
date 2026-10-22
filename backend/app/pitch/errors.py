"""
errors.py — The failure vocabulary of the pitch module.

Every error this module raises is one of these, so a caller can catch
`PitchError` and be sure it has caught everything the module throws.

The hierarchy is deliberately shallow:

    PitchError
      +- PitchUnavailableError   the detector has no usable weights
      +- InvalidFrameError       a frame was not a usable BGR image
      +- CalibrationError        the supplied corners cannot form a calibration

`PitchUnavailableError` is deliberately NOT the same as "no pitch was
found". A detector with no weights cannot find a pitch in any frame, and a
caller told "nothing detected" would quietly record an empty analysis
instead of a missing model.

`InvalidFrameError` and `CalibrationError` are ALSO `ValueError`, so code
that only cares that the argument was bad can catch `ValueError` and be
correct.
"""

from __future__ import annotations


class PitchError(Exception):
    """Base class for every error the pitch module raises."""


class PitchUnavailableError(PitchError):
    """
    Raised when the pitch detector has no usable model.

    Raised instead of loading weights from anywhere else: the module never
    downloads a checkpoint, so a missing file is a configuration fact the
    caller has to resolve, not a transient fault.
    """


class InvalidFrameError(PitchError, ValueError):
    """Raised when a frame is not a usable BGR image."""


class CalibrationError(PitchError, ValueError):
    """Raised when the supplied corners cannot form a calibration."""


__all__ = [
    "CalibrationError",
    "InvalidFrameError",
    "PitchError",
    "PitchUnavailableError",
]
