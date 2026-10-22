"""
geometry.py — The physical size of the pitch, and how that size is justified.

A calibration maps pixels to a pitch coordinate system. Which coordinate
system depends entirely on whether the pitch's real dimensions are known:

    normalized   0..1 across the quad. Always available. Says where the ball
                 was as a FRACTION of the pitch we can see. Knows nothing
                 about metres, and therefore claims nothing about metres.

    metres       The ground plane, in metres. Needs the pitch's dimensions,
                 and needs them to be justified.

This module owns the metres side: the dimensions, where they came from, and
the honest statement of what they mean.

THE ASSUMPTION IS NAMED, NOT HIDDEN
------------------------------------
`PitchGeometry` defaults to the dimensions the Laws of Cricket fix: 20.12 m
between the two creases and 2.64 m between the bowling creases. That default
carries `provenance = ASSUMED`, and it is a statement about EVERY pitch, not
about this one: the crease-to-crease distance is fixed by Law 6, but whether
the quad someone drew on the video spans exactly the creases (rather than
the whole mown strip) is a judgement about that video and that quad.

`provenance = MANUALLY_SUPPLIED` means a human measured this pitch and said
so. Only then is the metric conversion a measurement rather than a
restatement of the Laws.

THE AXIS IS NAMED TOO
---------------------
Which way the pitch runs in the picture is not something a camera reports.
`long_axis` says whether the pitch's long axis runs top-to-bottom in the
image (the view from behind the bowler) or left-to-right (a side-on view),
and the metric coordinates are laid out accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .constants import CREASE_WIDTH_M, PITCH_LENGTH_M
from .errors import CalibrationError

# A point in the metric pitch frame: (across, along) in metres.
MetricPoint = tuple[float, float]


class GeometryProvenance(str, Enum):
    """WHERE the pitch's dimensions came from."""

    #: The Laws of Cricket, applied as a general assumption.
    ASSUMED = "assumed"
    #: A human measured this pitch and said so.
    MANUALLY_SUPPLIED = "manually_supplied"


class LongAxis(str, Enum):
    """WHICH way the pitch's long axis runs in the image."""

    #: Top to bottom: the view from behind the bowler. The default, because
    #: it is the framing the upstream project's geometry was derived from.
    VERTICAL = "vertical"
    #: Left to right: a side-on view across the pitch.
    HORIZONTAL = "horizontal"


DEFAULT_JUSTIFICATION = (
    "Law 6 dimensions assumed: 20.12 m between the two creases and 2.64 m "
    "between the bowling creases. The quad is assumed to span exactly that "
    "strip; this has not been verified against this video."
)


@dataclass(frozen=True)
class PitchGeometry:
    """
    The size of the pitch, and the standing of that size.

    `length_m`   distance along the pitch, between the two creases.
    `width_m`    distance across the pitch, between the bowling creases.
    `long_axis`  which way the long axis runs in the image.
    `provenance` whether these numbers were measured or assumed.
    `justification` a sentence a human can read and disagree with.
    """

    length_m: float = PITCH_LENGTH_M
    width_m: float = CREASE_WIDTH_M
    long_axis: LongAxis = LongAxis.VERTICAL
    provenance: GeometryProvenance = GeometryProvenance.ASSUMED
    justification: str = DEFAULT_JUSTIFICATION

    @property
    def is_measured(self) -> bool:
        """Were these dimensions measured, rather than assumed from the Laws?"""
        return self.provenance is GeometryProvenance.MANUALLY_SUPPLIED

    def validate(self) -> tuple[bool, str]:
        """Is this geometry usable at all? `(True, "ok")` or `(False, reason)`."""
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            return False, "invalid_pitch_length"
        if not math.isfinite(self.width_m) or self.width_m <= 0.0:
            return False, "invalid_pitch_width"
        if self.long_axis not in (LongAxis.VERTICAL, LongAxis.HORIZONTAL):
            return False, "invalid_long_axis"
        return True, "ok"

    def target_quad(self) -> tuple[MetricPoint, MetricPoint, MetricPoint, MetricPoint]:
        """
        The four corners' coordinates in METRES, in the canonical corner order
        (top_left, top_right, bottom_right, bottom_left).

        The metric frame has two axes:

            across   metres across the pitch, measured from the middle of the
                     strip. Negative on the LEFT of the image, positive on
                     the RIGHT. The top edge of the quad is always at
                     `-width_m / 2` and the bottom edge at `+width_m / 2`.
            along    metres along the pitch, measured from the FAR end
                     (the top of the image for a vertical pitch, the left of
                     the image for a horizontal one) towards the NEAR end.

        Which end is the BOWLER's end is NOT decided here: the same quad read
        from either end gives the same numbers with the batter at the
        opposite end of them, and deciding it needs the ball's own travel.

        Raises `CalibrationError` when the geometry is not usable.
        """
        ok, reason = self.validate()
        if not ok:
            raise CalibrationError(f"pitch geometry is unusable: {reason}")
        half = self.width_m / 2.0
        length = self.length_m
        if self.long_axis is LongAxis.VERTICAL:
            return (
                (-half, 0.0),
                (half, 0.0),
                (half, length),
                (-half, length),
            )
        return (
            (-half, 0.0),
            (-half, length),
            (half, length),
            (half, 0.0),
        )

    def as_dict(self) -> dict:
        """The persisted form. Configuration, never a measurement."""
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "long_axis": self.long_axis.value,
            "provenance": self.provenance.value,
            "justification": self.justification,
        }


__all__ = [
    "DEFAULT_JUSTIFICATION",
    "GeometryProvenance",
    "LongAxis",
    "MetricPoint",
    "PitchGeometry",
]
