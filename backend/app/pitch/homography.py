"""
homography.py — Turning a pixel into a pitch coordinate.

THE CONVERSION IS A PURE FUNCTION
----------------------------------
`pixel_to_pitch(x, y, projection)` maps ONE source pixel to ONE pitch
coordinate and depends on nothing but its arguments: no globals, no
configuration lookup, no I/O, no state. The same pixel and the same
projection always give the same answer, which is what makes it testable
against hand-computed numbers instead of against a recording.

TRANSFORM DIRECTION
--------------------
Source pixels -> pitch coordinates. The projection is one-way in that
direction; `pitch_to_pixel` exists to prove it round-trips, not because
the application needs it.

UNITS
-----
A projection always knows which space it is in, and says so:

    PitchSpace.NORMALIZED   units "pitch_fraction". The two numbers are
                            (across, along) as a FRACTION of the pitch quad:
                            0.0 at one end of an axis, 1.0 at the other.
                            Resolution-independent and camera-independent.
                            This is the space that is always available.

    PitchSpace.METRES       units "metre". The two numbers are (across,
                            along) in metres on the ground plane. `across` is
                            measured from the middle of the strip; `along` is
                            measured from the FAR end of the pitch towards
                            the NEAR end.

BOTH SPACES USE THE SAME TWO AXES
---------------------------------
In both spaces the first number is `across` and the second is `along`.
Which way those run in the picture depends on the pitch's long axis, which
`PitchGeometry.long_axis` records: a vertical pitch (the view from behind
the bowler) runs across from left to right and along from top to bottom,
while a horizontal pitch (a side-on view) runs across from top to bottom
and along from left to right.

THE METRES ARE GATED
--------------------
A projection in metres is only built when the calibration can support it:
either the four corners were measured on this frame by a human, or the
pitch's dimensions were measured rather than assumed from the Laws. A
detected quad with assumed Law 6 dimensions produces a confident-looking
number that nobody checked, which is exactly the kind of number this
project refuses to publish. `metric_gate` states the reason out loud.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .calibration import CalibrationStatus, PitchCalibration
from .corners import PitchCorners, Point
from .geometry import GeometryProvenance, LongAxis, PitchGeometry

# A matrix that maps a point through a perspective transform.
Homography = np.ndarray


class PitchSpace(str, Enum):
    """Which coordinate system a projection produces."""

    #: Fractions of the pitch quad. Always available.
    NORMALIZED = "normalized"
    #: Metres on the ground plane. Only when the calibration supports it.
    METRES = "metres"


#: The unit of each space, spelled out for the persisted form.
SPACE_UNITS: dict[PitchSpace, str] = {
    PitchSpace.NORMALIZED: "pitch_fraction",
    PitchSpace.METRES: "metre",
}

#: The names of the two axes, in the order the projection returns them.
SPACE_AXES: dict[PitchSpace, tuple[str, str]] = {
    PitchSpace.NORMALIZED: ("across", "along"),
    PitchSpace.METRES: ("across", "along"),
}


def normalized_target(long_axis: LongAxis) -> tuple[Point, Point, Point, Point]:
    """
    The unit square, in the canonical corner order, for the given long axis.

    The square is a property of the QUAD, not of any real-world pitch size:
    it says "this is where the ball was, as a fraction of the pitch we can
    see". That is resolution-independent and camera-independent, and — unlike
    metres — it does not pretend the model told us how wide the pitch is.
    """
    if long_axis is LongAxis.VERTICAL:
        return ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    return ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))


@dataclass(frozen=True, eq=False)
class PitchProjection:
    """
    A ready-to-use pixel -> pitch mapping.

    `homography` the 3x3 matrix, mapping source pixels to pitch coordinates.
    `space`     which coordinate system it produces.
    `long_axis` which way the pitch's long axis runs in the image.
    `geometry`  the dimensions the projection was built against, when the
                space is metres.
    """

    homography: Homography
    space: PitchSpace
    long_axis: LongAxis = LongAxis.VERTICAL
    geometry: PitchGeometry | None = None

    @property
    def units(self) -> str:
        """The unit of the numbers this projection returns."""
        return SPACE_UNITS[self.space]

    @property
    def axes(self) -> tuple[str, str]:
        """The names of the two axes, in the order they are returned."""
        return SPACE_AXES[self.space]

    def project(self, x: float, y: float) -> tuple[float, float] | None:
        """Source pixel -> (across, along). None when the pixel cannot map."""
        return pixel_to_pitch(x, y, self)

    def unproject(self, across: float, along: float) -> tuple[float, float] | None:
        """(across, along) -> source pixel. The inverse direction."""
        return pitch_to_pixel(across, along, self)

    def contains(self, x: float, y: float, *, tolerance: float = 1e-6) -> bool:
        """Does this pixel fall inside the pitch?"""
        mapped = self.project(x, y)
        if mapped is None:
            return False
        across, along = mapped
        if self.space is PitchSpace.NORMALIZED:
            across_low, across_high = 0.0, 1.0
            along_low, along_high = 0.0, 1.0
        else:
            geometry = self.geometry
            if geometry is None:
                return False
            across_low = -geometry.width_m / 2.0
            across_high = geometry.width_m / 2.0
            along_low = 0.0
            along_high = geometry.length_m
        return (
            across_low - tolerance <= across <= across_high + tolerance
            and along_low - tolerance <= along <= along_high + tolerance
        )

    def homography_rows(self) -> list[list[float]]:
        """The matrix as a list of rows, for the persisted form."""
        return [[round(float(v), 8) for v in row] for row in self.homography]


def metric_gate(calibration: PitchCalibration) -> tuple[bool, str]:
    """
    May this calibration produce metre coordinates? `(True, "ok")` or
    `(False, reason)`.

    Metres need TWO things and are refused without either:

      * the pitch's real dimensions — measured, or the Law 6 defaults used
        as a labelled assumption; and
      * corners whose MEANING is known: the four corners of the pitch, in
        this picture.

    The second is why a detected quad is refused: a model can say where
    the pitch-shaped region is, but nobody has said that the quad's edges
    ARE the creases. `was_measured_on_this_frame` is the only evidence that
    somebody did.
    """
    if calibration.status is CalibrationStatus.INVALID:
        return False, "invalid_calibration"
    if calibration.was_measured_on_this_frame:
        return True, "ok"
    if calibration.geometry.provenance is GeometryProvenance.MANUALLY_SUPPLIED:
        return True, "ok"
    return False, "metric_geometry_not_justified"


def build_projection(
    calibration: PitchCalibration,
    *,
    space: PitchSpace = PitchSpace.NORMALIZED,
) -> tuple[PitchProjection | None, str]:
    """
    The projection a calibration justifies.

    Returns `(projection, "")` on success and `(None, reason)` on refusal.
    The reason is meant to be shown to whoever asked for the conversion: it
    says which of the two requirements was not met.
    """
    if calibration.status is CalibrationStatus.INVALID:
        return None, "invalid_calibration"

    geometry = calibration.geometry
    long_axis = geometry.long_axis

    if space is PitchSpace.NORMALIZED:
        target = normalized_target(long_axis)
    else:
        allowed, reason = metric_gate(calibration)
        if not allowed:
            return None, reason
        target = geometry.target_quad()

    matrix = build_homography(calibration.corners, target)
    if matrix is None:
        return None, "degenerate_homography"

    return (
        PitchProjection(
            homography=matrix,
            space=space,
            long_axis=long_axis,
            geometry=geometry if space is PitchSpace.METRES else None,
        ),
        "",
    )


def build_homography(
    corners: PitchCorners,
    target: Sequence[Point],
) -> Homography | None:
    """
    The perspective transform taking the four source corners to the four
    target points. None when no such transform exists.

    The transform is checked before it is returned: it must be finite, it
    must not be singular, and each source corner must land on its own
    target point. A matrix that passes those checks is a matrix that maps
    the quad it claims to map.
    """
    # `cv2.getPerspectiveTransform` requires CV_32F in OpenCV 5, so the
    # CORRESPONDENCES are single precision. The returned matrix is immediately
    # widened to float64 and every later operation is float64, which leaves a
    # relative error around 1e-7 -- a micrometre on a 20 m pitch. That is
    # documented rather than hidden: `PitchProjection.units` says "metre", and
    # the tests pin the metric answers to 0.1 mm.
    src = np.array(corners.points(), dtype=np.float32)
    dst = np.array(target, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        return None
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        return None
    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    if matrix is None or not np.isfinite(matrix).all():
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return None
    if not validate_homography(matrix, src, dst):
        return None
    return matrix


def validate_homography(
    matrix: Homography,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> bool:
    """
    Does this matrix really map each source point onto its target point?

    The round trip is the only honest check of a perspective transform: a
    matrix can be finite and non-singular and still be the wrong one.
    """
    for expected, point in zip(dst, src, strict=True):
        mapped = apply_homography(matrix, float(point[0]), float(point[1]))
        if mapped is None:
            return False
        if not np.allclose(mapped, np.asarray(expected, dtype=np.float64), atol=tolerance):
            return False
    return True


def invert_homography(matrix: Homography) -> Homography | None:
    """The inverse transform, or None when the matrix cannot be inverted."""
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    return inverse


def apply_homography(
    matrix: Homography, x: float, y: float
) -> tuple[float, float] | None:
    """
    One point through the transform. None when the point cannot be mapped.

    A point on the transform's horizon maps to infinity, and a point
    behind it maps through infinity to a mirrored position. Neither is a
    usable coordinate, so both are refused rather than returned as a huge
    number.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    vector = np.array([x, y, 1.0], dtype=np.float64)
    mapped = np.asarray(matrix, dtype=np.float64) @ vector
    w = float(mapped[2])
    if not math.isfinite(w) or abs(w) < 1e-12:
        return None
    u = float(mapped[0]) / w
    v = float(mapped[1]) / w
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v


def pixel_to_pitch(
    x: float, y: float, projection: PitchProjection
) -> tuple[float, float] | None:
    """
    Source pixel -> pitch coordinate, in the projection's space and units.

    PURE: the answer depends on the arguments and on nothing else. Returns
    None when the pixel has no finite image under the transform, which is
    the honest answer rather than a number with a huge magnitude.
    """
    return apply_homography(projection.homography, x, y)


def pitch_to_pixel(
    across: float, along: float, projection: PitchProjection
) -> tuple[float, float] | None:
    """
    Pitch coordinate -> source pixel. The inverse direction.

    Exists so the transform's direction can be checked, and so a client can
    draw a pitch-space marker back onto the picture.
    """
    inverse = invert_homography(projection.homography)
    if inverse is None:
        return None
    return apply_homography(inverse, across, along)


__all__ = [
    "SPACE_AXES",
    "SPACE_UNITS",
    "Homography",
    "PitchProjection",
    "PitchSpace",
    "apply_homography",
    "build_homography",
    "build_projection",
    "invert_homography",
    "metric_gate",
    "normalized_target",
    "pitch_to_pixel",
    "pixel_to_pitch",
    "validate_homography",
]
