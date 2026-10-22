"""
corners.py — The four pitch corners, and the rules that say whether they are
a pitch at all.

THE CANONICAL CORNER ORDER
---------------------------
Every pitch quad in this project is four corners in ONE order, and that
order is part of the coordinate contract rather than an implementation
detail:

    1. top_left      the corner nearest the TOP-LEFT of the image
    2. top_right     the corner nearest the TOP-RIGHT of the image
    3. bottom_right  the corner nearest the BOTTOM-RIGHT of the image
    4. bottom_left   the corner nearest the BOTTOM-LEFT of the image

The order is clockwise in image space, and the homography uses it to decide
which corner of the pitch coordinate system each pixel corner lands on. A
quad in a different order would silently rotate or mirror the pitch
coordinates, so the order is fixed here and every producer uses it.

HOW "NEAREST THE TOP-LEFT" IS DECIDED
-------------------------------------
By the extremes of two linear functionals, which is how the upstream
project ordered them too:

    top_left      min(x + y)
    bottom_right  max(x + y)
    top_right     max(x - y)
    bottom_left   min(x - y)

For a convex quad these four extremes are the four vertices. When they are
not — a triangle, a line, a bow-tie, or a quad rotated so far that the
extremes collide — the corners are REFUSED. A quad that cannot be ordered
in the canonical order has no unambiguous pitch coordinate system, and
inventing one would be worse than saying so.

AN APPROXIMATE BOX IS NOT A CORNER SET
--------------------------------------
`corners_from_box` turns a detection BOX into four corners. Those four
points are GUESSES: the box says where the pitch-shaped region is, not
where its corners are. They are labelled as guesses wherever they travel
(see `calibration.CornerMethod.BOX_ESTIMATE`), and a calibration built from
them is APPROXIMATE, never VALID.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .constants import (
    BOX_TOP_INSET,
    CORNER_OUT_OF_BOUNDS_SLACK,
    MAX_QUAD_AREA_FRACTION,
    MIN_QUAD_AREA_FRACTION,
)
from .errors import CalibrationError

# A point in source-frame pixels: (x, y), x to the right, y downwards.
Point = tuple[float, float]

# The canonical corner order. Never reorder this tuple: the homography, the
# metric target quad and every test depend on it.
CORNER_ORDER: tuple[str, str, str, str] = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)

# A frame's size, as (width, height).
ImageSize = tuple[int, int]


@dataclass(frozen=True)
class PitchCorners:
    """
    The four corners of the pitch in SOURCE-frame pixels.

    The coordinates are always in the same space as the frame they came from:
    a frame was never resized without rescaling the answer by the same
    factor.
    """

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    def points(self) -> tuple[Point, Point, Point, Point]:
        """The corners in the canonical order."""
        return (self.top_left, self.top_right, self.bottom_right, self.bottom_left)

    def as_dict(self) -> dict[str, list[float]]:
        """The persisted form: `{name: [x, y]}`, rounded to two decimals."""
        return {
            name: [round(float(p[0]), 2), round(float(p[1]), 2)]
            for name, p in zip(CORNER_ORDER, self.points(), strict=True)
        }

    def as_list(self) -> list[list[float]]:
        """The corners as a flat list in the canonical order."""
        return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.points()]


def order_corners(points: Sequence[Point]) -> PitchCorners:
    """
    Order four arbitrary points into the canonical corner order.

    Raises `CalibrationError` when the points cannot be ordered: fewer or
    more than four, a non-finite coordinate, an extreme reached twice, or a
    set that does not form a convex quad. Every one of those is a case where
    the pitch coordinate system would be a guess.
    """
    if len(points) != 4:
        raise CalibrationError(
            f"expected exactly four pitch corners, got {len(points)}"
        )

    parsed: list[Point] = []
    for raw in points:
        try:
            x, y = float(raw[0]), float(raw[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise CalibrationError(f"corner is not a point: {raw!r}") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CalibrationError(f"corner is not finite: {raw!r}")
        parsed.append((x, y))

    top_left = min(parsed, key=lambda p: (p[0] + p[1], p[0]))
    bottom_right = max(parsed, key=lambda p: (p[0] + p[1], p[0]))
    top_right = max(parsed, key=lambda p: (p[0] - p[1], -p[0]))
    bottom_left = min(parsed, key=lambda p: (p[0] - p[1], -p[0]))

    chosen = (top_left, top_right, bottom_right, bottom_left)
    if len(set(chosen)) != 4:
        raise CalibrationError(
            "the four corners do not have four distinct extremes; they do not "
            "describe a quad that can be ordered"
        )

    corners = PitchCorners(*chosen)
    if not is_convex(corners):
        raise CalibrationError(
            "the four corners do not form a convex quad in the canonical order"
        )
    return corners


def corners_from_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    top_inset: float = BOX_TOP_INSET,
) -> PitchCorners:
    """
    A detection BOX to four APPROXIMATE corners, in the canonical order.

    The near edge is the bottom of the box and the far edge is the top,
    pulled IN by `top_inset` of the box width, so the resulting trapezoid is
    narrower at the far end than at the near end — the way a pitch looks in
    perspective.

    These are guesses about where the pitch's CORNERS are, made from a box
    that only says where the pitch-shaped REGION is. They are labelled as
    such by the caller and never become a validated calibration.

    Raises `CalibrationError` for a box with no area, or an inset that would
    invert the far edge.
    """
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        raise CalibrationError("detection box coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise CalibrationError(
            f"detection box has no area: ({x1}, {y1}) to ({x2}, {y2})"
        )
    if not 0.0 <= top_inset < 0.5:
        raise CalibrationError(
            f"top_inset must be in [0, 0.5), got {top_inset}"
        )
    width = x2 - x1
    inset = top_inset * width
    return PitchCorners(
        top_left=(x1 + inset, y1),
        top_right=(x2 - inset, y1),
        bottom_right=(x2, y2),
        bottom_left=(x1, y2),
    )


def is_convex(corners: PitchCorners) -> bool:
    """
    Do the four corners form a convex quad in the canonical order?

    Every turn around the quad must go the same way, and no three corners
    may be collinear — three collinear corners are a triangle with a
    repeated vertex, and the homography through them is degenerate.
    """
    points = corners.points()
    count = len(points)
    signs: list[bool] = []
    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]
        c = points[(i + 2) % count]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0.0)
    return all(sign == signs[0] for sign in signs)


def polygon_area(corners: PitchCorners) -> float:
    """The area of the quad in square source pixels."""
    points = corners.points()
    count = len(points)
    total = 0.0
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def corners_in_frame(
    corners: PitchCorners,
    image_size: ImageSize,
    *,
    slack: float = CORNER_OUT_OF_BOUNDS_SLACK,
) -> bool:
    """
    Do all four corners lie inside the frame (plus a little slack)?

    The slack exists because a mask boundary can bleed a pixel or two past
    the edge of the picture. A corner halfway to the next frame is a
    different object, and is refused.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    slack_x = slack * width
    slack_y = slack * height
    return all(
        -slack_x <= x <= width + slack_x and -slack_y <= y <= height + slack_y
        for x, y in corners.points()
    )


def validate_corners(
    corners: PitchCorners,
    image_size: ImageSize,
    *,
    slack: float = CORNER_OUT_OF_BOUNDS_SLACK,
    min_area_fraction: float = MIN_QUAD_AREA_FRACTION,
    max_area_fraction: float = MAX_QUAD_AREA_FRACTION,
) -> tuple[bool, str]:
    """
    The geometric gate on a pitch quad. `(True, "ok")` or `(False, reason)`.

    The reasons are facts worth recording: a refusal that says only "no"
    cannot be told apart from a detector that found nothing.

    The order of the checks is the order of how badly a wrong answer would
    mislead: a quad that is not a quad first, then one that is not on this
    picture, then one that is the wrong size to be a pitch.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return False, "unknown_frame_size"

    points = corners.points()
    if len(points) != 4:
        return False, "not_four_corners"
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
        return False, "non_finite_corner"
    if len(set(points)) != 4:
        return False, "duplicate_corners"
    if not is_convex(corners):
        return False, "non_convex_corners"

    area = polygon_area(corners)
    if area <= 0.0:
        return False, "zero_area"

    if not corners_in_frame(corners, image_size, slack=slack):
        return False, "corners_outside_frame"

    fraction = area / float(width * height)
    if fraction < min_area_fraction:
        return False, "pitch_quad_too_small"
    if fraction > max_area_fraction:
        return False, "pitch_quad_too_large"
    return True, "ok"


__all__ = [
    "CORNER_ORDER",
    "ImageSize",
    "PitchCorners",
    "Point",
    "corners_from_box",
    "corners_in_frame",
    "is_convex",
    "order_corners",
    "polygon_area",
    "validate_corners",
]
