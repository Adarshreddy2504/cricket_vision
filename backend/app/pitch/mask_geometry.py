"""
mask_geometry.py — From a segmentation mask boundary to a four-corner quad.

Pure functions: no model, no I/O, no configuration. Given one mask's
boundary (as ultralytics reports it, in ORIGINAL frame pixels) they derive
the four-corner quad the calibration consumes, or say why they cannot.

THE QUAD COMES FROM THE MASK BOUNDARY, NEVER FROM THE FRAME
-----------------------------------------------------------
A segmentation model answers with a mask over the complete wicket-to-wicket
pitch. Four corners are then taken from THAT boundary: the convex hull,
`approxPolyDP` at a short list of simplification fractions until exactly
four corners survive, and — only if that fails — a greedy reduction of the
hull's flattest corner until four remain. No fixed inset, no axis-aligned
bounding box: the mask IS the boundary, and the corners are what
simplifying it produces.

THE ORDERING RULE IS THE PROJECT'S ONE
--------------------------------------
The corners this module returns are ordered by `corners.order_corners`,
the same rule the rest of the project uses. Two modules deriving corners
independently and disagreeing would make the quad a different shape
depending on which one produced it.

EVERY CHECK HERE CAN REFUSE
---------------------------
A triangle, a degenerate hull, a mask covering a third of the frame (a
crowd), a mask with NaN coordinates: all refused with a reason. A refusal
is a fact worth recording; a fabricated quad drawn over the video is not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .constants import (
    MASK_APPROX_POLYDP_FRACTIONS,
    MASK_MAX_AREA_FRACTION,
    MASK_MAX_ASPECT_RATIO,
    MASK_MAX_HULL_VERTICES,
    MASK_MIN_AREA_FRACTION,
    MASK_MIN_ASPECT_RATIO,
)
from .corners import ImageSize, Point, order_corners
from .errors import CalibrationError

# The four corners of a quad, in the canonical order.
Quad = tuple[Point, Point, Point, Point]


def mask_polygon_area_fraction(
    points: Sequence[Point], image_size: ImageSize
) -> float:
    """
    Share of the frame the mask's CONVEX HULL covers.

    The hull, not the raw polygon: a mask with a dent along one edge has
    the same area as the pitch it covers, and a share expressed against the
    hull is what the area bounds were measured against.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    points = _finite_points(points)
    if len(points) < 3:
        return 0.0
    array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(array)
    return float(cv2.contourArea(hull)) / float(width * height)


def mask_centroid(points: Sequence[Point]) -> tuple[float, float] | None:
    """Mean of the mask vertices — the consistency anchor between estimates."""
    points = _finite_points(points)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def quad_aspect_ratio(points: Sequence[Point]) -> float:
    """
    min-area-rect aspect ratio of the polygon: max(w, h) / min(w, h).

    Reported for the RAW MASK too (not only the quad) so a crowd-shaped
    blob can be refused before simplification even runs. A degenerate
    polygon (zero width or height) is infinite: the caller's ceiling
    refuses it.
    """
    points = _finite_points(points)
    if len(points) < 3:
        return float("inf")
    array = np.array(points, dtype=np.float32)
    (_, _), (rotated_w, rotated_h), _angle = cv2.minAreaRect(array)
    if rotated_w <= 1e-6 or rotated_h <= 1e-6:
        return float("inf")
    return float(max(rotated_w, rotated_h) / min(rotated_w, rotated_h))


def mask_in_bounds(points: Sequence[Point], image_size: ImageSize) -> bool:
    """Every finite vertex lies inside the frame."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    points = _finite_points(points)
    if not points:
        return False
    slack_x = 0.02 * width
    slack_y = 0.02 * height
    return all(
        -slack_x <= x <= width + slack_x and -slack_y <= y <= height + slack_y
        for x, y in points
    )


def validate_mask(
    points: Sequence[Point],
    image_size: ImageSize,
    *,
    min_area_fraction: float = MASK_MIN_AREA_FRACTION,
    max_area_fraction: float = MASK_MAX_AREA_FRACTION,
    min_aspect_ratio: float = MASK_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MASK_MAX_ASPECT_RATIO,
) -> tuple[bool, str]:
    """
    Shape-plausibility gate on the raw mask polygon, before simplification.

    Returns `(True, "ok")` or `(False, reason)`. Area and aspect bounds are
    the ones measured off real pitch masks versus the crowd false positive
    in the upstream project; nothing here is fitted to one video's framing.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return False, "unknown_frame_size"
    points = _finite_points(points)
    if len(points) < 3:
        return False, f"mask_polygon_too_small_{len(points)}"
    if not mask_in_bounds(points, image_size):
        return False, "mask_outside_frame"
    area = mask_polygon_area_fraction(points, image_size)
    if area < min_area_fraction:
        return False, "mask_area_too_small"
    if area > max_area_fraction:
        return False, "mask_area_too_large"
    aspect = quad_aspect_ratio(points)
    if aspect < min_aspect_ratio:
        return False, "mask_aspect_too_square"
    if aspect > max_aspect_ratio:
        return False, "mask_aspect_too_elongated"
    return True, "ok"


def quad_from_mask(points: Sequence[Point]) -> tuple[Quad | None, str]:
    """
    Four corners from the mask boundary, in the canonical order, plus which
    simplification produced them (`direct_hull`, `approx_polydp` or
    `hull_reduce`).

    `(None, reason)` when no honest quad exists: fewer than four finite
    vertices, a triangle hull, a boundary too complex to trust, or four
    points that cannot be ordered as a quad. The caller must record the
    refusal — inventing the missing corner would draw a boundary the model
    never reported.
    """
    points = _finite_points(points)
    if len(points) < 4:
        return None, "too_few_vertices"
    array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(array).reshape(-1, 2)
    hull_points = [(float(p[0]), float(p[1])) for p in hull]
    if len(hull_points) < 4:
        return None, "degenerate_hull"
    if len(hull_points) > MASK_MAX_HULL_VERTICES:
        return None, "hull_too_complex"

    # Exactly four hull vertices already: the mask is a quad.
    if len(hull_points) == 4:
        corners = _order_or_none(hull_points)
        if corners is not None:
            return corners, "direct_hull"

    # approxPolyDP at a short list of fractions until exactly four corners.
    for fraction in MASK_APPROX_POLYDP_FRACTIONS:
        epsilon = fraction * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
        if len(approx) != 4:
            continue
        candidate = [(float(p[0]), float(p[1])) for p in approx]
        if len(set(candidate)) != 4:
            continue
        corners = _order_or_none(candidate)
        if corners is not None:
            return corners, "approx_polydp"

    # Fallback: greedily drop the flattest hull corner until four remain.
    reduced = _greedy_hull_to_quad(hull_points)
    if reduced is None:
        return None, "quad_unavailable"
    corners = _order_or_none(reduced)
    if corners is None:
        return None, "quad_unavailable"
    return corners, "hull_reduce"


# ── internals ────────────────────────────────────────────────────────────────
def _finite_points(points: Sequence[Point]) -> list[Point]:
    """Drop NaN/inf/non-numeric vertices. A malformed vertex is not repaired."""
    out: list[Point] = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out.append((x, y))
    return out


def _order_or_none(points: Sequence[Point]) -> Quad | None:
    """The canonical ordering of four points, or None when they cannot be."""
    try:
        corners = order_corners(points)
    except CalibrationError:
        return None
    return corners.points()


def _greedy_hull_to_quad(hull: list[Point]) -> list[Point] | None:
    """
    Reduce a hull to four vertices by repeatedly dropping the flattest one.

    "Flattest" = the vertex whose two neighbours are closest to collinear
    with it, measured as the area the triangle (prev, vertex, next) covers.
    The smallest triangle is the corner the mask boundary bends the least
    at, so dropping it changes the covered region the least. Four vertices
    are left exactly; a hull that reaches three is degenerate and refused.
    """
    points = list(hull)
    if len(points) < 4:
        return None
    while len(points) > 4:
        worst_index = -1
        worst_area = float("inf")
        count = len(points)
        for index in range(count):
            a = points[(index - 1) % count]
            b = points[index]
            c = points[(index + 1) % count]
            area = abs(
                (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            ) / 2.0
            if area < worst_area:
                worst_area = area
                worst_index = index
        if worst_index < 0:
            return None
        points.pop(worst_index)
    if len(points) != 4 or len(set(points)) != 4:
        return None
    return points


__all__ = [
    "Quad",
    "mask_centroid",
    "mask_in_bounds",
    "mask_polygon_area_fraction",
    "quad_aspect_ratio",
    "quad_from_mask",
    "validate_mask",
]
