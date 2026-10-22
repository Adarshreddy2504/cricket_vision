"""
adapter.py — Part 1's ball trajectory, re-expressed in pitch coordinates.

THE ADAPTER IS STRUCTURAL, NOT DEPENDENT
----------------------------------------
This module imports NOTHING from the tracking module. It accepts anything
that looks like a Part 1 trajectory — an object with a `points` sequence, or
a sequence of points, where each point has `frame`, `x`, `y` and `source` —
and it says so with Protocols rather than by importing a type.

That is deliberate. The two parts were developed on separate branches, and
Part 2 must be usable and testable on its own. A structural adapter means
the same code works unchanged after the branches are merged, with no
import-order puzzle and no circular dependency between `app.tracking` and
`app.pitch`.

WHAT IT PRESERVES
-----------------
Every field of the ball point that a consumer might act on:

    frame       unchanged — the authoritative timebase
    x, y        unchanged — the source pixel, so the point can be drawn back
    source      unchanged — "detected" stays a measurement, "predicted" stays
                an extrapolation
    confidence  unchanged, and never invented
    track_id    unchanged

and it adds exactly two numbers: `across` and `along`.

WHAT IT REFUSES TO DO
---------------------
It never invents a point, never fills a gap, and never clamps a point into
the pitch square. A ball point the transform cannot map is COUNTED in
`skipped` and NAMED in `skipped_frames`; the frames it left behind show up
in `gaps` exactly as they do in the tracking document.

It also refuses to answer a question it cannot: asking for metres from a
calibration that does not justify them raises `CalibrationError` naming the
reason, rather than quietly handing back fractions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from app.schemas.pitch import PitchPoint, PitchTrajectory

from .calibration import PitchCalibration
from .errors import CalibrationError
from .homography import PitchProjection, PitchSpace, build_projection, pixel_to_pitch

# The two provenances Part 1 defines. Anything else is a schema mismatch
# between the parts and must be reported, not passed through.
_VALID_SOURCES = frozenset({"detected", "predicted"})


@runtime_checkable
class TrajectoryPointLike(Protocol):
    """The shape of one Part 1 trajectory point, structurally."""

    frame: int
    x: int
    y: int
    source: str
    confidence: float | None
    track_id: int | None


@runtime_checkable
class TrajectoryLike(Protocol):
    """The shape of a Part 1 trajectory, structurally."""

    points: Sequence[TrajectoryPointLike]


def project_points(
    points: Iterable[TrajectoryPointLike],
    projection: PitchProjection,
) -> tuple[list[PitchPoint], list[int]]:
    """
    Project a stream of ball points. Returns `(projected, skipped_frames)`.

    Raises `TypeError` when a point is not shaped like a trajectory point:
    that is a broken integration between the parts, and guessing at the
    missing field would hide it.
    """
    projected: list[PitchPoint] = []
    skipped_frames: list[int] = []

    for point in points:
        frame = _require_int(point, "frame")
        x = _require_int(point, "x")
        y = _require_int(point, "y")
        source = _require_source(point)

        mapped = pixel_to_pitch(float(x), float(y), projection)
        if mapped is None:
            skipped_frames.append(frame)
            continue

        across, along = mapped
        if not (math.isfinite(across) and math.isfinite(along)):
            skipped_frames.append(frame)
            continue

        projected.append(
            PitchPoint(
                frame=frame,
                x=x,
                y=y,
                across=round(across, 6),
                along=round(along, 6),
                space=projection.space.value,
                units=projection.units,
                source=source,
                confidence=getattr(point, "confidence", None),
                track_id=getattr(point, "track_id", None),
                inside_pitch=projection.contains(float(x), float(y)),
            )
        )

    return projected, skipped_frames


def project_trajectory(
    trajectory: TrajectoryLike | Sequence[TrajectoryPointLike],
    calibration: PitchCalibration,
    *,
    space: PitchSpace = PitchSpace.NORMALIZED,
) -> PitchTrajectory:
    """
    A Part 1 trajectory in the pitch coordinate system the calibration
    justifies.

    Raises `CalibrationError` when no such projection exists — a refused
    metric request says which requirement was not met, instead of silently
    falling back to fractions that look like metres in a different field.
    """
    projection, reason = build_projection(calibration, space=space)
    if projection is None:
        raise CalibrationError(
            f"cannot project the trajectory into {space.value} coordinates: {reason}"
        )

    points = _points_of(trajectory)
    projected, skipped_frames = project_points(points, projection)

    return PitchTrajectory(
        points=projected,
        space=projection.space.value,
        units=projection.units,
        calibration=calibration.as_dict(),
        skipped=len(skipped_frames),
        skipped_frames=skipped_frames,
    )


# ── internals ────────────────────────────────────────────────────────────────
def _points_of(
    trajectory: TrajectoryLike | Sequence[TrajectoryPointLike],
) -> Sequence[TrajectoryPointLike]:
    """
    The points of a trajectory object, or the sequence itself.

    Accepting both keeps the adapter usable by a caller that has already
    pulled the points out, without a second function name for it.
    """
    if isinstance(trajectory, Sequence):
        return trajectory
    points = getattr(trajectory, "points", None)
    if points is None:
        raise TypeError(
            "expected a trajectory (an object with a `points` sequence) or a "
            f"sequence of points, got {type(trajectory).__name__}"
        )
    return points


def _require_int(point, name: str) -> int:
    value = getattr(point, name, None)
    if value is None:
        raise TypeError(
            f"trajectory point {type(point).__name__} is missing `{name}`; "
            "the pitch adapter needs Part 1's trajectory schema"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"trajectory point `{name}` is not an integer: {value!r}"
        ) from exc


def _require_source(point) -> str:
    value = getattr(point, "source", None)
    if not isinstance(value, str):
        raise TypeError(
            f"trajectory point {type(point).__name__} is missing a string "
            f"`source`, got {value!r}"
        )
    if value not in _VALID_SOURCES:
        raise TypeError(
            f"trajectory point `source` must be one of "
            f"{sorted(_VALID_SOURCES)}, got {value!r}"
        )
    return value


__all__ = [
    "TrajectoryLike",
    "TrajectoryPointLike",
    "project_points",
    "project_trajectory",
]
