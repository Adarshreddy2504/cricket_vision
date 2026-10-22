"""
pitch_plane.py — Analytics on Part 2's pitch-plane trajectory, in metres.

WHAT THIS MODULE REQUIRES, AND WHY IT REFUSES EVERYTHING ELSE
------------------------------------------------------------
Part 2's `PitchTrajectory` carries `units` that are either `"metre"` or
`"pitch_fraction"`. A fraction of the pitch is not a distance: the same number
means a different physical length on a different pitch, and a speed derived
from it would be a confident fiction. This module therefore requires
`units == "metre"` and raises `AnalyticsInputError` for anything else, including
a missing units field. It never assumes.

The interface is structural: nothing here imports `app.pitch`. A Part 2
trajectory works because it exposes `.points`, `.units`, and points with
`frame`, `across`, `along`, `source`, and optionally `inside_pitch`. Mappings
with the same keys are accepted, so a stored JSON document can be analysed
directly.

WHAT THE NUMBERS MEAN
---------------------
`path_length_m` is the observed distance along the ground-plane projection,
using only steps between consecutive frames. `mean_speed_mps` is that distance
divided by the observed time, and `mean_speed_kmh` is simply `× 3.6`.

Both are ESTIMATES. The ball is airborne; Part 2 projects it onto the ground
plane because that is the only plane the calibration describes. The record
therefore always sets `speed_is_estimate=True` and `speed_is_calibrated=False`.
This module has no way to turn an estimate into a measurement, and it says so in
every document it produces.

`outside_pitch_count` counts points Part 2 flagged as outside the measured quad.
They are included in the distance (dropping them silently would create fake
gaps) and exposed so a consumer can judge whether to trust the result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..schemas.analytics import PitchPlaneSummary
from ._common import (
    POINT_SOURCES,
    as_coordinate,
    as_frame,
    as_source,
    check_frame_order,
    coerce_records,
    gaps,
    mean_speed,
    path_value,
    read_field,
    read_fps,
    walk,
)
from .errors import AnalyticsInputError

#: The only unit this module can turn into a physical distance.
PITCH_UNITS: str = "metre"

#: Metres per second to kilometres per hour.
MPS_TO_KMH: float = 3.6


@runtime_checkable
class PitchPointLike(Protocol):
    """The part of a Part 2 `PitchPoint` this module actually uses."""

    frame: int
    across: float
    along: float
    source: str


@runtime_checkable
class PitchTrajectoryLike(Protocol):
    """The part of a Part 2 `PitchTrajectory` this module actually uses."""

    points: Sequence[PitchPointLike]
    units: str


@dataclass(frozen=True)
class _PitchPoint:
    frame: int
    across: float
    along: float
    source: str
    inside_pitch: bool


def _declared_units(container: object, records: Sequence[object]) -> None:
    """Require metre units on the trajectory and/or on each point."""
    declared: list[tuple[str, object]] = []

    if not isinstance(container, (str, bytes)) and (
        isinstance(container, Mapping) or hasattr(container, "points")
    ):
        container_units = read_field(container, "units", default=None)
        if container_units is not None:
            declared.append(("trajectory", container_units))

    for index, record in enumerate(records):
        units = read_field(record, "units", default=None)
        if units is not None:
            declared.append((f"point {index}", units))

    if not declared:
        raise AnalyticsInputError(
            "pitch trajectory does not declare units; refusing to treat its "
            f"numbers as {PITCH_UNITS!r}"
        )

    for where, units in declared:
        if units != PITCH_UNITS:
            raise AnalyticsInputError(
                f"{where} declares units {units!r}; physical distances require "
                f"{PITCH_UNITS!r}"
            )


def _validate_points(pitch_trajectory: object) -> list[_PitchPoint]:
    records = coerce_records(pitch_trajectory, "pitch trajectory")
    _declared_units(pitch_trajectory, records)

    points = []
    for index, item in enumerate(records):
        inside = read_field(item, "inside_pitch", default=True)
        if not isinstance(inside, bool):
            raise AnalyticsInputError(
                f"point {index}: inside_pitch must be a bool, got {inside!r}"
            )
        points.append(
            _PitchPoint(
                frame=as_frame(read_field(item, "frame"), index),
                across=as_coordinate(read_field(item, "across"), "across", index),
                along=as_coordinate(read_field(item, "along"), "along", index),
                source=as_source(read_field(item, "source"), index, POINT_SOURCES),
                inside_pitch=inside,
            )
        )

    check_frame_order([point.frame for point in points])
    return points


def summarise_pitch_trajectory(
    pitch_trajectory: PitchTrajectoryLike | Sequence[PitchPointLike] | object,
    fps: float | None = None,
) -> PitchPlaneSummary:
    """
    Summarise a Part 2 pitch trajectory in projected metres.

    `fps` is optional: without it distances and gaps remain, while durations and
    speeds are None.

    Raises `AnalyticsInputError` for non-metre or missing units, malformed
    points, duplicate or decreasing frames, non-finite coordinates, an invalid
    source, or a non-positive fps.
    """
    points = _validate_points(pitch_trajectory)
    frames_per_second = read_fps(fps)

    if not points:
        return PitchPlaneSummary()

    frames = [point.frame for point in points]
    measured = walk((point.frame, point.across, point.along) for point in points)

    detected = [point for point in points if point.source == "detected"]
    detected_measured = walk(
        (point.frame, point.across, point.along) for point in detected
    )

    duration = None
    if frames_per_second is not None:
        duration = (frames[-1] - frames[0]) / frames_per_second

    speed_mps = mean_speed(len(points), measured, frames_per_second)
    detected_speed_mps = mean_speed(
        len(detected),
        detected_measured,
        frames_per_second,
    )

    return PitchPlaneSummary(
        point_count=len(points),
        detected_count=len(detected),
        predicted_count=len(points) - len(detected),
        frame_start=frames[0],
        frame_end=frames[-1],
        duration_s=duration,
        path_length_m=path_value(len(points), measured),
        detected_path_length_m=path_value(len(detected), detected_measured),
        net_displacement_m=math.hypot(
            points[-1].across - points[0].across,
            points[-1].along - points[0].along,
        ),
        mean_speed_mps=speed_mps,
        mean_speed_kmh=speed_mps * MPS_TO_KMH if speed_mps is not None else None,
        detected_mean_speed_mps=detected_speed_mps,
        detected_mean_speed_kmh=(
            detected_speed_mps * MPS_TO_KMH if detected_speed_mps is not None else None
        ),
        segments=max(0, len(points) - 1),
        adjacent_segments=measured.adjacent,
        unobserved_segments=measured.unobserved,
        unobserved_span_m=(
            measured.unobserved_span if measured.unobserved > 0 else None
        ),
        gaps=gaps(frames),
        outside_pitch_count=sum(1 for point in points if not point.inside_pitch),
    )


__all__ = [
    "MPS_TO_KMH",
    "PITCH_UNITS",
    "PitchPointLike",
    "PitchTrajectoryLike",
    "summarise_pitch_trajectory",
]
