"""
trajectory.py — Pure trajectory analytics, in pixels.

WHAT THIS MODULE IS ALLOWED TO SAY
----------------------------------
A trajectory of image coordinates supports counts, frame spans, gaps, path
lengths in pixels, and average pixel speed. It does not support metres, km/h,
pitch length, line, swing or bounce height, because none of those are derivable
from pixels without a calibration. Those claims live in `pitch_plane.py`, which
requires Part 2's metre-space document.

WHY PATH LENGTH EXCLUDES GAPS
-----------------------------
Part 1 does not invent points across a hole; a trajectory can jump from frame 3
to frame 40. Summing `dist(p3, p40)` would report a distance that was never
observed as if it were motion. Instead:

    path_length_px      only steps between CONSECUTIVE frames
    unobserved_span_px  the straight-line distances across the holes, reported
                        separately so the hole is visible rather than hidden

`mean_speed_px_s` divides the observed distance by the observed time (one
frame interval per adjacent step), so a gap does not dilute the average.

STRUCTURAL, NOT DEPENDENT
-------------------------
Nothing here imports `app.tracking`. A Part 1 `Trajectory` works because it
exposes `.points`, and each point exposes `frame`, `x`, `y` and `source`.
Mappings with the same keys are accepted too, so a stored JSON document can be
analysed directly after loading.

FRAME ORDER IS VALIDATED, NOT REPAIRED
--------------------------------------
Frames must be unique and strictly increasing. Sorting or merging duplicates
here would hide an inconsistency in the tracking pass; the caller gets an
`AnalyticsInputError` naming the offending frame instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..schemas.analytics import TrajectorySummary
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


@runtime_checkable
class TrajectoryPointLike(Protocol):
    """The part of a Part 1 `TrajectoryPoint` this module actually uses."""

    frame: int
    x: float
    y: float
    source: str


@runtime_checkable
class TrajectoryLike(Protocol):
    """The part of a Part 1 `Trajectory` this module actually uses."""

    points: Sequence[TrajectoryPointLike]


@dataclass(frozen=True)
class _Point:
    frame: int
    x: float
    y: float
    source: str


def _validate_points(trajectory: object) -> list[_Point]:
    points = [
        _Point(
            frame=as_frame(read_field(item, "frame"), index),
            x=as_coordinate(read_field(item, "x"), "x", index),
            y=as_coordinate(read_field(item, "y"), "y", index),
            source=as_source(read_field(item, "source"), index, POINT_SOURCES),
        )
        for index, item in enumerate(coerce_records(trajectory))
    ]
    check_frame_order([point.frame for point in points])
    return points


def summarise_trajectory(
    trajectory: TrajectoryLike | Sequence[TrajectoryPointLike] | object,
    fps: float | None = None,
) -> TrajectorySummary:
    """
    Summarise a pixel-space trajectory.

    `fps` is optional: without it the counts, path length and gaps are still
    available, while `duration_s` and every speed are None. That is a missing
    timebase, not a speed of zero.

    Raises `AnalyticsInputError` for malformed points, duplicate or decreasing
    frames, non-finite coordinates, an invalid source, or a non-positive fps.
    """
    points = _validate_points(trajectory)
    frames_per_second = read_fps(fps)

    if not points:
        return TrajectorySummary()

    frames = [point.frame for point in points]
    measured = walk((point.frame, point.x, point.y) for point in points)

    detected = [point for point in points if point.source == "detected"]
    detected_measured = walk((point.frame, point.x, point.y) for point in detected)

    duration = None
    if frames_per_second is not None:
        duration = (frames[-1] - frames[0]) / frames_per_second

    return TrajectorySummary(
        point_count=len(points),
        detected_count=len(detected),
        predicted_count=len(points) - len(detected),
        frame_start=frames[0],
        frame_end=frames[-1],
        duration_s=duration,
        path_length_px=path_value(len(points), measured),
        detected_path_length_px=path_value(len(detected), detected_measured),
        net_displacement_px=math.hypot(
            points[-1].x - points[0].x,
            points[-1].y - points[0].y,
        ),
        mean_speed_px_s=mean_speed(len(points), measured, frames_per_second),
        detected_mean_speed_px_s=mean_speed(
            len(detected),
            detected_measured,
            frames_per_second,
        ),
        segments=max(0, len(points) - 1),
        adjacent_segments=measured.adjacent,
        unobserved_segments=measured.unobserved,
        unobserved_span_px=measured.unobserved_span if measured.unobserved > 0 else None,
        gaps=gaps(frames),
    )


__all__ = [
    "POINT_SOURCES",
    "TrajectoryLike",
    "TrajectoryPointLike",
    "summarise_trajectory",
]
