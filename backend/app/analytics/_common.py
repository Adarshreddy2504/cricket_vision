"""
_common.py — Validation and segment maths shared by the analytics modules.

Private on purpose: the public surface is `trajectory.py` and `pitch_plane.py`.
Both operate on a sequence of records with a frame number, two coordinates and
a provenance value, so the checks and the gap-aware distance rule live once,
here, rather than being copied with small differences.

The distance rule is the important one: only steps between consecutive frames
count as observed motion. A jump across a gap is measured and reported
separately, never folded into a path length.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .errors import AnalyticsInputError

#: The only provenance values a point may carry, in either coordinate space.
POINT_SOURCES: tuple[str, ...] = ("detected", "predicted")

MISSING = object()


def read_field(record: object, name: str, default: object = MISSING):
    """Read a field from an object attribute or from a mapping key."""
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    else:
        try:
            return getattr(record, name)
        except AttributeError:
            pass
    if default is not MISSING:
        return default
    raise AnalyticsInputError(f"record is missing {name!r}")


def coerce_records(container: object, label: str = "trajectory") -> list[object]:
    """A trajectory object's `.points`, a mapping's `points`, or a sequence."""
    if isinstance(container, Mapping):
        if "points" not in container:
            raise AnalyticsInputError(f"{label} mapping has no 'points' key")
        raw = container["points"]
    elif hasattr(container, "points"):
        raw = container.points
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        raw = container
    else:
        raise AnalyticsInputError(
            f"{label} must expose .points or be a sequence of points, got "
            f"{type(container).__name__}"
        )

    if raw is None:
        raise AnalyticsInputError(f"{label} points are None")
    try:
        return list(raw)
    except TypeError as exc:
        raise AnalyticsInputError(f"{label} points are not iterable: {raw!r}") from exc


def as_frame(value: object, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise AnalyticsInputError(f"point {index}: frame must be an integer, got {value!r}")
    frame = int(value)
    if frame < 1:
        raise AnalyticsInputError(f"point {index}: frame must be >= 1, got {frame}")
    return frame


def as_coordinate(value: object, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise AnalyticsInputError(f"point {index}: {name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise AnalyticsInputError(f"point {index}: {name} must be finite, got {number!r}")
    return number


def as_source(value: object, index: int, sources: Sequence[str]) -> str:
    if value not in sources:
        raise AnalyticsInputError(
            f"point {index}: source must be one of {tuple(sources)}, got {value!r}"
        )
    return str(value)


def read_fps(fps: object) -> float | None:
    if fps is None:
        return None
    if isinstance(fps, bool) or not isinstance(fps, numbers.Real):
        raise AnalyticsInputError(f"fps must be a number, got {fps!r}")
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise AnalyticsInputError(f"fps must be finite and positive, got {fps!r}")
    return value


def check_frame_order(frames: Sequence[int]) -> None:
    """Duplicate or decreasing frames are an inconsistency, not something to sort."""
    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        if current == previous:
            raise AnalyticsInputError(
                f"duplicate frame {current}: a trajectory cannot have two points for one frame"
            )
        if current < previous:
            raise AnalyticsInputError(
                f"frames are out of order at index {index}: {previous} then {current}"
            )


def gaps(frames: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Inclusive runs of missing frames between consecutive points."""
    found: list[tuple[int, int]] = []
    for previous, current in zip(frames, frames[1:], strict=False):
        if current > previous + 1:
            found.append((previous + 1, current - 1))
    return tuple(found)


@dataclass(frozen=True)
class Walk:
    adjacent: int
    observed_length: float
    unobserved: int
    unobserved_span: float


def walk(steps: Iterable[tuple[int, float, float]]) -> Walk:
    """Measure consecutive-frame steps and gap-spanning steps separately."""
    ordered = list(steps)
    adjacent = 0
    observed = 0.0
    unobserved = 0
    unobserved_span = 0.0

    for (frame_a, u_a, v_a), (frame_b, u_b, v_b) in zip(
        ordered, ordered[1:], strict=False
    ):
        distance = math.hypot(u_b - u_a, v_b - v_a)
        if frame_b - frame_a == 1:
            adjacent += 1
            observed += distance
        else:
            unobserved += 1
            unobserved_span += distance

    return Walk(
        adjacent=adjacent,
        observed_length=observed,
        unobserved=unobserved,
        unobserved_span=unobserved_span,
    )


def path_value(point_count: int, measured: Walk) -> float | None:
    """
    Observed path length, distinguishing "no data" from "no movement".

    One point is a real observation of zero displacement. Two points separated
    only by a gap are not an observation of adjacent motion, so the observed
    length is unavailable rather than zero.
    """
    if point_count == 0:
        return None
    if point_count == 1:
        return 0.0
    if measured.adjacent == 0:
        return None
    return measured.observed_length


def mean_speed(point_count: int, measured: Walk, fps: float | None) -> float | None:
    if fps is None or point_count < 2 or measured.adjacent == 0:
        return None
    return measured.observed_length / (measured.adjacent / fps)


__all__ = [
    "MISSING",
    "POINT_SOURCES",
    "Walk",
    "as_coordinate",
    "as_frame",
    "as_source",
    "check_frame_order",
    "coerce_records",
    "gaps",
    "mean_speed",
    "path_value",
    "read_field",
    "read_fps",
    "walk",
]
