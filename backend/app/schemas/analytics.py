"""
analytics.py — Typed documents produced by the analytics pass.

These are PLAIN DATA, in the same sense as `schemas/shots.py`: no tracker, no
model and no OpenCV. The analytics modules compute them, the API serialises
them, and a stored analysis can be read back without either dependency.

THE ONE DISTINCTION THAT MATTERS
--------------------------------
`None` means "this was not measured". A number means it was. A field is never
zero-filled to look complete:

  * no points at all                  → path length is None, not 0
  * one point                          → path length is genuinely 0
  * two identical positions            → speed is genuinely 0, not None
  * a 3-point path with a 40-frame gap → speed is the average over real elapsed
                                          time, and `gaps` names the hole

Units are part of every record. Pixel distances are `"pixel"`; pitch-plane
distances are `"metre"` and are always marked as projected estimates that are
not calibrated physical measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from .shots import ShotClassification


@dataclass(frozen=True)
class TrajectorySummary:
    """
    What a pixel-space trajectory supports, and nothing more.

    frame_start/frame_end
        1-based source frame numbers of the first and last point.
    duration_s
        `(frame_end - frame_start) / fps`. Frame numbers are assumed evenly
        spaced at the supplied fps; variable-frame-rate footage is not modelled.
    path_length_px
        Sum of Euclidean distances between points on CONSECUTIVE frames. This is
        an observed polyline length in the image, not a physical distance. A
        step across a gap is excluded from it and reported separately as
        `unobserved_span_px`: the straight line between two points 40 frames
        apart is not motion that was seen.
    detected_path_length_px
        The same sum over steps where both endpoints are detected, skipping
        predicted points entirely.
    net_displacement_px
        Straight-line distance from the first to the last point, gaps included.
    mean_speed_px_s
        `path_length_px / (adjacent_segments / fps)` — the average of observed
        frame-to-frame motion, not an instantaneous or release speed. None when
        no consecutive pair exists (including a single point, or only gapped
        pairs).
    gaps
        Inclusive `(first_missing_frame, last_missing_frame)` runs, matching
        Part 1's `Trajectory.gaps`. No point is interpolated across a gap.
    """

    point_count: int = 0
    detected_count: int = 0
    predicted_count: int = 0
    frame_start: int | None = None
    frame_end: int | None = None
    duration_s: float | None = None
    path_length_px: float | None = None
    detected_path_length_px: float | None = None
    net_displacement_px: float | None = None
    mean_speed_px_s: float | None = None
    detected_mean_speed_px_s: float | None = None
    segments: int = 0
    adjacent_segments: int = 0
    unobserved_segments: int = 0
    unobserved_span_px: float | None = None
    gaps: tuple[tuple[int, int], ...] = ()
    units: str = "pixel"

    @property
    def available(self) -> bool:
        """True when at least one point was measured."""
        return self.point_count > 0

    @property
    def continuous(self) -> bool:
        """True when every frame between the first and last has a point."""
        return not self.gaps

    @property
    def time_available(self) -> bool:
        """True when an fps was supplied and a duration could be computed."""
        return self.duration_s is not None

    def as_dict(self) -> dict:
        return {
            "point_count": self.point_count,
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "duration_s": self.duration_s,
            "path_length_px": self.path_length_px,
            "detected_path_length_px": self.detected_path_length_px,
            "net_displacement_px": self.net_displacement_px,
            "mean_speed_px_s": self.mean_speed_px_s,
            "detected_mean_speed_px_s": self.detected_mean_speed_px_s,
            "segments": self.segments,
            "adjacent_segments": self.adjacent_segments,
            "unobserved_segments": self.unobserved_segments,
            "unobserved_span_px": self.unobserved_span_px,
            "gaps": [[start, end] for start, end in self.gaps],
            "continuous": self.continuous,
            "units": self.units,
            "available": self.available,
            "time_available": self.time_available,
        }


@dataclass(frozen=True)
class PitchPlaneSummary:
    """
    What a Part 2 pitch-plane trajectory supports, in projected metres.

    This is the physical-distance document, and it is deliberately pessimistic:

      * `units` must be `"metre"`. A trajectory in `"pitch_fraction"` is
        rejected rather than silently treated as metres.
      * `speed_is_calibrated` is always False. The ball is airborne; projecting
        it onto the ground plane is an estimate, not a radar or a calibrated
        measurement.
      * `outside_pitch_count` exposes points Part 2 flagged as outside the
        measured quad. They are not silently dropped — inspect the count before
        treating the path length as a measurement.

    Path length follows the same rule as `TrajectorySummary`: only steps between
    consecutive frames contribute. A jump across a gap is reported as
    `unobserved_span_m`, never folded into the observed distance.

    Zero/None semantics are identical to `TrajectorySummary`.
    """

    point_count: int = 0
    detected_count: int = 0
    predicted_count: int = 0
    frame_start: int | None = None
    frame_end: int | None = None
    duration_s: float | None = None
    path_length_m: float | None = None
    detected_path_length_m: float | None = None
    net_displacement_m: float | None = None
    mean_speed_mps: float | None = None
    mean_speed_kmh: float | None = None
    detected_mean_speed_mps: float | None = None
    detected_mean_speed_kmh: float | None = None
    segments: int = 0
    adjacent_segments: int = 0
    unobserved_segments: int = 0
    unobserved_span_m: float | None = None
    gaps: tuple[tuple[int, int], ...] = ()
    outside_pitch_count: int = 0
    units: str = "metre"
    speed_is_estimate: bool = True
    speed_is_calibrated: bool = False

    @property
    def available(self) -> bool:
        """True when at least one projected point was measured."""
        return self.point_count > 0

    @property
    def continuous(self) -> bool:
        """True when every frame between the first and last has a point."""
        return not self.gaps

    @property
    def time_available(self) -> bool:
        """True when an fps was supplied and a duration could be computed."""
        return self.duration_s is not None

    def as_dict(self) -> dict:
        return {
            "point_count": self.point_count,
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "duration_s": self.duration_s,
            "path_length_m": self.path_length_m,
            "detected_path_length_m": self.detected_path_length_m,
            "net_displacement_m": self.net_displacement_m,
            "mean_speed_mps": self.mean_speed_mps,
            "mean_speed_kmh": self.mean_speed_kmh,
            "detected_mean_speed_mps": self.detected_mean_speed_mps,
            "detected_mean_speed_kmh": self.detected_mean_speed_kmh,
            "segments": self.segments,
            "adjacent_segments": self.adjacent_segments,
            "unobserved_segments": self.unobserved_segments,
            "unobserved_span_m": self.unobserved_span_m,
            "gaps": [[start, end] for start, end in self.gaps],
            "continuous": self.continuous,
            "outside_pitch_count": self.outside_pitch_count,
            "units": self.units,
            "available": self.available,
            "time_available": self.time_available,
            "speed_is_estimate": self.speed_is_estimate,
            "speed_is_calibrated": self.speed_is_calibrated,
        }


@dataclass(frozen=True)
class ShotDistribution:
    """
    Counts and scores over a set of shot records.

    `counts` always contains every class in the vocabulary, including the ones
    with a genuine count of zero, so a consumer never has to distinguish
    "absent" from "not asked about". `shares` is empty when nothing was
    classified: a proportion with a zero denominator is undefined, not 0.
    """

    counts: dict[str, int] = field(default_factory=dict)
    classified: int = 0
    unclassified: int = 0
    total: int = 0
    shares: dict[str, float] = field(default_factory=dict)
    unclassified_reasons: dict[str, int] = field(default_factory=dict)
    scored: int = 0
    mean_score: float | None = None
    min_score: float | None = None
    max_score: float | None = None

    @property
    def available(self) -> bool:
        """True when there is at least one record, classified or not."""
        return self.total > 0

    def as_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "classified": self.classified,
            "unclassified": self.unclassified,
            "total": self.total,
            "shares": dict(self.shares),
            "unclassified_reasons": dict(self.unclassified_reasons),
            "scored": self.scored,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "available": self.available,
        }


@dataclass(frozen=True)
class DeliveryAnalytics:
    """
    One delivery's shot and trajectory evidence, plus what is missing.

    `unavailable` is keyed by field name ("shot", "trajectory", "pitch",
    "time") and explains why that field is None. It is never used to explain a
    genuine zero.
    """

    delivery_id: int | str
    shot: ShotClassification | None = None
    trajectory: TrajectorySummary | None = None
    pitch: PitchPlaneSummary | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def has_shot(self) -> bool:
        return self.shot is not None and self.shot.ok

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory is not None and self.trajectory.available

    @property
    def has_pitch(self) -> bool:
        return self.pitch is not None and self.pitch.available

    @property
    def speed_kmh(self) -> float | None:
        """The projected average speed, or None when it was not measured."""
        return self.pitch.mean_speed_kmh if self.pitch is not None else None

    def as_dict(self) -> dict:
        return {
            "delivery_id": self.delivery_id,
            "shot": self.shot.as_dict() if self.shot is not None else None,
            "trajectory": self.trajectory.as_dict() if self.trajectory is not None else None,
            "pitch": self.pitch.as_dict() if self.pitch is not None else None,
            "unavailable": dict(self.unavailable),
        }


@dataclass(frozen=True)
class AnalysisSummary:
    """An analysis-level roll-up that never hides an unavailable delivery."""

    deliveries: tuple[DeliveryAnalytics, ...] = ()
    shot_distribution: ShotDistribution = field(default_factory=ShotDistribution)
    delivery_count: int = 0
    with_shot: int = 0
    with_trajectory: int = 0
    with_pitch: int = 0
    with_speed_kmh: int = 0

    def as_dict(self) -> dict:
        return {
            "deliveries": [delivery.as_dict() for delivery in self.deliveries],
            "shot_distribution": self.shot_distribution.as_dict(),
            "delivery_count": self.delivery_count,
            "with_shot": self.with_shot,
            "with_trajectory": self.with_trajectory,
            "with_pitch": self.with_pitch,
            "with_speed_kmh": self.with_speed_kmh,
        }


__all__ = [
    "AnalysisSummary",
    "DeliveryAnalytics",
    "PitchPlaneSummary",
    "ShotDistribution",
    "TrajectorySummary",
]
