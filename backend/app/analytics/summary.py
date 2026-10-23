"""
summary.py — One delivery, then one analysis, assembled from the parts.

WHY THERE IS A MISSING-DATA MAP
-------------------------------
A roll-up is where honesty usually leaks: a failed shot classification becomes a
missing key, a trajectory with no fps quietly loses its speed, and the document
looks complete because every field it does have is filled in. `DeliveryAnalytics`
therefore carries `unavailable`, keyed by field name, saying why that part is
absent. A genuine zero never appears there.

Fps is validated by the trajectory summarisers, once. If it is absent, the
counts and distances survive and `unavailable["time"]` explains the missing
durations and speeds. If it is invalid (zero, negative, NaN), the summarisers
raise and the reason is recorded against the affected field.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..schemas.analytics import (
    AnalysisSummary,
    DeliveryAnalytics,
    PitchPlaneSummary,
    TrajectorySummary,
)
from ..schemas.shots import ShotClassification
from .errors import AnalyticsInputError
from .pitch_plane import summarise_pitch_trajectory
from .shots import summarise_shots
from .trajectory import summarise_trajectory


def summarise_delivery(
    delivery_id: int | str,
    *,
    shot: ShotClassification | None = None,
    trajectory: object | None = None,
    fps: float | None = None,
    pitch_trajectory: object | None = None,
) -> DeliveryAnalytics:
    """
    Assemble one delivery's analytics without inventing the missing parts.

    `trajectory` is anything Part 1's summariser accepts (a `Trajectory`, a
    sequence of points, or a stored mapping). `pitch_trajectory` is anything
    Part 2's summariser accepts. Neither is imported here, which is what keeps
    this module usable on a branch where Part 1 or Part 2 is absent.
    """
    unavailable: dict[str, str] = {}

    if shot is None:
        unavailable["shot"] = "no shot record supplied"
    elif not shot.ok:
        reason = shot.status if not shot.error else f"{shot.status}: {shot.error}"
        unavailable["shot"] = reason

    if fps is None:
        unavailable["time"] = "fps not supplied"

    trajectory_summary: TrajectorySummary | None = None
    if trajectory is None:
        unavailable["trajectory"] = "no trajectory supplied"
    else:
        try:
            trajectory_summary = summarise_trajectory(trajectory, fps)
        except AnalyticsInputError as exc:
            unavailable["trajectory"] = str(exc)

    pitch_summary: PitchPlaneSummary | None = None
    if pitch_trajectory is None:
        unavailable["pitch"] = "no pitch trajectory supplied"
    else:
        try:
            pitch_summary = summarise_pitch_trajectory(pitch_trajectory, fps)
        except AnalyticsInputError as exc:
            unavailable["pitch"] = str(exc)

    return DeliveryAnalytics(
        delivery_id=delivery_id,
        shot=shot,
        trajectory=trajectory_summary,
        pitch=pitch_summary,
        unavailable=unavailable,
    )


def summarise_analysis(deliveries: Iterable[DeliveryAnalytics]) -> AnalysisSummary:
    """
    Roll up already-summarised deliveries.

    Only shot records that were supplied take part in the distribution; a
    delivery with no shot at all is counted by `delivery_count` and visible in
    its own `unavailable` map, not silently counted as an unclassified shot.
    """
    if deliveries is None:
        raise AnalyticsInputError(
            "deliveries is None; pass an empty iterable for no deliveries"
        )

    items = tuple(deliveries)
    shot_records = [delivery.shot for delivery in items if delivery.shot is not None]

    return AnalysisSummary(
        deliveries=items,
        shot_distribution=summarise_shots(shot_records),
        delivery_count=len(items),
        with_shot=sum(1 for delivery in items if delivery.has_shot),
        with_trajectory=sum(1 for delivery in items if delivery.has_trajectory),
        with_pitch=sum(1 for delivery in items if delivery.has_pitch),
        with_speed_kmh=sum(1 for delivery in items if delivery.speed_kmh is not None),
    )


__all__ = ["summarise_analysis", "summarise_delivery"]
