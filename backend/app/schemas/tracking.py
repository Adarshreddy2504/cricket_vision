"""
tracking.py — Typed records produced by the tracking pass.

These dataclasses define the document the tracking module produces. The schema
is the contract between the tracking pass and every consumer of it, so it is
expressed in code rather than assembled ad hoc inside the service.

INVARIANTS ENFORCED HERE RATHER THAN TRUSTED
--------------------------------------------
  * Trajectory points are never synthesised. A point exists only if the
    tracker produced a position for that frame, and it always records which it
    was: a real measurement or a Kalman extrapolation.
  * A gap is never filled in. Frames with no point are simply not
    represented, and `Trajectory.gaps` names them so a consumer can see the
    hole instead of inferring a continuous path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

TrajectorySource = Literal["detected", "predicted"]


@dataclass
class TrajectoryPoint:
    """
    One tracked ball position.

    frame      : 1-based index of the frame in the input stream. This is the
                 authoritative timebase for the whole module.
    x, y       : pixel coordinates in the SAME space as the frames that were
                 handed in. A frame was never resized without rescaling the
                 answer by the same factor, so these are always source pixels.
    source     : "detected" (a real detector measurement on this frame) or
                 "predicted" (a Kalman extrapolation). Never fabricated.
    confidence : the detector's confidence for detected points, null for
                 predicted ones.
    track_id   : the identifier of the track this point belongs to. It is
                 stable while the ball is being followed — including across a
                 short occlusion — and changes when the ball is genuinely lost
                 and re-acquired.
    low_confidence : True when a real measurement was below the track-start
                 threshold and was accepted only because a live track already
                 existed and the Kalman gate agreed. The source stays
                 "detected"; this flag records the weaker provenance.
    """

    frame: int
    x: int
    y: int
    source: TrajectorySource
    confidence: float | None = None
    track_id: int | None = None
    low_confidence: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    """
    The ball's path through a stream of frames.

    Consumed by, unmodified: the overlay renderer, the trajectory view and
    any later analytic stage.
    """

    points: list[TrajectoryPoint] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def detected_count(self) -> int:
        return sum(1 for p in self.points if p.source == "detected")

    @property
    def predicted_count(self) -> int:
        return sum(1 for p in self.points if p.source == "predicted")

    @property
    def continuity(self) -> str:
        return "continuous" if not self.gaps else "gapped"

    @property
    def gaps(self) -> list[list[int]]:
        """
        Contiguous runs of frames with no tracked point at all.

        Computed from the points that exist; frames without a point are simply
        not represented. No point is ever interpolated to fill a gap.
        """
        if len(self.points) < 2:
            return []
        frames = sorted(p.frame for p in self.points)
        gaps: list[list[int]] = []
        run_start = None
        # `frames[1:]` is deliberately one element shorter than `frames`: the
        # zip walks the consecutive pairs.
        for prev, cur in zip(frames, frames[1:], strict=False):
            if cur - prev > 1:
                if run_start is None:
                    run_start = prev + 1
            elif run_start is not None:
                # `prev` itself HAS a point, so the gap ends at prev - 1.
                gaps.append([run_start, prev - 1])
                run_start = None
        if run_start is not None:
            gaps.append([run_start, frames[-1] - 1])
        return gaps

    def detected_points(self) -> list[TrajectoryPoint]:
        """Detection-confirmed points only — the valid input for physics fits."""
        return [p for p in self.points if p.source == "detected"]

    def as_dict(self) -> dict:
        return {
            "points": [p.as_dict() for p in self.points],
            "point_count": self.point_count,
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "continuity": self.continuity,
            "gaps": self.gaps,
        }
