"""
pitch.py — Typed records produced by the pitch projection.

These dataclasses define the document that re-expresses a ball trajectory in
pitch coordinates. They are PLAIN DATA: nothing here imports the pitch
geometry or the tracker, so the document can be serialised, stored and read
back without dragging a model or a homography along with it.

INVARIANTS ENFORCED HERE RATHER THAN TRUSTED
--------------------------------------------
  * No point is ever invented. A pitch point exists only because a ball point
    existed, and it keeps the ball point's `source` — "detected" or
    "predicted" — so a Kalman extrapolation can never be mistaken for a
    measurement after the conversion.
  * A point that could not be projected is not clamped, not averaged and not
    dropped silently. It is counted in `skipped` and named in
    `skipped_frames`, and the frames it belonged to appear in `gaps` exactly
    as they do in the tracking document.
  * `units` is always present. A number in "pitch_fraction" and a number in
    "metre" are different kinds of claim, and the document never leaves the
    distinction to be inferred from a field name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

#: Which pitch coordinate system the numbers are in.
PitchSpaceName = Literal["normalized", "metres"]

#: What the numbers are measured in.
PitchUnits = Literal["pitch_fraction", "metre"]

#: Where the underlying ball position came from. Part 1's vocabulary.
PitchPointSource = Literal["detected", "predicted"]


@dataclass
class PitchPoint:
    """
    One tracked ball position, re-expressed in pitch coordinates.

    frame         : 1-based frame index, inherited unchanged from the ball
                    point. This is the authoritative timebase.
    x, y          : the SOURCE PIXEL the ball was at, inherited unchanged.
                    Kept so a client can draw the point back onto the
                    picture without inverting the transform itself.
    across, along : the pitch coordinate. In `normalized` these are fractions
                    of the pitch quad (0.0 at one end of an axis, 1.0 at the
                    other); in `metres` they are metres, with `across`
                    measured from the middle of the strip and `along` from
                    the FAR end of the pitch towards the NEAR end.
    space, units  : which of the two the numbers are, in words.
    source        : "detected" or "predicted", inherited unchanged.
    confidence    : the detector's confidence, inherited unchanged. Null for
                    predicted points and for points that never had one.
    track_id      : the tracker's track identifier, inherited unchanged.
    inside_pitch  : whether the point fell inside the pitch quad. False is
                    information (the ball was wide of the strip, or the
                    projection put it outside), never an error.
    """

    frame: int
    x: int
    y: int
    across: float
    along: float
    space: PitchSpaceName
    units: PitchUnits
    source: PitchPointSource
    confidence: float | None = None
    track_id: int | None = None
    inside_pitch: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PitchTrajectory:
    """
    A ball trajectory in pitch coordinates, plus the calibration it used.

    `calibration` is the serialised calibration document, so a consumer can
    see how the numbers were produced without being handed the object that
    produced them. `skipped` and `skipped_frames` make the points that could
    not be projected visible instead of silently absent.
    """

    points: list[PitchPoint] = field(default_factory=list)
    space: PitchSpaceName = "normalized"
    units: PitchUnits = "pitch_fraction"
    calibration: dict = field(default_factory=dict)
    skipped: int = 0
    skipped_frames: list[int] = field(default_factory=list)

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
    def inside_count(self) -> int:
        return sum(1 for p in self.points if p.inside_pitch)

    @property
    def gaps(self) -> list[list[int]]:
        """
        Contiguous runs of frames with no projected point.

        The same computation the tracking document uses, and for the same
        reason: a hole must be visible rather than bridged. A frame with no
        point — whether the ball was lost or the projection refused it — is
        simply not represented.
        """
        if len(self.points) < 2:
            return []
        frames = sorted(p.frame for p in self.points)
        gaps: list[list[int]] = []
        run_start = None
        for prev, cur in zip(frames, frames[1:], strict=False):
            if cur - prev > 1:
                if run_start is None:
                    run_start = prev + 1
            elif run_start is not None:
                gaps.append([run_start, prev - 1])
                run_start = None
        if run_start is not None:
            gaps.append([run_start, frames[-1] - 1])
        return gaps

    @property
    def continuity(self) -> str:
        return "continuous" if not self.gaps else "gapped"

    def as_dict(self) -> dict:
        return {
            "space": self.space,
            "units": self.units,
            "points": [p.as_dict() for p in self.points],
            "point_count": self.point_count,
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "inside_count": self.inside_count,
            "continuity": self.continuity,
            "gaps": self.gaps,
            "skipped": self.skipped,
            "skipped_frames": self.skipped_frames,
            "calibration": self.calibration,
        }


__all__ = [
    "PitchPoint",
    "PitchPointSource",
    "PitchSpaceName",
    "PitchTrajectory",
    "PitchUnits",
]
