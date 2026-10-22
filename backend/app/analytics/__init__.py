"""
analytics — Pure calculations over Part 1 trajectories, Part 2 pitch geometry
and Part 3 shot records.

No module here opens a video, loads a model or imports a tracker. Each one takes
plain records and returns a `app.schemas.analytics` document, which makes the
whole package testable on hand-written points with known answers.

    summarise_trajectory        pixel counts, path length, gaps, pixel speed
    summarise_pitch_trajectory  projected metres, path length, estimated km/h
    summarise_shots             label counts, shares, score statistics
    summarise_delivery          one delivery, plus what is unavailable and why
    summarise_analysis          an analysis-level roll-up
"""

from __future__ import annotations

from .errors import AnalyticsError, AnalyticsInputError
from .pitch_plane import (
    MPS_TO_KMH,
    PITCH_UNITS,
    PitchPointLike,
    PitchTrajectoryLike,
    summarise_pitch_trajectory,
)
from .shots import summarise_shots
from .summary import summarise_analysis, summarise_delivery
from .trajectory import (
    POINT_SOURCES,
    TrajectoryLike,
    TrajectoryPointLike,
    summarise_trajectory,
)

__all__ = [
    "MPS_TO_KMH",
    "PITCH_UNITS",
    "POINT_SOURCES",
    "AnalyticsError",
    "AnalyticsInputError",
    "PitchPointLike",
    "PitchTrajectoryLike",
    "TrajectoryLike",
    "TrajectoryPointLike",
    "summarise_analysis",
    "summarise_delivery",
    "summarise_pitch_trajectory",
    "summarise_shots",
    "summarise_trajectory",
]
