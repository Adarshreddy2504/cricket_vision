"""
kalman_tracker.py — Constant-velocity Kalman filter for the cricket ball.

THE ONLY BALL TRACKER IN THIS MODULE.

WHAT THIS CLASS IS FOR
---------------------
It answers exactly one question: "where is the ball right now". It is a
smoother, NOT a segmenter. It does not decide where one flight ends and the
next begins — that is a separate concern, and keeping it out of here is what
makes the provenance of every point unambiguous.

The filter is a 4-state constant-velocity model, `[x, y, vx, vy]`. A cricket
ball travels in a near-straight line over the span of a single frame, so
modelling gravity inside one frame step would add noise rather than signal.

THE PROVENANCE RULE
-------------------
Every answer carries where it came from:

    "detected"   the position derives from a real detector measurement on
                 this frame.
    "predicted"  the position is a Kalman extrapolation. There was no
                 measurement, or the measurement was refused by the gate.
    None         the ball is lost. There is no position and no source.

A downstream stage that fits a physics curve MUST use only the "detected"
points. A predicted point is an estimate of where the ball was, not evidence
that it was there, and mixing the two produces confident nonsense.

THE ASSOCIATION GATE
--------------------
A detection further than `GATE_RADIUS_PX` from the current prediction is
almost always a false positive — a player's boot, a sponsor logo, the ball
from the NEXT delivery. It is refused rather than snapped to the prediction,
because snapping would drag the estimate onto a wrong answer.

Two rules about that refusal matter, and both are easy to get wrong:

  1. A refused measurement is evidence AGAINST the current track, so it
     advances the missed-frame counter. If it did not, a stream of detections
     belonging to the next ball would keep the current track alive forever.

  2. The gate is applied only while the track is still healthy. Once the
     filter has declared the ball lost, it no longer refuses measurements
     against a prediction it has already disowned — that is a livelock, not a
     rejection policy, because the refusal would keep the counter high and
     the track could never come back.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from ..core.constants import GATE_RADIUS, MAX_MISSED, MAX_TRAIL

# ── Defaults ──────────────────────────────────────────────────────────────────
# Module-level aliases, so a caller can read the settings the module ships
# with without reaching into `core.constants`.
MAX_TRAIL_POINTS = MAX_TRAIL       # maximum trajectory points kept in the trail
MAX_MISSED_FRAMES = MAX_MISSED      # frames without a measurement before the ball is lost
GATE_RADIUS_PX = GATE_RADIUS       # pixel radius — detections further away are refused


class TrackResult(NamedTuple):
    """
    The outcome of one tracker update.

    position : (x, y) in full-resolution source pixel space, or None when the
               ball is lost.
    source   : "detected" — the position comes from a real measurement on this
               frame.
               "predicted" — the position is a Kalman extrapolation: either
               there was no measurement, or the measurement was refused by the
               gate.
               None — the ball is lost; there is no position.
    gated_out: True when a real detection existed on this frame but fell
               outside the gate and was therefore refused. The position
               returned is then the previous estimate, and the source is
               "predicted" — the measurement was seen and deliberately not
               used. Reported separately so a consumer can tell "nothing was
               in this frame" apart from "something was in this frame and it
               was not where I expected", which are different events.
    """

    position: tuple | None
    source: str | None
    gated_out: bool = False


class KalmanBallTracker:
    """
    The 4-state filter itself: `[x, y, vx, vy]`.

    Kept separate from `BallTracker` so the linear algebra can be reasoned
    about — and tested — on its own.
    """

    def __init__(self) -> None:
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)

        # State transition (constant-velocity model)
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = 1.0   # x += vx
        self.F[1, 3] = 1.0   # y += vy

        # Measurement matrix (we observe x, y only)
        self.H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance — larger = trust the measurement more.
        self.Q = np.diag([5.0, 5.0, 20.0, 20.0]).astype(np.float32)

        # Measurement noise covariance.
        self.R = np.diag([10.0, 10.0]).astype(np.float32)

        # Estimate covariance.
        self.P = np.eye(4, dtype=np.float32) * 500.0

        self.initialized = False

    def init(self, pt: tuple) -> None:
        """Seed the filter with the first observed position."""
        self.state = np.array(
            [[pt[0]], [pt[1]], [0.0], [0.0]], dtype=np.float32
        )
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.initialized = True

    def predict(self) -> tuple:
        """Time-update step. Returns the predicted (x, y)."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return int(self.state[0, 0]), int(self.state[1, 0])

    def update(self, pt: tuple) -> None:
        """Measurement-update step."""
        z = np.array([[pt[0]], [pt[1]]], dtype=np.float32)
        y = z - self.H @ self.state                  # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def position(self) -> tuple:
        """The current estimated position (x, y)."""
        return int(self.state[0, 0]), int(self.state[1, 0])

    def velocity(self) -> tuple:
        """The current estimated velocity (vx, vy), in pixels per frame."""
        return float(self.state[2, 0]), float(self.state[3, 0])


class BallTracker:
    """
    The high-level tracker: the filter plus the state around it.

    It owns:
      * the filter itself,
      * the rolling trail of positions,
      * the missed-frame counter that decides when the ball is lost.

    Every setting is a constructor argument with the shipped defaults, so a
    caller can tighten the coast window or the gate for a particular clip
    without editing the module.
    """

    def __init__(
        self,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        gate_radius_px: int = GATE_RADIUS_PX,
        max_trail_points: int = MAX_TRAIL_POINTS,
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.gate_radius_px = int(gate_radius_px)
        self.max_trail_points = int(max_trail_points)

        self.kf = KalmanBallTracker()
        self.trajectory: deque = deque(maxlen=self.max_trail_points)
        self.missed_frames = 0

    # ── Per-frame update ──────────────────────────────────────────────────────
    def update(self, detection: tuple | None, frame_no: int) -> TrackResult:
        """
        Feed one frame's detection.

        `detection` is the measured (x, y) in source pixels, or None when the
        detector found nothing on this frame.

        Returns a `TrackResult(position, source, gated_out)`. See that class
        for the meaning of each field — the short version is that "detected"
        means a real measurement and "predicted" means an extrapolation, and
        the two must never be confused.

        NOTE: when a detection exists but is refused by the gate, the returned
        position is the previous estimate — an extrapolation of an ignored
        measurement — and is reported as "predicted".
        """
        if detection is not None:
            # The gate may only be applied while this filter still stands
            # behind its prediction. Once the track is lost the filter has
            # disowned that prediction, and refusing against it would be a
            # livelock: the refusal advances the counter, so the track could
            # never come back. Instead the detection falls through to the
            # cold-start path and re-initialises the filter.
            if self.kf.initialized and self.missed_frames <= self.max_missed_frames:
                px, py = self.kf.predict()
                dist = ((detection[0] - px) ** 2 + (detection[1] - py) ** 2) ** 0.5
                if dist > self.gate_radius_px:
                    # Likely a false positive. Skip the update but keep the
                    # prediction, AND advance the missed counter: a refused
                    # measurement is evidence against the current track, not
                    # evidence for it. Without this, a stream of detections
                    # belonging to the next ball keeps this track alive
                    # indefinitely.
                    self.missed_frames += 1
                    smoothed = self.kf.position()
                    self.trajectory.append(smoothed)
                    return TrackResult(smoothed, "predicted", gated_out=True)
                self.kf.update(detection)
            else:
                self.kf.init(detection)

            self.missed_frames = 0
            pos = self.kf.position()
            self.trajectory.append(pos)
            return TrackResult(pos, "detected")

        # No detection — predict only.
        self.missed_frames += 1
        if self.kf.initialized and self.missed_frames <= self.max_missed_frames:
            pos = self.kf.predict()
            self.trajectory.append(pos)
            return TrackResult(pos, "predicted")

        # The ball is genuinely lost. No position, and — crucially — nothing
        # invented to fill the hole.
        return TrackResult(None, None)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def reseed(self, pt: tuple, frame_no: int) -> TrackResult:
        """
        Re-acquire the ball without a gap.

        `MAX_MISSED_FRAMES` bounds how long the filter will extrapolate, and
        once it expires `update()` returns no position at all. That is correct
        for the filter but wrong for the flight: a ball passing behind the
        batter's bat, or behind a fielder's arm, routinely disappears for
        longer than that and then comes straight back. Treated as a lost ball
        that ends one flight and starts another, so a single stroke is cut in
        half.

        `reseed` restarts the filter on the ball that is actually visible.
        Provenance is unaffected: this IS a real measurement, so the returned
        source is "detected".
        """
        self.kf.init(pt)
        self.missed_frames = 0
        pos = self.kf.position()
        self.trajectory.append(pos)
        return TrackResult(pos, "detected")

    def reset(self) -> None:
        """Forget everything: the filter, the trail and the miss counter."""
        self.kf.initialized = False
        self.missed_frames = 0
        self.trajectory.clear()

    # ── Read-only views ───────────────────────────────────────────────────────
    @property
    def is_lost(self) -> bool:
        """True once the ball has been missing for longer than the coast window."""
        return self.missed_frames > self.max_missed_frames

    @property
    def traj_points(self) -> list:
        """The rolling trail, as a plain list."""
        return list(self.trajectory)


__all__ = [
    "GATE_RADIUS_PX",
    "MAX_MISSED_FRAMES",
    "MAX_TRAIL_POINTS",
    "BallTracker",
    "KalmanBallTracker",
    "TrackResult",
]
