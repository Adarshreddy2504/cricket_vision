"""
pitch_helpers.py — synthetic frames, hand-checkable quads and stub models.

The stubs exist so the pitch module can be tested deterministically and
without the YOLO weights, which are deliberately NOT tracked in git (see
docs/PITCH.md). They are not a shortcut around the real detector: the stub
model implements the `ultralytics.YOLO` contract the detector calls —

    model.predict(source=..., conf=..., imgsz=..., device=..., verbose=...)
    -> a sequence whose first element has `boxes` and `masks`

— so the extraction, validation and selection logic under test is the real
one. `StubBoxes` mirrors `ultralytics.engine.results.Boxes` (`xyxy`,
`conf`, `cls`, `__len__`) and `StubMasks` mirrors `Masks.xy`.

WHY THESE SHAPES
----------------
`RECTANGLE_POINTS` is an axis-aligned rectangle. Its homography to the unit
square is affine, so every interior point maps to a coordinate that can be
computed by hand — which is what makes the conversion tests evidence rather
than a recording of whatever the code happened to print.

`TRAPEZOID_POINTS` is a perspective quad, so the tests also cover the case
where the mapping is genuinely projective and an interior point is NOT the
naive linear interpolation of its position in the bounding box.
"""

from __future__ import annotations

import numpy as np

from app.pitch.corners import PitchCorners, order_corners

#: An axis-aligned rectangle inside a 640x360 frame, already in canonical order.
RECTANGLE_POINTS: tuple[tuple[float, float], ...] = (
    (100.0, 100.0),
    (300.0, 100.0),
    (300.0, 300.0),
    (100.0, 300.0),
)

#: A perspective quad: the far edge is narrower than the near edge.
TRAPEZOID_POINTS: tuple[tuple[float, float], ...] = (
    (100.0, 100.0),
    (300.0, 100.0),
    (320.0, 300.0),
    (80.0, 300.0),
)

#: The frame both quads above live in.
FRAME_SIZE: tuple[int, int] = (640, 360)

#: The pitch dimensions the tests compute against (Law 6, as in constants.py).
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 2.64


def make_frame(width: int = 640, height: int = 360, value: int = 30) -> np.ndarray:
    """A flat BGR frame of the requested size."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_pitch_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """A frame with a green band down the middle, vaguely pitch-like."""
    frame = make_frame(width, height, value=40)
    left = int(width * 0.40)
    right = int(width * 0.60)
    top = int(height * 0.12)
    bottom = int(height * 0.92)
    frame[top:bottom, left:right] = (60, 140, 60)
    return frame


def rectangle() -> PitchCorners:
    """The axis-aligned test rectangle, in canonical order."""
    return order_corners(RECTANGLE_POINTS)


def trapezoid() -> PitchCorners:
    """The perspective test quad, in canonical order."""
    return order_corners(TRAPEZOID_POINTS)


def pitch_mask_polygon(
    *,
    top_y: float = 60.0,
    bottom_y: float = 340.0,
    top_half_width: float = 50.0,
    bottom_half_width: float = 80.0,
    centre_x: float = 320.0,
) -> list[tuple[float, float]]:
    """
    A trapezoid mask polygon over the pitch, in drawing order.

    At 640x360 with the default arguments this covers 0.158 of the frame with
    a 1.75:1 aspect ratio, so it passes the shape gate; the detectors derive
    the four corners (270, 60), (370, 60), (400, 340), (240, 340) from it.
    """
    return [
        (centre_x - top_half_width, top_y),
        (centre_x + top_half_width, top_y),
        (centre_x + bottom_half_width, bottom_y),
        (centre_x - bottom_half_width, bottom_y),
    ]


class StubBoxes:
    """The subset of `ultralytics...Boxes` the pitch detector reads."""

    def __init__(self, rows: list[tuple[float, float, float, float, float, float]]) -> None:
        self.xyxy = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32)
        self.conf = np.array([r[4] for r in rows], dtype=np.float32)
        self.cls = np.array([r[5] for r in rows], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])


class StubMasks:
    """The subset of `ultralytics...Masks` the pitch detector reads."""

    def __init__(self, polygons: list[list[tuple[float, float]]]) -> None:
        self.xy = [np.asarray(p, dtype=np.float32) for p in polygons]


class StubResult:
    """One ultralytics result: `boxes` and `masks`, either may be None."""

    def __init__(self, boxes=None, masks=None) -> None:
        self.boxes = boxes
        self.masks = masks


class _RecordingModel:
    """Shared bookkeeping: how many calls, and the last `predict` keyword arguments."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict = {}

    def _record(self, kwargs) -> None:
        self.calls += 1
        self.last_kwargs = dict(kwargs)


class StubModel(_RecordingModel):
    """
    A deterministic stand-in for `ultralytics.YOLO`.

    Returns ONE response — a list of results — on every call. Records how it
    was called, so the tests can assert that the real frame reached the model
    at its original size and that the detector never resized it first.
    """

    def __init__(self, results) -> None:
        super().__init__()
        self._results = list(results)

    def predict(self, **kwargs):
        self._record(kwargs)
        return self._results


class StubSequenceModel(_RecordingModel):
    """
    A stand-in model that answers differently on successive calls.

    `responses` is a list of responses, one consumed per `predict` call, with
    the last one repeated once the queue runs out. This is what lets a test
    give ONE detector a sequence of different outcomes (a detour, then a tie,
    then a failure) without rebuilding the detector.
    """

    def __init__(self, responses) -> None:
        super().__init__()
        self._responses = [list(response) for response in responses]

    def predict(self, **kwargs):
        self._record(kwargs)
        if not self._responses:
            return []
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class StubTrajectoryPoint:
    """
    A stand-in for Part 1's `TrajectoryPoint`, field for field.

    Kept here rather than imported because Part 1 lives on another branch.
    `test_pitch_adapter.py` additionally runs against the REAL Part 1
    dataclass when it is importable, so the stand-in is a convenience and
    not the only evidence.
    """

    def __init__(
        self,
        frame: int,
        x: int,
        y: int,
        source: str,
        confidence: float | None = None,
        track_id: int | None = 1,
    ) -> None:
        self.frame = frame
        self.x = x
        self.y = y
        self.source = source
        self.confidence = confidence
        self.track_id = track_id


class StubTrajectory:
    """A stand-in for Part 1's `Trajectory`: an object with a `points` list."""

    def __init__(self, points: list[StubTrajectoryPoint]) -> None:
        self.points = points


def sample_trajectory() -> StubTrajectory:
    """
    A short flight across the test rectangle.

    Frame 1 is inside the pitch, frame 2 is inside, frame 3 is predicted,
    and frame 4 is wide of the strip (outside the quad) — so the adapter's
    provenance pass-through and its `inside_pitch` reporting are both
    exercised by one fixture.
    """
    return StubTrajectory(
        [
            StubTrajectoryPoint(1, 120, 120, "detected", 0.91, 7),
            StubTrajectoryPoint(2, 200, 200, "detected", 0.84, 7),
            StubTrajectoryPoint(3, 240, 240, "predicted", None, 7),
            StubTrajectoryPoint(4, 400, 120, "detected", 0.72, 7),
        ]
    )
