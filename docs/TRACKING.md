# Ball tracking (Part 1)

The ball-tracking module: it finds the cricket ball in each frame of a
video, follows it across frames, and reports where it went — with an
honest account of which positions were measured and which were estimated.

This is Part 1 of CricketVision. Pitch detection, shot classification,
delivery analytics, the HTTP API and the frontend are **not** part of this
branch.

---

## What it does

Given a video (or a stream of frames), the module produces a trajectory of
the ball.

Three components, each with one job:

| Component | File | Job |
|---|---|---|
| `BallDetector` | `backend/app/tracking/detector.py` | Finds the ball in a single frame. YOLO11, one class, full-resolution output. |
| `BallTracker` | `backend/app/tracking/kalman_tracker.py` | Smooths those findings into a trajectory. 4-state constant-velocity Kalman filter. |
| `TrackingService` | `backend/app/tracking/tracking_service.py` | Drives the two over a stream of frames and produces the result. |

### The rule everything else follows: provenance

Every point in a trajectory says where it came from.

| `source` | Meaning | `confidence` |
|---|---|---|
| `"detected"` | A real detector measurement on that frame. | The detector's confidence. |
| `"predicted"` | A Kalman extrapolation — either no detection that frame, or a detection refused by the association gate. | `null`. There was no measurement, so there is no confidence. |
| `null` | The ball is lost. There is no position, and no point at all. | `null`. |

Nothing is interpolated to fill a hole. When the ball is lost the affected
frames simply carry no point, and `Trajectory.gaps` names them. A gap in the
trajectory is information about the footage, not something to paper over.

A consequence worth stating plainly: **only `"detected"` points may be used
to fit a physics curve.** Fitting release speed to a mix of measurements and
extrapolations produces a confident, wrong number.

### The association gate

A detection further than 120 px from the current prediction is almost always
a false positive — a player's boot, a sponsor logo, the ball from the *next*
delivery. It is refused rather than snapped to the prediction.

Two rules govern that refusal, and both matter:

1. **A refused detection counts towards the track dying.** A refused
   measurement is evidence *against* the current track, not for it.
   Otherwise a stream of detections belonging to the next ball would keep
   the current track alive indefinitely, merging two separate flights into
   one.
2. **The gate is not applied while the track is already lost.** Refusing
   against a prediction the filter has already disowned is a livelock: the
   refusal advances the miss counter, so the track can never come back.
   Instead the detection re-initialises the filter.

### Track identity

The `track_id` is stable while the ball is being followed — including across
a short occlusion where the ball is hidden behind the bat — and changes once
the ball has genuinely been lost and re-acquired.

A track must collect `MIN_TRACK_DETECTIONS` (3) *detected* points before it
is presented as a confirmed flight. A single stray false positive is not a
flight.

---

## Requirements

- **Python 3.10 or newer.** (Developed and tested on 3.14.4; the module uses
  `X | Y` type syntax and `from __future__ import annotations`.)
- The dependencies in [`requirements.txt`](../requirements.txt):

  | Package | Used for |
  |---|---|
  | `numpy` | Frame arrays, the filter's linear algebra. |
  | `opencv-python` | Video decoding, frame resizing for inference. |
  | `ultralytics` | The YOLO11 runtime the detector is loaded from. |
  | `torch` | Ultralytics' inference backend. Imported lazily, so the module imports and its pure-maths parts test without a GPU. |
  | `torchvision` | Required by ultralytics. |
  | `python-dotenv` | Optional. Reads a `.env` from the project root. Never overrides a real environment variable. |
  | `pytest` | The test suite. |

```bash
pip install -r requirements.txt
```

For a CPU-only install, use the PyTorch CPU index and drop the minimum pins:

```bash
pip install "torch>=2.5" "torchvision>=0.20" --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` uses compatible ranges rather than exact pins so the module
also installs against a newer environment. The exact versions the upstream
project was verified against are recorded as comments next to each entry.

---

## Model weights

**The checkpoint is not tracked in git.** `*.pt` and friends are ignored by
`.gitignore`, and the weights are not redistributed here.

| File | Size | Purpose | MD5 |
|---|---|---|---|
| `ball_detector.pt` | 6,225,450 B | YOLO11 ball detection, single class `ball` | `adde61b5409fce5ec6583867d00c070c` |

Requirements on the checkpoint:

- A YOLO11 **detection** model loadable with `ultralytics.YOLO`, exported to
  a `.pt` file.
- **Single class**, and it must be class index `0`. The detector refuses every
  other class — a second class in the checkpoint is ignored rather than
  mis-read as the ball.
- The model is fed BGR frames; ultralytics handles letterboxing and
  normalisation. The frame is resized to 640×360 by this module *before*
  inference, and the answer is scaled back afterwards.

A checkpoint loaded with the wrong task loads without error and returns
tensors the tracker cannot read, so verify the checksum after copying it.

### Pointing the module at the weights

Two environment variables, checked in this order:

```bash
export CRICKET_MODELS_DIR=/path/to/models        # the directory
export CRICKET_BALL_MODEL=/path/to/detector.pt    # or one file, overriding the directory
```

Without either, the module looks for `backend/models/ball_detector.pt`.

| Variable | Default | Meaning |
|---|---|---|
| `CRICKET_MODELS_DIR` | `backend/models` | Directory holding the weights. |
| `CRICKET_BALL_MODEL` | `$CRICKET_MODELS_DIR/ball_detector.pt` | One weights file, overriding the directory. |

**The module never downloads anything.** If the file is not on disk the
detector reports itself unavailable with the reason, and the tracking service
refuses to run with `DetectorUnavailableError`. A missing model must never
turn into a network fetch, and must never be reported as "no ball was found"
— a detector with no weights cannot find a ball in *any* frame, and saying
otherwise would record a missing model as an empty analysis.

### Verifying a copy

```bash
python -c "import hashlib,pathlib; \
p=pathlib.Path('backend/models/ball_detector.pt'); \
print(hashlib.md5(p.read_bytes()).hexdigest())"
```

---

## Configuration

Every tunable can be overridden from the environment. Values are resolved
once, at import.

| Variable | Default | Meaning |
|---|---|---|
| `CRICKET_BALL_CONF` | `0.30` | Minimum confidence for a box to count. |
| `CRICKET_BALL_RESIZE_W` | `640` | Inference width. |
| `CRICKET_BALL_RESIZE_H` | `360` | Inference height. |
| `CRICKET_BALL_MIN_DIAG` | `4` | Smallest plausible ball-box diagonal, in resized pixels. |
| `CRICKET_BALL_MAX_DIAG` | `80` | Largest plausible ball-box diagonal, in resized pixels. |
| `CRICKET_BALL_DEVICE` | `auto` | `auto` picks CUDA when available, CPU otherwise. Any other value is passed to ultralytics verbatim. |
| `CRICKET_BALL_GATE_RADIUS` | `120` | Association-gate radius, in source pixels. |
| `CRICKET_BALL_MAX_MISSED` | `8` | Frames the filter may coast before the ball is lost. |
| `CRICKET_BALL_MAX_TRAIL` | `120` | Positions kept in the rolling trail. |
| `CRICKET_BALL_MIN_TRACK_DETECTIONS` | `3` | Detections needed before a track is confirmed. |

The same values are reachable in code through
`app.core.config.TrackingConfig`, which is a frozen dataclass — use
`dataclasses.replace` to build a modified copy:

```python
from dataclasses import replace
from app.core.config import TRACKING_CONFIG

tight = replace(TRACKING_CONFIG, gate_radius_px=60)
```

`TRACKING_CONFIG.as_metadata()` returns the whole configuration as plain
data. It contains no credentials, and is safe to write next to a result.

---

## How to call it

### From a video

```python
from app.tracking import TrackingService

service = TrackingService()
result = service.track_video("delivery.mp4")

print(result.frames_decoded, "frames")
for track in result.confirmed_tracks:
    print(f"flight {track.track_id}: frames {track.start_frame}-{track.end_frame}")
```

### From frames you already have

```python
import cv2

service = TrackingService()
cap = cv2.VideoCapture("delivery.mp4")
frames = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
cap.release()

result = service.track_frames(frames)
```

### One frame at a time

```python
outcome = service.process_frame(frame)          # the service numbers the frames
outcome = service.process_frame(frame, 42)       # or you number them

outcome.position    # (x, y) in source pixels, or None when the ball is lost
outcome.source      # "detected" | "predicted" | None
outcome.confidence  # float on detected points, else None
outcome.track_id    # int, or None when the ball is lost
outcome.detection   # the raw detector output, before filtering
outcome.gated_out   # True when a detection existed but was refused
```

### The result

```python
result.points            # every point, in frame order
result.tracks            # the points grouped by flight
result.confirmed_tracks  # the flights backed by enough real detections
result.trajectory        # the same points, with .gaps / .continuity
result.frames_decoded
result.detection_frames
result.predicted_frames
result.video             # {"path", "width", "height", "fps", ...}
result.as_dict()         # the whole thing as plain, serialisable data
```

### Using the pieces on their own

```python
from app.tracking import BallDetector, BallTracker

detector = BallDetector()
cx, cy, conf = detector.detect_with_confidence(frame)   # source pixels

tracker = BallTracker()
result = tracker.update((cx, cy), frame_no)
result.position, result.source, result.gated_out
```

---

## Input and output formats

### Input

- **Video**: anything OpenCV can decode — `.mp4`, `.mov`, `.avi`, `.mkv`.
  The container's advertised frame count is recorded as a diagnostic only; the
  **1-based decode counter is the authoritative frame numbering**, because
  that count is frequently wrong for the formats cricket footage arrives in.
- **Frames**: BGR `numpy.ndarray` of shape `H × W × 3`, non-empty. A frame
  that is `None`, not an array, greyscale (`H × W`), or empty raises
  `InvalidFrameError` — which is also a `ValueError`, so code that only cares
  that the argument was bad can catch `ValueError` and be correct.

### Output — the trajectory document

`TrackingResult.as_dict()`:

```json
{
  "schema_version": "1.0",
  "frames_decoded": 240,
  "detection_frames": 118,
  "predicted_frames": 31,
  "point_count": 149,
  "confirmed_track_count": 1,
  "points": [
    {
      "frame": 42,
      "x": 913,
      "y": 388,
      "source": "detected",
      "confidence": 0.87,
      "track_id": 1
    }
  ],
  "tracks": [
    {
      "track_id": 1,
      "start_frame": 38,
      "end_frame": 96,
      "point_count": 59,
      "detected_count": 51,
      "predicted_count": 8,
      "is_confirmed": true,
      "points": [ ... ]
    }
  ],
  "trajectory": {
    "points": [ ... ],
    "point_count": 149,
    "detected_count": 118,
    "predicted_count": 31,
    "continuity": "gapped",
    "gaps": [[71, 74]]
  },
  "video": {
    "path": "delivery.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 25.0,
    "declared_frame_count": 240,
    "frames_decoded": 240
  }
}
```

Coordinates are **always in the pixel space of the frames that were handed
in**. The detector's output is scaled back by exactly the factor the frame
was downscaled by, so a caller never has to think about it. A detector that
returned resized coordinates would produce a ball that appears to move at half
its real speed.

`gaps` lists contiguous runs of frames with no tracked point, as inclusive
`[first, last]` pairs.

### Errors

| Error | Raised when |
|---|---|
| `TrackingError` | Base class for everything below. |
| `InvalidFrameError` | The frame is not a usable BGR image. Also a `ValueError`. |
| `VideoOpenError` | The video could not be opened. |
| `DetectorUnavailableError` | The detector has no usable model. |

---

## Running the tests

```bash
pytest backend
```

Or from the repository root:

```bash
python -m pytest backend -q
```

`backend/conftest.py` puts `backend/` on `sys.path`, so `pytest` works from
the repository root as well as from `backend/` itself.

### Linting

```bash
ruff check backend
```

The rules and the line length are pinned in `ruff.toml` at the repository
root. The lint configuration lives there rather than in a `pyproject.toml`
because the project has no packaging metadata yet.

**The tests do not need the model weights.** The stub in
`backend/tests/helpers.py` implements the exact `ultralytics.YOLO` contract
the detector uses, so the real selection, filtering and coordinate-scaling
logic is what gets exercised. The weights are only needed to run the module
against real footage.

The suite covers:

| File | What it pins |
|---|---|
| `test_ball_detector.py` | Box selection, the class and size filters, coordinate scaling, the missing-model path, and invalid frames. |
| `test_kalman_tracker.py` | Provenance, the association gate, coasting, loss, recovery and re-acquisition, and the tracker's state hygiene. |
| `test_tracking_service.py` | Frame numbering, provenance end to end, track identity across an occlusion and after a loss, the per-track records, the counters, gaps, invalid input and the video path. |

---

## Limitations

Known and deliberate, rather than accidental:

- **No accuracy claims are made here.** This branch contains no benchmark
  against real footage, and the constants are the upstream project's values,
  not values fitted to any particular clip. The detection and tracking
  behaviour on real broadcast footage has not been measured in this
  repository.
- **The model weights are external.** Nothing here can find a ball without
  them, and the module will refuse to run rather than pretend.
- **One ball at a time.** The tracker follows a single ball. Two balls in
  frame (a throw and a fielder's hand, say) will contend for the same
  filter; the association gate is what keeps the wrong one out.
- **The filter is constant-velocity.** Gravity is not modelled. Over a
  single frame that is a reasonable approximation; over a long arc the
  extrapolation drifts, which is one reason the coast window is short
  (8 frames ≈ 0.32 s at 25 fps).
- **Coasting is an estimate, and it is labelled as one.** A `"predicted"`
  point says where the ball *probably* was. Do not fit a curve to it.
- **Track boundaries are the tracker's, not a segmenter's.** This module
  answers "where did the ball go"; deciding where one delivery ends and the
  next begins, and whether a gap was a brief occlusion or a genuinely lost
  ball, is a separate concern that is **not** in this branch.
- **No pitch geometry.** Positions are raw pixels. Nothing here knows where
  the pitch is, so nothing here can report a speed in metres per second or a
  release point on the pitch. That is Part 2.
- **Coordinates are pixels, not metres.** A speed in px/frame is not a speed.
- **No output rendering.** There is no overlay, no clip writing and no
  analytics.
- **The detector's size filter is tuned for broadcast footage.** A ball
  rendered very large (close-up replay) or very small (ultra-wide stadium
  shot) can fall outside `CRICKET_BALL_MIN_DIAG`/`CRICKET_BALL_MAX_DIAG`.
  Both are configurable.

---

## Layout

```
backend/
  app/
    core/
      constants.py    every value the module applies, stated once
      config.py       environment-driven configuration and filesystem layout
    schemas/
      tracking.py     TrajectoryPoint, Trajectory — the output contract
    tracking/
      detector.py     BallDetector
      kalman_tracker.py  BallTracker, KalmanBallTracker
      tracking_service.py  TrackingService, TrackingResult, Track
      errors.py       TrackingError and friends
      frames.py       the frame-validation rule, shared by detector and service
  tests/
    helpers.py        synthetic frames and stub models
    unit/             the unit tests
```
