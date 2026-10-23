# Pitch detection and calibration (Part 2)

Locate the pitch in a frame, and turn source-frame pixels into coordinates on
that pitch.

The module does two things, and it keeps them apart:

1. **Calibration.** Four corners of the pitch, in source-frame pixels, plus a
   record of where those corners came from and how far they can be trusted.
2. **Conversion.** A homography from those pixels into a documented pitch
   coordinate system — and a refusal when the calibration does not justify
   the system being asked for.

Everything in between follows from one rule: **a number carries the standing
of the thing that produced it.** A quad guessed from a bounding box is
labelled a guess everywhere it travels; a quad measured on the frame may be a
validated calibration; a conversion to metres that nobody measured is
refused rather than reported.

---

## What it does

| Component | File | Purpose |
|---|---|---|
| `PitchCorners` / `order_corners` | `backend/app/pitch/corners.py` | The four corners in source pixels, in one canonical order, with the geometric checks. |
| `PitchGeometry` | `backend/app/pitch/geometry.py` | The pitch's dimensions (Law 6 by default), their provenance, and the metre target quad. |
| `PitchCalibration` | `backend/app/pitch/calibration.py` | One frame's calibration: corners, provenance, method, **status**, reasons, confidence. |
| `PitchProjection` / `build_projection` | `backend/app/pitch/homography.py` | The pixel ↔ pitch mapping the calibration justifies, in one of two spaces. |
| `PitchBoxDetector` | `backend/app/pitch/detector.py` | A YOLO detection model over the pitch-shaped region. Corners derived from the box are **approximate**. |
| `PitchSegmentationDetector` | `backend/app/pitch/detector.py` | A YOLO segmentation model over the pitch. Mask corners are a **measurement**. |
| `project_trajectory` | `backend/app/pitch/adapter.py` | A Part 1 ball trajectory re-expressed in pitch coordinates. |
| `PitchTrajectory` | `backend/app/schemas/pitch.py` | The document that adapter produces. |

### The calibration record, and the first thing to read

`status` is the first field of `PitchCalibration.as_dict()`, because it is the
field a consumer must not skip.

| `status` | Meaning | What a client may do |
|---|---|---|
| `valid` | The corners were measured on this frame: a human marked them (`manual`) or a model reported the pitch outline (`segmentation`, `keypoints`). | Draw it as a validated calibration. |
| `approximate` | The quad says where the pitch is, not where its corners are: it was guessed from a detection box (`box_estimate`), or assumed from a template (`template`). | Draw it, **labelled as an estimate**. Never as a validated calibration. |
| `invalid` | The geometry was refused. `reasons` says which check failed. | Use nothing. There is no pitch. |

A box-derived calibration **cannot** be `valid`, however confident the model
was. That is enforced by the method label, not by the caller remembering to
say so: `APPROXIMATE_METHODS = {box_estimate, template}`.

`provenance` answers *where the corners came from* — `detected`,
`manually_supplied`, or `assumed`. `method` is finer-grained, because two
detected sources are not equally trustworthy. When `provenance` is `assumed`,
the reason is `assumed_corners`; when the method is `box_estimate`, it is
`approximate_corners`. Both are `approximate`.

Confidence is recorded **only** for `keypoints` and `segmentation`, and only
when the status is not `invalid`. A detection box's score is evidence about
the box, not about the four points derived from it, so a box-derived
calibration carries `confidence: null` rather than a borrowed number.

### The canonical corner order

Every quad in the project is four points in one order, and the order is part
of the contract rather than an implementation detail:

| Position | Meaning |
|---|---|
| `top_left` | nearest the top-left of the image |
| `top_right` | nearest the top-right |
| `bottom_right` | nearest the bottom-right |
| `bottom_left` | nearest the bottom-left |

The order is clockwise in image space. It is decided by the extremes of two
linear functionals, so it works for a rotated quad:

```
top_left     = min(x + y)
top_right    = max(x - y)
bottom_right = max(x + y)
bottom_left  = min(x - y)
```

For a convex quad those four extremes are the four vertices. When they are
not — a triangle, a line, a bow-tie, or a quad whose extremes collide — the
corners are **refused** with `CalibrationError`. A quad that cannot be
ordered has no unambiguous pitch coordinate system, and inventing one would
be worse than saying so.

The homography uses this order to decide which corner of the pitch system
each pixel corner lands on. A quad in a different order would silently rotate
or mirror the pitch coordinates.

### The two coordinate systems

Both spaces return **`(across, along)`** and both have units, spelled out in
the record. The direction of the axes depends on `PitchGeometry.long_axis`.

| | `normalized` | `metres` |
|---|---|---|
| Units (`units`) | `pitch_fraction` | `metre` |
| Available | Always, for any usable quad | Only when the calibration justifies it (below) |
| `across` | 0.0 at the left edge of the quad, 1.0 at the right (a vertical pitch) | metres from the middle of the strip: `-width/2 … +width/2`, i.e. `-1.32 … +1.32` |
| `along` | 0.0 at the **far** end of the quad, 1.0 at the near end | 0.0 at the **far** end, `length` `= 20.12` at the near end |
| `long_axis = vertical` | `along` runs top → bottom of the image | same |
| `long_axis = horizontal` | `along` runs left → right of the image | same |

`normalized` says *"this is where the ball was as a fraction of the pitch we
can see"*. It is resolution-independent, camera-independent, and — unlike
metres — it does not pretend the model measured the width of anything.

`metres` says *"this is where the ball was on a 20.12 m × 2.64 m strip"*. The
dimensions are real (Law 6), but whether **the quad someone drew spans exactly
the bowling creases** is a judgement about that video. That judgement is what
`PitchGeometry.provenance` records.

Which end is the batter's is deliberately **not** decided. The same quad read
from either end gives the same numbers with the batter at the opposite end of
them, and settling it needs the ball's own travel.

### Which numbers are metres, and when

`metric_gate(calibration)` returns `(True, "ok")` only when one of these
holds:

- the corners were **measured on this frame by a human**
  (`provenance = manually_supplied`), or
- the pitch's **dimensions** were measured rather than assumed
  (`geometry.provenance = manually_supplied`).

Otherwise: `(False, "metric_geometry_not_justified")`, and
`build_projection(..., space=METRES)` returns `None` with that reason. A
detected quad plus assumed Law 6 dimensions produces a confident-looking
metre value that nobody checked, so it is refused rather than published.

The module knows exactly **one** set of pitch dimensions — 20.12 m between
the creases, 2.64 m between the bowling creases — and it labels them
`assumed`. They are never presented as something the camera told us.

---

## Requirements

- **Python 3.10 or newer.** (Developed and tested on 3.14.4; the module uses
  `X | Y` type syntax and `from __future__ import annotations`.)
- The dependencies in [`requirements.txt`](../requirements.txt):

  | Package | Used for |
  |---|---|
  | `numpy` | Frame validation, homography arithmetic, mask geometry. |
  | `opencv-python` | The perspective transform, convex hulls, polygon simplification. |
  | `ultralytics` | The YOLO runtime the two detectors are loaded from. Imported lazily. |
  | `torch` | Ultralytics' inference backend, and the CUDA availability probe. |
  | `python-dotenv` | Optional. Reads a `.env` from the project root. Never overrides a real environment variable. |
  | `pytest` | The test suite. |

  Everything except the detector needs only `numpy` and `opencv-python`: the
  geometry, the calibration record and the homography are importable and
  testable without a model or a GPU.

```bash
pip install -r requirements.txt
```

---

## Model weights

Both supplied checkpoints are tracked in git under `backend/models/`. Other
`*.pt` files remain ignored, so a local experiment cannot be committed by
accident. The loaders never download weights.

| File | Size | Task | MD5 |
|---|---|---|---|
| `pitch_detect.pt` | 22,517,987 B | YOLO **detection** over the pitch-shaped region | `fe980582cb5446062db5e65c238a8afb` |
| `pitch_seg_yolo11s.pt` | 20,521,757 B | YOLO **segmentation** over the pitch outline | `9235c4b3ed956ef56d4c80e2f6bb61a8` |

These are the copies this module was exercised against. They are **not**
shipped with the repository, and no accuracy figure is claimed for them here:
see [Limitations](#limitations).

Requirements on a checkpoint:

- Loadable with `ultralytics.YOLO(...)`, exported to a `.pt` file.
- The pitch must be **class index 0**. Any other class is refused with
  `wrong_class` rather than mis-read as a pitch.
- The segmentation checkpoint must be a **segmentation** model: the corners
  are simplified out of its mask boundary. A detection checkpoint used as a
  segmentation model yields no masks and is reported as `no_pitch_mask`.

Unlike the Part 1 ball detector, **the frame is not resized before
inference.** It goes to the model at its original resolution and ultralytics
letterboxes it to `imgsz` itself. Pre-resizing the frame is what squashed
frames and broke the aspect ratio in an earlier version of this project. The
corners that come back are therefore in the coordinate system of the frame
that was passed in.

### Pointing the module at the weights

| Variable | Default | Meaning |
|---|---|---|
| `CRICKET_MODELS_DIR` | `backend/models` | Directory holding the weights. |
| `CRICKET_PITCH_BOX_MODEL` | `$CRICKET_MODELS_DIR/pitch_detect.pt` | The box detector's file, overriding the directory. |
| `CRICKET_PITCH_SEG_MODEL` | `$CRICKET_MODELS_DIR/pitch_seg_yolo11s.pt` | The segmentation detector's file, overriding the directory. |

A bare file name is resolved against the project directory, so the module
finds the same files however it was started. These paths are resolved **at
import**; set the environment before importing `app.pitch.config`.

The resolvers name every path they tried:

```python
from app.pitch.config import resolve_pitch_segmentation_model_path

resolve_pitch_segmentation_model_path()   # returns a Path, or FileNotFoundError
```

**The module never downloads anything.** `ultralytics.YOLO("name.pt")`
fetches a checkpoint from the internet when the file is not present, so both
detectors check that the weights exist as a **file** before constructing a
model, and pass an absolute path when they do. A missing checkpoint is an
explicit `unavailable` state, never a surprise download and never a crash.

### Verifying a copy

```bash
# Linux / macOS
md5sum backend/models/pitch_detect.pt backend/models/pitch_seg_yolo11s.pt

# Windows
certutil -hashfile backend\models\pitch_detect.pt MD5
```

A checkpoint loaded with the wrong task loads without error and then reports
`no_pitch_mask`. Verify the checksum after copying it.

### When the weights are missing or broken

Two different failures, both explicit and both counted:

| Situation | `status` | `reason` |
|---|---|---|
| The file is not on disk | `unavailable` | `weights not found at <path>` |
| The file exists but cannot be loaded (truncated, not a checkpoint) | `unavailable` | the load error, naming the path and the parser's complaint |

`detector.load_error` exposes the second case, and `stats()["model_found"]`
tells the two apart (`False` for missing, `True` for present but unusable).
The load failure is cached, so a bad checkpoint costs one attempt rather than
one per frame.

`unavailable` is deliberately **not** `no_detection`. A detector with no
usable weights cannot find a pitch in *any* frame, and reporting it as "no
pitch in this frame" would record a missing model as an empty analysis.

---

## Configuration

`PitchConfig` bundles the settings. All of it is read from the environment
once, at import; `dataclasses.replace` builds a modified copy.

| Setting | Env variable | Default | Used by |
|---|---|---|---|
| `box_model_path` | `CRICKET_PITCH_BOX_MODEL` | `backend/models/pitch_detect.pt` | `PitchBoxDetector` |
| `segmentation_model_path` | `CRICKET_PITCH_SEG_MODEL` | `backend/models/pitch_seg_yolo11s.pt` | `PitchSegmentationDetector` |
| `conf_threshold` | `CRICKET_PITCH_CONF` | `0.25` | Both detectors |
| `imgsz` | `CRICKET_PITCH_IMGSZ` | `640` | Both detectors |
| `device` | `CRICKET_PITCH_DEVICE` | `auto` | Both detectors |
| `min_area_fraction` | `CRICKET_PITCH_MIN_AREA_FRACTION` | `0.01` | Segmentation mask gate |
| `max_area_fraction` | `CRICKET_PITCH_MAX_AREA_FRACTION` | `0.25` | Segmentation mask gate |
| `min_aspect_ratio` | `CRICKET_PITCH_MIN_ASPECT_RATIO` | `1.0` | Segmentation mask gate |
| `max_aspect_ratio` | `CRICKET_PITCH_MAX_ASPECT_RATIO` | `6.0` | Segmentation mask gate |

`device = "auto"` picks CUDA when `torch` reports it available, otherwise
CPU. An explicit `cuda` that cannot be honoured falls back to CPU with a
warning, because a detector that refuses to run is worse than a slow one.

The detectors take the same values as keyword arguments:

```python
from app.pitch.config import PITCH_CONFIG
from app.pitch.detector import PitchSegmentationDetector

detector = PitchSegmentationDetector(
    PITCH_CONFIG.segmentation_model_path,
    conf=PITCH_CONFIG.conf_threshold,
    imgsz=PITCH_CONFIG.imgsz,
    device=PITCH_CONFIG.device,
    min_area_fraction=PITCH_CONFIG.min_area_fraction,
    max_area_fraction=PITCH_CONFIG.max_area_fraction,
    min_aspect_ratio=PITCH_CONFIG.min_aspect_ratio,
    max_aspect_ratio=PITCH_CONFIG.max_aspect_ratio,
)
```

The four shape bounds apply to the **segmentation** detector's mask gate. The
box detector's size gate is fixed by its own constants
(`MIN_BOX_AREA_FRACTION`, `MAX_QUAD_AREA_FRACTION`), because a box and a mask
cover different shares of the frame.

### Two masks, one pitch

When a model returns several candidates, the highest confidence wins. Two
masks whose confidences are within `DETECTOR_CONF_TIE_BAND` (0.05) are a
**tie**, and a tie is broken by consistency with the last accepted estimate —
the pitch does not teleport between adjacent frames under a split screen. A
real confidence gap is never overruled by continuity: confidence is the
model's evidence, position is only ours. Every refused candidate is tallied
in `rejections`.

---

## How to call it

### Calibrate from four corners you know

```python
import cv2

from app.pitch.calibration import manual_calibration
from app.pitch.corners import order_corners
from app.pitch.homography import PitchSpace, build_projection, pixel_to_pitch

frame = cv2.imread("frame.png")
height, width = frame.shape[:2]

# Spoken in any order; the canonical order is applied here.
corners = order_corners([(412, 208), (768, 206), (980, 806), (240, 812)])

calibration = manual_calibration(
    corners,
    image_size=(width, height),
    frame_number=0,
)
print(calibration.status)          # valid  (a human measured this frame)
print(calibration.as_dict())

projection, reason = build_projection(calibration, space=PitchSpace.NORMALIZED)
if projection is None:
    raise SystemExit(f"cannot project: {reason}")

print(pixel_to_pitch(560, 430, projection))   # (across, along), pitch fractions
```

`order_corners` raises `CalibrationError` for anything that is not four
distinct, finite points forming a convex quad. `manual_calibration` does not
raise for a well-formed but implausible quad — it returns `status=invalid`
with reasons, so the record can be persisted and shown.

### Detect the pitch in a frame

```python
from app.pitch.config import PITCH_CONFIG
from app.pitch.detector import (
    DetectionStatus,
    PitchSegmentationDetector,
    detection_to_calibration,
)

detector = PitchSegmentationDetector(
    PITCH_CONFIG.segmentation_model_path,
    conf=PITCH_CONFIG.conf_threshold,
)

detection = detector.detect(frame, frame_number=0)

if detection.status is DetectionStatus.UNAVAILABLE:
    # No usable weights. This is NOT "no pitch in this frame".
    print(detection.reason, detector.load_error)
elif detection.status is DetectionStatus.NO_DETECTION:
    print("the model ran and found no pitch:", detection.reason)
elif detection.status is DetectionStatus.ERROR:
    print("the attempt failed:", detection.reason)
else:
    print(detection.corners.as_list())
    print(detection.confidence, detection.quad_method)
    calibration = detection_to_calibration(detection)   # None if nothing found
    print(calibration.status)     # valid for segmentation, approximate for a box
```

Both detectors share that shape. `detection.as_dict()` is the persisted form;
`detector.stats()` is the tally, and
`frames_seen == detections + no_detections + unavailable + errors` always
holds. No frame is ever dropped silently.

### Turn pixels into pitch coordinates

```python
from app.pitch.homography import PitchSpace, build_projection, pixel_to_pitch, pitch_to_pixel

projection, reason = build_projection(calibration, space=PitchSpace.METRES)
if projection is None:
    print("metres refused:", reason)      # e.g. metric_geometry_not_justified
else:
    print(projection.units)               # "metre"
    print(projection.axes)                # ("across", "along")
    print(pixel_to_pitch(560, 430, projection))
    print(pitch_to_pixel(0.0, 10.06, projection))    # back to the source pixel
    print(projection.homography_rows())              # 3x3, for the record
```

`pixel_to_pitch` is **pure**: the answer depends on its arguments and nothing
else. It returns `None` when the pixel has no finite image under the
transform — a point on the horizon, for instance — rather than a number with
a huge magnitude. `pitch_to_pixel` is the inverse direction, so a client can
draw a pitch-space marker back onto the picture.

`projection.contains(x, y)` answers "was this pixel inside the pitch quad",
which is what sets `inside_pitch` on a projected point. A ball wide of the
strip is information, never an error.

### Put a Part 1 trajectory on the pitch

```python
from app.pitch.adapter import project_trajectory
from app.pitch.homography import PitchSpace

pitch_trajectory = project_trajectory(
    trajectory,                 # a Part 1 Trajectory, or a list of its points
    calibration,
    space=PitchSpace.NORMALIZED,
)

print(pitch_trajectory.space, pitch_trajectory.units)
print(pitch_trajectory.point_count, pitch_trajectory.skipped)
print(pitch_trajectory.gaps, pitch_trajectory.continuity)
print(pitch_trajectory.as_dict())
```

The adapter is **structural, not dependent**: it imports nothing from
`app.tracking`. It accepts anything with a `points` sequence — or a sequence
of points — where each point has `frame`, `x`, `y` and `source`. That is what
lets Part 2 be tested on its own branch and merge without an import-order
puzzle.

Every field a consumer might act on is preserved: `frame`, the source pixel
`x`/`y`, `source` (`"detected"` stays a measurement, `"predicted"` stays an
extrapolation), `confidence` (never invented), and `track_id`. It adds
exactly two numbers: `across` and `along`.

It never invents a point, fills a gap, or clamps a point into the pitch
square. A point the transform cannot map is counted in `skipped` and named in
`skipped_frames`, and the frames it belonged to appear in `gaps`.

Asking for metres from a calibration that does not justify them raises
`CalibrationError` naming the reason — it does not quietly hand back
fractions that look like metres in a different field:

```
CalibrationError: cannot project the trajectory into metres coordinates:
metric_geometry_not_justified
```

### Using the pieces on their own

| Call | Answers |
|---|---|
| `order_corners(points)` | The canonical order, or `CalibrationError`. |
| `validate_corners(corners, image_size)` | `(ok, reason)` for the quad gate. |
| `build_calibration(corners, ..., provenance, method)` | A record with the status derived from the labels. |
| `build_homography(corners, target)` | The 3×3 matrix, or `None` when no usable transform exists. |
| `validate_homography(matrix, src, dst)` | Whether the matrix maps the corners it claims to. |
| `metric_gate(calibration)` | `(ok, reason)` for the metre conversion. |
| `validate_mask(points, image_size, ...)` | `(ok, reason)` for the mask shape gate. |
| `quad_from_mask(points)` | Four corners plus the simplification used, or `(None, reason)`. |
| `validate_frame(frame)` | `(width, height)`, or `InvalidFrameError`. |
| `resolve_device("auto")` | The device inference will actually use. |

---

## Input and output formats

### The calibration document

`PitchCalibration.as_dict()`. `status` is deliberately the first key:

```json
{
  "status": "valid",
  "is_valid": true,
  "reasons": [],
  "frame_number": 0,
  "image_size": [1920, 1080],
  "corners": {
    "top_left": [412.0, 208.0],
    "top_right": [768.0, 206.0],
    "bottom_right": [980.0, 806.0],
    "bottom_left": [240.0, 812.0]
  },
  "provenance": "detected",
  "method": "segmentation",
  "confidence": 0.9123,
  "geometry": {
    "length_m": 20.12,
    "width_m": 2.64,
    "long_axis": "vertical",
    "provenance": "assumed",
    "justification": "Law 6 dimensions assumed: 20.12 m between the two creases and 2.64 m between the bowling creases. The quad is assumed to span exactly that strip; this has not been verified against this video."
  }
}
```

Corners are in **source-frame pixels**, `x` to the right and `y` downwards,
`(0, 0)` at the top-left of the frame. `image_size` is `[width, height]`, and
the corners belong to that exact frame — a frame is never resized without
rescaling the answer by the same factor.

An approximate record differs in exactly the fields that matter:

```json
{
  "status": "approximate",
  "is_valid": false,
  "reasons": ["approximate_corners"],
  "provenance": "detected",
  "method": "box_estimate",
  "confidence": null
}
```

### The detection outcome

`PitchDetection.as_dict()`:

```json
{
  "status": "no_detection",
  "frame_number": 0,
  "image_size": [1920, 1080],
  "method": null,
  "corners": null,
  "confidence": null,
  "reason": "no_pitch_mask",
  "quad_method": null,
  "rejections": {"mask_area_too_small": 1, "wrong_class": 2}
}
```

`status` is one of `ok`, `no_detection`, `unavailable`, `error`.
`quad_method` says how four corners were derived from a detection:
`box_estimate`, `direct_hull`, `approx_polydp`, or `hull_reduce`.

Refusal reasons are facts worth recording, and each is distinguishable:

| Stage | Reasons |
|---|---|
| Frame | `InvalidFrameError` (not an ndarray, not 3-dimensional, not 3 channels, no pixels, not `uint8`) |
| Box | `no_pitch_box`, `wrong_class`, `pitch_box_too_small`, `pitch_box_too_large`, `below_confidence` |
| Mask shape | `mask_outside_frame`, `mask_area_too_small`, `mask_area_too_large`, `mask_aspect_too_square`, `mask_aspect_too_elongated`, `unknown_frame_size` |
| Mask → quad | `too_few_vertices`, `degenerate_hull`, `hull_too_complex`, `quad_unavailable` |
| Corner gate | `duplicate_corners`, `non_finite_corner`, `non_convex_corners`, `zero_area`, `corners_outside_frame`, `pitch_quad_too_small`, `pitch_quad_too_large`, `unknown_frame_size` |
| Projection | `invalid_calibration`, `metric_geometry_not_justified`, `degenerate_homography` |

`mask_aspect_too_square` is unreachable with the default
`min_aspect_ratio = 1.0`; it exists for a caller who raises that bound. A
square mask is **not** rejected by default: an elevated camera can legitimately
see the pitch as nearly square in the frame.

### The pitch trajectory document

`PitchTrajectory.as_dict()`:

```json
{
  "space": "normalized",
  "units": "pitch_fraction",
  "points": [
    {
      "frame": 12,
      "x": 640,
      "y": 512,
      "across": 0.482113,
      "along": 0.731904,
      "space": "normalized",
      "units": "pitch_fraction",
      "source": "detected",
      "confidence": 0.87,
      "track_id": 1,
      "inside_pitch": true
    }
  ],
  "point_count": 1,
  "detected_count": 1,
  "predicted_count": 0,
  "inside_count": 1,
  "continuity": "continuous",
  "gaps": [],
  "skipped": 0,
  "skipped_frames": [],
  "calibration": { "...": "the calibration document above" }
}
```

`units` is always present, because a number in `pitch_fraction` and a number
in `metre` are different kinds of claim. `source` is inherited from Part 1,
so a Kalman extrapolation can never be mistaken for a measurement after the
conversion. `gaps` uses the same computation as the tracking document: a
hole is visible rather than bridged.

### Errors

| Exception | Raised for |
|---|---|
| `CalibrationError` | Corners that cannot be ordered; unusable geometry; a projection the calibration does not justify. |
| `InvalidFrameError` | A frame that is not a usable BGR image. That is a caller's mistake, not a fact about the world, so it raises rather than returning a status. |
| `FileNotFoundError` | `resolve_pitch_*_model_path()` when no checkpoint is on disk. The detectors themselves never raise for this; they return `unavailable`. |
| `TypeError` | The adapter, when a point is not shaped like a trajectory point. Guessing at the missing field would hide a broken integration. |

---

## Running the tests

From the project root:

```bash
python -m pytest backend -q
python -m ruff check backend
```

On `feature/pitch-detection` on its own:

```
158 passed, 2 skipped, 69 subtests passed
All checks passed!
```

The two skips are the **real-Part-1-schema** compatibility tests: they import
`app.schemas.tracking`, which lives on `feature/ball-tracking`. They skip
rather than fail there and run for real once the branches are together.

A focused run:

```bash
python -m pytest backend/tests/unit/test_pitch_homography.py -q
```

| Test module | Covers |
|---|---|
| `test_pitch_corners.py` | Canonical order under shuffling, every refusal, the size/position gate, box-derived corners. |
| `test_pitch_calibration.py` | `valid` / `approximate` / `invalid` per method and provenance, confidence rules, the persisted form, geometry. |
| `test_pitch_homography.py` | Pixel → pitch against hand-computed values, transform direction, round-trip, the projective diagonal property, the metric gate, refusals. |
| `test_pitch_detector.py` | Frame validation, missing **and corrupt** weights, box and mask extraction, every rejection reason, selection tie-breaks, stats, configuration. |
| `test_pitch_adapter.py` | Part 1 field preservation, no invented points, named gaps, refusals, and the real Part 1 dataclasses when importable. |

The detector tests inject a stub model, so the real extraction, validation
and selection code runs with no checkpoints on disk. Metric assertions are
pinned to 0.1 mm rather than micro-precision, because
`cv2.getPerspectiveTransform` requires single-precision (`CV_32F`)
correspondences in OpenCV 5; the matrix and all later arithmetic are float64,
which leaves about a micrometre of noise at pitch scale.

`ruff format` is not a gate for this project: the committed Part 1 code is
not format-clean either, and the documented check is `ruff check`.

---

## Limitations

These are properties of the method, not bugs to be fixed later:

- **No accuracy figure is claimed.** There is no cricket video in this
  checkout, so neither detector has been evaluated against real footage here.
  What *is* verified: both checkpoints load and run, and they return
  `no_detection` on a synthetic non-pitch frame. Precision and recall on
  broadcast cricket are unknown.
- **The box detector is approximate by construction.** Its four corners are a
  trapezoid derived from a rectangle with a fixed inset (`BOX_TOP_INSET`).
  They are a plausible shape, not the pitch's corners, and can never be a
  validated calibration.
- **One homography per calibration.** The calibration describes one frame and
  assumes a static camera and a planar pitch. Camera pan, zoom, focus pulls
  and lens distortion all invalidate it; the module does not correct for any
  of them and does not detect that they happened.
- **The metric conversion is a judgement, not a measurement.** Unless a human
  supplied the corners or the dimensions, metres are refused. Even when they
  are supplied, the quad is assumed to span exactly the bowling creases and
  the crease-to-crease length.
- **Confidence is the model's instance score**, not a calibration error bar.
  There is no covariance, no per-corner uncertainty, and no error propagation
  into the trajectory.
- **Mask corners can be wrong on partial or occluded pitches.** A mask that
  stops at the batter reports the visible outline; the simplification then
  produces four corners as if that outline were the whole pitch. The geometric
  plausibility gate catches absurd results, not subtly wrong ones.
- **No temporal smoothing of the calibration.** Each `detect()` call is
  independent except for the tie-break; nothing accumulates evidence across
  frames or detects that the calibration jumped.
- **Which end is the batter's is not decided.** `along` runs from the far end
  of the image, whichever end that physically is.
- **The class must be index 0.** A checkpoint that puts the pitch anywhere
  else is refused, which is a safe but silent-until-inspected failure.
- **The Part 2 branch does not contain Part 1.** The adapter is verified
  structurally here; the end-to-end run happens after the branches merge.

---

## What is not implemented yet

- **A keypoint detector.** `CornerMethod.KEYPOINTS` exists and is treated as
  a measurement, but there is no keypoint model and no `PitchDetector`
  implementation that would use one.
- **Calibration over a video.** Callers drive `detect()` frame by frame; there
  is no service that picks a calibration frame, refreshes it, or reports
  when it drifted.
- **Real-footage evaluation.** No numbers, no comparison to a hand-labelled
  ground truth, no footage in the repository.
- **Provenance of the checkpoints.** The files are included in this repository,
  but their licence and provenance are not recorded here. Confirm both before
  redistributing them outside this project.

---

## Layout

```
backend/
  app/
    pitch/
      __init__.py       module map; re-exports nothing
      constants.py      dimensions, sanity limits, detector defaults
      errors.py         CalibrationError, InvalidFrameError
      corners.py        PitchCorners, order_corners, corners_from_box, validate_corners
      geometry.py       PitchGeometry, LongAxis, GeometryProvenance
      calibration.py    PitchCalibration, its builders, statuses and reasons
      homography.py     build_projection, pixel_to_pitch, pitch_to_pixel, metric_gate
      mask_geometry.py  mask validation, mask boundary -> four corners
      config.py         weights locations, PitchConfig, resolvers
      detector.py       PitchDetector protocol, PitchBoxDetector, PitchSegmentationDetector
      adapter.py        a Part 1 trajectory in pitch coordinates
    schemas/
      pitch.py          PitchPoint, PitchTrajectory
  tests/
    pitch_helpers.py    stubs and synthetic frames shared by the pitch tests
    unit/
      test_pitch_corners.py
      test_pitch_calibration.py
      test_pitch_homography.py
      test_pitch_detector.py
      test_pitch_adapter.py
```
