# CricketVision

Cricket video analysis: detect the ball, follow it, and turn the footage into
something a coach can read.

This repository is being built in parts. Each part is a self-contained,
independently testable slice of the system, developed on its own branch.

| Part | Branch | Contents |
|---|---|---|
| 1 — Ball tracking | `feature/ball-tracking` | Ball detection and Kalman-based tracking of the ball across frames. |
| 2 — Pitch detection | `feature/pitch-detection` | Pitch corners, calibration and validity, and pixel → pitch coordinates. |
| 3+ | *(planned)* | Delivery analytics, shot classification, the API and the frontend. |

**Current status:** the repository scaffold, the Python backend package layout,
and Parts 1 and 2 — ball tracking (`feature/ball-tracking`) and pitch detection
with calibration (`feature/pitch-detection`), each on its own branch. Shot
classification, delivery analytics, the HTTP API and the frontend are **not**
in this repository yet.

## Layout

```
backend/
  app/
    core/       # constants and runtime configuration
    schemas/    # typed records persisted by the tracking and pitch passes
    pitch/      # pitch corners, calibration, homography and the pitch detectors
    tracking/   # the ball detector, the Kalman tracker and the tracking service
  tests/        # unit tests
docs/           # per-module documentation
```

`backend/` is the import root: `app.*` and `tests.*` are imported from there.

## Requirements

- Python 3.10 or newer
- The dependencies in [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

## Model weights

The ball detector runs a YOLO11 checkpoint that is **not tracked in git**. Part
2 runs two further YOLO checkpoints — a detection model and a segmentation
model — which are **not tracked in git** either. See
[`docs/TRACKING.md`](docs/TRACKING.md) and [`docs/PITCH.md`](docs/PITCH.md) for
how to supply the weights and what happens when they are missing.

## Running the tests

```bash
pytest backend
```

## Documentation

- [`docs/TRACKING.md`](docs/TRACKING.md) — the ball-tracking module: setup,
  configuration, API, input/output formats and known limitations.
- [`docs/PITCH.md`](docs/PITCH.md) — the pitch module: the corner order,
  calibration status and provenance, the two coordinate systems, model
  configuration, input/output formats and known limitations.

## Licence

The code in this repository is provided under the terms that accompany the
upstream CricketVision project. Model weights are **not** redistributed here;
see the tracking documentation for how to obtain them.
