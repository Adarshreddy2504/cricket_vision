# CricketVision

Cricket video analysis: detect the ball, follow it, and turn the footage into
something a coach can read.

This repository is being built in parts. Each part is a self-contained,
independently testable slice of the system, developed on its own branch.

| Part | Branch | Contents |
|---|---|---|
| 1 — Ball tracking | `feature/ball-tracking` | Ball detection and Kalman-based tracking of the ball across frames. |
| 2 — Pitch detection | `feature/pitch-detection` | Pitch corners, calibration and validity, and pixel → pitch coordinates. |
| 3 — Shot classification and analytics | `feature/shot-classification` | The shot classifier — checkpoint loading, the input contract, inference records — and pure analytics over trajectories, pitch geometry and shot labels. |
| 4+ | *(planned)* | The HTTP API and the frontend. |

**Current status:** the repository scaffold, the Python backend package layout,
and Parts 1–3 — ball tracking (`feature/ball-tracking`), pitch detection with
calibration (`feature/pitch-detection`), and shot classification with analytics
(`feature/shot-classification`), each on its own branch. The HTTP API and the
frontend are **not** in this repository yet.

## Layout

```
backend/
  app/
    core/       # constants and runtime configuration
    schemas/    # typed records persisted by the tracking, pitch and shot passes
    analytics/  # pure calculations over tracking, pitch and shot records
    pitch/      # pitch corners, calibration, homography and the pitch detectors
    shots/      # shot checkpoint config, preprocessing, architecture and inference
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

No model weights are tracked in git. The ball detector runs a YOLO11
checkpoint; Part 2 runs two further YOLO checkpoints — a detection model and a
segmentation model; Part 3 runs a 10-class shot classifier. See
[`docs/TRACKING.md`](docs/TRACKING.md), [`docs/PITCH.md`](docs/PITCH.md) and
[`docs/SHOTS.md`](docs/SHOTS.md) for how to supply the weights and what happens
when they are missing. Nothing is downloaded automatically.

## Running the tests

```bash
pytest backend
python -m ruff check backend
```

## Documentation

- [`docs/TRACKING.md`](docs/TRACKING.md) — the ball-tracking module: setup,
  configuration, API, input/output formats and known limitations.
- [`docs/PITCH.md`](docs/PITCH.md) — the pitch module: the corner order,
  calibration status and provenance, the two coordinate systems, model
  configuration, input/output formats and known limitations.
- [`docs/SHOTS.md`](docs/SHOTS.md) — the shot classifier and the analytics
  pass: the checkpoint and its class order, the exact input tensor, the
  classification record, configuration, the zero-versus-unavailable rules and
  known limitations.

## Licence

The code in this repository is provided under the terms that accompany the
upstream CricketVision project. Model weights are **not** redistributed here;
see the tracking documentation for how to obtain them.
