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
| 4 — Frontend | `feature/frontend` | The React/TypeScript browser client for schema 2.1. |
| 5 — Backend integration | `feature/backend-integration` | The FastAPI server, upload/status/result/clip routes, the real analysis pipeline and the frontend contract tests. |

**Current status:** all five parts are implemented on their own branches.
`feature/backend-integration` combines Parts 1–4 and implements the six-route
HTTP contract. It does not add delivery segmentation, bounce detection, metric
calibration or a server-side overlay renderer; those gaps are reported
explicitly in each result rather than filled with plausible-looking values.

## Layout

```
backend/
  main.py       # Uvicorn entry point
  app/
    api/        # FastAPI app, upload validation, jobs, routes, health
    pipeline/   # tracking → clips → pitch → shot → analytics orchestration
    core/       # constants and runtime configuration
    schemas/    # typed records persisted by the tracking, pitch and shot passes
    analytics/  # pure calculations over tracking, pitch and shot records
    pitch/      # pitch corners, calibration, homography and the pitch detectors
    shots/      # shot checkpoint config, preprocessing, architecture and inference
    tracking/   # the ball detector, the Kalman tracker and the tracking service
  tests/        # unit tests and the Part 5 integration suite
frontend/       # React/TypeScript client (Part 4)
docs/           # per-module and API documentation
```

`backend/` is the import root: `app.*` and `tests.*` are imported from there.

## Requirements

- Python 3.10 or newer
- The dependencies in [`requirements.txt`](requirements.txt)
- Node 18 or newer for the frontend

```bash
pip install -r requirements.txt
```

## Running the API

From the repository root:

```bash
python backend/main.py
```

The API listens on `http://127.0.0.1:8000` by default. `CRICKET_HOST` and
`CRICKET_PORT` change the address; `CRICKET_DATA_DIR` moves generated files.
OpenAPI documentation is at `/docs`. The endpoints, schemas, error statuses and
processing limitations are documented in [`docs/API.md`](docs/API.md).

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

The client defaults to `http://127.0.0.1:8000`; set `VITE_API_URL` at build time
to use another API origin. The frontend's development origin is included in the
default CORS allow-list.

## Model weights

No model weights are tracked in git. The ball detector (`ball_detector.pt`) is
required; the two pitch YOLO checkpoints (`pitch_seg_yolo11s.pt`,
`pitch_detect.pt`) and the 10-class shot classifier (`shot_classifier.ckpt`)
degrade to explicit unavailable states when absent. Place them in
`backend/models/` or set `CRICKET_MODELS_DIR` / the per-model overrides. See
[`docs/TRACKING.md`](docs/TRACKING.md), [`docs/PITCH.md`](docs/PITCH.md),
[`docs/SHOTS.md`](docs/SHOTS.md) and [`docs/API.md`](docs/API.md). Nothing is
downloaded automatically.

## Running the tests

```bash
pytest backend
python -m ruff check backend

cd frontend
npm test
npm run typecheck
npm run build
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
- [`docs/API.md`](docs/API.md) — the six HTTP routes, request/response schemas,
  model configuration, job lifecycle, cleanup guarantees and the integration
  limitations (candidate delivery boundaries, unavailable bounce detection,
  uncalibrated metrics and client-rendered overlays).

## Licence

The code in this repository is provided under the terms that accompany the
upstream CricketVision project. Model weights are **not** redistributed here;
see the tracking documentation for how to obtain them.
