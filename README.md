# CricketVision

CricketVision is a cricket-video analysis application. It detects the ball,
tracks its flight, detects and calibrates the pitch, classifies the shot, and
shows the evidence in a React browser client. The API and browser contract are
schema **2.1**; the processing pipeline is version **1.1.0**.

This repository is the Part 5 integration branch,
`feature/backend-integration`. It combines the five independently tested parts
into one FastAPI + React application without rewriting the original branches.

| Part | Branch | Contribution |
|---|---|---|
| 1 - Ball tracking | `feature/ball-tracking` | YOLO ball detection, Kalman tracking, trajectory provenance, confidence filtering. |
| 2 - Pitch detection | `feature/pitch-detection` | Pitch segmentation/box detection, calibration states, homography, pixel-to-pitch projection. |
| 3 - Shot classification and analytics | `feature/shot-classification` | 10-class shot checkpoint contract and trajectory/pitch/shot analytics. |
| 4 - Frontend | `feature/frontend` | React/TypeScript client for schema 2.1. |
| 5 - Backend integration | `feature/backend-integration` | Upload validation, job lifecycle, the real pipeline, clip serving and the HTTP contract. |

The supplied model weights are included in `backend/models/`. The default
Docker setup is CPU-based and has been built, started and exercised in this
repository. A native CUDA path has also been exercised; Docker GPU execution
has **not** been verified on this host because NVIDIA Container Toolkit/CDI
discovery failed. See [GPU guide](#gpu-guide).

## What is implemented

| Area | What works |
|---|---|
| Ball tracking | YOLO detection, Kalman smoothing, low-confidence association, provenance. |
| Pitch | Segmentation/box detection, calibration validation, normalized projection. |
| Shot | Checkpoint loading, preprocessing, inference. |
| Analytics | Trajectory, pitch-geometry and shot summaries. |
| Jobs | Upload validation, async jobs, measured progress, cancellation, deadline, restart recovery, cleanup. |
| Output | Result/diagnostics serialization and clip serving. |
| Frontend | Upload, polling, cancellation, playback, overlays, filters, aggregate cards. |
| Packaging | CPU backend image, nginx frontend, Compose network, health checks, data volume, read-only models. |

Plumbing is covered end to end by automated tests. The three model boundaries
(ball, pitch, shot) are replaced by deterministic doubles there, so model
accuracy on real footage is not asserted by the suite.

Deliberately **not** implemented:

| Gap | What the result shows instead |
|---|---|
| Delivery segmentation | One confirmed track becomes a *candidate* delivery; the boundary flag is set. |
| Bounce detection | `bounce.detected: false`; every bounce field is `null`. |
| Metric calibration | Speed, line, length, swing and angles stay `null`. |
| Server-side overlay | The browser draws trajectory and pitch in SVG. |
| Variable frame rate | Frame index is authoritative; VFR is not modelled. |

These gaps are reported in the result or diagnostics, never filled with
plausible-looking values. See
[processing semantics](#processing-semantics-and-known-gaps).

## Architecture

The Docker deployment has two services and no database or queue:

```text
Browser
  |
  | http://localhost:8080
  v
frontend (nginx-unprivileged)
  |  /api/  ->  backend:8000
  v
backend (FastAPI + Uvicorn, one worker)
  |-- /data volume: uploads, status.json, result.json, diagnostics.json, clips
  |-- /models (read-only bind): the four checkpoints
  |-- job registry + daemon worker thread (process-local)
  `-- AnalysisPipeline: tracking -> candidate clips -> pitch -> shot -> analytics
```

- **One Uvicorn worker.** The job registry and model lifecycle are
  process-local; multiple workers would split status polling across processes.
- **Persistent data.** The named volume `cricketvision_cricket_data` keeps
  analysis state across container restarts.
- **Read-only models.** Weights are mounted from `./backend/models`, not baked
  into either image.

### Pipeline data flow

```text
Browser: Upload page
        │  POST /api/analyze   (multipart field: video)
        ▼
Upload validation
        │  extension · size ceiling · OpenCV probe
        ▼
JobRegistry
        │  daemon thread · analyses serialized
        ▼
AnalysisPipeline
        │
        ├─▶ probe video — width · height · fps · frame count
        │
        ├─▶ track every decoded frame
        │     BallDetector (640x360) → Kalman filter → confirmed tracks
        │                                              (>= 3 detections)
        │
        ├─▶ candidate windows — +/-10 frames · max 600
        │     └─▶ ClipExtractor → clips/delivery_NNN.webm
        │
        ├─▶ pitch detection — segmentation → box fallback
        │     └─▶ calibration (valid / approximate / invalid)
        │           └─▶ normalized projection (across, along)
        │
        ├─▶ shot classification — 30 frames at 224x224
        │
        └─▶ analytics
              └─▶ result.json + diagnostics.json
                        │
        ┌───────────────┴───────────────────┐
        ▼                                   ▼
        GET .../result                      GET .../clips/{name}
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
                        Browser: Results page
                        adaptResult → ClipStage · PitchMap · charts
```

| Where state lives | What it is |
|---|---|
| `GET /api/analysis/{id}/status` | In-memory `Job` document (13 keys), updated during the run. |
| `status.json` | Written on accept and at terminal state; used for restart recovery, not live polling. |
| `result.json` | Small browser contract. |
| `diagnostics.json` | Full evidence trail; never served to the browser. |

## Quick start with Docker

### Prerequisites

- Git.
- Docker Engine with Compose v2, or Docker Desktop. On Windows, make sure the
  Docker engine is running before using the commands below.
- About 4 GB of free disk space for the images (the CPU backend image measured
  2.89 GB on this host) plus space for uploads and generated results.
- Ports `8080` and `8000` free, unless you change them in `.env`.
- An NVIDIA GPU is optional. The default image is CPU-only.

### Fresh clone

```bash
git clone --branch feature/backend-integration https://github.com/Adarshreddy2504/cricket_vision.git
cd cricket_vision
```

The four checkpoints are part of the checkout. Confirm them against the
checksums in [Model weights](#model-weights) if the clone may have been
modified.

### Start the stack

From the repository root:

```bash
docker compose up --build -d
```

The first build downloads Python, PyTorch CPU wheels and Node packages. It can
take several minutes. Wait for both services to become healthy:

```bash
docker compose ps
```

Open:

- Application: <http://localhost:8080>
- API documentation: <http://localhost:8000/docs>
- API health: <http://localhost:8000/api/health>
- Through the frontend proxy: <http://localhost:8080/api/health>

`docker compose ps` should show `backend` and `frontend` as `healthy`. The
first request that loads a model can take longer than later requests.

### Logs and stop

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose down
```

`docker compose down` stops and removes the containers and network but keeps
the `cricketvision_cricket_data` volume. To remove generated analysis state as
well:

```bash
docker compose down -v
```

That deletes uploads, status/result documents, diagnostics and clips in the
volume. It does not delete the weights in `backend/models`.

### Change host ports

Copy the example environment file and edit it:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Then set, for example:

```dotenv
CRICKET_FRONTEND_PORT=8081
CRICKET_API_PORT=8001
```

Recreate the stack with `docker compose up -d`. The frontend still talks to the
backend through the internal Compose network, so the browser only needs the
frontend port. Direct API calls use `CRICKET_API_PORT`.

`.env` is optional for the defaults. `.env` is ignored by git; `.env.example`
is the safe template. Compose also reads `.env` for variable substitution.
Inside the backend container the data and model directories are fixed to
`/data` and `/models`; changing `CRICKET_DATA_DIR` or `CRICKET_MODELS_DIR` on
the host does not remap those container paths.

## Configuration

The Python app reads an optional `.env` from the repository root (native use),
or environment variables supplied by Compose (Docker use). A shell-exported
variable wins over a value in `.env`.

The main settings are:

| Setting | Default | Meaning |
|---|---|---|
| `CRICKET_HOST` / `CRICKET_PORT` | `127.0.0.1` / `8000` | Native API bind address. Docker binds `0.0.0.0:8000` inside the container. |
| `CRICKET_API_PORT` | `8000` | Host port mapped to the Docker backend. |
| `CRICKET_FRONTEND_PORT` | `8080` | Host port mapped to the Docker frontend. |
| `CRICKET_DATA_DIR` | `./data` | Native uploads, runs, status/result documents, diagnostics and clips. Docker uses `/data`. |
| `CRICKET_MODELS_DIR` | `./backend/models` | Directory holding the four weights. Docker uses `/models`. |
| `CRICKET_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Upload body ceiling. |
| `CRICKET_MAX_JOB_SECONDS` | `3600` | Processing safety net, checked between measured progress updates. |
| `CRICKET_TEMP_MAX_AGE_SECONDS` | `86400` | Abandoned upload fragments removed at startup. |
| `CRICKET_CLIP_MAX_FRAMES` | `600` | Hard cap on a candidate delivery clip window. |
| `CRICKET_CLIP_PRE_ROLL` / `POST_ROLL` | `10` / `10` | Context frames around a confirmed track. |
| `CRICKET_BALL_CONF` | `0.30` | Minimum confidence to start a track. |
| `CRICKET_BALL_ASSOC_CONF` | `0.10` | Lower floor used only to extend a live track through the Kalman gate. |
| `CRICKET_BALL_GATE_RADIUS` | `120` px | Maximum detection-to-prediction distance for association. |
| `CRICKET_BALL_MAX_MISSED` | `8` frames | Coast window before a track is declared lost. |
| `CRICKET_BALL_MIN_TRACK_DETECTIONS` | `3` | Real detections needed before a track is reported as a flight. |
| `CRICKET_BALL_RESIZE_W` / `H` | `640` / `360` | Inference resolution; answers are rescaled to source pixels. |
| `CRICKET_BALL_MIN_DIAG` / `MAX_DIAG` | `4` / `80` px | Plausible ball-box diagonal range. |
| `CRICKET_BALL_DEVICE`, `CRICKET_PITCH_DEVICE`, `CRICKET_SHOT_DEVICE` | `auto` | Use CUDA when the installed PyTorch build can see it, otherwise CPU. |
| `CRICKET_CORS_ORIGINS` | localhost/127.0.0.1 ports 5173, 3000, 8080 | Browser origins allowed to call the API. Compose uses the 8080 origins by default. |

The complete list, including pitch thresholds and checksum options, is in
[`.env.example`](.env.example). Paths are resolved against the repository root,
not the current working directory. Nothing downloads a model automatically.

## Model weights

The four supplied checkpoints are tracked in Git so a fresh clone can analyse
footage without a separate download. Other `*.pt`, `*.ckpt` and similar files
are ignored, so a local experiment is not committed by accident. The images
do **not** contain the weights: Compose mounts `./backend/models` read-only at
`/models`.

| File | Default location | Required? | Purpose | MD5 |
|---|---|---|---|---|
| `ball_detector.pt` | `backend/models/ball_detector.pt` | **Yes** | YOLO11 single-class ball detector. | `ADDE61B5409FCE5EC6583867D00C070C` |
| `pitch_seg_yolo11s.pt` | `backend/models/pitch_seg_yolo11s.pt` | No | YOLO11 segmentation model for the full pitch. | `9235C4B3ED956EF56D4C80E2F6BB61A8` |
| `pitch_detect.pt` | `backend/models/pitch_detect.pt` | No | YOLO11 pitch-box detector; approximate fallback. | `FE980582CB5446062DB5E65C238A8AFB` |
| `shot_classifier.ckpt` | `backend/models/shot_classifier.ckpt` | No | EfficientNet-B0 + temporal transformer, 10 shot classes. | `2974EFDCF449C8447BEF107D94633217` |

Sizes: `ball_detector.pt` 6,225,450 B; `pitch_detect.pt` 22,517,987 B;
`pitch_seg_yolo11s.pt` 20,521,757 B; `shot_classifier.ckpt` 72,010,842 B.

Verify a checkout:

```powershell
# Windows PowerShell
Get-FileHash backend\models\ball_detector.pt, backend\models\pitch_detect.pt, backend\models\pitch_seg_yolo11s.pt, backend\models\shot_classifier.ckpt -Algorithm MD5
```

```bash
# macOS / Linux
md5sum backend/models/ball_detector.pt backend/models/pitch_detect.pt backend/models/pitch_seg_yolo11s.pt backend/models/shot_classifier.ckpt
```

If the ball detector is absent or unloadable, an analysis fails loudly rather
than returning an empty result. Missing pitch or shot weights produce explicit
unavailable states and diagnostic flags, leaving those values `null`. Nothing
invents a label or a measurement.

Override the directory or individual files with:

```bash
export CRICKET_MODELS_DIR=/path/to/models
export CRICKET_BALL_MODEL=/path/to/ball_detector.pt
export CRICKET_PITCH_BOX_MODEL=/path/to/pitch_detect.pt
export CRICKET_PITCH_SEG_MODEL=/path/to/pitch_seg_yolo11s.pt
export CRICKET_SHOT_MODEL=/path/to/shot_classifier.ckpt
```

`CRICKET_SHOT_MODEL_MD5` is an optional checksum gate; it is not enforced
unless you set it. For Docker, change the read-only bind mount in
`compose.yaml` or point the environment variables into a different mounted
directory. See [`backend/models/README.md`](backend/models/README.md).

The files are included for this project. Their provenance and licence are not
recorded here; confirm both before redistributing them outside this project.

## Native installation (without Docker)

### Requirements

- Python 3.10 or newer. This branch was tested natively with Python 3.14.4.
- Node.js 18 or newer and npm. Tested with Node 24.14.1 and npm 11.11.0.
- OpenCV-compatible video codecs. Clips prefer WebM (VP8, then VP9) and fall
  back to MP4 (`mp4v`). A build without those encoders can still analyse a
  video, but clip encoding reports the failure explicitly.
- An NVIDIA GPU is optional. Device selection defaults to `auto`.

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, allow it for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install PyTorch first, then the rest

The PyTorch wheel decides whether the installation uses CPU or CUDA. Install
the matching wheel before `requirements.txt`; the pinned torch/torchvision
versions are then already satisfied.

CPU-only:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt
```

NVIDIA CUDA 12.6 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt
```

Confirm the device selection:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

CPU installs commonly print `2.13.0+cpu`, `None`, `False`. CUDA installs print
`2.13.0+cu126`, `12.6`, `True` when the driver and GPU are visible.

### Start the backend

From the repository root:

```bash
python backend/main.py
```

- API: <http://127.0.0.1:8000>
- OpenAPI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Health: <http://127.0.0.1:8000/api/health>

Equivalent Uvicorn invocation:

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

### Start the frontend

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

The Vite development server listens on <http://localhost:8080>. It calls
`http://127.0.0.1:8000` by default. Set `VITE_API_URL` at build/dev time to use
another API origin. The default CORS list includes the Vite origin.

The native frontend and the Docker frontend both default to port 8080. Run only
one at a time, or change the port for one of them.

Production frontend build:

```bash
cd frontend
npm run build
npm run preview
```

## Using the application

1. Open <http://localhost:8080> (Docker) or the Vite URL (native).
2. Choose an `.mp4`, `.avi`, `.mov`, `.mkv` or `.webm` file.
3. Upload it. The API returns `202` with an analysis id and the browser polls
   the status document.
4. Watch the measured stages: `uploading`, `tracking`, `validation`,
   `shot_classification`, `overlay`, `persisting`.
5. When the job completes, open the results page. Candidate delivery clips,
   trajectory overlays, pitch data and shot evidence are displayed. A failed
   job shows an explicit error instead of a fabricated result.
6. Cancellation is a cooperative request. It is checked at progress boundaries;
   a blocking model call must return before the job can stop.

The frontend can also be driven through the API. For a Docker stack:

```bash
curl -F "video=@my_clip.mp4" http://localhost:8080/api/analyze
curl http://localhost:8080/api/analysis/ANALYSIS_ID/status
curl http://localhost:8080/api/analysis/ANALYSIS_ID/result
```

For a native backend, use `http://127.0.0.1:8000` instead. `/api/analyze` uses
the multipart field `video`; `/api/upload` is an alias using `file`.

### HTTP API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | Upload a video as multipart field `video`; returns `202` and an analysis id. |
| `POST` | `/api/upload` | Alias of `/analyze` using multipart field `file`. |
| `GET` | `/api/analysis/{id}/status` | 13-key progress/status document. |
| `POST` | `/api/analysis/{id}/cancel` | Request cooperative cancellation; returns the current status. |
| `GET` | `/api/analysis/{id}/result` | Result envelope for a completed job. |
| `GET` | `/api/analysis/{id}/clips/{name}` | Browser-playable clip bytes with HTTP range support. |
| `GET` | `/api/health` | Versions, stage vocabulary and model availability. |

The status document always has exactly these keys: `analysis_id`, `status`,
`stage`, `substage`, `progress`, `detail`, `error`, `frames_done`,
`frames_total`, `source_name`, `created_at`, `started_at`, `finished_at`.
`status` is one of `queued`, `running`, `completed`, `failed`.

Selected status codes:

- `202` - upload accepted, or cancellation requested.
- `400` - unsupported extension, empty file, or unreadable/non-video upload.
- `404` - unknown analysis id.
- `409` - result requested while queued/running, or after a failed job.
- `413` - upload exceeds the configured ceiling.
- `500` - result missing/unreadable/mismatched, or the analysis could not start.
- `503` - the analysis service is not ready.

Cancellation and deadlines use the terminal `failed` status with an explicit
`cancelled by client` or `processing deadline exceeded (...)` reason. Full
schemas, status codes, the result envelope and clip behaviour are in
[`docs/API.md`](docs/API.md).

## Processing semantics and known gaps

The right column names the field or flag that makes each limitation visible.

| Topic | What the code does | What you see |
|---|---|---|
| Candidate deliveries | One confirmed track becomes one candidate window with context frames. No segmentation model exists. | `candidate_delivery_boundary_unvalidated`, `requires_human_review: true` |
| Trajectory provenance | Each point is `detected` or `predicted`. Gaps are not interpolated. A detection below `CRICKET_BALL_CONF` may only extend a live track and must pass the Kalman gate. | `source`, `confidence` (`null` when predicted), `low_confidence_points`, `trajectory_contains_low_confidence_detections` |
| Bounce | No bounce detector. `false` is a capability sentinel, not a measured "no bounce". | `bounce.detected: false`, all bounce fields `null`, `bounce_detection_unavailable` |
| Bowling metrics | `metric_gate()` rejects detected quads and assumed geometry without manually justified metric calibration. Normalized projection is still produced; it is not a metric measurement. | `speed_kmh`, `line`, `length`, `swing`, `release_angle`, `bounce_angle` are `null`; `calibrated: false` |
| Pixel speeds | Diagnostics may carry pixel-space summaries. | `units: "pixel"` — not km/h, never presented as bowling speed |
| Pitch calibration | Valid segmentation → calibrated. Box geometry → approximate. One accepted frame per delivery; no temporal fusion. Unprojectable points are omitted. | `state`, `calibrated`, `pitch_calibration_approximate`, `pitch_calibration_single_frame`, `pitch_projection_skipped_points` |
| Shot | Ten frozen labels: Cover, Defense, Flick, Hook, Late Cut, Lofted, Pull, Square Cut, Straight, Sweep. Missing/corrupt checkpoint leaves label and confidence `null`. | `model_unavailable` or the inference error in diagnostics |
| Overlay | Client-rendered. No server artifact or URL. | Browser draws trajectory and pitch in SVG |
| Timebase | Frame index is authoritative. `CAP_PROP_FPS` supplies the rate; seconds derive from frame counts. No PTS; constant frame rate assumed; VFR not modelled. | Diagnostics `video.timebase`; `invalid_video_fps` if the container reports no usable fps |

## Limits, timeouts and cleanup

| Limit | Where enforced | Behaviour when reached |
|---|---|---|
| 2 GiB upload | `backend/app/api/upload.py` | `413`; no analysis directory is created. |
| Allowed extensions | `backend/app/api/upload.py` | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`; others get `400`. |
| At least one decodable frame | OpenCV probe | `400` during upload. |
| 1-hour job deadline (configurable) | `backend/app/api/jobs.py` | Job marked `failed` with a deadline error; partial clips/documents removed. |
| Cooperative cancellation | `backend/app/api/jobs.py` | Checked at progress boundaries. A single blocking model call is not interrupted. |
| 600-frame clip cap | `backend/app/pipeline/clips.py` | Window is truncated; diagnostics record the candidate window. |
| 24-hour temp fragment age | startup cleanup | Stale upload fragments removed. |
| Restart recovery | `backend/app/api/jobs.py` | A persisted `queued`/`running` job becomes `failed` with a restart reason; partial clips are removed. |

The processing deadline is a safety net, not a guarantee that an individual
OpenCV/torch call will return within a fixed time. The API cannot pre-empt a
blocking C++/CUDA call from another thread.

Browser-side request timeouts are finite for status/result calls: 10 s for
status polling, 15 s for ordinary JSON, 60 s for the result fetch on the
processing page and 30 s on the results page. Upload uses `XMLHttpRequest`
progress reporting and **no fixed request timeout**, so a slow large upload is
not cut off mid-transfer. Polling is unbounded but backs off to 8 s and surfaces
lost-contact warnings; cancellation is an explicit user action, not a side
effect of leaving the page.

Nginx uses larger upload/read/send timeouts for the proxy. Those are transport
limits, separate from `CRICKET_MAX_JOB_SECONDS`. The body limit in
`frontend/nginx.conf` is 2 GiB; keep it at least as large as
`CRICKET_MAX_UPLOAD_BYTES`.

## Metrics: measured versus estimated

| Output | Status |
|---|---|
| Detected ball positions and confidences | Measured, with per-point provenance. |
| Predicted positions | Kalman extrapolation; confidence is `null`. |
| Normalized pitch positions | Projection through a validated or approximate homography; not metric. |
| Trajectory duration / pixel speed (diagnostics) | Computed from frame indices and container fps; pixel units only. |
| Shot label/confidence | Measured only when a compatible checkpoint loads. |
| Bounce | Unavailable; sentinel plus explicit flag. |
| Bowling speed/line/length/swing/angles | Unavailable without metric calibration; all `null`. |

No physical measurement is inferred from the pitch dimensions alone. The Laws
dimensions are used only inside explicit, labelled calibration assumptions.

## GPU guide

### Native CUDA

The native CUDA path was exercised on this machine:

- NVIDIA RTX 3050 Laptop GPU, 6 GB.
- NVIDIA driver 610.74.
- PyTorch `2.13.0+cu126`; `torch.cuda.is_available()` returned `True`.
- All four checkpoints loaded; the ball and pitch detectors ran on a synthetic
  frame and the shot classifier reported `device=cuda`.
- This is a loading/inference-path check, not an accuracy or throughput
  benchmark.

Install the CUDA wheels with the command in
[Install PyTorch first](#install-pytorch-first-then-the-rest), leave
`CRICKET_*_DEVICE=auto`, and start the backend normally. Confirm from a Python
shell:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

### Docker with NVIDIA

`compose.gpu.yaml` selects the CUDA 12.6 wheel index and requests an NVIDIA
device. **This path has not been verified on this host.** The prerequisite
check failed here:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Before using the override, make the prerequisite pass on the host. That means a
working NVIDIA driver and NVIDIA Container Toolkit configured for Docker (or
Docker Desktop's GPU support). Then:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec backend python -c "import torch; print(torch.cuda.is_available())"
```

The GPU image is much larger than the CPU image because it downloads CUDA
wheels. `CRICKET_*_DEVICE=auto` selects CUDA when the container can see it and
CPU otherwise, but the Compose device reservation itself requires the NVIDIA
runtime; if that reservation fails, the container will not start. Do not claim
GPU container support until both the `docker run --gpus all ... nvidia-smi`
check and the Compose health/inference checks pass.

## Testing

Run from the repository root.

### Native backend

```bash
python -m pytest backend
python -m ruff check backend
```

The integration tests use a deterministic fake pipeline for the HTTP contract
and the real tracking/pitch/projection/analytics code with deterministic model
doubles. They do not require network access or model inference.

### Native frontend

```bash
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
```

`npm run lint` currently reports three pre-existing `react-refresh` warnings;
the command exits successfully. The production build uses Vite and TypeScript.

### Docker verification

```bash
docker compose run --rm --no-deps -T backend sh -lc "cd backend && python -m pytest -q"
docker compose run --rm --no-deps -T backend sh -lc "python -m ruff check backend"
```

The backend tests and lint were run inside the CPU image in this repository.
Frontend tests were run natively; the frontend image is a production build
served by nginx, and the test suite is not installed into that image.

Measured results in this repository:

| Check | Result |
|---|---|
| Backend tests, native | 492 passed, 1 skipped, 123 subtests passed |
| Backend tests, CPU container | 492 passed, 1 skipped, 1 warning, 123 subtests passed |
| Backend lint, native and CPU container | clean |
| Frontend tests | 83 passed |
| Frontend typecheck | clean |
| Frontend lint | clean with three pre-existing `react-refresh` warnings |
| Frontend production build | passed, 1617 modules transformed |
| CPU backend image | 2.89 GB |
| Frontend image | 74.2 MB |
| Docker stack health | backend and frontend both healthy |
| Docker persistence | `status.json`, `result.json`, `diagnostics.json` survived `docker compose down` followed by `up`; status/result routes still served the job |
| Missing weights | `/api/health` reported all four unavailable with reasons; an uploaded job failed with an explicit ball-detector error |

Docker smoke test: a synthetic 90-frame, 640x360 clip was uploaded through the
frontend proxy. The job progressed through tracking and validation to
`completed` and the result route returned HTTP 200 with zero deliveries. That
is expected for synthetic footage and is **not** an accuracy or throughput
benchmark. The four real checkpoints were also loaded and run on blank
synthetic frames in both native CUDA and Docker CPU configurations: ball and
pitch returned `no_detection`, and the shot classifier returned a label (a
reminder that a model output on meaningless pixels is not evidence). No
representative cricket footage is present, so real accuracy and throughput
remain unmeasured.

## Troubleshooting

| Symptom | Check |
|---|---|
| `docker compose` cannot connect | Start Docker Desktop/Engine. On Windows, make sure the Linux container engine is running. |
| Port 8080 or 8000 already in use | Change `CRICKET_FRONTEND_PORT` / `CRICKET_API_PORT` in `.env`, then recreate the stack. |
| Frontend loads but API calls fail | Check `docker compose ps`; both services should be healthy. Open <http://localhost:8080/api/health> and `docker compose logs backend`. |
| `/api/health` shows `ball_detector.available: false` | Confirm `backend/models/ball_detector.pt` exists on the host. The Compose mount is `./backend/models:/models:ro`. |
| Upload returns `503` | The app lifespan did not initialise the registry; check the server log. |
| Upload returns `400` | Extension, empty file, or OpenCV cannot decode at least one frame. Re-encode the video. |
| Upload returns `413` | The file exceeds `CRICKET_MAX_UPLOAD_BYTES` or nginx's `client_max_body_size`. |
| Job fails immediately with a missing-weights message | The ball detector is required. Pitch/shot weights are optional but produce unavailable states. |
| Result returns `409` | The job is queued/running or failed. Poll the status document; a failed reason is in `error`. |
| Clips fail to encode | The local OpenCV build cannot open the WebM or MP4 writer. Check the diagnostics `video`/clip error. |
| Frontend cannot reach a native API | Confirm the backend origin, `VITE_API_URL`, and that the browser origin is in `CRICKET_CORS_ORIGINS`. |
| Cancellation seems slow | Cancellation is cooperative and checked at progress boundaries; a blocking model call must finish first. |
| Analysis exceeds the deadline | Inspect the source length/device and raise `CRICKET_MAX_JOB_SECONDS` if the job is legitimately long. |
| Docker GPU override fails to start | Run the `docker run --rm --gpus all ... nvidia-smi` prerequisite check. This host failed CDI discovery, so the GPU override was not verified. |
| A fresh stack has no old results | `docker compose down -v` deletes the data volume by design. Use `docker compose down` to keep it. |

## Cross-platform and repository notes

- Runtime and model paths use `pathlib`; relative environment paths resolve
  against the repository root.
- Generated files, local weights, videos, datasets, caches and `.env` files are
  ignored by git. The four supplied checkpoints are explicit exceptions.
- Native development was exercised on Windows. The backend and frontend Linux
  containers were built and started in WSL2 Ubuntu. Native Linux/macOS parity is
  supported by the code and commands but was not exercised on those hosts.
- The frontend image builds with an empty `VITE_API_URL`, so browser requests
  are same-origin `/api/...` and nginx proxies them to `backend:8000`.
- The backend image runs as an unprivileged user and uses one Uvicorn worker.
- The data volume is initialised from `/data` in the image; the model mount is
  read-only.

## Repository layout

```text
backend/
  main.py                 Uvicorn entry point
  Dockerfile              CPU/GPU-selectable backend image
  models/                 the four supplied checkpoints + notes
  app/
    api/                  FastAPI app, upload validation, jobs, routes, health
    pipeline/             tracking -> clips -> pitch -> shot -> analytics
    core/                 shared constants, environment and model paths
    schemas/              typed records for tracking, pitch and shots
    analytics/            pure calculations over those records
    pitch/                corners, calibration, homography, pitch detectors
    shots/                checkpoint config, preprocessing, architecture, inference
    tracking/             ball detector, Kalman tracker, tracking service
  tests/                  unit and integration tests
frontend/
  Dockerfile              Vite build + unprivileged nginx
  nginx.conf              same-origin /api proxy, SPA routing, timeouts
  src/                    React/TypeScript client
  ...                     Vitest tests and Vite config
docs/                     module and HTTP API documentation
compose.yaml              default CPU stack
compose.gpu.yaml          optional CUDA override (not verified on this host)
.env.example              documented configuration template
data/                     generated at runtime; ignored by git
```

`backend/` is the Python import root: `app.*` and `tests.*` are imported from
there.

## Documentation

- [`docs/TRACKING.md`](docs/TRACKING.md) - ball detection, Kalman tracking,
  configuration, output formats and limitations.
- [`docs/PITCH.md`](docs/PITCH.md) - corner order, calibration states,
  coordinate systems, model configuration and limitations.
- [`docs/SHOTS.md`](docs/SHOTS.md) - checkpoint contract, class order, input
  tensor, classification records, configuration and zero-versus-unavailable
  rules.
- [`docs/API.md`](docs/API.md) - routes, schemas, job lifecycle, cancellation,
  limits, cleanup and integration limitations.
- [`backend/models/README.md`](backend/models/README.md) - checkpoint checksums,
  loaders, optionality and Docker mounts.

## Branch and merge order

The integration work is on `feature/backend-integration`. The branches remain
independent; no history was rewritten and `main` was not merged. A sensible
integration order is:

1. `feature/ball-tracking`
2. `feature/pitch-detection`
3. `feature/shot-classification`
4. `feature/frontend`
5. `feature/backend-integration`

## Licence and model weights

The code in this repository is provided under the terms that accompany the
upstream CricketVision project. The four checkpoint files are included in this
repository for a fresh-clone install, but their provenance and licence are not
recorded here. Confirm the upstream terms before using, redistributing or
publishing any checkpoint.
