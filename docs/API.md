# HTTP API and backend integration (Part 5)

Part 5 puts a real FastAPI server in front of Parts 1–3 and connects them to the
Part 4 browser client. It implements the six routes in the frontend contract at
schema **2.1** and pipeline version **1.1.0**.

The server does **not** invent the stages that are absent from the feature
branches. Delivery segmentation, bounce detection, metric calibration and a
server-side overlay renderer are not implemented. The API reports those
capabilities as unavailable or unvalidated instead of manufacturing values.

---

## Running the server

From the repository root:

```bash
pip install -r requirements.txt
python backend/main.py
```

The default address is `http://127.0.0.1:8000`. Set `CRICKET_HOST` and
`CRICKET_PORT` to change it. The equivalent Uvicorn invocation is:

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `/docs` and `/redoc`.

The frontend defaults to `http://127.0.0.1:8000` (`VITE_API_URL`); its Vite
development server runs on port 8080, which is in the default CORS allow-list.
Use `CRICKET_CORS_ORIGINS` for any other browser origin.

## Health

`GET /api/health` returns versions, the stage vocabulary and whether each model
file can be found. It does not load a model, download weights or import torch.

```json
{
  "status": "ok",
  "schema_version": "2.1",
  "pipeline_version": "1.1.0",
  "analyses_tracked_in_memory": 0,
  "stages": [
    "uploading",
    "tracking",
    "validation",
    "shot_classification",
    "overlay",
    "persisting"
  ],
  "models": {
    "ball_detector": { "available": false, "reason": "..." },
    "pitch_segmentation": { "available": false, "reason": "..." },
    "pitch_box": { "available": false, "reason": "..." },
    "shot_classifier": { "available": false, "reason": "...", "md5": null }
  }
}
```

`available: true` means the file exists and passed the resolver's checks. It
does not guarantee that the checkpoint is compatible with the installed model
runtime. A load failure is reported when the job runs.

## Endpoints

| Method | Path | Request | Success |
|---|---|---|---|
| `POST` | `/api/analyze` | multipart field `video` | `202` analysis accepted |
| `POST` | `/api/upload` | multipart field `file` | `202` alias with the same behaviour |
| `GET` | `/api/analysis/{id}/status` | — | `200` status document |
| `GET` | `/api/analysis/{id}/result` | — | `200` result envelope |
| `GET` | `/api/analysis/{id}/clips/{name}` | — | `200` clip bytes, byte ranges supported |
| `GET` | `/api/health` | — | `200` health document |

### Upload rules

- Accepted extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
- Maximum body size: 2 GiB by default (`CRICKET_MAX_UPLOAD_BYTES`).
- The upload is streamed to a temporary file, size-checked, then opened with
  OpenCV and probed for at least one decodable frame.
- A rejected upload creates no analysis directory and leaves no partial file.
- The source filename is reduced to its basename; no client path is used.
- The original source is deleted after processing unless
  `CRICKET_KEEP_SOURCE=true`.

A successful upload returns:

```json
{
  "analysis_id": "8c5d…",
  "status": "queued",
  "progress": 0.0,
  "stage": "uploading",
  "bytes": 1234567,
  "status_url": "/api/analysis/8c5d…/status",
  "result_url": "/api/analysis/8c5d…/result"
}
```

Upload errors are JSON `{ "detail": "…" }`:

| Status | Meaning |
|---|---|
| `400` | Unsupported extension, empty file, or an unreadable/non-video file |
| `413` | Upload exceeds the configured ceiling |
| `500` | The analysis could not be started; partial state is removed |
| `503` | The application is not ready (registry unavailable) |

### Status document

The status document has exactly these keys:

```json
{
  "analysis_id": "8c5d…",
  "status": "running",
  "stage": "tracking",
  "substage": null,
  "progress": 37.5,
  "detail": "tracked frame 420 of 900",
  "error": null,
  "frames_done": 420,
  "frames_total": 900,
  "source_name": "net.mp4",
  "created_at": "2026-10-22T10:00:00+00:00",
  "started_at": "2026-10-22T10:00:01+00:00",
  "finished_at": null
}
```

`status` is one of `queued`, `running`, `completed`, `failed`. `stage` follows
the health vocabulary while running and is `completed` or `failed` when
terminal. `progress` is measured from bytes, frames, clips and deliveries; it
never decreases and never uses an assumed time fraction. `frames_done` and
`frames_total` are populated during tracking only. `frames_total` is `null` if
the container does not declare a frame count.

A failed job stores the exception text in `error` and leaves `result.json`
absent. `GET …/result` returns `409` while queued/running and for a failed job.

### Result envelope

The result is served only for a completed analysis whose stored document still
matches that analysis id. The envelope has exactly four top-level keys:

```json
{
  "analysis_id": "8c5d…",
  "status": "completed",
  "pitch": { "...": "pitch block of the last calibrated delivery" },
  "deliveries": []
}
```

Each delivery has `id`, `video`, `trajectory`, `bounce`, `bowling`, `shot`,
`quality` and `pitch`. A value that was not measured is `null`; it is never
replaced by `0`, `false`, an empty label or a guessed geometry.

```json
{
  "id": 1,
  "video": {
    "clip_url": "/api/analysis/8c5d…/clips/delivery_001.webm",
    "start_frame": 101,
    "end_frame": 160,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_seconds": 2.0
  },
  "trajectory": {
    "points": [
      {
        "frame": 101,
        "x": 812,
        "y": 604,
        "source": "detected",
        "confidence": 0.91,
        "pitch_position": { "x": 0.48, "y": 0.12, "inside_pitch": true }
      }
    ],
    "detected_points": 42,
    "predicted_points": 6,
    "continuity": "continuous"
  },
  "bounce": {
    "detected": false,
    "frame": null,
    "x": null,
    "y": null,
    "ground_x_m": null,
    "ground_y_m": null
  },
  "bowling": {
    "speed_kmh": null,
    "line": null,
    "length": null,
    "swing": null,
    "release_angle": null,
    "bounce_angle": null,
    "calibrated": false
  },
  "shot": { "classification": null, "confidence": null },
  "quality": {
    "trajectory_valid": true,
    "tracking_confidence": 0.87,
    "requires_human_review": true,
    "flags": [
      "candidate_delivery_boundary_unvalidated",
      "metric_calibration_required",
      "bounce_detection_unavailable",
      "pitch_calibration_single_frame",
      "shot_model_unavailable"
    ]
  },
  "pitch": { "...": "pitch block" }
}
```

`trajectory.points` preserves detector provenance. A `"predicted"` point has
`confidence: null`; there is no measurement to attach. `continuity` is
`"continuous"` or `"gapped"`. Gaps are not interpolated.

The pitch block has `state`, `detected`, `calibrated`, `confidence`,
`using_previous_calibration`, `keypoints`, `corners`, `center`,
`homography_available`, `orientation`, `source` and `geometry_reason`.

- `state` is `calibrated`, `approximate` or `no_calibration`.
- `calibrated` is true only for a validated segmentation calibration.
- `orientation` is `{axis, batting_end, travel_delta, calibration_frame}` when
  the projected travel direction is unambiguous; otherwise `null`.
- `homography_available` means a normalized pitch projection exists. It does
  **not** mean the camera was metrically calibrated.

### Result errors

| Status | Meaning |
|---|---|
| `404` | Unknown analysis id |
| `409` | The analysis is still running or has failed |
| `500` | `result.json` is missing, unreadable or belongs to another id |

### Clip route

Clips are written during validation from the source video in a single pass.
WebM with VP8 (falling back to VP9) is preferred; `mp4v` MP4 is used only when
the local OpenCV build cannot open a WebM writer. The response is served by
Starlette's `FileResponse`, which supports HTTP byte ranges for `<video>`
scrubbing. Clips are available only for completed analyses and only by their
generated basename.

## Model weights

No weights are tracked in git and none are downloaded automatically.

| File | Environment override | Required? | Behaviour when missing or unloadable |
|---|---|---|---|
| `ball_detector.pt` | `CRICKET_BALL_MODEL` | **Yes** | The job fails loudly; no result is fabricated. |
| `pitch_seg_yolo11s.pt` | `CRICKET_PITCH_SEG_MODEL` | No | Pitch falls back to the box detector or to `no_calibration`. |
| `pitch_detect.pt` | `CRICKET_PITCH_BOX_MODEL` | No | Pitch falls back to `no_calibration`. |
| `shot_classifier.ckpt` | `CRICKET_SHOT_MODEL` | No | `shot.classification` and `confidence` stay `null`, with a `shot_model_unavailable` flag. |

`CRICKET_MODELS_DIR` points at the directory holding all four files (default
`backend/models/`). `CRICKET_SHOT_MODEL_MD5` optionally requires the documented
shot checkpoint MD5 to match. Device selection is controlled by
`CRICKET_BALL_DEVICE`, `CRICKET_PITCH_DEVICE` and `CRICKET_SHOT_DEVICE`.

## Processing semantics and limitations

### Candidate deliveries, not validated boundaries

`TrackingService.track_video()` produces whole-video trajectories. A confirmed
track is a continuous flight, not a delivery boundary. The pipeline therefore
turns each confirmed track into a **candidate delivery**: a clip window with
10 frames of context before and after by default, capped at 600 frames.
Every delivery carries `candidate_delivery_boundary_unvalidated` and
`requires_human_review: true`. No delivery segmentation model exists in the
combined branches.

### Bounce detection is unavailable

There is no bounce detector, so `bounce.detected` is `false` and every other
bounce field is `null`. The `bounce_detection_unavailable` flag makes clear that
`false` is a capability sentinel, not a measured "no bounce" finding.

### Bowling measurements are uncalibrated

`metric_gate()` rejects detected quads and assumed geometry without manually
justified metric calibration. Consequently `speed_kmh`, `line`, `length`,
`swing`, `release_angle` and `bounce_angle` are `null` and `calibrated` is
`false`, with `metric_calibration_required`. The normalized pitch projection is
still produced when the pitch calibration is usable; that projection is not a
metric measurement.

### Pitch calibration

- Segmentation corners can produce a `valid` calibration.
- Box-estimated corners produce `state: "approximate"`, `calibrated: false`,
  and `pitch_calibration_approximate`.
- One accepted frame calibrates a delivery. There is no temporal fusion, hence
  `pitch_calibration_single_frame`.
- Points that cannot be projected are omitted from `pitch_position`, and
  `pitch_projection_skipped_points` is added when that happens.
- Approximate calibration is never presented as validated.

### Shot classification

A successful classifier returns one of the frozen ten labels. If the
checkpoint is absent, corrupt or incompatible, the label and confidence remain
`null`; `shot.status` in the diagnostics document records
`model_unavailable`. Input or inference failures are also recorded there. The
result document never invents a label from the pitch or the trajectory.

### Overlay

The `overlay` stage is reported as client-rendered. No server overlay artifact
or URL is produced. The browser draws the trajectory and pitch overlay from the
result data.

### Analytics

Delivery analytics are computed for diagnostics: trajectory duration, pixel
speed, pitch-space summaries and a shot record, each with its own unavailable
map when an input was missing. They are written to `diagnostics.json`, not
added as new result fields. The frontend computes its aggregate cards from the
result deliveries. Metrics that require metric calibration remain unavailable
in the analytics roll-up.

## Persistence, cleanup and restarts

Generated files live under `data/` by default (`CRICKET_DATA_DIR` to override):

```
data/
  tmp/                              # streamed uploads, never inside a run
  uploads/<id>/source.<ext>         # deleted after the run unless kept
  runs/<id>/status.json
  runs/<id>/result.json             # clean browser document
  runs/<id>/diagnostics.json        # decisions, models, boundaries; never served
  runs/<id>/clips/delivery_NNN.webm
```

Rejected uploads create no `runs/<id>` directory. Temporary files older than
24 hours are removed at startup. On restart, a persisted status that is still
`queued` or `running` is marked `failed` with
`"the server restarted before this analysis finished"`; it is never resumed
under a new process. Analyses are executed on daemon threads but serialized
around the shared model instances.

## Verification

```bash
# All backend tests, including the Part 5 integration suite
python -m pytest backend

# Lint
python -m ruff check backend

# Frontend contract, types and production build
cd frontend
npm test
npm run typecheck
npm run build
```

The integration tests use a deterministic fake pipeline for the HTTP contract
and real tracking, pitch, projection and analytics modules with deterministic
model doubles for the pipeline tests. They do not require model weights.
