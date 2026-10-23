# CricketVision frontend

The browser interface for CricketVision: choose a clip, watch the run progress,
then read the tracked trajectory, pitch map, shot labels and delivery records.

This is **Part 4**. The six HTTP routes it consumes are the documented contract
(schema 2.1); the server that implements them is **Part 5**. Until that server
is running, live uploads fail honestly and the interface offers a clearly
labelled synthetic dataset for review.

## Stack

- Vite 5, React 18, TypeScript (strict)
- Tailwind CSS with the project's dark-first design tokens
- React Router 6
- Radix Slot, `class-variance-authority`, `lucide-react`
- Vitest 3, Testing Library, jsdom
- ESLint 9 with the TypeScript and React Hooks rules

No state library, chart library or component framework is used. The trajectory
and pitch views are SVG; aggregates are computed in pure TypeScript.

## Quick start

```bash
cd frontend
npm install
npm run dev
```

The dev server prints a local URL (Vite defaults to `http://localhost:5173`).
Copy `.env.example` to `.env` only if the API is not at the default origin:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
```

`VITE_API_URL` is read at build time. It defaults to `http://127.0.0.1:8000`,
with the `/api` prefix added by the client.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite development server with HMR |
| `npm run build` | `tsc --noEmit` followed by a production build into `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run typecheck` | Strict TypeScript check, no emit |
| `npm run lint` | ESLint over the project |
| `npm test` | Run the full Vitest suite once |
| `npm run test:watch` | Vitest in watch mode |

## Routes

| Path | Screen | Purpose |
|---|---|---|
| `/` | Dashboard | API health, quick start, recent analysis, demo entry |
| `/upload` | Upload | Drag/drop or pick a video, validate it locally, upload with progress, choose demo mode |
| `/processing` | Processing | Poll status, show stages and progress, handle completion, failure and unreachable backend |
| `/results` | Results | Summary, pitch map, shot distribution, clip stage, delivery explorer, export |
| `*` | Not found | Route back to a real screen |

Navigation carries the analysis id in router state. It is also stored in
`localStorage` so a refresh can resume `/processing` or `/results`. A cached
result is shown only for the id it was written under.

## What is implemented

### Dashboard

- Calls `GET /api/health`. A failure is shown as an unreachable backend, not as
  a healthy state.
- Links to upload and, when a stored analysis exists, back to its processing or
  results screen.
- Demo entry is marked as synthetic on the card and on every subsequent screen.
- Responsive feature and status cards; keyboard-visible focus states.

### Upload

- Drag/drop and file picker with the documented accepted types
  (`mp4`, `avi`, `mov`, `mkv`, `webm`) and the 2 GB ceiling.
- Local checks before sending: empty file, unsupported extension, oversized
  file and a non-video MIME type. Errors name the problem and the accepted
  formats; they are announced in an alert and via `aria-live`.
- Local preview via `URL.createObjectURL` with cleanup on replace/unmount.
- Upload progress from XHR, cancel, backend error detail, connection failure and
  contract mismatch. Nothing is faked if the request fails.
- On success the analysis id is stored and the browser moves to `/processing`.

### Processing

- Polls `GET /api/analysis/{id}/status` using the returned stage vocabulary.
- Shows per-stage state, elapsed time, frame counts, measured progress and the
  backend's `detail` text. A missing percentage produces an indeterminate bar,
  not a made-up number.
- Terminal `failed` stops polling and shows the backend's error reason.
- Terminal `completed` fetches the result document and moves to `/results`.
- Handles the backend disappearing mid-run, a 404 for an unknown id, malformed
  status payloads, pause/resume, retry and copy-id.
- Demo analyses skip the unavailable API and open the labelled demo result.

### Results

- Computes all aggregates from the deliveries in the current document only.
- Summary cards: delivery count, measured average speed and sample size, fastest
  measured ball, dominant length, dominant line, classified-shot count, mean
  confidence and processing time.
- Trajectory overlay drawn in source-frame pixels with a dashed predicted
  segment. Pitch corners and keypoints are drawn only when the contract supplies
  them and calibration is valid.
- Top-down pitch map with bounce dots. A dot requires both a pitch position and
  an orientation naming the batting end; no dot is inferred from pixels.
  Uncalibrated deliveries are listed separately.
- Five published length zones, shot distribution over the frozen 10-class
  vocabulary, and out-of-vocabulary labels counted and flagged rather than
  silently dropped.
- Clip stage with playback, scrubbing, frame stepping, mute, fullscreen and
  overlay toggles. Real clips come from `video.clip_url`; the demo has no
  footage, so it draws a **synthetic tracking preview** instead.
- Delivery explorer with search, length/line filters and a detail panel for the
  selected record. `null` renders as `—`; missing labels say so.
- States for no result, no deliveries, no clip media, incomplete/legacy result
  fields, cached-but-unreachable backend and contract mismatch.
- JSON export. Demo exports are wrapped with `{"demo": true, "notice": ...,
  "result": ...}` so an exported file cannot be mistaken for a model run.

### Accessibility and responsiveness

- Semantic landmarks, headings and lists; one visible `h1` per screen.
- All interactive controls are native buttons/links/inputs; the file input is
  reachable by keyboard and the dropzone exposes a button.
- Status changes use `role="status"` / `aria-live`, and progress exposes
  `role="progressbar"` with an accessible name.
- Overlays are marked decorative where the same facts are in text; colour is
  never the only signal.
- Layouts collapse from three columns to one; tables become cards; controls
  wrap at narrow widths.
- `prefers-reduced-motion` disables non-essential animation.

## API contract

The client is written against schema **2.1** and the routes in the backend's
`docs/API_CONTRACT.md`. No endpoint is invented.

| Route | Used for |
|---|---|
| `POST /api/analyze` | Upload (`multipart` field `video`), returns `analysis_id` and status URLs |
| `POST /api/upload` | Documented alternative upload (`multipart` field `file`); typed but unused by the UI |
| `GET /api/analysis/{id}/status` | Lifecycle, stage, progress, detail, error, frame counts |
| `GET /api/analysis/{id}/result` | Result envelope: pitch block, deliveries, per-delivery trajectory/bounce/video/shot |
| `GET /api/analysis/{id}/clips/{name}` | Clip bytes for the stage; consumed via `video.clip_url` |
| `GET /api/health` | Service status, schema version, stage vocabulary |

Statuses: `queued`, `running`, `completed`, `failed`.
Stages: `uploading`, `tracking`, `validation`, `shot_classification`, `overlay`,
`persisting`.

The API layer in `src/lib/api.ts`:

- normalises network/HTTP/contract/abort failures into `ApiError`;
- validates response envelopes before the UI sees them;
- encodes analysis ids in paths and makes root-relative clip URLs absolute;
- returns `null` for absent values instead of substituting defaults;
- uses XHR only for upload progress, `fetch` everywhere else.

## Demo mode (synthetic data)

`src/lib/demo.ts` is the only source of demo content. It is **not** a model run,
a recorded match or a benchmark set.

- Analysis id is `demo-analysis`; `isDemoAnalysisId` gates the mode.
- Every screen renders a permanent `DemoBanner` with `DEMO_NOTICE`.
- The result is the same `ResultDocument` shape the real route serves, so live
  and demo data exercise exactly the same components.
- The 10 deliveries intentionally include missing values, one uncalibrated
  delivery and two unclassified shots to exercise honest empty states.
- No video is shipped. The stage shows a synthetic trajectory preview with
  working timeline controls and a visible **SYNTHETIC PREVIEW** label.
- Export wraps the payload and repeats the notice.

Demo mode is for interface review. It must not be described as a working
analysis, and it must be removed from any claim about model performance.

## Data honesty rules

These are enforced in `src/lib/analysis.ts` and covered by tests:

- A `null` measurement stays `null`; renderers show `—`, never `0` or
  "Unknown".
- Averages report their sample size ("measured on 9 of 10 deliveries").
- Shares are undefined (empty state) when nothing was classified, not `0%`.
- A bounce dot exists only with both a pitch position and a valid orientation.
- Predicted trajectory is dashed; observed points are solid.
- Cached results are labelled as cached; refresh failures are shown alongside
  the cached data, not hidden.
- A contract mismatch is an error, not a partially rendered dashboard.

## Project structure

```
frontend/
  src/
    components/        # shared views and shadcn-style primitives (ui/)
    lib/
      api.ts           # contract types, fetch/XHR client, validation
      analysis.ts      # pure delivery adaptation and aggregates
      demo.ts          # synthetic dataset and demo labels
      format.ts        # formatting; DASH for absent values
      storage.ts       # analysis id + cached result
      validation.ts    # pre-upload file checks
    pages/             # Dashboard, Upload, Processing, Results, NotFound
    test/              # unit and component tests, jsdom setup, helpers
  .env.example
  tailwind.config.ts
  vite.config.ts
  vitest.config.ts
```

Components do not fetch. Data access lives in `src/lib`; projection and
aggregation are pure functions; pages own effects and routing.

## Testing

```bash
npm test          # 83 tests
npm run lint      # 0 errors; 3 fast-refresh warnings from primitive exports
npm run typecheck
npm run build
```

Coverage includes: formatting and absent values; delivery adaptation, bounce
orientation, stages, summaries and distribution; upload validation; API
encoding, HTTP/network/contract failures; storage scoping; demo labelling;
summary and distribution rendering; upload staging/removal/demo entry;
processing failure, indeterminate progress and 404; results demo, empty,
contract-error, cached-offline and no-id states.

The suite mocks `fetch` at the client boundary. There is no end-to-end test
against a running server, because that server is Part 5.

## Part 5 work

The frontend is ready to point at a real service, but the following is **not**
implemented here:

1. Implement and serve the six contract routes (schema 2.1), including the
   `analysis_id` envelope and the documented error shape.
2. Run the Parts 1–3 pipeline behind `/api/analyze` (or `/api/upload`) and
   publish stage/progress updates that match the vocabulary above.
3. Serve clip bytes with HTTP range support so the stage can seek; return
   absolute or root-relative `clip_url` values.
4. Persist analyses and results so a refresh, restart or second browser can
   resume them. The frontend currently relies on router state and
   `localStorage`.
5. Configure CORS for the frontend origin and any proxy used in deployment.
6. Replace or hide demo entry in production, or keep it explicitly disabled.
7. Add end-to-end tests against the live service and a deployment story for the
   SPA (history fallback, cache headers, `VITE_API_URL` per environment).

Until then, treat every number shown in demo mode as synthetic interface
filler, and every live screen as unavailable rather than successful.
