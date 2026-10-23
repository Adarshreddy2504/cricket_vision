/**
 * api.ts — the one place the browser talks to the CricketVision backend.
 *
 * CONTRACT
 * --------
 * The paths and payloads below are the documented contract (schema 2.1,
 * docs/API.md in the repository). No endpoint is invented here:
 *
 *   POST /api/analyze                              multipart `video`
 *   POST /api/upload                               multipart `file`
 *   GET  /api/analysis/{id}/status                 progress document
 *   POST /api/analysis/{id}/cancel                 cooperative cancellation
 *   GET  /api/analysis/{id}/result                 result envelope
 *   GET  /api/analysis/{id}/clips/{name}           clip bytes (byte-range)
 *   GET  /api/health                               versions + stage vocabulary
 *
 * If the backend is not running, `uploadVideo` reports a connection failure
 * instead of pretending to analyse anything. The UI offers the clearly-labelled
 * demo dataset in `demo.ts`.
 *
 * HONESTY RULES ENCODED HERE
 * --------------------------
 *   * A `null` measurement stays `null`; the adapter layer turns it into an
 *     absent value rendered as "—". It is never coerced to 0 or to a label
 *     such as "Unknown".
 *   * A result envelope is accepted only for the analysis id that was asked
 *     for. A server document for another id is a contract error, not data.
 *   * Errors are typed (`network`, `http`, `contract`, `aborted`) so the UI can
 *     distinguish "backend is not running" from "backend rejected this video".
 */

export const API_BASE = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** The contract this client is written against. */
export const CONTRACT_SCHEMA_VERSION = "2.1";

/** The documented accepted extensions and upload ceiling. */
export const ACCEPTED_VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"] as const;
export const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;

export type ApiErrorKind = "network" | "http" | "contract" | "aborted";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(message: string, kind: ApiErrorKind = "http", status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE}/api${path.startsWith("/") ? path : `/${path}`}`;
}

/**
 * Make a URL returned by the API absolute. The backend serves root-relative
 * clip paths such as `/api/analysis/…/clips/delivery_004.mp4`; the page is
 * usually served from a different origin in development.
 */
export function mediaUrl(url?: string | null): string | null {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return `${API_BASE}${url}`;
  return `${API_BASE}/${url}`;
}

/* ── Served types (schema 2.1) ───────────────────────────────────────────── */

export type AnalysisLifecycle = "queued" | "running" | "completed" | "failed";

export interface AnalyzeResponse {
  analysis_id: string;
  status: AnalysisLifecycle | string;
  progress?: number | null;
  stage?: string | null;
  bytes?: number | null;
  status_url?: string | null;
  result_url?: string | null;
}

export interface AnalysisStatus {
  analysis_id: string;
  status: AnalysisLifecycle | string;
  stage?: string | null;
  substage?: string | null;
  /** 0..100, measured units — never an assumed time fraction. */
  progress?: number | null;
  detail?: string | null;
  error?: string | null;
  frames_done?: number | null;
  frames_total?: number | null;
  source_name?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface Health {
  status: string;
  schema_version?: string;
  pipeline_version?: string;
  analyses_tracked_in_memory?: number;
  stages?: string[];
}

export interface PitchPosition {
  /** Fraction across the detected pitch quad, 0..1. */
  x: number;
  /** Fraction along the detected pitch quad, 0..1. */
  y: number;
  inside_pitch: boolean;
}

/**
 * Which end of the detected quad the batter stood at. A top-down pitch map
 * without this can put a yorker at the bowler's end, so `null` means no dot
 * may be drawn.
 */
export interface PitchOrientation {
  axis: "u" | "v" | string;
  batting_end: number;
  travel_delta: number | null;
  calibration_frame: number | null;
}

export interface PitchBlock {
  state:
    | "calibrated"
    | "temporary_loss"
    | "no_calibration"
    | "calibration_expired"
    | string;
  detected: boolean;
  calibrated: boolean;
  confidence: number | null;
  using_previous_calibration: boolean;
  /** name → [x, y] in SOURCE frame pixels. */
  keypoints: Record<string, number[]>;
  /** corner name → [x, y] in SOURCE frame pixels; {} when no quad exists. */
  corners: Record<string, number[]>;
  center: number[] | null;
  homography_available: boolean;
  orientation: PitchOrientation | null;
  source?: string | null;
  geometry_reason?: string | null;
}

export interface TrajectoryPoint {
  /** 1-based SOURCE frame number. */
  frame: number;
  x: number;
  y: number;
  /** "detected" is a measurement; "predicted" is the tracker coasting. */
  source: string;
  confidence: number | null;
  pitch_position: PitchPosition | null;
}

export interface DeliveryVideo {
  clip_url: string | null;
  start_frame: number | null;
  end_frame: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
}

export interface DeliveryTrajectory {
  points: TrajectoryPoint[];
  detected_points: number;
  predicted_points: number;
  continuity: string;
}

export interface DeliveryBounce {
  detected: boolean;
  frame: number | null;
  x: number | null;
  y: number | null;
  ground_x_m: number | null;
  ground_y_m: number | null;
}

export interface DeliveryBowling {
  speed_kmh: number | null;
  line: string | null;
  length: string | null;
  swing: string | null;
  release_angle: number | null;
  bounce_angle: number | null;
  calibrated: boolean;
}

export interface DeliveryShot {
  classification: string | null;
  confidence: number | null;
}

export interface DeliveryQuality {
  trajectory_valid: boolean;
  tracking_confidence: number | null;
  requires_human_review: boolean;
  flags: string[];
}

export interface Delivery {
  id: number;
  video: DeliveryVideo;
  trajectory: DeliveryTrajectory;
  bounce: DeliveryBounce;
  bowling: DeliveryBowling;
  shot: DeliveryShot;
  quality: DeliveryQuality;
  pitch: PitchBlock;
}

export interface ResultDocument {
  analysis_id: string;
  status: string;
  /** The calibration of the last delivery that had one. */
  pitch: PitchBlock;
  deliveries: Delivery[];
}

export interface CachedResult extends ResultDocument {
  processing_time?: number;
}

/* ── Parsing and validation ──────────────────────────────────────────────── */

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function contractError(message: string): ApiError {
  return new ApiError(
    `The server response does not match API contract ${CONTRACT_SCHEMA_VERSION}: ${message}`,
    "contract",
    200,
  );
}

/**
 * Validate just enough of the envelope that the UI cannot crash on a malformed
 * or truncated response. Measurements are allowed to be null; missing blocks
 * are not, because the contract says they are always present.
 */
export function parseResultDocument(value: unknown, expectedId?: string): ResultDocument {
  if (!isObject(value)) throw contractError("expected a JSON object");
  if (typeof value.analysis_id !== "string" || !value.analysis_id) {
    throw contractError("analysis_id is missing");
  }
  if (expectedId && value.analysis_id !== expectedId) {
    throw contractError(
      `the server returned analysis ${value.analysis_id} instead of ${expectedId}`,
    );
  }
  if (!Array.isArray(value.deliveries)) throw contractError("deliveries is not an array");
  if (!isObject(value.pitch)) throw contractError("the analysis pitch block is missing");

  value.deliveries.forEach((entry, index) => {
    const where = `deliveries[${index}]`;
    if (!isObject(entry)) throw contractError(`${where} is not an object`);
    if (typeof entry.id !== "number" || !Number.isFinite(entry.id)) {
      throw contractError(`${where}.id is not a number`);
    }
    for (const key of ["video", "trajectory", "bowling", "shot", "quality", "pitch"]) {
      if (!isObject(entry[key])) throw contractError(`${where}.${key} is missing`);
    }
    const trajectory = entry.trajectory as Record<string, unknown>;
    if (!Array.isArray(trajectory.points)) {
      throw contractError(`${where}.trajectory.points is not an array`);
    }
  });

  return value as unknown as ResultDocument;
}

export function isResultDocument(value: unknown, expectedId?: string): value is ResultDocument {
  try {
    parseResultDocument(value, expectedId);
    return true;
  } catch {
    return false;
  }
}

function parseAnalyzeResponse(value: unknown): AnalyzeResponse {
  if (!isObject(value) || typeof value.analysis_id !== "string" || !value.analysis_id) {
    throw contractError("the upload response has no analysis_id");
  }
  return value as unknown as AnalyzeResponse;
}

/** Pull the most useful message out of a FastAPI error body. */
function errorDetail(body: unknown): string {
  if (typeof body === "string") return body;
  if (!isObject(body)) return "";
  const detail = body.detail ?? body.error;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (isObject(item) && typeof item.msg === "string") return item.msg;
        return "";
      })
      .filter(Boolean)
      .join("; ");
  }
  return "";
}

interface RequestOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

/**
 * A fetch with a timeout and a typed failure. The caller's abort signal is
 * forwarded; an abort caused by the timeout is reported as a network timeout,
 * while a caller abort stays an abort.
 */
async function requestJson<T>(
  path: string,
  init: RequestInit,
  options: RequestOptions,
  parse: (value: unknown) => T,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutMs = options.timeoutMs ?? 15_000;

  const onAbort = () => controller.abort();
  if (options.signal) {
    if (options.signal.aborted) controller.abort();
    else options.signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  let response: Response;
  try {
    response = await fetch(apiUrl(path), { ...init, signal: controller.signal });
  } catch {
    if (timedOut) {
      throw new ApiError(
        `The backend did not answer within ${Math.round(timeoutMs / 1000)}s.`,
        "network",
      );
    }
    if (options.signal?.aborted) {
      throw new ApiError("Request cancelled.", "aborted");
    }
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Check that it is running.`,
      "network",
    );
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", onAbort);
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail = errorDetail(body);
    throw new ApiError(
      detail || `${response.status} ${response.statusText}`,
      "http",
      response.status,
    );
  }
  if (body === null && response.status !== 204) {
    throw contractError("expected a JSON body but the response was empty or not JSON");
  }
  return parse(body);
}

/** `POST /api/analyze` — accepts the video and returns an analysis id. */
export function uploadVideo(
  file: File,
  options: {
    signal?: AbortSignal;
    onProgress?: (progress: { loaded: number; total: number; percent: number | null }) => void;
  } = {},
): Promise<AnalyzeResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/analyze"));
    xhr.responseType = "text";

    const abort = () => xhr.abort();
    options.signal?.addEventListener("abort", abort, { once: true });

    const cleanup = () => options.signal?.removeEventListener("abort", abort);

    xhr.upload.onprogress = (event) => {
      options.onProgress?.({
        loaded: event.loaded,
        total: event.total,
        percent: event.lengthComputable && event.total > 0 ? (event.loaded / event.total) * 100 : null,
      });
    };

    xhr.onerror = () => {
      cleanup();
      reject(
        new ApiError(
          `Cannot reach the backend at ${API_BASE}. Check that it is running.`,
          "network",
        ),
      );
    };

    xhr.onabort = () => {
      cleanup();
      reject(new ApiError("Upload cancelled.", "aborted"));
    };

    xhr.onload = () => {
      cleanup();
      let body: unknown = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        body = null;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(
          new ApiError(
            errorDetail(body) || `${xhr.status} ${xhr.statusText || "Upload failed"}`,
            "http",
            xhr.status,
          ),
        );
        return;
      }
      try {
        resolve(parseAnalyzeResponse(body));
      } catch (error) {
        reject(error);
      }
    };

    xhr.send(form);
  });
}

function parseStatus(body: unknown): AnalysisStatus {
  if (!isObject(body) || typeof body.analysis_id !== "string") {
    throw contractError("the status response has no analysis_id");
  }
  return body as unknown as AnalysisStatus;
}

/** `GET /api/analysis/{id}/status`. */
export function fetchStatus(
  analysisId: string,
  options: RequestOptions = {},
): Promise<AnalysisStatus> {
  return requestJson(
    `/analysis/${encodeURIComponent(analysisId)}/status`,
    { method: "GET" },
    options,
    parseStatus,
  );
}

/**
 * `POST /api/analysis/{id}/cancel` — cooperative cancellation.
 *
 * The response is the current status document. A running job may remain
 * `running` until the pipeline reaches its next progress check; callers must
 * keep polling rather than assume the job stopped immediately.
 */
export function cancelAnalysis(
  analysisId: string,
  options: RequestOptions = {},
): Promise<AnalysisStatus> {
  return requestJson(
    `/analysis/${encodeURIComponent(analysisId)}/cancel`,
    { method: "POST" },
    options,
    parseStatus,
  );
}

/** `GET /api/analysis/{id}/result`. */
export function fetchResult(
  analysisId: string,
  options: RequestOptions = {},
): Promise<ResultDocument> {
  return requestJson(
    `/analysis/${encodeURIComponent(analysisId)}/result`,
    { method: "GET" },
    options,
    (body) => parseResultDocument(body, analysisId),
  );
}

/** `GET /api/health` — liveness, versions and the stage vocabulary. */
export function fetchHealth(options: RequestOptions = {}): Promise<Health> {
  return requestJson(
    "/health",
    { method: "GET" },
    { timeoutMs: 4_000, ...options },
    (body) => {
      if (!isObject(body) || typeof body.status !== "string") {
        throw contractError("the health response has no status");
      }
      return body as unknown as Health;
    },
  );
}
