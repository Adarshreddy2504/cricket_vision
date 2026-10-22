import type { ResultDocument } from "@/lib/api";

/**
 * A minimal stand-in for a fetch Response. The API client only reads `ok`,
 * `status`, `statusText` and `json()`, so the tests do not need a real body
 * stream (and jsdom's Response support varies by Node version).
 */
export function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  } as Response;
}

export function emptyResponse(status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "OK",
    json: async () => {
      throw new SyntaxError("Unexpected end of JSON input");
    },
  } as unknown as Response;
}

export function resultFor(id: string): ResultDocument {
  return {
    analysis_id: id,
    status: "completed",
    pitch: {
      state: "no_calibration",
      detected: false,
      calibrated: false,
      confidence: null,
      using_previous_calibration: false,
      keypoints: {},
      corners: {},
      center: null,
      homography_available: false,
      orientation: null,
      source: null,
      geometry_reason: "no_pitch_quad_in_frame",
    },
    deliveries: [],
  };
}
