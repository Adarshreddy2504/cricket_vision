/**
 * demo.ts — a synthetic result used ONLY to review the interface without the
 * Part 5 backend.
 *
 * This is not a model run, not a recorded match and not a benchmarking set.
 * Every screen that renders it shows `DEMO_NOTICE`, the analysis id starts
 * with `demo-`, and exporting it writes a wrapper that repeats the notice. The
 * numbers are plausible shapes chosen to exercise the interface (measured and
 * missing values, calibrated and uncalibrated deliveries), not measurements.
 *
 * Real runs replace this wholesale: the components consume the same
 * `ResultDocument` contract that `/api/analysis/{id}/result` serves.
 */

import type {
  Delivery,
  DeliveryBounce,
  DeliveryBowling,
  DeliveryShot,
  PitchBlock,
  ResultDocument,
  TrajectoryPoint,
} from "./api";

export const DEMO_ANALYSIS_ID = "demo-analysis";
export const DEMO_SOURCE_NAME = "demo-broadcast.mp4";
export const DEMO_PROCESSING_SECONDS = 41.8;
export const DEMO_NOTICE =
  "Demo data — synthetic deliveries for interface review. No model was run and these numbers are not a real analysis.";

export function isDemoAnalysisId(id: string | null | undefined): boolean {
  return typeof id === "string" && (id === DEMO_ANALYSIS_ID || id.startsWith("demo-"));
}

const FRAME_WIDTH = 1280;
const FRAME_HEIGHT = 720;
const FPS = 25;

const CORNERS = {
  top_left: [772, 318],
  top_right: [896, 314],
  bottom_right: [1072, 700],
  bottom_left: [528, 700],
};

function calibratedPitch(): PitchBlock {
  return {
    state: "calibrated",
    detected: true,
    calibrated: true,
    confidence: 0.87,
    using_previous_calibration: false,
    keypoints: {
      top_left: [...CORNERS.top_left],
      top_right: [...CORNERS.top_right],
      bottom_right: [...CORNERS.bottom_right],
      bottom_left: [...CORNERS.bottom_left],
      popping_crease_left: [612, 638],
      popping_crease_right: [1004, 638],
    },
    corners: {
      top_left: [...CORNERS.top_left],
      top_right: [...CORNERS.top_right],
      bottom_right: [...CORNERS.bottom_right],
      bottom_left: [...CORNERS.bottom_left],
    },
    center: [800, 509],
    homography_available: true,
    orientation: {
      axis: "v",
      batting_end: 1,
      travel_delta: 0.72,
      calibration_frame: 1000,
    },
    source: "demo",
    geometry_reason: null,
  };
}

function uncalibratedPitch(): PitchBlock {
  return {
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
  };
}

interface DemoSpec {
  speedKmh: number | null;
  length: string | null;
  line: string | null;
  swing: string | null;
  shot: string | null;
  confidence: number | null;
  calibrated: boolean;
  bounceX: number;
  bounceY: number;
  swingDirection: number;
}

const SPECS: DemoSpec[] = [
  { speedKmh: 142.6, length: "Good Length", line: "Off Side", swing: "Outswing", shot: "Cover", confidence: 0.82, calibrated: true, bounceX: 0.56, bounceY: 0.58, swingDirection: 0.06 },
  { speedKmh: 131.2, length: "Full", line: "Middle", swing: "Inswing", shot: "Defense", confidence: 0.71, calibrated: true, bounceX: 0.48, bounceY: 0.74, swingDirection: -0.05 },
  { speedKmh: 147.5, length: "Short", line: "Leg Side", swing: "Seam", shot: "Pull", confidence: 0.66, calibrated: true, bounceX: 0.38, bounceY: 0.32, swingDirection: 0.01 },
  { speedKmh: 118.9, length: "Yorker", line: "Off Side", swing: "Inswing", shot: "Straight", confidence: 0.58, calibrated: true, bounceX: 0.52, bounceY: 0.9, swingDirection: -0.04 },
  { speedKmh: null, length: null, line: null, swing: null, shot: null, confidence: null, calibrated: false, bounceX: 0.5, bounceY: 0.5, swingDirection: 0 },
  { speedKmh: 138.4, length: "Good Length", line: "Middle", swing: "Seam", shot: "Cover", confidence: 0.77, calibrated: true, bounceX: 0.5, bounceY: 0.56, swingDirection: 0.01 },
  { speedKmh: 135.1, length: "Good Length", line: "Wide Off", swing: "Outswing", shot: "Late Cut", confidence: 0.52, calibrated: true, bounceX: 0.68, bounceY: 0.52, swingDirection: 0.07 },
  { speedKmh: 128.0, length: "Full", line: "Wide Leg", swing: "Inswing", shot: "Flick", confidence: 0.63, calibrated: true, bounceX: 0.3, bounceY: 0.76, swingDirection: -0.07 },
  { speedKmh: 140.3, length: "Good Length", line: "Off Side", swing: "Seam", shot: "Lofted", confidence: 0.49, calibrated: true, bounceX: 0.6, bounceY: 0.57, swingDirection: 0.02 },
  { speedKmh: 126.7, length: "Short", line: "Middle", swing: "Outswing", shot: null, confidence: null, calibrated: true, bounceX: 0.46, bounceY: 0.35, swingDirection: 0.04 },
];

const START = 1000;

function buildTrajectory(spec: DemoSpec, index: number): {
  points: TrajectoryPoint[];
  bounceFrame: number;
  startFrame: number;
  endFrame: number;
} {
  const startFrame = START + index * 180;
  const bounceFrame = startFrame + 24;
  const endFrame = startFrame + 58;
  const total = endFrame - startFrame;

  const releaseX = 762 + (index % 3) * 6;
  const releaseY = 330;
  const bouncePixelX = 780 + spec.bounceX * 260;
  const bouncePixelY = 420 + spec.bounceY * 250;
  const endX = bouncePixelX + 46;
  const endY = Math.max(300, bouncePixelY - 120);

  const points: TrajectoryPoint[] = [];
  for (let offset = 0; offset <= total; offset += 1) {
    const frame = startFrame + offset;
    const beforeBounce = offset <= 24;
    const t = beforeBounce ? offset / 24 : (offset - 24) / (total - 24);
    const x = beforeBounce
      ? releaseX + (bouncePixelX - releaseX) * t
      : bouncePixelX + (endX - bouncePixelX) * t;
    const y = beforeBounce
      ? releaseY + (bouncePixelY - releaseY) * Math.pow(t, 1.55)
      : bouncePixelY - (bouncePixelY - endY) * Math.pow(t, 0.85);

    // A little lateral drift before bounce, so the drawn path is not a ruler.
    const drift = beforeBounce ? spec.swingDirection * 60 * Math.sin(Math.PI * t) : 0;

    const detected = offset <= 20;
    const pitchFraction = Math.max(0.04, Math.min(0.96, spec.bounceY + (t - 0.6) * 0.08));

    points.push({
      frame,
      x: Math.round(x + drift),
      y: Math.round(y),
      source: detected ? "detected" : "predicted",
      confidence: detected ? Number((0.9 - Math.abs(t - 0.5) * 0.25).toFixed(4)) : null,
      pitch_position: spec.calibrated
        ? {
            x: Math.max(0.05, Math.min(0.95, spec.bounceX + drift / 240)),
            y: pitchFraction,
            inside_pitch: true,
          }
        : null,
    });
  }

  return { points, bounceFrame, startFrame, endFrame };
}

function buildDelivery(spec: DemoSpec, index: number): Delivery {
  const id = index + 1;
  const { points, bounceFrame, startFrame, endFrame } = buildTrajectory(spec, index);
  const pitch = spec.calibrated ? calibratedPitch() : uncalibratedPitch();
  const duration = (endFrame - startFrame + 1) / FPS;

  const bowling: DeliveryBowling = {
    speed_kmh: spec.speedKmh,
    line: spec.line,
    length: spec.length,
    swing: spec.swing,
    release_angle: spec.calibrated ? Number((-88 - index * 0.7).toFixed(1)) : null,
    bounce_angle: spec.calibrated ? Number((14 + index * 1.3).toFixed(1)) : null,
    calibrated: spec.calibrated,
  };

  const shot: DeliveryShot = {
    classification: spec.shot,
    confidence: spec.confidence,
  };

  const bounce: DeliveryBounce = {
    detected: spec.calibrated,
    frame: spec.calibrated ? bounceFrame : null,
    x: spec.calibrated ? points[24]?.x ?? null : null,
    y: spec.calibrated ? points[24]?.y ?? null : null,
    ground_x_m: null,
    ground_y_m: null,
  };

  return {
    id,
    video: {
      // Demo mode ships no media. The UI shows a clearly-labelled synthetic
      // trajectory preview instead of pretending this is footage.
      clip_url: null,
      start_frame: startFrame,
      end_frame: endFrame,
      fps: FPS,
      width: FRAME_WIDTH,
      height: FRAME_HEIGHT,
      duration_seconds: Number(duration.toFixed(2)),
    },
    trajectory: {
      points,
      detected_points: points.filter((point) => point.source === "detected").length,
      predicted_points: points.filter((point) => point.source === "predicted").length,
      continuity: "continuous",
    },
    bounce,
    bowling,
    shot,
    quality: {
      trajectory_valid: true,
      tracking_confidence: spec.calibrated ? 0.78 : 0.61,
      requires_human_review: !spec.calibrated || spec.shot === null,
      flags: [
        ...(spec.calibrated ? [] : ["pitch_calibration_required"]),
        ...(spec.shot === null ? ["shot_model_unavailable"] : []),
      ],
    },
    pitch,
  };
}

/** Deterministic; a fresh document each call so no view can mutate the others. */
export function buildDemoResult(): ResultDocument {
  return {
    analysis_id: DEMO_ANALYSIS_ID,
    status: "completed",
    pitch: calibratedPitch(),
    deliveries: SPECS.map(buildDelivery),
  };
}

export const DEMO_FRAME = { width: FRAME_WIDTH, height: FRAME_HEIGHT, fps: FPS };
