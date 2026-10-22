/**
 * analysis.ts — pure data processing between the API contract and the UI.
 *
 * Components never read `delivery.bowling.speed_kmh` directly; they consume a
 * `DeliveryView` produced here. Keeping the projection in one pure module
 * makes the honesty rules testable:
 *
 *   * `null` in the contract becomes `null` here, never 0 or "Unknown";
 *   * a bounce dot exists only when the record carries a pitch-model position
 *     AND an orientation that says which end is the batting end;
 *   * shares are undefined (empty) when nothing was classified, not 0%.
 */

import type {
  Delivery,
  DeliveryBounce,
  DeliveryQuality,
  DeliveryVideo,
  PitchBlock,
  ResultDocument,
  TrajectoryPoint,
} from "./api";
import { isFiniteNumber } from "./format";

/** The frozen 10-way vocabulary of the Part 3 classifier. */
export const SHOT_CLASSES = [
  "Cover",
  "Defense",
  "Flick",
  "Hook",
  "Late Cut",
  "Lofted",
  "Pull",
  "Square Cut",
  "Straight",
  "Sweep",
] as const;

export type ShotClass = (typeof SHOT_CLASSES)[number];

/** Length labels published by the metric geometry gate. */
export const LENGTH_ZONES = [
  "Yorker",
  "Full",
  "Good Length",
  "Short",
  "Very Short",
] as const;

/** Older records may still carry these two labels. */
export const LEGACY_LENGTH_ZONES = ["Beamer", "Bouncer"] as const;

export const ALL_LENGTH_ZONES = [...LENGTH_ZONES, ...LEGACY_LENGTH_ZONES];

export const LINE_ZONES = ["Wide Leg", "Leg Side", "Middle", "Off Side", "Wide Off"] as const;

export interface DeliveryView {
  /** Stable backend id; quarantined ids are absent, so ids are not renumbered. */
  id: number;
  speedKmh: number | null;
  length: string | null;
  line: string | null;
  swing: string | null;
  shot: string | null;
  shotConfidence: number | null;
  releaseAngle: number | null;
  bounceAngle: number | null;
  /** Bounce position on the pitch map, already oriented (batting end y = 1). */
  bounceX: number | null;
  bounceY: number | null;
  bounceInsidePitch: boolean | null;
  /** The geometry gate accepted this delivery's calibration. */
  calibrated: boolean;
  /** The gate's recorded refusal reason, when it refused. */
  geometryReason: string | null;
  trajectory: TrajectoryPoint[];
  video: DeliveryVideo;
  bounce: DeliveryBounce;
  quality: DeliveryQuality;
  pitch: PitchBlock;
}

function finiteOrNull(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

/**
 * Map a bounce onto the top-down pitch, but only when the record supports it.
 *
 * Required: a pitch-position on the trajectory point at the bounce frame, and
 * an orientation naming the batting end. The old frame-pixel fallback is gone:
 * a pixel is not a place on a pitch, and plotting one invents calibration.
 */
export function bouncePosition(delivery: Delivery): {
  x: number | null;
  y: number | null;
  insidePitch: boolean | null;
} {
  const orientation = delivery.pitch?.orientation;
  if (!orientation) return { x: null, y: null, insidePitch: null };
  if (orientation.axis !== "u" && orientation.axis !== "v") {
    return { x: null, y: null, insidePitch: null };
  }

  const bounceFrame = delivery.bounce?.frame;
  if (!isFiniteNumber(bounceFrame)) return { x: null, y: null, insidePitch: null };

  const point = (delivery.trajectory?.points ?? []).find((entry) => entry.frame === bounceFrame);
  const position = point?.pitch_position;
  if (!position || !isFiniteNumber(position.x) || !isFiniteNumber(position.y)) {
    return { x: null, y: null, insidePitch: null };
  }

  const u = position.x;
  const v = position.y;
  const towardsBattingEnd = orientation.batting_end === 1;
  // The top-down map always has the batting end at y = 1.
  if (orientation.axis === "v") {
    return {
      x: u,
      y: towardsBattingEnd ? v : 1 - v,
      insidePitch: Boolean(position.inside_pitch),
    };
  }
  return {
    x: v,
    y: towardsBattingEnd ? u : 1 - u,
    insidePitch: Boolean(position.inside_pitch),
  };
}

export function adaptDelivery(delivery: Delivery): DeliveryView {
  const bowling = delivery.bowling;
  const bounce = bouncePosition(delivery);
  return {
    id: delivery.id,
    speedKmh: finiteOrNull(bowling?.speed_kmh),
    length: bowling?.length ?? null,
    line: bowling?.line ?? null,
    swing: bowling?.swing ?? null,
    shot: delivery.shot?.classification ?? null,
    shotConfidence: finiteOrNull(delivery.shot?.confidence),
    releaseAngle: finiteOrNull(bowling?.release_angle),
    bounceAngle: finiteOrNull(bowling?.bounce_angle),
    bounceX: bounce.x,
    bounceY: bounce.y,
    bounceInsidePitch: bounce.insidePitch,
    calibrated: Boolean(delivery.pitch?.calibrated),
    geometryReason: delivery.pitch?.geometry_reason ?? null,
    trajectory: Array.isArray(delivery.trajectory?.points) ? delivery.trajectory.points : [],
    video: delivery.video,
    bounce: delivery.bounce,
    quality: delivery.quality,
    pitch: delivery.pitch,
  };
}

export function adaptDeliveries(deliveries: Delivery[]): DeliveryView[] {
  return deliveries.map(adaptDelivery);
}

export function adaptResult(document: ResultDocument): DeliveryView[] {
  return adaptDeliveries(document.deliveries);
}

export interface AnalysisSummary {
  deliveryCount: number;
  measuredSpeeds: number;
  averageSpeedKmh: number | null;
  fastest: { id: number; speedKmh: number } | null;
  slowest: { id: number; speedKmh: number } | null;
  calibrated: number;
  dominantLength: string | null;
  dominantLine: string | null;
  classifiedShots: number;
  meanShotConfidence: number | null;
}

function dominant(values: Array<string | null>): string | null {
  const counts = new Map<string, number>();
  values.forEach((value) => {
    if (value) counts.set(value, (counts.get(value) ?? 0) + 1);
  });
  let best: string | null = null;
  let bestCount = 0;
  counts.forEach((count, value) => {
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  });
  return best;
}

/**
 * Aggregates only over measurements that exist. With three speeds out of ten
 * deliveries, the average is those three; the other seven contribute nothing
 * and are not silently counted as zero.
 */
export function summariseDeliveries(deliveries: DeliveryView[]): AnalysisSummary {
  const measured = deliveries
    .filter((delivery) => isFiniteNumber(delivery.speedKmh))
    .map((delivery) => ({ id: delivery.id, speedKmh: delivery.speedKmh as number }));

  const average =
    measured.length > 0
      ? measured.reduce((total, entry) => total + entry.speedKmh, 0) / measured.length
      : null;

  const fastest = measured.reduce<AnalysisSummary["fastest"]>((best, entry) => {
    if (!best || entry.speedKmh > best.speedKmh) return entry;
    return best;
  }, null);

  const slowest = measured.reduce<AnalysisSummary["slowest"]>((best, entry) => {
    if (!best || entry.speedKmh < best.speedKmh) return entry;
    return best;
  }, null);

  const confidences = deliveries
    .map((delivery) => delivery.shotConfidence)
    .filter((value): value is number => isFiniteNumber(value));

  return {
    deliveryCount: deliveries.length,
    measuredSpeeds: measured.length,
    averageSpeedKmh: average,
    fastest,
    slowest,
    calibrated: deliveries.filter((delivery) => delivery.calibrated).length,
    dominantLength: dominant(deliveries.map((delivery) => delivery.length)),
    dominantLine: dominant(deliveries.map((delivery) => delivery.line)),
    classifiedShots: deliveries.filter((delivery) => delivery.shot !== null).length,
    meanShotConfidence:
      confidences.length > 0
        ? confidences.reduce((total, value) => total + value, 0) / confidences.length
        : null,
  };
}

export interface ShotRow {
  label: string;
  count: number;
  /** Share of classified deliveries, 0..1; null when nothing was classified. */
  share: number | null;
}

export interface ShotDistribution {
  rows: ShotRow[];
  classified: number;
  unclassified: number;
  total: number;
  scored: number;
  meanScore: number | null;
  /** False when the run produced no labels at all. */
  hasLabels: boolean;
  /** Labels returned by the server that are outside the frozen vocabulary. */
  unknownLabels: string[];
}

/**
 * Counts and shares over the shot labels. Every known class is present, even
 * with a count of zero, so a consumer can distinguish "zero" from "not asked".
 * Shares are null when the denominator is zero: a proportion of nothing is
 * undefined, not 0%.
 */
export function shotDistribution(
  deliveries: DeliveryView[],
  classes: readonly string[] = SHOT_CLASSES,
): ShotDistribution {
  const counts = new Map<string, number>();
  classes.forEach((label) => counts.set(label, 0));

  const unknownLabels: string[] = [];
  deliveries.forEach((delivery) => {
    if (delivery.shot === null) return;
    if (!counts.has(delivery.shot)) {
      counts.set(delivery.shot, 0);
      unknownLabels.push(delivery.shot);
    }
    counts.set(delivery.shot, (counts.get(delivery.shot) ?? 0) + 1);
  });

  const classified = deliveries.filter((delivery) => delivery.shot !== null).length;

  const rows: ShotRow[] = Array.from(counts.entries()).map(([label, count]) => ({
    label,
    count,
    share: classified > 0 ? count / classified : null,
  }));

  const scores = deliveries
    .map((delivery) => delivery.shotConfidence)
    .filter((value): value is number => isFiniteNumber(value));

  return {
    rows,
    classified,
    unclassified: deliveries.length - classified,
    total: deliveries.length,
    scored: scores.length,
    meanScore: scores.length > 0 ? scores.reduce((total, value) => total + value, 0) / scores.length : null,
    hasLabels: classified > 0,
    unknownLabels,
  };
}

export interface DeliveryFilters {
  query?: string;
  length?: string | null;
  shot?: string | null;
  lengthZone?: string | null;
}

export function filterDeliveries(
  deliveries: DeliveryView[],
  filters: DeliveryFilters,
): DeliveryView[] {
  const query = (filters.query ?? "").trim().toLowerCase();
  const length = filters.length ?? filters.lengthZone ?? null;
  const shot = filters.shot ?? null;

  return deliveries.filter((delivery) => {
    if (length && delivery.length !== length) return false;
    if (shot && delivery.shot !== shot) return false;
    if (!query) return true;
    return [String(delivery.id), delivery.shot, delivery.length, delivery.line, delivery.swing]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(query));
  });
}

/* ── Processing stages (docs/API_CONTRACT.md §2) ─────────────────────────── */

export const PIPELINE_STAGES = [
  { id: "uploading", label: "Upload", description: "Video accepted" },
  { id: "tracking", label: "Track", description: "Detect and track the ball" },
  { id: "validation", label: "Validate", description: "Check each delivery clip" },
  { id: "shot_classification", label: "Classify", description: "Shot labels and analytics" },
  { id: "overlay", label: "Render", description: "Draw the tracking overlay" },
  { id: "persisting", label: "Save", description: "Write the result document" },
] as const;

export type PipelineStageId = (typeof PIPELINE_STAGES)[number]["id"];
export type StageState = "done" | "active" | "pending" | "failed";

export function stageIndex(stage: string | null | undefined): number {
  if (!stage) return -1;
  return PIPELINE_STAGES.findIndex((entry) => entry.id === stage);
}

export function stageStates(
  stage: string | null | undefined,
  lifecycle: string | null | undefined,
): StageState[] {
  if (lifecycle === "completed") return PIPELINE_STAGES.map(() => "done");
  const index = stageIndex(stage);

  if (lifecycle === "failed") {
    return PIPELINE_STAGES.map((_, position) => {
      if (index < 0) return position === 0 ? "failed" : "pending";
      if (position < index) return "done";
      if (position === index) return "failed";
      return "pending";
    });
  }

  if (lifecycle === "queued") return PIPELINE_STAGES.map(() => "pending");

  return PIPELINE_STAGES.map((_, position) => {
    if (index < 0) return "pending";
    if (position < index) return "done";
    if (position === index) return "active";
    return "pending";
  });
}

/** A safe progress value for ARIA and the progress bar. */
export function clampProgress(value: number | null | undefined): number {
  if (!isFiniteNumber(value)) return 0;
  return Math.max(0, Math.min(100, value));
}
