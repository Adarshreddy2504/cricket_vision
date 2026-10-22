import { describe, expect, it } from "vitest";
import type { Delivery } from "@/lib/api";
import {
  LENGTH_ZONES,
  adaptDeliveries,
  adaptResult,
  bouncePosition,
  clampProgress,
  filterDeliveries,
  shotDistribution,
  stageStates,
  summariseDeliveries,
} from "@/lib/analysis";
import { buildDemoResult } from "@/lib/demo";

function demoViews() {
  return adaptResult(buildDemoResult());
}

function cloneDelivery(delivery: Delivery): Delivery {
  return JSON.parse(JSON.stringify(delivery)) as Delivery;
}

describe("adaptDelivery", () => {
  it("keeps absent measurements absent", () => {
    const views = demoViews();
    const uncalibrated = views.find((view) => view.id === 5);
    expect(uncalibrated).toBeDefined();
    expect(uncalibrated?.speedKmh).toBeNull();
    expect(uncalibrated?.length).toBeNull();
    expect(uncalibrated?.line).toBeNull();
    expect(uncalibrated?.shot).toBeNull();
    expect(uncalibrated?.bounceX).toBeNull();
    expect(uncalibrated?.calibrated).toBe(false);
  });

  it("maps recorded values without converting them", () => {
    const views = demoViews();
    const first = views[0];
    expect(first.speedKmh).toBe(142.6);
    expect(first.length).toBe("Good Length");
    expect(first.shot).toBe("Cover");
    expect(first.shotConfidence).toBe(0.82);
    expect(first.trajectory.length).toBeGreaterThan(0);
  });
});

describe("bouncePosition", () => {
  it("uses the pitch-model position and the batting-end orientation", () => {
    const delivery = cloneDelivery(buildDemoResult().deliveries[0]);
    const position = bouncePosition(delivery);
    expect(position.x).not.toBeNull();
    expect(position.y).not.toBeNull();
    expect(position.x).toBeGreaterThanOrEqual(0);
    expect(position.x).toBeLessThanOrEqual(1);
    expect(position.y).toBeGreaterThanOrEqual(0);
    expect(position.y).toBeLessThanOrEqual(1);
  });

  it("flips the v axis when the batting end is the far end", () => {
    const delivery = cloneDelivery(buildDemoResult().deliveries[0]);
    delivery.pitch.orientation = {
      axis: "v",
      batting_end: 0,
      travel_delta: 0.5,
      calibration_frame: 1,
    };
    const point = delivery.trajectory.points.find(
      (entry) => entry.frame === delivery.bounce.frame,
    );
    expect(point).toBeDefined();
    if (point) point.pitch_position = { x: 0.4, y: 0.2, inside_pitch: true };

    expect(bouncePosition(delivery)).toEqual({ x: 0.4, y: 0.8, insidePitch: true });
  });

  it("maps the u axis into the top-down map", () => {
    const delivery = cloneDelivery(buildDemoResult().deliveries[0]);
    delivery.pitch.orientation = {
      axis: "u",
      batting_end: 1,
      travel_delta: 0.5,
      calibration_frame: 1,
    };
    const point = delivery.trajectory.points.find(
      (entry) => entry.frame === delivery.bounce.frame,
    );
    if (point) point.pitch_position = { x: 0.2, y: 0.4, inside_pitch: true };

    expect(bouncePosition(delivery)).toMatchObject({ x: 0.4, y: 0.2 });
  });

  it("refuses to invent a dot without an orientation or a bounce position", () => {
    const noOrientation = cloneDelivery(buildDemoResult().deliveries[0]);
    noOrientation.pitch.orientation = null;
    expect(bouncePosition(noOrientation)).toEqual({
      x: null,
      y: null,
      insidePitch: null,
    });

    const noPosition = cloneDelivery(buildDemoResult().deliveries[0]);
    noPosition.trajectory.points.forEach((point) => {
      point.pitch_position = null;
    });
    expect(bouncePosition(noPosition)).toEqual({
      x: null,
      y: null,
      insidePitch: null,
    });
  });
});

describe("summariseDeliveries", () => {
  it("averages only measured speeds and names the fastest measured ball", () => {
    const summary = summariseDeliveries(demoViews());
    expect(summary.deliveryCount).toBe(10);
    expect(summary.measuredSpeeds).toBe(9);
    expect(summary.averageSpeedKmh).toBeCloseTo(134.3, 1);
    expect(summary.fastest).toEqual({ id: 3, speedKmh: 147.5 });
    expect(summary.slowest).toEqual({ id: 4, speedKmh: 118.9 });
    expect(summary.calibrated).toBe(9);
    expect(summary.dominantLength).toBe("Good Length");
    expect(summary.dominantLine).toBe("Off Side");
    expect(summary.classifiedShots).toBe(8);
    expect(summary.meanShotConfidence).toBeCloseTo(0.6475, 3);
  });

  it("reports nothing rather than zero for an empty analysis", () => {
    const summary = summariseDeliveries([]);
    expect(summary).toMatchObject({
      deliveryCount: 0,
      measuredSpeeds: 0,
      averageSpeedKmh: null,
      fastest: null,
      slowest: null,
      dominantLength: null,
      dominantLine: null,
      classifiedShots: 0,
      meanShotConfidence: null,
    });
  });
});

describe("shotDistribution", () => {
  it("counts labels and keeps zero-count vocabulary rows", () => {
    const distribution = shotDistribution(demoViews());
    expect(distribution.total).toBe(10);
    expect(distribution.classified).toBe(8);
    expect(distribution.unclassified).toBe(2);
    expect(distribution.hasLabels).toBe(true);

    const cover = distribution.rows.find((row) => row.label === "Cover");
    expect(cover).toEqual({ label: "Cover", count: 2, share: 0.25 });

    const sweep = distribution.rows.find((row) => row.label === "Sweep");
    expect(sweep).toEqual({ label: "Sweep", count: 0, share: 0 });

    const shares = distribution.rows.reduce((total, row) => total + (row.share ?? 0), 0);
    expect(shares).toBeCloseTo(1, 6);
    expect(distribution.meanScore).toBeCloseTo(0.6475, 3);
    expect(distribution.unknownLabels).toEqual([]);
  });

  it("reports undefined shares when nothing was classified", () => {
    const views = demoViews().map((view) => ({
      ...view,
      shot: null,
      shotConfidence: null,
    }));
    const distribution = shotDistribution(views);
    expect(distribution.classified).toBe(0);
    expect(distribution.hasLabels).toBe(false);
    expect(distribution.meanScore).toBeNull();
    expect(distribution.rows.every((row) => row.share === null)).toBe(true);
    expect(distribution.rows.every((row) => row.count === 0)).toBe(true);
  });

  it("counts labels outside the frozen vocabulary instead of hiding them", () => {
    const views = demoViews();
    views[0] = { ...views[0], shot: "Switch Hit" };
    const distribution = shotDistribution(views);
    expect(distribution.unknownLabels).toEqual(["Switch Hit"]);
    expect(distribution.rows.find((row) => row.label === "Switch Hit")?.count).toBe(1);
    expect(distribution.classified).toBe(8);
  });
});

describe("filterDeliveries", () => {
  it("filters by query, length and shot", () => {
    const views = demoViews();
    expect(filterDeliveries(views, { query: "ball 4" })).toHaveLength(0);
    expect(filterDeliveries(views, { query: "4" }).map((view) => view.id)).toContain(4);
    expect(filterDeliveries(views, { query: "cover" }).map((view) => view.id)).toEqual([1, 6]);
    expect(filterDeliveries(views, { length: "Yorker" }).map((view) => view.id)).toEqual([4]);
    expect(filterDeliveries(views, { shot: "No such shot" })).toEqual([]);
  });

  it("accepts the legacy lengthZone field", () => {
    const views = demoViews();
    expect(filterDeliveries(views, { lengthZone: "Short" }).map((view) => view.id)).toEqual([3, 10]);
  });
});

describe("stageStates", () => {
  it("marks every stage done for a completed analysis", () => {
    expect(stageStates("persisting", "completed")).toEqual([
      "done",
      "done",
      "done",
      "done",
      "done",
      "done",
    ]);
  });

  it("marks the reported stage active and earlier stages done", () => {
    expect(stageStates("shot_classification", "running")).toEqual([
      "done",
      "done",
      "done",
      "active",
      "pending",
      "pending",
    ]);
  });

  it("marks the reported stage failed", () => {
    expect(stageStates("overlay", "failed")).toEqual([
      "done",
      "done",
      "done",
      "done",
      "failed",
      "pending",
    ]);
  });

  it("shows queued work as pending rather than pretending it is active", () => {
    expect(stageStates(null, "queued").every((state) => state === "pending")).toBe(true);
  });
});

describe("clampProgress", () => {
  it("keeps reported progress inside the drawable range", () => {
    expect(clampProgress(null)).toBe(0);
    expect(clampProgress(Number.NaN)).toBe(0);
    expect(clampProgress(-5)).toBe(0);
    expect(clampProgress(42)).toBe(42);
    expect(clampProgress(150)).toBe(100);
  });
});

describe("length vocabulary", () => {
  it("ships the five published zones", () => {
    expect(Array.from(LENGTH_ZONES)).toEqual([
      "Yorker",
      "Full",
      "Good Length",
      "Short",
      "Very Short",
    ]);
  });
});

describe("adaptDeliveries", () => {
  it("keeps the input order and identity", () => {
    const document = buildDemoResult();
    const views = adaptDeliveries(document.deliveries);
    expect(views.map((view) => view.id)).toEqual(document.deliveries.map((entry) => entry.id));
  });
});
