import { describe, expect, it } from "vitest";
import {
  DASH,
  formatAngle,
  formatBytes,
  formatClock,
  formatCount,
  formatDuration,
  formatNumber,
  formatPercent,
  formatSpeed,
  isFiniteNumber,
  titleCase,
} from "@/lib/format";

describe("format", () => {
  it("renders absent numbers as a dash rather than zero", () => {
    expect(formatNumber(null)).toBe(DASH);
    expect(formatNumber(undefined)).toBe(DASH);
    expect(formatNumber(Number.NaN)).toBe(DASH);
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe(DASH);
    expect(formatSpeed(null)).toBe(DASH);
    expect(formatPercent(undefined)).toBe(DASH);
    expect(formatAngle(null)).toBe(DASH);
  });

  it("formats measured values with their units", () => {
    expect(formatNumber(12.345, 1)).toBe("12.3");
    expect(formatSpeed(141.27)).toBe("141.3 km/h");
    expect(formatPercent(0.734, 1)).toBe("73.4%");
    expect(formatAngle(13.25)).toBe("13.3°");
    expect(formatCount(1, "ball")).toBe("1 ball");
    expect(formatCount(4, "ball")).toBe("4 balls");
    expect(formatCount(2, "delivery", "deliveries")).toBe("2 deliveries");
  });

  it("formats byte counts without inventing a value", () => {
    expect(formatBytes(null)).toBe(DASH);
    expect(formatBytes(-1)).toBe(DASH);
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(900)).toBe("900 B");
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(2 * 1024 * 1024 * 1024)).toBe("2.0 GB");
  });

  it("formats durations and clip clocks", () => {
    expect(formatDuration(null)).toBe(DASH);
    expect(formatDuration(3.84)).toBe("0:04");
    expect(formatDuration(65)).toBe("1:05");
    expect(formatDuration(3725)).toBe("1:02:05");
    expect(formatClock(3.84)).toBe("0:03.8");
    expect(formatClock(null)).toBe(DASH);
  });

  it("narrows only finite numbers", () => {
    expect(isFiniteNumber(0)).toBe(true);
    expect(isFiniteNumber(Number.NaN)).toBe(false);
    expect(isFiniteNumber("3")).toBe(false);
    expect(isFiniteNumber(null)).toBe(false);
  });

  it("turns snake_case labels into readable text", () => {
    expect(titleCase("shot_classification")).toBe("Shot Classification");
  });
});
