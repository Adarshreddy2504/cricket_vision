/**
 * format.ts — presentation-only helpers. Kept free of React and fetch so they
 * can be unit-tested directly and reused by every view.
 *
 * The most important rule lives here: an absent measurement renders as "—".
 * It is never "0", "N/A" or a word such as "Unknown", because those read as
 * answers the run produced when in fact it produced nothing.
 */

export const DASH = "—";

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  return isFiniteNumber(value) ? value.toFixed(digits) : DASH;
}

export function formatSpeed(value: number | null | undefined): string {
  return isFiniteNumber(value) ? `${value.toFixed(1)} km/h` : DASH;
}

export function formatPercent(value: number | null | undefined, digits = 0): string {
  return isFiniteNumber(value) ? `${(value * 100).toFixed(digits)}%` : DASH;
}

export function formatAngle(value: number | null | undefined): string {
  return isFiniteNumber(value) ? `${value.toFixed(1)}°` : DASH;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!isFiniteNumber(bytes) || bytes < 0) return DASH;
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

/** "1:05" or "1:02:03" for longer recordings. */
export function formatDuration(seconds: number | null | undefined): string {
  if (!isFiniteNumber(seconds) || seconds < 0) return DASH;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

/** "0:03.4" — used by the clip timeline, where tenths are meaningful. */
export function formatClock(seconds: number | null | undefined): string {
  if (!isFiniteNumber(seconds) || seconds < 0) return DASH;
  const whole = Math.floor(seconds);
  const tenths = Math.floor((seconds - whole) * 10);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}.${tenths}`;
}

export function formatCount(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function titleCase(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
