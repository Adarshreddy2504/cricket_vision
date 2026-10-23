/**
 * storage.ts — the browser's memory of the current analysis.
 *
 * The analysis id and the cached result are deliberately separate entries and
 * are only trusted together: a cached envelope is returned only for the id it
 * was written under. Starting a new analysis clears the old envelope so the
 * previous run can never paint while the new one is processing.
 */

import { isResultDocument, type CachedResult, type ResultDocument } from "./api";

const ANALYSIS_ID_KEY = "cricketvision:analysis-id";
const RESULT_KEY = "cricketvision:result";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private mode or a full quota: navigation state still carries the id.
  }
}

function remove(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // Nothing to clear.
  }
}

export function getStoredAnalysisId(): string | null {
  return read(ANALYSIS_ID_KEY);
}

export function setStoredAnalysisId(id: string): void {
  write(ANALYSIS_ID_KEY, id);
}

/** Start over: no current id and no cached envelope. */
export function clearStoredAnalysis(): void {
  remove(ANALYSIS_ID_KEY);
  remove(RESULT_KEY);
}

export function cacheResult(result: ResultDocument, processingTime?: number): void {
  const value: CachedResult = { ...result, ...(processingTime ? { processing_time: processingTime } : {}) };
  write(RESULT_KEY, JSON.stringify(value));
}

export function clearCachedResult(): void {
  remove(RESULT_KEY);
}

/**
 * The cached envelope for `analysisId`, or null. A copy belonging to another
 * analysis is not this analysis's result, however recent it is.
 */
export function loadCachedResult(analysisId: string | null | undefined): CachedResult | null {
  if (!analysisId) return null;
  const raw = read(RESULT_KEY);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isResultDocument(parsed, analysisId)) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function hasCachedResult(analysisId: string | null | undefined): boolean {
  return loadCachedResult(analysisId) !== null;
}
