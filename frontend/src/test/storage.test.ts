import { beforeEach, describe, expect, it } from "vitest";
import { buildDemoResult, DEMO_ANALYSIS_ID } from "@/lib/demo";
import {
  cacheResult,
  clearCachedResult,
  clearStoredAnalysis,
  getStoredAnalysisId,
  hasCachedResult,
  loadCachedResult,
  setStoredAnalysisId,
} from "@/lib/storage";

describe("storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("remembers the current analysis id", () => {
    expect(getStoredAnalysisId()).toBeNull();
    setStoredAnalysisId("analysis-123");
    expect(getStoredAnalysisId()).toBe("analysis-123");
  });

  it("returns a cached envelope only for its own analysis id", () => {
    const result = buildDemoResult();
    cacheResult(result, 12.5);

    const cached = loadCachedResult(DEMO_ANALYSIS_ID);
    expect(cached).not.toBeNull();
    expect(cached?.analysis_id).toBe(DEMO_ANALYSIS_ID);
    expect(cached?.deliveries).toHaveLength(10);
    expect(cached?.processing_time).toBe(12.5);

    expect(loadCachedResult("another-analysis")).toBeNull();
    expect(hasCachedResult("another-analysis")).toBe(false);
    expect(hasCachedResult(DEMO_ANALYSIS_ID)).toBe(true);
  });

  it("does not trust a corrupted cache entry", () => {
    localStorage.setItem("cricketvision:result", "{not json");
    expect(loadCachedResult(DEMO_ANALYSIS_ID)).toBeNull();
    expect(hasCachedResult(DEMO_ANALYSIS_ID)).toBe(false);
  });

  it("rejects a cached envelope whose analysis id does not match", () => {
    const result = buildDemoResult();
    result.analysis_id = "other-analysis";
    localStorage.setItem("cricketvision:result", JSON.stringify(result));
    expect(loadCachedResult(DEMO_ANALYSIS_ID)).toBeNull();
    expect(loadCachedResult("other-analysis")).not.toBeNull();
  });

  it("clears the cached envelope without forgetting the current id", () => {
    setStoredAnalysisId("analysis-123");
    cacheResult(buildDemoResult());
    clearCachedResult();
    expect(getStoredAnalysisId()).toBe("analysis-123");
    expect(loadCachedResult(DEMO_ANALYSIS_ID)).toBeNull();
  });

  it("clears both the id and the cache when a new analysis starts", () => {
    setStoredAnalysisId(DEMO_ANALYSIS_ID);
    cacheResult(buildDemoResult());
    clearStoredAnalysis();
    expect(getStoredAnalysisId()).toBeNull();
    expect(loadCachedResult(DEMO_ANALYSIS_ID)).toBeNull();
  });
});
