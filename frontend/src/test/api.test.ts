import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  CONTRACT_SCHEMA_VERSION,
  apiUrl,
  fetchHealth,
  fetchResult,
  fetchStatus,
  isResultDocument,
  mediaUrl,
  parseResultDocument,
} from "@/lib/api";
import { buildDemoResult } from "@/lib/demo";
import { emptyResponse, jsonResponse } from "./helpers";

function catchApiError(action: () => unknown): ApiError {
  try {
    action();
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error("expected an ApiError to be thrown");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseResultDocument", () => {
  it("accepts the contract-shaped demo document", () => {
    const document = buildDemoResult();
    const parsed = parseResultDocument(document, document.analysis_id);
    expect(parsed.deliveries).toHaveLength(10);
    expect(isResultDocument(parsed, document.analysis_id)).toBe(true);
  });

  it("rejects an envelope for a different analysis id", () => {
    const error = catchApiError(() => parseResultDocument(buildDemoResult(), "someone-else"));
    expect(error.kind).toBe("contract");
    expect(error.status).toBe(200);
    expect(error.message).toContain(`contract ${CONTRACT_SCHEMA_VERSION}`);
    expect(error.message).toContain("instead of someone-else");
  });

  it("rejects a document without deliveries", () => {
    const error = catchApiError(() =>
      parseResultDocument({ analysis_id: "a", status: "completed", pitch: {} }),
    );
    expect(error.kind).toBe("contract");
    expect(error.message).toContain("deliveries is not an array");
  });

  it("rejects a delivery without the required blocks", () => {
    const document = buildDemoResult() as unknown as {
      analysis_id: string;
      status: string;
      pitch: unknown;
      deliveries: Array<Record<string, unknown>>;
    };
    delete (document.deliveries[0] as Record<string, unknown>).pitch;
    const error = catchApiError(() => parseResultDocument(document));
    expect(error.kind).toBe("contract");
    expect(error.message).toContain("deliveries[0].pitch is missing");
  });

  it("rejects a trajectory that is not an array of points", () => {
    const document = buildDemoResult() as unknown as {
      deliveries: Array<{ trajectory: { points: unknown } }>;
    };
    document.deliveries[0].trajectory.points = null;
    const error = catchApiError(() => parseResultDocument(document));
    expect(error.message).toContain("deliveries[0].trajectory.points is not an array");
  });

  it("isResultDocument reports false instead of throwing", () => {
    expect(isResultDocument(null)).toBe(false);
    expect(isResultDocument({ analysis_id: "" })).toBe(false);
    expect(isResultDocument(buildDemoResult())).toBe(true);
  });
});

describe("URL helpers", () => {
  it("joins api paths exactly once", () => {
    expect(apiUrl("/health")).toMatch(/\/api\/health$/);
    expect(apiUrl("health")).toMatch(/\/api\/health$/);
  });

  it("makes root-relative clip paths absolute", () => {
    const absolute = mediaUrl("/api/analysis/a/clips/delivery_001.mp4");
    expect(absolute).toMatch(/\/api\/analysis\/a\/clips\/delivery_001\.mp4$/);
    expect(mediaUrl("api/analysis/a/clips/x.mp4")).toMatch(/\/api\/analysis\/a\/clips\/x\.mp4$/);
    expect(mediaUrl("https://cdn.example.com/x.mp4")).toBe("https://cdn.example.com/x.mp4");
    expect(mediaUrl(null)).toBeNull();
    expect(mediaUrl("")).toBeNull();
  });
});

describe("fetchResult", () => {
  it("requests the documented result route and parses the envelope", async () => {
    const document = buildDemoResult();
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(document));
    vi.stubGlobal("fetch", fetchMock);

    const parsed = await fetchResult(document.analysis_id);

    expect(parsed.analysis_id).toBe(document.analysis_id);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toMatch(
      /\/api\/analysis\/demo-analysis\/result$/,
    );
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "GET" });
  });

  it("encodes the analysis id in the route", async () => {
    const document = buildDemoResult();
    document.analysis_id = "a/b";
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(document));
    vi.stubGlobal("fetch", fetchMock);

    await fetchResult("a/b");

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/analysis\/a%2Fb\/result$/);
  });

  it("reports the FastAPI detail on an HTTP failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: [{ msg: "field required" }, { msg: "not a video" }] }, 400),
      ),
    );

    await expect(fetchResult("a")).rejects.toMatchObject({
      kind: "http",
      status: 400,
      message: "field required; not a video",
    });
  });

  it("reports a connection failure instead of inventing a result", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(fetchResult("a")).rejects.toMatchObject({ kind: "network" });
    await expect(fetchResult("a")).rejects.toThrow(/Cannot reach the backend/);
  });

  it("rejects an empty or non-JSON body as a contract failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse(200)));

    await expect(fetchResult("a")).rejects.toMatchObject({ kind: "contract" });
  });
});

describe("fetchStatus and fetchHealth", () => {
  it("validates that a status response names an analysis", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "running" })));
    await expect(fetchStatus("a")).rejects.toMatchObject({ kind: "contract" });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ analysis_id: "a", status: "running", stage: "tracking", progress: 42 }),
      ),
    );
    await expect(fetchStatus("a")).resolves.toMatchObject({
      analysis_id: "a",
      status: "running",
      stage: "tracking",
      progress: 42,
    });
  });

  it("validates the health document", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ status: "ok", schema_version: "2.1", stages: ["tracking"] }),
      ),
    );
    await expect(fetchHealth()).resolves.toMatchObject({ status: "ok", schema_version: "2.1" });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({})));
    await expect(fetchHealth()).rejects.toMatchObject({ kind: "contract" });
  });
});
