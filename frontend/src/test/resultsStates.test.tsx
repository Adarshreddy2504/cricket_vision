import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Results from "@/pages/Results";
import { buildDemoResult, DEMO_ANALYSIS_ID } from "@/lib/demo";
import { cacheResult } from "@/lib/storage";
import { jsonResponse, resultFor } from "./helpers";

function renderResults(state?: Record<string, unknown>, path = "/results") {
  return render(
    <MemoryRouter initialEntries={[{ pathname: path, state }]}>
      <Routes>
        <Route path="/results" element={<Results />} />
        <Route path="/upload" element={<div>upload target</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  localStorage.clear();
});

describe("Results — demo mode", () => {
  it("labels the synthetic result everywhere and renders the full dashboard", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    renderResults({ analysis_id: DEMO_ANALYSIS_ID, demo: true });

    expect(screen.getByTestId("demo-banner")).toHaveTextContent(
      "Demo data — not a real analysis",
    );
    expect(
      screen.getByRole("heading", { name: "Demo delivery analysis" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Analysis summary")).toBeInTheDocument();
    expect(screen.getByText("Pitch map")).toBeInTheDocument();
    expect(screen.getByText("Shot distribution")).toBeInTheDocument();
    expect(screen.getByText("Delivery explorer")).toBeInTheDocument();
    expect(screen.getAllByText("Ball 1").length).toBeGreaterThan(0);
    expect(
      screen.getByRole("img", { name: /Synthetic pitch preview — not real footage/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/SYNTHETIC PREVIEW/)).toBeInTheDocument();

    await Promise.resolve();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders dashes for the uncalibrated delivery rather than filling a metric", () => {
    renderResults({ analysis_id: DEMO_ANALYSIS_ID, demo: true });

    const uncalibratedCard = screen.getByTestId("delivery-card-5");
    expect(uncalibratedCard).toHaveTextContent("uncalibrated");
    expect(uncalibratedCard).toHaveTextContent("no shot label");
    expect(uncalibratedCard).toHaveTextContent("—");
  });

  it("exports a wrapped demo payload with the notice (not a genuine result)", async () => {
    const createObjectURL = vi.mocked(URL.createObjectURL);
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    renderResults({ analysis_id: DEMO_ANALYSIS_ID, demo: true });

    const button = screen.getByRole("button", { name: /Export JSON/i });
    button.click();

    expect(createObjectURL).toHaveBeenCalled();
    const blob = createObjectURL.mock.calls[createObjectURL.mock.calls.length - 1][0] as Blob;
    const text = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(blob);
    });
    const payload = JSON.parse(text) as { demo?: boolean; notice?: string; result?: unknown };
    expect(payload.demo).toBe(true);
    expect(payload.notice).toContain("synthetic");
    expect(payload.result).toBeTruthy();
    expect(click).toHaveBeenCalled();
  });
});

describe("Results — truthful empty/error states", () => {
  it("explains a completed run with no deliveries instead of drawing an empty dashboard", async () => {
    const document = resultFor("empty-1");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(document)));

    renderResults({ analysis_id: "empty-1" });

    expect(await screen.findByText("No deliveries detected")).toBeInTheDocument();
    expect(screen.queryByText("Analysis summary")).not.toBeInTheDocument();
  });

  it("reports a contract mismatch as a contract mismatch", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ analysis_id: "bad-1", status: "completed", pitch: {}, deliveries: "no" }),
      ),
    );

    renderResults({ analysis_id: "bad-1" });

    expect(
      await screen.findByText("The result did not match the API contract"),
    ).toBeInTheDocument();
    expect(screen.getByText(/does not match API contract/)).toBeInTheDocument();
  });

  it("shows the cached result with a warning when the refresh fails", async () => {
    const cached = buildDemoResult();
    cached.analysis_id = "live-1";
    cacheResult(cached, 12.5);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    renderResults({ analysis_id: "live-1" });

    expect(screen.getByRole("heading", { name: "Ball-by-ball analysis" })).toBeInTheDocument();
    expect(await screen.findByText("Showing the cached result")).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the backend/)).toBeInTheDocument();
  });

  it("asks for an upload when no analysis id exists", () => {
    renderResults();

    expect(screen.getByText("No result to show")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Analyse a video" })).toHaveAttribute(
      "href",
      "/upload",
    );
  });
});
