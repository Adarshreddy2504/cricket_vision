import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Processing from "@/pages/Processing";
import { jsonResponse } from "./helpers";

function renderProcessing(state?: Record<string, unknown>) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: "/processing", state }]}>
      <Routes>
        <Route path="/processing" element={<Processing />} />
        <Route path="/results" element={<div>result target</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Processing", () => {
  it("asks for a video when no analysis id is available", () => {
    renderProcessing();
    expect(screen.getByText("No analysis to track")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Choose a video" })).toHaveAttribute(
      "href",
      "/upload",
    );
  });

  it("shows the backend's failure reason and stops polling", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        analysis_id: "run-1",
        status: "failed",
        stage: "tracking",
        error: "The video could not be decoded.",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderProcessing({ analysis_id: "run-1" });

    expect(
      await screen.findByRole("heading", { name: "Analysis failed" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("processing-failed")).toHaveTextContent(
      "The video could not be decoded.",
    );
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("shows the reported stage and an indeterminate progress bar when no percentage exists", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        analysis_id: "run-2",
        status: "running",
        stage: "tracking",
        progress: null,
        detail: "Tracking the ball through the clip.",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const { unmount } = renderProcessing({ analysis_id: "run-2" });

    expect(await screen.findByText("Tracking the ball through the clip.")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Analysis progress" })).toHaveAttribute(
      "data-indeterminate",
      "true",
    );
    unmount();
  });

  it("reports a missing analysis id from the backend with a fatal state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "analysis not found" }, 404)),
    );

    renderProcessing({ analysis_id: "gone-1" });

    expect(await screen.findByText("This analysis cannot be tracked")).toBeInTheDocument();
    expect(screen.getByText(/No analysis with id gone-1 exists/)).toBeInTheDocument();
  });
});
