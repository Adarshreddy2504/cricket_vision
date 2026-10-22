import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import UploadPage from "@/pages/Upload";
import { DEMO_ANALYSIS_ID } from "@/lib/demo";

function renderUpload() {
  return render(
    <MemoryRouter initialEntries={["/upload"]}>
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/results" element={<div>demo result target</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("UploadPage", () => {
  it("shows a specific error for an unsupported file and does not stage it", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderUpload();

    await user.upload(
      screen.getByTestId("video-input"),
      new File(["not a video"], "notes.txt", { type: "text/plain" }),
    );

    expect(await screen.findByTestId("validation-error")).toHaveTextContent(
      "Unsupported file type",
    );
    expect(screen.queryByText("ready to upload")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Start analysis/i })).not.toBeInTheDocument();
    expect(screen.getByText("Drop a video here")).toBeInTheDocument();
  });

  it("stages a valid video, shows a local preview and can remove it", async () => {
    const user = userEvent.setup();
    renderUpload();

    await user.upload(
      screen.getByTestId("video-input"),
      new File(["video-bytes"], "clip.mp4", { type: "video/mp4" }),
    );

    expect(screen.getByText("clip.mp4")).toBeInTheDocument();
    expect(screen.getByText("ready to upload")).toBeInTheDocument();
    expect(screen.getByLabelText("Local preview of clip.mp4")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start analysis/i })).toBeInTheDocument();
    expect(screen.queryByTestId("validation-error")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Remove/i }));
    expect(screen.getByText("Drop a video here")).toBeInTheDocument();
    expect(screen.queryByText("clip.mp4")).not.toBeInTheDocument();
  });

  it("opens the labelled demo result from the upload page", async () => {
    const user = userEvent.setup();
    renderUpload();

    await user.click(screen.getByRole("button", { name: /Open demo result/i }));

    expect(await screen.findByText("demo result target")).toBeInTheDocument();
    expect(localStorage.getItem("cricketvision:analysis-id")).toBe(DEMO_ANALYSIS_ID);
  });

  it("rejects a file with a video extension but a non-video MIME type", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderUpload();

    await user.upload(
      screen.getByTestId("video-input"),
      new File(["plain text"], "pretend.mp4", { type: "text/plain" }),
    );

    expect(await screen.findByTestId("validation-error")).toHaveTextContent(
      "does not look like a video",
    );
  });
});
