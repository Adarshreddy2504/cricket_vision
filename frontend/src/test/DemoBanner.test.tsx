import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import DemoBanner from "@/components/DemoBanner";
import { DEMO_NOTICE } from "@/lib/demo";

describe("DemoBanner", () => {
  it("states that the data is demo data and repeats the notice", () => {
    render(
      <MemoryRouter>
        <DemoBanner />
      </MemoryRouter>,
    );

    const banner = screen.getByTestId("demo-banner");
    expect(banner).toHaveTextContent("Demo data — not a real analysis");
    expect(banner).toHaveTextContent(DEMO_NOTICE);
    expect(screen.getByRole("status")).toBe(banner);
  });

  it("links back to a real upload", () => {
    render(
      <MemoryRouter>
        <DemoBanner />
      </MemoryRouter>,
    );

    expect(screen.getByRole("link", { name: /Analyse a real video/i })).toHaveAttribute(
      "href",
      "/upload",
    );
  });
});
