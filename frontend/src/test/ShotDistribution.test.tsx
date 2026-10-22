import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ShotDistribution from "@/components/ShotDistribution";
import { adaptResult, shotDistribution } from "@/lib/analysis";
import { buildDemoResult } from "@/lib/demo";

describe("ShotDistribution", () => {
  it("shows counts, shares and confidence for a classified run", () => {
    const distribution = shotDistribution(adaptResult(buildDemoResult()));
    render(<ShotDistribution distribution={distribution} />);

    expect(screen.getByTestId("shot-distribution")).toBeInTheDocument();
    expect(screen.getByText("8 classified")).toBeInTheDocument();
    expect(screen.getByText("2 without a label")).toBeInTheDocument();
    expect(screen.getByText("Cover")).toBeInTheDocument();
    expect(screen.getByText("2 balls · 25%")).toBeInTheDocument();
    expect(screen.getByText("Sweep").closest("li")).toHaveTextContent("0 balls · 0%");
    expect(screen.getByText(/mean confidence 64\.\d%/)).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Shot label distribution" })).toBeInTheDocument();
  });

  it("states that no labels were returned instead of drawing zero bars", () => {
    const deliveries = adaptResult(buildDemoResult()).map((delivery) => ({
      ...delivery,
      shot: null,
      shotConfidence: null,
    }));
    render(<ShotDistribution distribution={shotDistribution(deliveries)} />);

    expect(screen.getByText("No shot labels were returned")).toBeInTheDocument();
    expect(screen.queryByTestId("shot-distribution")).not.toBeInTheDocument();
  });

  it("states that there are no deliveries to chart", () => {
    render(<ShotDistribution distribution={shotDistribution([])} />);
    expect(screen.getByText("No deliveries to chart")).toBeInTheDocument();
  });

  it("counts a label outside the vocabulary rather than dropping it", () => {
    const deliveries = adaptResult(buildDemoResult());
    deliveries[0] = { ...deliveries[0], shot: "Switch Hit" };
    render(<ShotDistribution distribution={shotDistribution(deliveries)} />);

    expect(screen.getByText(/outside the documented vocabulary/i)).toHaveTextContent(
      "Switch Hit",
    );
    expect(screen.getByText("Switch Hit")).toBeInTheDocument();
  });
});
