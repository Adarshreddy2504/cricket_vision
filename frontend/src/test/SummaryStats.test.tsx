import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import SummaryStats from "@/components/SummaryStats";
import { adaptResult, summariseDeliveries } from "@/lib/analysis";
import { buildDemoResult } from "@/lib/demo";

describe("SummaryStats", () => {
  it("renders the measured aggregates and names the sample size", () => {
    render(<SummaryStats summary={summariseDeliveries(adaptResult(buildDemoResult()))} />);

    expect(screen.getByTestId("summary-speed")).toHaveTextContent("134.3 km/h");
    expect(screen.getByTestId("summary-speed")).toHaveTextContent(
      "Measured on 9 of 10 deliveries",
    );
    expect(screen.getByTestId("summary-fastest")).toHaveTextContent("147.5 km/h");
    expect(screen.getByTestId("summary-fastest")).toHaveTextContent("Ball 3");
    expect(screen.getByTestId("summary-length")).toHaveTextContent("Good Length");
    expect(screen.getByTestId("summary-line")).toHaveTextContent("Off Side");
    expect(screen.getByTestId("summary-shots")).toHaveTextContent("8/10");
    expect(screen.getByText(/Mean confidence 64\.\d%/)).toBeInTheDocument();
  });

  it("renders dashes and an explicit reason when nothing was measured", () => {
    render(<SummaryStats summary={summariseDeliveries([])} />);

    expect(screen.getByTestId("summary-speed")).toHaveTextContent("—");
    expect(screen.getByTestId("summary-speed")).toHaveTextContent("No speed was measured");
    expect(screen.getByTestId("summary-fastest")).toHaveTextContent("—");
    expect(screen.getByTestId("summary-fastest")).toHaveTextContent("No measured speed");
    expect(screen.getByTestId("summary-length")).toHaveTextContent("—");
    expect(screen.getByTestId("summary-length")).toHaveTextContent("Length not measured");
    expect(screen.getByTestId("summary-line")).toHaveTextContent("Line not measured");
    expect(screen.getByTestId("summary-shots")).toHaveTextContent("0/0");
    expect(screen.getByText(/A dash \(—\) means the run did not measure that value/)).toBeInTheDocument();
  });

  it("shows the processing time only when it is known", () => {
    const summary = summariseDeliveries(adaptResult(buildDemoResult()));
    const { rerender } = render(<SummaryStats summary={summary} />);
    expect(screen.queryByText(/Processed in/)).not.toBeInTheDocument();

    rerender(<SummaryStats summary={summary} processingTime={41.8} />);
    expect(screen.getByText("Processed in 41.8 s")).toBeInTheDocument();
  });
});
