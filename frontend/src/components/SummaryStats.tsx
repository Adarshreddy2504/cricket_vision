/**
 * SummaryStats — aggregates computed only from the deliveries on the page.
 * Nothing is read from storage or a previous analysis. With three measured
 * speeds out of ten deliveries, the average is those three; the other seven
 * contribute nothing and the card says so.
 */

import { Crosshair, Gauge, Layers, MapPin, Target, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { AnalysisSummary } from "@/lib/analysis";
import { DASH, formatNumber, formatPercent, formatSpeed } from "@/lib/format";

interface Props {
  summary: AnalysisSummary;
  processingTime?: number;
}

const SummaryStats = ({ summary, processingTime }: Props) => {
  const cards = [
    {
      key: "deliveries",
      label: "Deliveries",
      value: String(summary.deliveryCount),
      detail:
        summary.deliveryCount === 0
          ? "No deliveries in this result"
          : `${summary.calibrated} of ${summary.deliveryCount} with calibrated geometry`,
      icon: Layers,
    },
    {
      key: "speed",
      label: "Average speed",
      value: formatSpeed(summary.averageSpeedKmh),
      detail:
        summary.measuredSpeeds === 0
          ? "No speed was measured"
          : `Measured on ${summary.measuredSpeeds} of ${summary.deliveryCount} deliveries`,
      icon: Gauge,
    },
    {
      key: "fastest",
      label: "Fastest ball",
      value: summary.fastest ? formatSpeed(summary.fastest.speedKmh) : DASH,
      detail: summary.fastest ? `Ball ${summary.fastest.id}` : "No measured speed",
      icon: Zap,
    },
    {
      key: "length",
      label: "Most common length",
      value: summary.dominantLength ?? DASH,
      detail: summary.dominantLength ? "Among measured deliveries" : "Length not measured",
      icon: MapPin,
    },
    {
      key: "line",
      label: "Most common line",
      value: summary.dominantLine ?? DASH,
      detail: summary.dominantLine ? "Among measured deliveries" : "Line not measured",
      icon: Target,
    },
    {
      key: "shots",
      label: "Shot labels",
      value: `${summary.classifiedShots}/${summary.deliveryCount}`,
      detail:
        summary.meanShotConfidence !== null
          ? `Mean confidence ${formatPercent(summary.meanShotConfidence, 1)}`
          : "No classifier confidence",
      icon: Crosshair,
    },
  ];

  return (
    <section aria-labelledby="summary-heading" className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="summary-heading" className="text-xl font-bold tracking-tight md:text-2xl">
            Analysis summary
          </h2>
          <p className="text-sm text-muted-foreground">
            Each value is computed from this result only. Missing measurements are never filled.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {typeof processingTime === "number" && (
            <Badge variant="outline">Processed in {formatNumber(processingTime, 1)} s</Badge>
          )}
          <Badge variant="outline">schema 2.1</Badge>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {cards.map(({ key, label, value, detail, icon: Icon }) => (
          <div
            key={key}
            className="group rounded-2xl border border-border gradient-card p-4 shadow-soft transition-smooth hover:-translate-y-0.5 hover:border-primary/30"
            data-testid={`summary-${key}`}
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {label}
              </p>
              <Icon className="h-4 w-4 text-primary/80" aria-hidden="true" />
            </div>
            <p className="mt-2 text-xl font-extrabold leading-none tracking-tight md:text-2xl">
              {value}
            </p>
            <p className="mt-2 text-[11px] leading-snug text-muted-foreground">{detail}</p>
          </div>
        ))}
      </div>

      <p className="text-xs text-muted-foreground">
        Speed is an estimate from the measured ground plane even when geometry is calibrated. A
        dash ({DASH}) means the run did not measure that value.
      </p>
    </section>
  );
};

export default SummaryStats;
