/**
 * ShotDistribution — shot label counts, shares and confidence from this result.
 *
 * Every label in the 10-way vocabulary is listed, including zero counts. When
 * no delivery was classified the panel says so explicitly and shows no 0%
 * bars: a share of zero classified deliveries is undefined, not zero.
 */

import { BarChart3, Crosshair } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import EmptyState from "./EmptyState";
import type { ShotDistribution as Distribution } from "@/lib/analysis";
import { DASH, formatPercent } from "@/lib/format";

interface Props {
  distribution: Distribution;
}

const ShotDistribution = ({ distribution }: Props) => {
  if (distribution.total === 0) {
    return (
      <EmptyState
        icon={BarChart3}
        title="No deliveries to chart"
        description="This result contains no deliveries, so there is no shot distribution to show."
        compact
      />
    );
  }

  if (!distribution.hasLabels) {
    return (
      <EmptyState
        icon={Crosshair}
        title="No shot labels were returned"
        description="None of the deliveries in this result carries a shot classification. This usually means the classifier was unavailable or produced no usable frames, so no distribution is shown rather than a set of zero bars."
        compact
      />
    );
  }

  const max = Math.max(1, ...distribution.rows.map((row) => row.count));

  return (
    <div className="space-y-4" data-testid="shot-distribution">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="default">{distribution.classified} classified</Badge>
        <Badge variant="outline">{distribution.unclassified} without a label</Badge>
        {distribution.meanScore !== null && (
          <Badge variant="outline">
            mean confidence {formatPercent(distribution.meanScore, 1)}
          </Badge>
        )}
      </div>

      <ul className="space-y-2.5" aria-label="Shot label distribution">
        {distribution.rows.map((row) => (
          <li key={row.label}>
            <div className="mb-1 flex items-baseline justify-between gap-3 text-sm">
              <span className="font-medium">{row.label}</span>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {row.count} {row.count === 1 ? "ball" : "balls"} ·{" "}
                {row.share === null ? DASH : formatPercent(row.share, 0)}
              </span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-secondary/70">
              <div
                className="h-full rounded-full gradient-accent transition-[width] duration-500"
                style={{ width: `${(row.count / max) * 100}%` }}
                role="presentation"
              />
            </div>
          </li>
        ))}
      </ul>

      {distribution.unknownLabels.length > 0 && (
        <p className="text-xs text-warning">
          The server returned label{distribution.unknownLabels.length === 1 ? "" : "s"} outside the
          documented vocabulary: {distribution.unknownLabels.join(", ")}. They are counted rather
          than hidden.
        </p>
      )}

      <p className="text-xs text-muted-foreground">
        Shares are of the {distribution.classified} classified deliveries. Confidence is the
        classifier's own score for the chosen label, not an accuracy figure.
      </p>
    </div>
  );
};

export default ShotDistribution;
