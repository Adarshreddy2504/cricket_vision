/**
 * PitchMap.tsx — a top-down pitch with the deliveries' bounce positions.
 *
 * A dot exists only when `analysis.ts` produced a coordinate, which requires
 * both a pitch-model position at the bounce frame and an orientation naming the
 * batting end. The batting end is always at the bottom (y = 1). No dot is
 * inferred from frame pixels.
 *
 * Length zones are the five published labels, drawn in pitch proportion rather
 * than as equal decorative bands: a yorker lands at the batter's feet, a short
 * ball roughly halfway down.
 */

import { useMemo, useState } from "react";
import { Crosshair, MapPin } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import EmptyState from "./EmptyState";
import { type DeliveryView } from "@/lib/analysis";
import { formatPercent, formatSpeed } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  deliveries: DeliveryView[];
}

const ZONES = [
  { label: "Yorker", from: 0.86, to: 0.99, color: "#f87171" },
  { label: "Full", from: 0.7, to: 0.86, color: "#fb923c" },
  { label: "Good Length", from: 0.44, to: 0.7, color: "#4ade80" },
  { label: "Short", from: 0.2, to: 0.44, color: "#38bdf8" },
  { label: "Very Short", from: 0.03, to: 0.2, color: "#a78bfa" },
];

const VIEW = { width: 320, height: 470, topLeft: 112, topRight: 208, bottomLeft: 46, bottomRight: 274 };

function halfWidthAt(y: number): { left: number; right: number } {
  // y = 0 at the far (bowler) end, y = 1 at the near (batting) end.
  const ratio = Math.max(0, Math.min(1, y));
  const halfTop = (VIEW.topRight - VIEW.topLeft) / 2;
  const halfBottom = (VIEW.bottomRight - VIEW.bottomLeft) / 2;
  const centerTop = (VIEW.topLeft + VIEW.topRight) / 2;
  const centerBottom = (VIEW.bottomLeft + VIEW.bottomRight) / 2;
  const half = halfTop + (halfBottom - halfTop) * ratio;
  const center = centerTop + (centerBottom - centerTop) * ratio;
  return { left: center - half, right: center + half };
}

function pointOnPitch(x: number, y: number) {
  const { left, right } = halfWidthAt(y);
  const inset = 10;
  return {
    x: left + inset + (right - left - inset * 2) * Math.max(0, Math.min(1, x)),
    y: 36 + (VIEW.height - 72) * Math.max(0, Math.min(1, y)),
  };
}

const PitchMap = ({ deliveries }: Props) => {
  const [selected, setSelected] = useState<number | null>(null);
  const [zone, setZone] = useState<string | null>(null);

  const mappable = useMemo(
    () =>
      deliveries.filter(
        (delivery) =>
          delivery.bounceX !== null &&
          delivery.bounceY !== null &&
          Number.isFinite(delivery.bounceX) &&
          Number.isFinite(delivery.bounceY),
      ),
    [deliveries],
  );

  const visible = zone ? mappable.filter((delivery) => delivery.length === zone) : mappable;
  const selectedDelivery = deliveries.find((delivery) => delivery.id === selected) ?? null;

  const counts = useMemo(() => {
    const map = new Map<string, number>();
    mappable.forEach((delivery) => {
      if (delivery.length) map.set(delivery.length, (map.get(delivery.length) ?? 0) + 1);
    });
    return map;
  }, [mappable]);

  if (mappable.length === 0) {
    return (
      <EmptyState
        icon={MapPin}
        title="Pitch map unavailable"
        description="No delivery carries a calibrated bounce position. A dot is only drawn when the pitch model recorded a position at the bounce frame and an orientation for the pitch quad — this run has neither, so the map stays empty rather than inventing positions from frame pixels."
        compact
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setZone(null)}
          aria-pressed={zone === null}
          className={cn(
            "rounded-full border px-3 py-1 text-xs font-semibold transition-colors",
            zone === null
              ? "border-primary/40 bg-primary/15 text-primary"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          All ({mappable.length})
        </button>
        {ZONES.map((entry) => {
          const count = counts.get(entry.label) ?? 0;
          return (
            <button
              key={entry.label}
              type="button"
              onClick={() => setZone(zone === entry.label ? null : entry.label)}
              aria-pressed={zone === entry.label}
              disabled={count === 0}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                zone === entry.label
                  ? "border-primary/40 bg-primary/15 text-primary"
                  : "border-border text-muted-foreground hover:text-foreground",
              )}
            >
              {entry.label} ({count})
            </button>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
        <div className="relative overflow-hidden rounded-2xl border border-border bg-[#0a241b] p-2">
          <svg
            viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
            className="mx-auto h-auto w-full max-w-[360px]"
            role="img"
            aria-label={`Top-down pitch with ${visible.length} bounce positions; batting end at the bottom`}
          >
            <defs>
              <linearGradient id="pitch-surface" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8a6a3f" />
                <stop offset="100%" stopColor="#c9a267" />
              </linearGradient>
              <filter id="dot-glow" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {ZONES.map((entry) => {
              const top = pointOnPitch(0, entry.from).y;
              const bottom = pointOnPitch(0, entry.to).y;
              const topHalf = halfWidthAt(entry.from);
              const bottomHalf = halfWidthAt(entry.to);
              return (
                <g key={entry.label}>
                  <polygon
                    points={`${topHalf.left},${top} ${topHalf.right},${top} ${bottomHalf.right},${bottom} ${bottomHalf.left},${bottom}`}
                    fill={entry.color}
                    fillOpacity={0.09}
                  />
                  <line
                    x1={topHalf.left}
                    y1={top}
                    x2={topHalf.right}
                    y2={top}
                    stroke={entry.color}
                    strokeOpacity={0.28}
                    strokeWidth="1"
                    strokeDasharray="4 5"
                  />
                </g>
              );
            })}

            <polygon
              points={`${VIEW.topLeft},36 ${VIEW.topRight},36 ${VIEW.bottomRight},${VIEW.height - 36} ${VIEW.bottomLeft},${VIEW.height - 36}`}
              fill="url(#pitch-surface)"
              fillOpacity={0.22}
              stroke="#fef3c7"
              strokeOpacity={0.35}
              strokeWidth="1.5"
            />
            <line
              x1={VIEW.topLeft}
              y1={36}
              x2={VIEW.topRight}
              y2={36}
              stroke="#fef3c7"
              strokeOpacity={0.7}
              strokeWidth="2"
            />
            <line
              x1={VIEW.bottomLeft}
              y1={VIEW.height - 36}
              x2={VIEW.bottomRight}
              y2={VIEW.height - 36}
              stroke="#fef3c7"
              strokeOpacity={0.7}
              strokeWidth="2"
            />

            {[0.12, 0.5, 0.88].map((y) => {
              const { left, right } = halfWidthAt(y);
              return (
                <line
                  key={y}
                  x1={left + 14}
                  y1={pointOnPitch(0, y).y}
                  x2={right - 14}
                  y2={pointOnPitch(0, y).y}
                  stroke="#fef3c7"
                  strokeOpacity="0.16"
                  strokeWidth="1"
                />
              );
            })}

            {visible.map((delivery) => {
              const point = pointOnPitch(delivery.bounceX ?? 0.5, delivery.bounceY ?? 0.5);
              const isSelected = selected === delivery.id;
              const color =
                ZONES.find((entry) => entry.label === delivery.length)?.color ?? "#94a3b8";
              return (
                <g key={delivery.id}>
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={isSelected ? 11 : 8}
                    fill={color}
                    fillOpacity={delivery.bounceInsidePitch === false ? 0.25 : 0.9}
                    stroke={isSelected ? "#ffffff" : "#0b1220"}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                    filter={isSelected ? "url(#dot-glow)" : undefined}
                  />
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={isSelected ? 17 : 14}
                    fill="transparent"
                    className="cursor-pointer"
                    role="button"
                    tabIndex={0}
                    aria-label={`Delivery ${delivery.id}, ${delivery.length ?? "length unavailable"}, ${formatSpeed(delivery.speedKmh)}`}
                    onClick={() => setSelected(delivery.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelected(delivery.id);
                      }
                    }}
                  />
                </g>
              );
            })}

            <text
              x={VIEW.width / 2}
              y={VIEW.height - 12}
              textAnchor="middle"
              fill="#fef3c7"
              fillOpacity="0.65"
              fontSize="10"
              fontWeight="700"
              letterSpacing="2"
            >
              BATTING END
            </text>
            <text
              x={VIEW.width / 2}
              y={22}
              textAnchor="middle"
              fill="#fef3c7"
              fillOpacity="0.45"
              fontSize="9"
              fontWeight="600"
              letterSpacing="1.5"
            >
              BOWLING END
            </text>
          </svg>
          <p className="mt-1 text-center text-[10px] text-white/50">
            Positions are projected onto the measured pitch quad; they are estimates, not radar.
          </p>
        </div>

        <div className="space-y-3">
          {selectedDelivery ? (
            <div className="rounded-2xl border border-border bg-card p-4">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Selected delivery
              </p>
              <p className="mt-1 text-2xl font-extrabold">Ball {selectedDelivery.id}</p>
              <dl className="mt-3 space-y-1.5 text-sm">
                <div className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">Speed</dt>
                  <dd className="font-semibold">{formatSpeed(selectedDelivery.speedKmh)}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">Length</dt>
                  <dd className="font-semibold">{selectedDelivery.length ?? "—"}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">Line</dt>
                  <dd className="font-semibold">{selectedDelivery.line ?? "—"}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">Shot</dt>
                  <dd className="font-semibold">
                    {selectedDelivery.shot ?? "—"}
                    {selectedDelivery.shotConfidence !== null && (
                      <span className="ml-1 text-xs font-normal text-muted-foreground">
                        {formatPercent(selectedDelivery.shotConfidence)}
                      </span>
                    )}
                  </dd>
                </div>
              </dl>
              {selectedDelivery.bounceInsidePitch === false && (
                <p className="mt-3 text-xs text-warning">
                  This projected bounce falls outside the measured pitch.
                </p>
              )}
              <button
                type="button"
                className="mt-3 text-xs font-semibold text-primary underline-offset-4 hover:underline"
                onClick={() => setSelected(null)}
              >
                Clear selection
              </button>
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-border bg-card/40 p-4 text-sm text-muted-foreground">
              <Crosshair className="mb-2 h-5 w-5" aria-hidden="true" />
              Select a bounce dot to see the delivery. Dots are coloured by recorded length.
            </div>
          )}

          <div className="rounded-2xl border border-border bg-card p-4">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Length zones
            </p>
            <ul className="mt-2 space-y-1.5">
              {ZONES.map((entry) => (
                <li key={entry.label} className="flex items-center justify-between gap-2 text-xs">
                  <span className="flex items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: entry.color }}
                      aria-hidden="true"
                    />
                    {entry.label}
                  </span>
                  <span className="font-mono text-muted-foreground">{counts.get(entry.label) ?? 0}</span>
                </li>
              ))}
            </ul>
            <Badge variant="outline" className="mt-3">
              {visible.length} shown
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PitchMap;
