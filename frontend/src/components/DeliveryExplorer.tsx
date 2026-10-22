/**
 * DeliveryExplorer — search and filter the delivery records, then open one in
 * the clip stage. Filtering is presentation only; every card is a projection
 * of a real record (or, in demo mode, the labelled synthetic one).
 */

import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Crosshair, Filter, Play, RotateCcw, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import EmptyState from "./EmptyState";
import { filterDeliveries, type DeliveryView } from "@/lib/analysis";
import { formatPercent, formatSpeed } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  deliveries: DeliveryView[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

const ALL = "__all__";
const NONE = "__none__";

const DeliveryExplorer = ({ deliveries, selectedId, onSelect }: Props) => {
  const [query, setQuery] = useState("");
  const [length, setLength] = useState(ALL);
  const [shot, setShot] = useState(ALL);

  const lengths = useMemo(
    () =>
      Array.from(
        new Set(deliveries.map((delivery) => delivery.length).filter((value): value is string => Boolean(value))),
      ).sort(),
    [deliveries],
  );
  const shots = useMemo(
    () =>
      Array.from(
        new Set(deliveries.map((delivery) => delivery.shot).filter((value): value is string => Boolean(value))),
      ).sort(),
    [deliveries],
  );

  const filtered = useMemo(() => {
    const base = filterDeliveries(deliveries, {
      query,
      length: length === ALL ? null : length,
    });
    if (shot === ALL) return base;
    if (shot === NONE) return base.filter((delivery) => delivery.shot === null);
    return base.filter((delivery) => delivery.shot === shot);
  }, [deliveries, length, query, shot]);

  const reset = () => {
    setQuery("");
    setLength(ALL);
    setShot(ALL);
  };

  return (
    <section aria-labelledby="deliveries-heading" className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="deliveries-heading" className="text-xl font-bold tracking-tight md:text-2xl">
            Delivery explorer
          </h2>
          <p className="text-sm text-muted-foreground">
            {filtered.length} of {deliveries.length} deliveries shown. Select one to load it in the
            clip stage.
          </p>
        </div>
        {(query || length !== ALL || shot !== ALL) && (
          <Button variant="ghost" size="sm" onClick={reset}>
            <RotateCcw /> Clear filters
          </Button>
        )}
      </div>

      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_180px_180px]">
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <label htmlFor="delivery-search" className="sr-only">
            Search deliveries
          </label>
          <Input
            id="delivery-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search ball, shot, length or line…"
            className="pl-9"
          />
        </div>

        <div>
          <label htmlFor="length-filter" className="sr-only">
            Filter by length
          </label>
          <select
            id="length-filter"
            value={length}
            onChange={(event) => setLength(event.target.value)}
            className="h-10 w-full rounded-xl border border-input bg-background/60 px-3 text-sm"
          >
            <option value={ALL}>All lengths</option>
            {lengths.map((value) => (
              <option key={value} value={value as string}>
                {value}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="shot-filter" className="sr-only">
            Filter by shot
          </label>
          <select
            id="shot-filter"
            value={shot}
            onChange={(event) => setShot(event.target.value)}
            className="h-10 w-full rounded-xl border border-input bg-background/60 px-3 text-sm"
          >
            <option value={ALL}>All shots</option>
            <option value={NONE}>No shot label</option>
            {shots.map((value) => (
              <option key={value} value={value as string}>
                {value}
              </option>
            ))}
          </select>
        </div>
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          icon={Filter}
          title="No deliveries match"
          description="Adjust the search or filters to see deliveries from this analysis."
          action={
            <Button variant="outline" onClick={reset}>
              Clear filters
            </Button>
          }
          compact
        />
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((delivery) => {
            const active = delivery.id === selectedId;
            return (
              <article
                key={delivery.id}
                data-testid={`delivery-card-${delivery.id}`}
                className={cn(
                  "flex flex-col rounded-2xl border bg-card p-4 shadow-soft transition-smooth",
                  active ? "border-primary/60 ring-1 ring-primary/30" : "border-border hover:border-primary/30",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Delivery
                    </p>
                    <h3 className="text-lg font-extrabold leading-tight">Ball {delivery.id}</h3>
                  </div>
                  <Badge variant={delivery.calibrated ? "success" : "warning"}>
                    {delivery.calibrated ? (
                      <>
                        <CheckCircle2 /> calibrated
                      </>
                    ) : (
                      <>
                        <AlertTriangle /> uncalibrated
                      </>
                    )}
                  </Badge>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  <Metric label="Speed" value={formatSpeed(delivery.speedKmh)} />
                  <Metric label="Length" value={delivery.length ?? "—"} />
                  <Metric label="Line" value={delivery.line ?? "—"} />
                  <Metric label="Swing" value={delivery.swing ?? "—"} />
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {delivery.shot ? (
                    <Badge variant="accent">
                      <Crosshair /> {delivery.shot}
                      {delivery.shotConfidence !== null && (
                        <span className="font-normal opacity-80">
                          {formatPercent(delivery.shotConfidence)}
                        </span>
                      )}
                    </Badge>
                  ) : (
                    <Badge variant="outline">no shot label</Badge>
                  )}
                  {delivery.quality.requires_human_review && (
                    <Badge variant="warning">review</Badge>
                  )}
                  {delivery.video.clip_url && <Badge variant="outline">clip</Badge>}
                </div>

                {!delivery.calibrated && delivery.geometryReason && (
                  <p className="mt-3 text-xs text-warning">
                    Metrics unavailable: {delivery.geometryReason.replace(/_/g, " ")}
                  </p>
                )}

                <div className="mt-4 flex items-center justify-between gap-2 border-t border-border/60 pt-3">
                  <span className="text-[11px] text-muted-foreground">
                    {delivery.trajectory.length} tracked points · {delivery.trajectory.filter((p) => p.source === "predicted").length} predicted
                  </span>
                  <Button
                    size="sm"
                    variant={active ? "default" : "outline"}
                    onClick={() => onSelect(delivery.id)}
                    aria-label={`View delivery ${delivery.id} in the clip stage`}
                  >
                    <Play /> {active ? "Viewing" : "View"}
                  </Button>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
};

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-secondary/40 px-2.5 py-1.5">
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="truncate text-sm font-semibold" title={value}>
        {value}
      </p>
    </div>
  );
}

export default DeliveryExplorer;
