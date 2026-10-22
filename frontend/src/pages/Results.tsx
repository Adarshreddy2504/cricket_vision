import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Copy,
  Download,
  FlaskConical,
  Gauge,
  Layers,
  Loader2,
  MapPin,
  RefreshCw,
  SearchX,
  Target,
  VideoOff,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import ClipStage from "@/components/ClipStage";
import DeliveryExplorer from "@/components/DeliveryExplorer";
import DemoBanner from "@/components/DemoBanner";
import EmptyState from "@/components/EmptyState";
import PitchMap from "@/components/PitchMap";
import ShotDistributionChart from "@/components/ShotDistribution";
import SummaryStats from "@/components/SummaryStats";
import { ApiError, fetchResult, type ResultDocument } from "@/lib/api";
import {
  adaptResult,
  shotDistribution,
  summariseDeliveries,
  type DeliveryView,
} from "@/lib/analysis";
import { DEMO_ANALYSIS_ID, DEMO_NOTICE, buildDemoResult, isDemoAnalysisId } from "@/lib/demo";
import { formatNumber, formatPercent, formatSpeed } from "@/lib/format";
import { cacheResult, getStoredAnalysisId, loadCachedResult } from "@/lib/storage";

interface NavState {
  analysis_id?: string;
  processing_time?: number;
  demo?: boolean;
}

const PITCH_STATE_LABEL: Record<string, string> = {
  calibrated: "Pitch calibrated",
  temporary_loss: "Previous calibration in use",
  no_calibration: "No pitch calibration",
  calibration_expired: "Calibration expired",
};

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const navState = (location.state ?? null) as NavState | null;
  const analysisId = navState?.analysis_id ?? getStoredAnalysisId();
  const demo = isDemoAnalysisId(analysisId);

  const cached = useMemo(
    () => (demo ? null : loadCachedResult(analysisId)),
    [analysisId, demo],
  );
  const demoResult = useMemo(() => (demo ? buildDemoResult() : null), [demo]);

  const [remote, setRemote] = useState<ResultDocument | null>(null);
  const [loading, setLoading] = useState(Boolean(analysisId) && !demo && !cached);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contractError, setContractError] = useState(false);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const processingTime = navState?.processing_time ?? cached?.processing_time;

  // A different analysis id means a different result. Clear the in-memory copy
  // before the next fetch so the previous analysis can never paint under the
  // new id while the new request is in flight.
  useEffect(() => {
    setRemote(null);
    setError(null);
    setContractError(false);
    setRefreshing(false);
  }, [analysisId]);

  useEffect(() => {
    if (!analysisId || demo) return;
    let cancelled = false;
    const controller = new AbortController();
    if (cached) setRefreshing(true);
    else setLoading(true);
    setError(null);
    setContractError(false);

    fetchResult(analysisId, { signal: controller.signal, timeoutMs: 30_000 })
      .then((document) => {
        if (cancelled) return;
        setRemote(document);
        cacheResult(document, processingTime);
        setLoading(false);
        setRefreshing(false);
      })
      .catch((caught: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        setLoading(false);
        setRefreshing(false);
        if (caught instanceof ApiError && caught.kind === "contract") {
          setContractError(true);
        }
        setError(
          caught instanceof Error ? caught.message : "The result could not be loaded.",
        );
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
    // `cached` and `processingTime` are inputs to this one fetch; re-running on
    // a cache write would loop. They are intentionally not dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisId, demo, reloadNonce]);

  const resultDoc: ResultDocument | null = demo ? demoResult : remote ?? cached ?? null;
  const views = useMemo(() => (resultDoc ? adaptResult(resultDoc) : []), [resultDoc]);
  const summary = useMemo(() => summariseDeliveries(views), [views]);
  const distribution = useMemo(() => shotDistribution(views), [views]);

  useEffect(() => {
    if (views.length === 0) {
      setSelectedId(null);
      return;
    }
    if (!views.some((view) => view.id === selectedId)) {
      setSelectedId(views[0].id);
    }
  }, [selectedId, views]);

  const selectedIndex = views.findIndex((view) => view.id === selectedId);
  const selected: DeliveryView | null = selectedIndex >= 0 ? views[selectedIndex] : views[0] ?? null;

  const scrollToStage = useCallback(() => {
    if (!stageRef.current) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    stageRef.current.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, []);

  const selectDelivery = (id: number) => {
    setSelectedId(id);
    scrollToStage();
  };

  const copyId = async () => {
    if (!analysisId) return;
    try {
      await navigator.clipboard.writeText(analysisId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const exportResult = () => {
    if (!resultDoc) return;
    const payload = demo
      ? { demo: true, notice: DEMO_NOTICE, result: resultDoc }
      : resultDoc;
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = demo
      ? "cricketvision-demo-result.json"
      : `cricketvision-${analysisId ?? "result"}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  if (!analysisId) {
    return (
      <div className="container py-12">
        <EmptyState
          icon={SearchX}
          title="No result to show"
          description="There is no analysis in this tab. Upload a video to produce one, or open the demo result to review the interface."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Button asChild>
                <Link to="/upload">Analyse a video</Link>
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  navigate("/results", {
                    state: { analysis_id: DEMO_ANALYSIS_ID, demo: true },
                  });
                }}
              >
                <FlaskConical /> Demo result
              </Button>
            </div>
          }
        />
      </div>
    );
  }

  if (loading && !resultDoc) {
    return <ResultsSkeleton />;
  }

  if (!resultDoc) {
    return (
      <div className="container py-10">
        <div className="mx-auto max-w-2xl space-y-4">
          <Alert variant="destructive">
            <AlertTitle>
              {contractError ? "The result did not match the API contract" : "Result unavailable"}
            </AlertTitle>
            <AlertDescription>
              {error ??
                "The backend returned no result for this analysis. It may have expired or still be processing."}
            </AlertDescription>
          </Alert>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => setReloadNonce((value) => value + 1)}>
              <RefreshCw /> Try again
            </Button>
            <Button asChild variant="outline">
              <Link to="/upload">Analyse a video</Link>
            </Button>
            <Button asChild variant="ghost">
              <Link to="/">Back to dashboard</Link>
            </Button>
            <Button
              variant="ghost"
              onClick={() =>
                navigate("/results", {
                  state: { analysis_id: DEMO_ANALYSIS_ID, demo: true },
                })
              }
            >
              <FlaskConical /> Demo result
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            An analysis id is only meaningful on the backend that produced it. A new backend start
            can clear in-memory analyses.
          </p>
        </div>
      </div>
    );
  }

  const uncalibrated = views.filter((view) => !view.calibrated).length;
  const reviewCount = views.filter((view) => view.quality.requires_human_review).length;
  const pitchLabel = PITCH_STATE_LABEL[resultDoc.pitch.state] ?? resultDoc.pitch.state;
  const total = views.length;

  return (
    <div className="container space-y-8 py-8 md:py-12">
      <header className="space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Badge variant={demo ? "warning" : "success"}>
                {demo ? <FlaskConical /> : <ClipboardCheck />}
                {demo ? "Demo data" : "Analysis complete"}
              </Badge>
              <Badge variant="outline">{total} deliveries</Badge>
              {typeof processingTime === "number" && (
                <Badge variant="outline">
                  processed in {formatNumber(processingTime, 1)} s
                </Badge>
              )}
              {refreshing && (
                <Badge variant="secondary">
                  <Loader2 className="animate-spin" /> refreshing
                </Badge>
              )}
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">
              {demo ? "Demo delivery analysis" : "Ball-by-ball analysis"}
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-sm text-muted-foreground">
              <span>Analysis</span>
              <code className="max-w-[60vw] truncate rounded bg-secondary/60 px-1.5 py-0.5 font-mono text-xs">
                {analysisId}
              </code>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => void copyId()}
                aria-label="Copy analysis id"
              >
                {copied ? <ClipboardCheck /> : <Copy />}
              </Button>
              <span className="sr-only" aria-live="polite">
                {copied ? "Analysis id copied" : ""}
              </span>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={exportResult}>
              <Download /> Export JSON
            </Button>
            {!demo && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setReloadNonce((value) => value + 1)}
                disabled={refreshing}
              >
                <RefreshCw className={refreshing ? "animate-spin" : undefined} /> Refresh
              </Button>
            )}
            <Button asChild size="sm">
              <Link to="/upload">New analysis</Link>
            </Button>
          </div>
        </div>
      </header>

      {demo && <DemoBanner />}

      {error && resultDoc && (
        <Alert variant="warning">
          <AlertTitle>Showing the cached result</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center gap-3">
            <span>{error}</span>
            <Button size="sm" variant="outline" onClick={() => setReloadNonce((value) => value + 1)}>
              <RefreshCw /> Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {total === 0 ? (
        <EmptyState
          icon={VideoOff}
          title="No deliveries detected"
          description="The analysis completed, but the pipeline did not segment any deliveries from this video. That is a real result, not a loading state — no summary or overlays are invented to fill the page. Try a longer clip or clearer footage."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Button asChild>
                <Link to="/upload">Try another video</Link>
              </Button>
              <Button asChild variant="outline">
                <Link to="/">Dashboard</Link>
              </Button>
            </div>
          }
        />
      ) : (
        <>
          <SummaryStats summary={summary} processingTime={processingTime} />

          <div ref={stageRef} className="scroll-mt-24">
            <section aria-labelledby="clip-heading" className="grid gap-6 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
              <div className="space-y-3">
                <div className="flex flex-wrap items-end justify-between gap-3">
                  <div>
                    <h2 id="clip-heading" className="text-xl font-bold tracking-tight md:text-2xl">
                      Delivery replay
                    </h2>
                    <p className="text-sm text-muted-foreground">
                      {selected
                        ? `Ball ${selected.id} of ${total} · detected path solid, predicted tail dashed`
                        : "Select a delivery"}
                    </p>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="icon-sm"
                      aria-label="Previous delivery"
                      disabled={selectedIndex <= 0}
                      onClick={() =>
                        selectedIndex > 0 && setSelectedId(views[selectedIndex - 1].id)
                      }
                    >
                      <ChevronLeft />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon-sm"
                      aria-label="Next delivery"
                      disabled={selectedIndex < 0 || selectedIndex >= views.length - 1}
                      onClick={() =>
                        selectedIndex >= 0 &&
                        selectedIndex < views.length - 1 &&
                        setSelectedId(views[selectedIndex + 1].id)
                      }
                    >
                      <ChevronRight />
                    </Button>
                  </div>
                </div>

                {selected && (
                  <ClipStage
                    key={selected.id}
                    video={selected.video}
                    trajectory={selected.trajectory}
                    bounceFrame={selected.bounce.frame}
                    pitch={selected.pitch}
                    title={`Ball ${selected.id}`}
                    demo={demo}
                  />
                )}
              </div>

              <DeliveryFacts delivery={selected} pitchLabel={pitchLabel} pitch={resultDoc.pitch} />
            </section>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <CardTitle className="flex items-center gap-2">
                    <MapPin className="h-5 w-5 text-primary" /> Pitch map
                  </CardTitle>
                  <Badge variant={resultDoc.pitch.calibrated ? "success" : "warning"}>
                    {pitchLabel}
                  </Badge>
                </div>
                <CardDescription>
                  Bounce positions projected onto the measured pitch. Batting end at the bottom.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <PitchMap deliveries={views} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" /> Shot distribution
                </CardTitle>
                <CardDescription>
                  Labels and confidence as returned by the classifier. Counts are deliveries, not
                  scoring shots.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ShotDistributionChart distribution={distribution} />
              </CardContent>
            </Card>
          </div>

          <DeliveryExplorer
            deliveries={views}
            selectedId={selected?.id ?? null}
            onSelect={selectDelivery}
          />

          <section aria-labelledby="notes-heading" className="space-y-3">
            <h2 id="notes-heading" className="sr-only">
              Analysis notes
            </h2>
            {!resultDoc.pitch.calibrated && (
              <Alert variant="warning">
                <AlertTitle>Pitch geometry is not calibrated</AlertTitle>
                <AlertDescription>
                  {resultDoc.pitch.geometry_reason
                    ? `The geometry gate reported: ${resultDoc.pitch.geometry_reason.replace(/_/g, " ")}. `
                    : ""}
                  Speed, line and length are left blank rather than estimated from uncalibrated
                  pixels. The pitch map only plots positions the model recorded.
                </AlertDescription>
              </Alert>
            )}
            {uncalibrated > 0 && resultDoc.pitch.calibrated && (
              <Alert variant="info">
                <AlertTitle>{uncalibrated} of {total} deliveries lack calibrated geometry</AlertTitle>
                <AlertDescription>
                  Those deliveries show a dash for the measurements that depend on the ground
                  plane. The remaining deliveries are unaffected.
                </AlertDescription>
              </Alert>
            )}
            {!distribution.hasLabels && (
              <Alert variant="warning">
                <AlertTitle>No shot labels were returned</AlertTitle>
                <AlertDescription>
                  None of the {total} deliveries carries a classification. This usually means the
                  shot model was unavailable or produced no usable frames. The page states that
                  outcome instead of showing guessed labels or an empty set of 0% bars.
                </AlertDescription>
              </Alert>
            )}
            {reviewCount > 0 && (
              <Alert variant="info">
                <AlertTitle>{reviewCount} deliveries are flagged for review</AlertTitle>
                <AlertDescription>
                  Review flags come from the backend's quality block. They mark tracking or
                  calibration conditions that deserve a human look; they do not change any
                  measurement.
                </AlertDescription>
              </Alert>
            )}

            <p className="text-xs leading-relaxed text-muted-foreground">
              Every measurement on this page was read from the result document. Absent values are
              rendered as <span className="font-mono">—</span> and never replaced with a guess.
              Speed is an estimate from the measured ground plane, even when the pitch is
              calibrated. Demo data is synthetic and labelled as such.
            </p>
          </section>
        </>
      )}
    </div>
  );
};

function DeliveryFacts({
  delivery,
  pitchLabel,
  pitch,
}: {
  delivery: DeliveryView | null;
  pitchLabel: string;
  pitch: ResultDocument["pitch"];
}) {
  const detected = delivery?.trajectory.filter((point) => point.source === "detected").length ?? 0;
  const predicted =
    delivery?.trajectory.filter((point) => point.source === "predicted").length ?? 0;

  return (
    <Card className="flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-base">Delivery facts</CardTitle>
          {delivery && (
            <Badge variant={delivery.calibrated ? "success" : "warning"}>
              {delivery.calibrated ? "calibrated" : "uncalibrated"}
            </Badge>
          )}
        </div>
        <CardDescription>
          {delivery ? `Recorded values for ball ${delivery.id}` : "Select a delivery"}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 space-y-4">
        {delivery ? (
          <>
            <div className="rounded-xl border border-border/70 bg-secondary/30 p-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Shot
              </p>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                {delivery.shot ? (
                  <>
                    <span className="text-lg font-extrabold">{delivery.shot}</span>
                    <Badge variant="accent">
                      {formatPercent(delivery.shotConfidence, 1)} confidence
                    </Badge>
                  </>
                ) : (
                  <span className="text-sm font-semibold text-muted-foreground">
                    No label returned
                  </span>
                )}
              </div>
            </div>

            <dl className="grid grid-cols-2 gap-2">
              <Fact icon={Gauge} label="Speed" value={formatSpeed(delivery.speedKmh)} />
              <Fact icon={Layers} label="Length" value={delivery.length ?? "—"} />
              <Fact icon={Target} label="Line" value={delivery.line ?? "—"} />
              <Fact icon={RefreshCw} label="Swing" value={delivery.swing ?? "—"} />
            </dl>

            <div className="space-y-1.5 text-xs">
              <Row label="Pitch state" value={pitchLabel} />
              <Row
                label="Calibration confidence"
                value={
                  pitch.confidence !== null ? formatPercent(pitch.confidence, 0) : "not reported"
                }
              />
              <Row label="Tracked points" value={`${detected} detected · ${predicted} predicted`} />
              <Row
                label="Camera frame"
                value={
                  delivery.video.width !== null && delivery.video.height !== null
                    ? `${delivery.video.width}×${delivery.video.height}`
                    : "not reported"
                }
              />
              <Row
                label="Clip frames"
                value={
                  delivery.video.start_frame !== null && delivery.video.end_frame !== null
                    ? `${delivery.video.start_frame}–${delivery.video.end_frame}`
                    : "not reported"
                }
              />
            </div>

            {!delivery.calibrated && delivery.geometryReason && (
              <p className="rounded-lg border border-warning/30 bg-warning/10 p-2 text-xs text-warning">
                Geometry refused: {delivery.geometryReason.replace(/_/g, " ")}
              </p>
            )}

            {delivery.quality.flags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {delivery.quality.flags.map((flag) => (
                  <Badge key={flag} variant="outline" className="text-[10px]">
                    {flag.replace(/_/g, " ")}
                  </Badge>
                ))}
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            Choose a delivery from the explorer to inspect its recorded values.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function Fact({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-card px-2.5 py-2">
      <dt className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        <Icon className="h-3 w-3" aria-hidden="true" /> {label}
      </dt>
      <dd className="mt-0.5 truncate text-sm font-bold" title={value}>
        {value}
      </dd>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

function ResultsSkeleton() {
  return (
    <div className="container space-y-6 py-8 md:py-12" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading analysis result</span>
      <div className="space-y-3">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-10 w-2/3 max-w-md" />
        <Skeleton className="h-4 w-56" />
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {Array.from({ length: 6 }).map((_, index) => (
          <Skeleton key={index} className="h-28 w-full" />
        ))}
      </div>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <Skeleton className="aspect-video w-full rounded-3xl" />
        <Skeleton className="h-full min-h-72 w-full rounded-2xl" />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-80 w-full rounded-2xl" />
        <Skeleton className="h-80 w-full rounded-2xl" />
      </div>
    </div>
  );
}

export default Results;
