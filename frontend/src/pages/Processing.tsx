import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Check,
  CircleDashed,
  Copy,
  Crosshair,
  FlaskConical,
  Layers,
  Loader2,
  Pause,
  Play,
  Radar,
  RefreshCw,
  Save,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ApiError,
  cancelAnalysis,
  fetchResult,
  fetchStatus,
  type AnalysisStatus,
} from "@/lib/api";
import {
  PIPELINE_STAGES,
  clampProgress,
  stageStates,
  type StageState,
} from "@/lib/analysis";
import { isDemoAnalysisId, DEMO_ANALYSIS_ID, DEMO_NOTICE } from "@/lib/demo";
import { formatDuration, formatNumber } from "@/lib/format";
import { cacheResult, getStoredAnalysisId } from "@/lib/storage";
import { cn } from "@/lib/utils";

interface NavState {
  analysis_id?: string;
  source_name?: string;
  started_at?: number;
}

const STAGE_ICONS: Record<string, LucideIcon> = {
  uploading: UploadCloud,
  tracking: Radar,
  validation: ShieldCheck,
  shot_classification: Crosshair,
  overlay: Layers,
  persisting: Save,
};

const STATE_STYLES: Record<StageState, string> = {
  done: "border-success/40 bg-success/15 text-success",
  active: "border-primary/50 bg-primary/15 text-primary animate-pulse-glow",
  pending: "border-border bg-secondary/40 text-muted-foreground",
  failed: "border-destructive/50 bg-destructive/15 text-destructive",
};

const Processing = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const navState = (location.state ?? null) as NavState | null;
  const analysisId = navState?.analysis_id ?? getStoredAnalysisId();
  const startedAtRef = useRef(navState?.started_at ?? Date.now());

  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [paused, setPaused] = useState(false);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [pollWarning, setPollWarning] = useState<string | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const [loadingResult, setLoadingResult] = useState(false);
  const [resultError, setResultError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());

  const terminal = status?.status === "completed" || status?.status === "failed";

  // Elapsed time is shown as elapsed time, never as a completion estimate.
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (isDemoAnalysisId(analysisId)) {
      navigate("/results", {
        replace: true,
        state: { analysis_id: analysisId, demo: true },
      });
    }
  }, [analysisId, navigate]);

  useEffect(() => {
    if (!analysisId || paused || terminal || isDemoAnalysisId(analysisId)) return;

    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();
    let failures = 0;

    const tick = async () => {
      let delay = 1500;
      try {
        const next = await fetchStatus(analysisId, {
          signal: controller.signal,
          timeoutMs: 10_000,
        });
        if (cancelled) return;
        failures = 0;
        setPollWarning(null);
        setStatus(next);
        if (next.status === "failed" || next.status === "completed") return;
        delay = next.status === "queued" ? 2000 : 1500;
      } catch (caught) {
        if (cancelled || controller.signal.aborted) return;
        failures += 1;
        const message =
          caught instanceof Error ? caught.message : "The status request failed.";
        if (caught instanceof ApiError && caught.status === 404) {
          setFatalError(
            `No analysis with id ${analysisId} exists on this backend. It may have expired or been cleared.`,
          );
          setPaused(true);
          return;
        }
        setPollWarning(
          `Lost contact with the backend (attempt ${failures}). ${message}`,
        );
        delay = Math.min(8000, 1500 * failures);
      }
      if (!cancelled) timer = window.setTimeout(() => void tick(), delay);
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [analysisId, paused, retryNonce, terminal]);

  useEffect(() => {
    if (!analysisId || status?.status !== "completed" || isDemoAnalysisId(analysisId)) return;

    let cancelled = false;
    const controller = new AbortController();
    setLoadingResult(true);
    setResultError(null);

    fetchResult(analysisId, { signal: controller.signal, timeoutMs: 60_000 })
      .then((document) => {
        if (cancelled) return;
        const started = Date.parse(status.started_at ?? "");
        const finished = Date.parse(status.finished_at ?? "");
        const measured =
          Number.isFinite(started) && Number.isFinite(finished) && finished >= started
            ? (finished - started) / 1000
            : (Date.now() - startedAtRef.current) / 1000;
        const processingTime = Math.max(0, Number(measured.toFixed(1)));
        cacheResult(document, processingTime);
        setLoadingResult(false);
        navigate("/results", {
          replace: true,
          state: { analysis_id: analysisId, processing_time: processingTime },
        });
      })
      .catch((caught: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        setLoadingResult(false);
        setResultError(
          caught instanceof Error
            ? caught.message
            : "The analysis finished but its result could not be loaded.",
        );
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [analysisId, navigate, retryNonce, status]);

  const cancelJob = async () => {
    if (!analysisId || cancelling) return;
    setCancelling(true);
    setCancelError(null);
    try {
      const next = await cancelAnalysis(analysisId, { timeoutMs: 15_000 });
      setStatus(next);
      setPaused(false);
      setRetryNonce((value) => value + 1);
    } catch (caught) {
      setCancelError(
        caught instanceof Error
          ? caught.message
          : "The cancellation request could not be sent.",
      );
    } finally {
      setCancelling(false);
    }
  };

  const states = useMemo(
    () => stageStates(status?.stage, status?.status),
    [status?.stage, status?.status],
  );

  const activeIndex = Math.max(
    0,
    states.findIndex((state) => state === "active" || state === "failed"),
  );
  const currentStage =
    PIPELINE_STAGES[activeIndex] ?? PIPELINE_STAGES[PIPELINE_STAGES.length - 1];

  const elapsed = Math.max(0, (now - startedAtRef.current) / 1000);
  const progress = status?.progress;
  const frameText =
    typeof status?.frames_done === "number" && typeof status?.frames_total === "number"
      ? `${status.frames_done.toLocaleString()} / ${status.frames_total.toLocaleString()} frames`
      : null;

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

  if (!analysisId) {
    return (
      <div className="container py-12">
        <Card className="mx-auto max-w-lg text-center">
          <CardHeader>
            <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary text-muted-foreground">
              <CircleDashed className="h-6 w-6" aria-hidden="true" />
            </span>
            <CardTitle className="mt-3">No analysis to track</CardTitle>
            <CardDescription>
              There is no analysis id in this tab. Start by choosing a video, or open the demo
              result.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 sm:flex-row sm:justify-center">
            <Button asChild>
              <Link to="/upload">Choose a video</Link>
            </Button>
            <Button asChild variant="outline">
              <Link to="/">Back to dashboard</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container py-8 md:py-12">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <Button asChild variant="ghost" size="sm" className="-ml-2">
          <Link to="/upload">
            <ArrowLeft /> New analysis
          </Link>
        </Button>
        <StatusBadge status={status?.status} paused={paused} />
      </div>

      <header className="mb-8 max-w-3xl">
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">
          {status?.status === "failed"
            ? "Analysis failed"
            : status?.status === "completed"
              ? "Analysis complete"
              : "Processing your video"}
        </h1>
        <p className="mt-2 text-muted-foreground">
          {status?.detail ??
            (status?.status === "queued"
              ? "The analysis is queued behind other work. This page updates itself."
              : "Tracking, calibration, shot classification and result assembly.")}
        </p>
      </header>

      {fatalError ? (
        <Card className="mx-auto max-w-2xl">
          <CardContent className="space-y-4 p-6">
            <Alert variant="destructive">
              <AlertTitle>This analysis cannot be tracked</AlertTitle>
              <AlertDescription>{fatalError}</AlertDescription>
            </Alert>
            <div className="flex flex-wrap gap-2">
              <Button asChild>
                <Link to="/upload">Upload a video</Link>
              </Button>
              <Button asChild variant="outline">
                <Link to="/results">Open the stored result</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
          <Card className="overflow-hidden">
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <CardTitle>Pipeline</CardTitle>
                  <CardDescription>
                    The backend stage vocabulary comes from the contract, not a hardcoded guess.
                  </CardDescription>
                </div>
                <span className="font-mono text-xs text-muted-foreground">
                  {formatDuration(elapsed)} elapsed
                </span>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col items-center gap-4 rounded-2xl border border-border/70 bg-secondary/20 px-4 py-8 text-center sm:flex-row sm:text-left">
                <div className="relative flex h-24 w-24 shrink-0 items-center justify-center">
                  <span className="absolute inset-0 rounded-full border border-primary/20" aria-hidden="true" />
                  <span className="absolute inset-2 rounded-full border border-primary/30" aria-hidden="true" />
                  <span
                    className={cn(
                      "absolute inset-5 rounded-full",
                      status?.status === "failed"
                        ? "bg-destructive/20"
                        : "gradient-primary opacity-90 animate-pulse-glow",
                    )}
                    aria-hidden="true"
                  />
                  <Activity className="relative h-8 w-8 text-primary-foreground" aria-hidden="true" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-bold">
                    {status?.status === "failed"
                      ? "The pipeline stopped"
                      : status?.status === "completed"
                        ? "All stages finished"
                        : `${currentStage.label} — ${currentStage.description}`}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {status?.stage
                      ? `Stage: ${status.stage.replace(/_/g, " ")}`
                      : "Waiting for the first status update"}
                    {frameText ? ` · ${frameText}` : ""}
                  </p>
                  <div className="mt-3 space-y-1.5">
                    <Progress
                      value={progress ?? null}
                      label="Analysis progress"
                      className="h-2.5"
                      aria-valuetext={
                        typeof progress === "number"
                          ? `${Math.round(clampProgress(progress))}% reported by the backend`
                          : "Progress has not been reported"
                      }
                    />
                    <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                      <span>
                        {typeof progress === "number"
                          ? `${Math.round(clampProgress(progress))}% reported`
                          : "Progress not reported"}
                      </span>
                      <span>{status?.substage ? status.substage.replace(/_/g, " ") : ""}</span>
                    </div>
                  </div>
                </div>
              </div>

              <ol className="space-y-0" aria-label="Processing stages">
                {PIPELINE_STAGES.map((stage, index) => {
                  const state = states[index] ?? "pending";
                  const Icon = STAGE_ICONS[stage.id] ?? CircleDashed;
                  return (
                    <li key={stage.id} className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <span
                          className={cn(
                            "flex h-9 w-9 shrink-0 items-center justify-center rounded-full border",
                            STATE_STYLES[state],
                          )}
                          aria-hidden="true"
                        >
                          {state === "done" ? (
                            <Check className="h-4 w-4" />
                          ) : state === "failed" ? (
                            <X className="h-4 w-4" />
                          ) : state === "active" ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Icon className="h-4 w-4" />
                          )}
                        </span>
                        {index < PIPELINE_STAGES.length - 1 && (
                          <span
                            className={cn(
                              "my-1 h-6 w-px",
                              state === "done" ? "bg-success/50" : "bg-border",
                            )}
                            aria-hidden="true"
                          />
                        )}
                      </div>
                      <div className="flex min-w-0 flex-1 items-start justify-between gap-3 pb-4">
                        <div>
                          <p
                            className={cn(
                              "text-sm font-semibold",
                              state === "pending" && "text-muted-foreground",
                            )}
                          >
                            {stage.label}
                          </p>
                          <p className="text-xs text-muted-foreground">{stage.description}</p>
                        </div>
                        <span className="sr-only">
                          {state === "done"
                            ? "complete"
                            : state === "active"
                              ? "in progress"
                              : state === "failed"
                                ? "failed"
                                : "pending"}
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ol>

              {pollWarning && (
                <Alert variant="warning">
                  <AlertTitle>Connection interrupted</AlertTitle>
                  <AlertDescription className="flex flex-wrap items-center gap-3">
                    <span>{pollWarning}</span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        setPaused(false);
                        setRetryNonce((value) => value + 1);
                      }}
                    >
                      <RefreshCw /> Retry now
                    </Button>
                  </AlertDescription>
                </Alert>
              )}

              {status?.status === "failed" && (
                <Alert variant="destructive" data-testid="processing-failed">
                  <AlertTitle>The analysis failed</AlertTitle>
                  <AlertDescription>
                    {status.error?.trim() || "The backend did not provide a reason."}
                  </AlertDescription>
                </Alert>
              )}

              {resultError && (
                <Alert variant="destructive">
                  <AlertTitle>Result could not be loaded</AlertTitle>
                  <AlertDescription className="flex flex-wrap items-center gap-3">
                    <span>{resultError}</span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setRetryNonce((value) => value + 1)}
                    >
                      <RefreshCw /> Try again
                    </Button>
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <aside className="space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">Run details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Analysis id
                  </p>
                  <div className="mt-1 flex items-center gap-1.5">
                    <code className="min-w-0 flex-1 truncate rounded-lg bg-secondary/60 px-2 py-1.5 font-mono text-xs">
                      {analysisId}
                    </code>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => void copyId()}
                      aria-label="Copy analysis id"
                    >
                      {copied ? <Check /> : <Copy />}
                    </Button>
                  </div>
                  <span className="sr-only" aria-live="polite">
                    {copied ? "Analysis id copied" : ""}
                  </span>
                </div>
                <Detail
                  label="Source"
                  value={status?.source_name ?? navState?.source_name ?? "not reported"}
                />
                <Detail
                  label="Elapsed"
                  value={`${formatDuration(elapsed)}${
                    status?.status === "completed" || status?.status === "failed" ? "" : " and counting"
                  }`}
                />
                <Detail
                  label="Progress"
                  value={
                    typeof progress === "number"
                      ? `${formatNumber(clampProgress(progress), 0)}%`
                      : "not reported"
                  }
                />
                <Detail label="Frame count" value={frameText ?? "not reported"} />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-2 p-4">
                {!terminal && (
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => {
                      setPaused((value) => !value);
                      setRetryNonce((value) => value + 1);
                    }}
                  >
                    {paused ? (
                      <>
                        <Play /> Resume polling
                      </>
                    ) : (
                      <>
                        <Pause /> Pause polling
                      </>
                    )}
                  </Button>
                )}
                {!terminal && (
                  <Button
                    variant="destructive"
                    className="w-full"
                    onClick={() => void cancelJob()}
                    disabled={cancelling}
                  >
                    {cancelling ? <Loader2 className="animate-spin" /> : <X />}
                    {cancelling ? "Cancelling…" : "Cancel backend analysis"}
                  </Button>
                )}
                {cancelError && (
                  <p className="text-[11px] leading-snug text-destructive" role="alert">
                    {cancelError}
                  </p>
                )}
                <Button asChild variant="ghost" className="w-full">
                  <Link to="/upload">Analyse another video</Link>
                </Button>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Pausing only stops this page from polling. Cancelling asks the
                  backend to stop; the current model call must finish before the
                  job reaches its terminal state.
                </p>
              </CardContent>
            </Card>

            {loadingResult && (
              <Card>
                <CardContent className="space-y-3 p-4" aria-live="polite">
                  <p className="flex items-center gap-2 text-sm font-semibold">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
                    Loading the completed result…
                  </p>
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-2/3" />
                  <Skeleton className="h-24 w-full" />
                </CardContent>
              </Card>
            )}

            <Card className="border-warning/30 bg-warning/5">
              <CardContent className="space-y-2 p-4">
                <p className="flex items-center gap-2 text-sm font-semibold">
                  <FlaskConical className="h-4 w-4 text-warning" /> Demo alternative
                </p>
                <p className="text-xs text-muted-foreground">{DEMO_NOTICE}</p>
                <Button asChild variant="outline" className="w-full">
                  <Link to="/results" state={{ analysis_id: DEMO_ANALYSIS_ID, demo: true }}>
                    Open demo result
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </aside>
        </div>
      )}

      <p className="mt-6 text-xs text-muted-foreground">
        <AlertTriangle className="mr-1 inline h-3 w-3" aria-hidden="true" />
        The progress bar shows the backend's reported progress. If none is reported, the bar is
        indeterminate and elapsed time is shown instead of an invented time estimate.
      </p>
    </div>
  );
};

function StatusBadge({ status, paused }: { status?: string | null; paused: boolean }) {
  if (!status) return <Badge variant="secondary">waiting for status</Badge>;
  if (status === "completed") {
    return (
      <Badge variant="success">
        <Check /> completed
      </Badge>
    );
  }
  if (status === "failed") {
    return (
      <Badge variant="destructive">
        <X /> failed
      </Badge>
    );
  }
  return (
    <Badge variant={paused ? "warning" : "default"}>
      {paused ? <Pause /> : <Loader2 className="animate-spin" />}
      {paused ? "polling stopped" : status}
    </Badge>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="max-w-[60%] break-words text-right text-xs font-medium">{value}</span>
    </div>
  );
}

export default Processing;
