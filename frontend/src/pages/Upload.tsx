import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  CheckCircle2,
  FileVideo,
  FlaskConical,
  Loader2,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { uploadVideo, ApiError } from "@/lib/api";
import { DEMO_ANALYSIS_ID, DEMO_NOTICE } from "@/lib/demo";
import { formatBytes } from "@/lib/format";
import { clearStoredAnalysis, setStoredAnalysisId } from "@/lib/storage";
import { fileExtension, validateVideoFile } from "@/lib/validation";
import { cn } from "@/lib/utils";

type UploadPhase = "idle" | "uploading" | "error";

interface UploadProgress {
  loaded: number;
  total: number;
  percent: number | null;
}

const LIMITS = [
  "MP4, AVI, MOV, MKV or WEBM",
  "Up to 2 GB per video",
  "A single delivery or a full spell",
  "Processed locally by the backend",
];

const NEXT_STEPS = [
  { title: "Validate", detail: "The type, size and container are checked before upload." },
  { title: "Track", detail: "The pipeline detects the ball and segments each delivery." },
  { title: "Analyse", detail: "Pitch geometry, shot labels and bowling metrics are computed." },
  { title: "Review", detail: "You land on the result with playback, overlays and filters." },
];

const UploadPage = () => {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);
  const objectUrlRef = useRef<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const releasePreview = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    abortRef.current?.abort();
  }, []);

  const chooseFile = useCallback(
    (next: File | null) => {
      if (!next) return;
      setError(null);
      setProgress(null);
      setPhase("idle");

      const validation = validateVideoFile(next);
      if (!validation.ok) {
        releasePreview();
        setFile(null);
        setValidationError(validation.error);
        return;
      }

      setValidationError(null);
      setFile(next);
      releasePreview();
      const url = URL.createObjectURL(next);
      objectUrlRef.current = url;
      setPreviewUrl(url);
    },
    [releasePreview],
  );

  const removeFile = () => {
    if (phase === "uploading") return;
    setFile(null);
    setValidationError(null);
    setError(null);
    setProgress(null);
    releasePreview();
    if (inputRef.current) inputRef.current.value = "";
  };

  const startUpload = async () => {
    if (!file || phase === "uploading") return;
    const validation = validateVideoFile(file);
    if (!validation.ok) {
      setValidationError(validation.error);
      return;
    }

    // A new analysis invalidates the previous result before the first byte
    // leaves, so the old result cannot appear on the processing screen.
    clearStoredAnalysis();
    setError(null);
    setPhase("uploading");
    setProgress({ loaded: 0, total: file.size, percent: 0 });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await uploadVideo(file, {
        signal: controller.signal,
        onProgress: (value) => setProgress(value),
      });
      setStoredAnalysisId(response.analysis_id);
      navigate("/processing", {
        state: {
          analysis_id: response.analysis_id,
          source_name: file.name,
          started_at: Date.now(),
        },
      });
    } catch (caught) {
      if (caught instanceof ApiError && caught.kind === "aborted") {
        setPhase("idle");
        setProgress(null);
        setError("Upload cancelled. The video was not submitted.");
      } else {
        setPhase("error");
        setError(
          caught instanceof Error
            ? caught.message
            : "The upload failed for an unknown reason. Try again.",
        );
      }
    } finally {
      abortRef.current = null;
    }
  };

  const cancelUpload = () => {
    abortRef.current?.abort();
  };

  const openDemo = () => {
    setStoredAnalysisId(DEMO_ANALYSIS_ID);
    navigate("/results", { state: { analysis_id: DEMO_ANALYSIS_ID, demo: true } });
  };

  return (
    <div className="container py-8 md:py-12">
      <header className="mb-8 max-w-3xl">
        <Badge variant="outline" className="mb-3 border-primary/30 text-primary">
          <UploadCloud className="h-3 w-3" /> New analysis
        </Badge>
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Analyse a video</h1>
        <p className="mt-2 text-muted-foreground">
          Choose one clip. It is validated here, uploaded once and processed into tracked
          deliveries, a calibrated pitch, shot labels and bowling metrics.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-5">
          <input
            ref={inputRef}
            id="video-file"
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.webm,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
            className="sr-only"
            tabIndex={-1}
            onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
            data-testid="video-input"
          />

          {!file ? (
            <div
              onDragEnter={(event) => {
                event.preventDefault();
                dragDepth.current += 1;
                setDragging(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = "copy";
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                dragDepth.current = Math.max(0, dragDepth.current - 1);
                if (dragDepth.current === 0) setDragging(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                dragDepth.current = 0;
                setDragging(false);
                const dropped = event.dataTransfer.files?.[0];
                if (dropped) chooseFile(dropped);
              }}
              className={cn(
                "rounded-3xl border-2 border-dashed transition-smooth",
                dragging
                  ? "border-primary bg-primary/10 shadow-elegant"
                  : "border-border bg-card/40 hover:border-primary/40",
              )}
            >
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="flex w-full flex-col items-center justify-center gap-4 rounded-3xl px-6 py-16 text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background md:py-24"
                aria-describedby="upload-limits"
              >
                <span className="flex h-16 w-16 items-center justify-center rounded-2xl gradient-primary shadow-glow">
                  <UploadCloud className="h-8 w-8 text-primary-foreground" aria-hidden="true" />
                </span>
                <span className="space-y-1">
                  <span className="block text-lg font-bold">Drop a video here</span>
                  <span className="block text-sm text-muted-foreground">
                    or click to choose a file
                  </span>
                </span>
                <span id="upload-limits" className="text-xs text-muted-foreground">
                  MP4 · AVI · MOV · MKV · WEBM — up to 2 GB
                </span>
              </button>
            </div>
          ) : (
            <Card className="overflow-hidden">
              <CardHeader className="border-b border-border/60 pb-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/15 text-primary">
                      <FileVideo className="h-5 w-5" aria-hidden="true" />
                    </span>
                    <div className="min-w-0">
                      <CardTitle className="truncate text-base" title={file.name}>
                        {file.name}
                      </CardTitle>
                      <CardDescription>
                        {formatBytes(file.size)} · .{fileExtension(file.name) ?? "video"}
                        {phase === "uploading" ? " · uploading" : ""}
                      </CardDescription>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={removeFile}
                    disabled={phase === "uploading"}
                  >
                    <X /> Remove
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                {previewUrl && (
                  <video
                    src={previewUrl}
                    controls
                    className="max-h-[420px] w-full rounded-xl border border-border bg-black"
                    aria-label={`Local preview of ${file.name}`}
                  />
                )}

                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="success">
                    <CheckCircle2 /> ready to upload
                  </Badge>
                  <Badge variant="outline">single file</Badge>
                  <Badge variant="outline">validated locally</Badge>
                </div>

                {phase === "uploading" && progress && (
                  <div className="space-y-2" aria-live="polite" data-testid="upload-progress">
                    <div className="flex items-center justify-between gap-3 text-sm">
                      <span className="flex items-center gap-2 font-medium">
                        <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
                        Uploading
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {progress.percent === null
                          ? formatBytes(progress.loaded)
                          : `${Math.round(progress.percent)}%`}
                      </span>
                    </div>
                    <Progress
                      value={progress.percent ?? 0}
                      label="Upload progress"
                      aria-valuetext={
                        progress.percent === null
                          ? `${formatBytes(progress.loaded)} uploaded`
                          : `${Math.round(progress.percent)}% uploaded`
                      }
                    />
                    <div className="flex justify-between gap-3 text-[11px] text-muted-foreground">
                      <span>
                        {formatBytes(progress.loaded)}
                        {progress.total > 0 ? ` of ${formatBytes(progress.total)}` : ""}
                      </span>
                      <span>Keep this tab open until processing starts.</span>
                    </div>
                  </div>
                )}

                {error && (
                  <Alert variant="destructive">
                    <AlertTitle>Upload failed</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <div className="flex flex-wrap gap-2">
                  {phase === "uploading" ? (
                    <Button variant="outline" onClick={cancelUpload}>
                      <X /> Cancel upload
                    </Button>
                  ) : (
                    <Button onClick={() => void startUpload()}>
                      {phase === "error" ? "Try upload again" : "Start analysis"} <ArrowRight />
                    </Button>
                  )}
                  <Button asChild variant="outline">
                    <Link to="/">Cancel</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {validationError && (
            <Alert variant="destructive" data-testid="validation-error">
              <AlertTitle>That file cannot be analysed</AlertTitle>
              <AlertDescription>{validationError}</AlertDescription>
            </Alert>
          )}

          <Alert variant="info">
            <AlertTitle>Uploads need the processing API</AlertTitle>
            <AlertDescription>
              This build wires the documented <span className="font-mono">POST /api/analyze</span>{" "}
              route. If the backend is not running, the upload fails with a connection message
              instead of pretending to analyse your file. The demo dataset is the labelled
              alternative.
            </AlertDescription>
          </Alert>
        </div>

        <aside className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">What happens next</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-3">
                {NEXT_STEPS.map((step, index) => (
                  <li key={step.title} className="flex gap-3">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
                      {index + 1}
                    </span>
                    <div>
                      <p className="text-sm font-semibold">{step.title}</p>
                      <p className="text-xs leading-snug text-muted-foreground">{step.detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Accepted input</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                {LIMITS.map((limit) => (
                  <li key={limit} className="flex items-start gap-2">
                    <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
                    {limit}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="border-warning/30 bg-warning/5">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <FlaskConical className="h-4 w-4 text-warning" /> No backend handy?
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs leading-relaxed text-muted-foreground">{DEMO_NOTICE}</p>
              <Button variant="outline" className="w-full" onClick={openDemo}>
                <FlaskConical /> Open demo result
              </Button>
            </CardContent>
          </Card>

          <Alert variant="warning">
            <AlertDescription>
              Uploading does not guarantee a result. The backend may reject a video it cannot
              decode; that failure is shown on the processing screen with its reason.
            </AlertDescription>
          </Alert>
        </aside>
      </div>
    </div>
  );
};

export default UploadPage;
