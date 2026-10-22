import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  CircleDot,
  Clock3,
  FlaskConical,
  Gauge,
  Layers,
  MapPin,
  Server,
  ShieldCheck,
  Upload,
  WifiOff,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  API_BASE,
  CONTRACT_SCHEMA_VERSION,
  fetchHealth,
  type Health,
} from "@/lib/api";
import { DEMO_ANALYSIS_ID, DEMO_NOTICE, isDemoAnalysisId } from "@/lib/demo";
import { loadCachedResult, getStoredAnalysisId, setStoredAnalysisId } from "@/lib/storage";
import { formatCount } from "@/lib/format";

const FEATURES = [
  {
    icon: CircleDot,
    title: "Ball track",
    description:
      "Detected points are drawn solid; the tracker's predicted tail is dashed. The two are never merged into one confident line.",
  },
  {
    icon: MapPin,
    title: "Pitch calibration",
    description:
      "Corners, keypoints and the batting-end orientation appear when the pitch model found them. Three keypoints never become an invented fourth corner.",
  },
  {
    icon: BarChart3,
    title: "Shot analytics",
    description:
      "The frozen 10-class vocabulary, label confidence and the unclassified count. An unavailable classifier is stated, not filled with a guess.",
  },
  {
    icon: Gauge,
    title: "Bowling metrics",
    description:
      "Speed, line, length and swing appear only where the measured ground plane supports them. Speed is always labelled an estimate.",
  },
];

const STEPS = [
  { title: "Upload a clip", detail: "MP4, AVI, MOV, MKV or WEBM up to 2 GB. Rejected at the door, not minutes later." },
  { title: "Track once", detail: "One decode pass: detect the ball, track it, segment deliveries." },
  { title: "Calibrate and classify", detail: "Pitch geometry, shot labels and bowling analytics travel with every delivery." },
  { title: "Review the result", detail: "Clip timeline, overlays, pitch map and a searchable delivery explorer." },
];

type HealthState =
  | { state: "checking" }
  | { state: "online"; health: Health }
  | { state: "offline"; message: string };

const Dashboard = () => {
  const navigate = useNavigate();
  const [health, setHealth] = useState<HealthState>({ state: "checking" });
  const storedId = getStoredAnalysisId();
  const cached = loadCachedResult(storedId);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth({ signal: controller.signal })
      .then((value) => setHealth({ state: "online", health: value }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setHealth({
          state: "offline",
          message: error instanceof Error ? error.message : "The backend did not answer.",
        });
      });
    return () => controller.abort();
  }, []);

  const openDemo = () => {
    setStoredAnalysisId(DEMO_ANALYSIS_ID);
    navigate("/results", { state: { analysis_id: DEMO_ANALYSIS_ID, demo: true } });
  };

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[520px] gradient-hero opacity-70" aria-hidden="true" />
      <div className="pointer-events-none absolute -left-40 top-20 h-96 w-96 rounded-full bg-primary/10 blur-3xl animate-float" aria-hidden="true" />

      <div className="container relative py-10 md:py-16">
        <section className="grid items-center gap-10 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <div className="animate-fade-in">
            <Badge variant="outline" className="mb-4 border-primary/30 bg-background/60 text-primary">
              <Activity className="h-3 w-3" /> Cricket video intelligence
            </Badge>
            <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
              See every ball.
              <br />
              <span className="text-gradient">Understand every delivery.</span>
            </h1>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground md:text-lg">
              One video in: ball tracking, pitch calibration, shot classification and bowling
              analytics out. Every number is tied to the evidence that produced it.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Button asChild size="lg">
                <Link to="/upload">
                  <Upload /> Analyse a video
                </Link>
              </Button>
              <Button size="lg" variant="outline" onClick={openDemo}>
                <FlaskConical /> Explore demo data
              </Button>
              {storedId && (
                <Button asChild size="lg" variant="ghost">
                  <Link to="/results">
                    Open current result <ArrowRight />
                  </Link>
                </Button>
              )}
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Demo mode is labelled everywhere and never presented as a model run.
            </p>

            <dl className="mt-8 grid max-w-lg grid-cols-3 gap-3">
              {[
                { value: "10", label: "shot classes" },
                { value: "1", label: "decode pass" },
                { value: "6", label: "API routes" },
              ].map((item) => (
                <div key={item.label} className="rounded-xl border border-border/70 bg-card/60 p-3 backdrop-blur">
                  <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">{item.label}</dt>
                  <dd className="mt-1 text-2xl font-extrabold">{item.value}</dd>
                </div>
              ))}
            </dl>
          </div>

          <HeroVisual />
        </section>

        <section aria-labelledby="status-heading" className="mt-12 grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
          <Card className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle id="status-heading">Pipeline at a glance</CardTitle>
                <Badge variant="outline">schema {CONTRACT_SCHEMA_VERSION}</Badge>
              </div>
              <CardDescription>
                The frontend speaks the documented six-route contract. Part 5 brings the processing
                server online behind it.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ol className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {STEPS.map((step, index) => (
                  <li key={step.title} className="rounded-xl border border-border/70 bg-secondary/20 p-3">
                    <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
                      {index + 1}
                    </span>
                    <p className="mt-2 text-sm font-semibold">{step.title}</p>
                    <p className="mt-1 text-xs leading-snug text-muted-foreground">{step.detail}</p>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Server className="h-4 w-4 text-primary" /> Backend status
                </CardTitle>
              </CardHeader>
              <CardContent>
                {health.state === "checking" ? (
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-3 w-full" />
                    <span className="sr-only">Checking backend</span>
                  </div>
                ) : health.state === "online" ? (
                  <div className="space-y-2">
                    <Badge variant="success">
                      <CheckCircle2 /> online
                    </Badge>
                    <p className="break-all font-mono text-xs text-muted-foreground">{API_BASE}</p>
                    <p className="text-xs text-muted-foreground">
                      schema {health.health.schema_version ?? "unknown"} · pipeline{" "}
                      {health.health.pipeline_version ?? "unknown"}
                    </p>
                    <Button asChild size="sm" className="mt-1 w-full">
                      <Link to="/upload">Start an analysis</Link>
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Badge variant="warning">
                      <WifiOff /> not connected
                    </Badge>
                    <p className="text-xs leading-relaxed text-muted-foreground">{health.message}</p>
                    <p className="break-all font-mono text-[11px] text-muted-foreground">{API_BASE}</p>
                    <Button size="sm" variant="outline" className="w-full" onClick={openDemo}>
                      <FlaskConical /> Use demo data instead
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {storedId && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Clock3 className="h-4 w-4 text-primary" /> Current analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={isDemoAnalysisId(storedId) ? "warning" : "success"}>
                      {isDemoAnalysisId(storedId) ? "demo" : "stored locally"}
                    </Badge>
                    {cached && (
                      <span className="text-xs text-muted-foreground">
                        {formatCount(cached.deliveries.length, "delivery", "deliveries")}
                      </span>
                    )}
                  </div>
                  <p className="break-all font-mono text-xs text-muted-foreground">{storedId}</p>
                  <Button asChild size="sm" variant="outline" className="w-full">
                    <Link to="/results">
                      Open result <ArrowRight />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </section>

        <section aria-labelledby="features-heading" className="mt-12">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 id="features-heading" className="text-2xl font-bold tracking-tight">
                Built around evidence
              </h2>
              <p className="text-sm text-muted-foreground">
                The interface reflects what the pipeline measured — including what it could not.
              </p>
            </div>
            <Badge variant="outline">
              <ShieldCheck /> honest by construction
            </Badge>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {FEATURES.map(({ icon: Icon, title, description }) => (
              <Card key={title} className="transition-smooth hover:-translate-y-1 hover:border-primary/30">
                <CardHeader className="pb-2">
                  <span className="mb-1 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/12 text-primary">
                    <Icon className="h-5 w-5" aria-hidden="true" />
                  </span>
                  <CardTitle className="text-base">{title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-muted-foreground">{description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="mt-12">
          <Alert variant="info">
            <AlertTitle>What this build is</AlertTitle>
            <AlertDescription>
              Part 4 is the complete frontend: dashboard, upload and validation, processing states,
              playback and timeline, trajectory and pitch overlays, shot analytics and delivery
              search. The API client targets the documented contract; until the Part 5 server is
              running, a live upload reports the connection failure and the demo dataset is the
              explicitly labelled way to review the interface.
            </AlertDescription>
          </Alert>
          <p className="mt-3 text-xs text-muted-foreground">{DEMO_NOTICE}</p>
        </section>
      </div>
    </div>
  );
};

function HeroVisual() {
  return (
    <div className="relative mx-auto w-full max-w-md animate-scale-in">
      <div className="absolute inset-0 -z-10 rounded-[2rem] bg-primary/20 blur-3xl" aria-hidden="true" />
      <div className="overflow-hidden rounded-3xl border border-border/80 bg-[#071019] shadow-elegant">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <span className="flex items-center gap-2 text-xs font-semibold text-white/80">
            <Layers className="h-4 w-4 text-primary" /> Delivery preview
          </span>
          <span className="rounded-full border border-white/15 px-2 py-0.5 text-[10px] font-semibold text-white/60">
            ILLUSTRATIVE
          </span>
        </div>
        <svg viewBox="0 0 480 300" className="w-full" role="img" aria-label="Illustrative trajectory and pitch overlay">
          <defs>
            <linearGradient id="hero-field" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#08131f" />
              <stop offset="100%" stopColor="#0d2f27" />
            </linearGradient>
            <linearGradient id="hero-trajectory" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#fb923c" />
            </linearGradient>
          </defs>
          <rect width="480" height="300" fill="url(#hero-field)" />
          <ellipse cx="240" cy="240" rx="210" ry="110" fill="#0f3b2d" />
          <polygon points="196,96 236,96 312,286 136,286" fill="#b58a52" fillOpacity="0.32" stroke="#fef3c7" strokeOpacity="0.25" />
          <g stroke="#fef3c7" strokeOpacity="0.2">
            <line x1="205" y1="120" x2="229" y2="120" />
            <line x1="174" y1="246" x2="270" y2="246" />
          </g>
          <path
            d="M 232 108 C 250 150, 258 206, 246 250"
            fill="none"
            stroke="url(#hero-trajectory)"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <path
            d="M 246 250 C 240 262, 234 268, 228 274"
            fill="none"
            stroke="#94a3b8"
            strokeWidth="2.5"
            strokeDasharray="6 6"
            strokeLinecap="round"
          />
          {[
            [232, 108, "#38bdf8"],
            [250, 158, "#38bdf8"],
            [252, 206, "#38bdf8"],
            [246, 250, "#f472b6"],
            [228, 274, "#94a3b8"],
          ].map(([cx, cy, color], index) => (
            <circle
              key={index}
              cx={cx as number}
              cy={cy as number}
              r={index === 3 ? 7 : 4}
              fill={color as string}
              stroke="#050b16"
              strokeWidth="1.5"
            />
          ))}
          <g transform="translate(310,52)">
            <rect width="126" height="58" rx="10" fill="#050b16" fillOpacity="0.82" stroke="#ffffff" strokeOpacity="0.12" />
            <text x="12" y="20" fill="#94a3b8" fontSize="9" fontWeight="700" letterSpacing="1">TRACKED</text>
            <text x="12" y="37" fill="#38bdf8" fontSize="11" fontWeight="700"> </text>
            <circle cx="16" cy="42" r="3" fill="#38bdf8" />
            <text x="26" y="45" fill="#e2e8f0" fontSize="9">detected</text>
            <circle cx="84" cy="42" r="3" fill="#94a3b8" />
            <text x="94" y="45" fill="#e2e8f0" fontSize="9">predicted</text>
          </g>
          <text x="20" y="280" fill="#fef3c7" fillOpacity="0.4" fontSize="8" letterSpacing="2">NOT A REAL ANALYSIS</text>
        </svg>
      </div>
    </div>
  );
}

export default Dashboard;
