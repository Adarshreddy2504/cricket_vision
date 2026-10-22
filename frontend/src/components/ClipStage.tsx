/**
 * ClipStage.tsx — clip playback, timeline controls and the two overlays.
 *
 * Real analyses play the delivery clip served at `video.clip_url` and draw the
 * trajectory/pitch in the clip's own source-pixel coordinate system.
 *
 * Demo mode ships no footage. It renders a *synthetic tracking preview* —
 * labelled as synthetic on the stage — driven by the same timeline controls.
 * That keeps playback and the overlays reviewable without shipping a video or
 * pretending a clip exists.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Eye,
  EyeOff,
  Film,
  FlaskConical,
  Maximize,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import TrajectoryOverlay from "./TrajectoryOverlay";
import PitchOverlay from "./PitchOverlay";
import { mediaUrl, type DeliveryVideo, type PitchBlock, type TrajectoryPoint } from "@/lib/api";
import { formatClock, formatNumber } from "@/lib/format";

interface Props {
  video: DeliveryVideo;
  trajectory: TrajectoryPoint[];
  bounceFrame: number | null;
  pitch: PitchBlock;
  title: string;
  /** Demo mode: no clip is shipped, so render the labelled synthetic stage. */
  demo?: boolean;
}

const RATES = [0.25, 0.5, 1, 1.5, 2];

const ClipStage = ({ video, trajectory, bounceFrame, pitch, title, demo = false }: Props) => {
  const clipUrl = mediaUrl(video.clip_url);
  const synthetic = !clipUrl && demo;

  const fps = typeof video.fps === "number" && video.fps > 0 ? video.fps : null;
  const startFrame = video.start_frame ?? 1;
  const endFrame =
    video.end_frame ??
    (fps && typeof video.duration_seconds === "number"
      ? startFrame + Math.max(1, Math.round(video.duration_seconds * fps)) - 1
      : startFrame + 95);
  const duration =
    typeof video.duration_seconds === "number" && video.duration_seconds > 0
      ? video.duration_seconds
      : fps
        ? (endFrame - startFrame + 1) / fps
        : 0;

  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [rate, setRate] = useState(1);
  const [showOverlay, setShowOverlay] = useState(true);
  const [videoError, setVideoError] = useState(false);
  const [videoKey, setVideoKey] = useState(0);

  const currentFrame = fps ? Math.min(endFrame, startFrame + Math.round(time * fps)) : startFrame;
  const progress = duration > 0 ? Math.min(1, time / duration) : 0;

  const seek = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(duration, next));
      setTime(clamped);
      if (videoRef.current) {
        try {
          videoRef.current.currentTime = clamped;
        } catch {
          // Metadata may not be loaded yet; the next timeupdate corrects us.
        }
      }
    },
    [duration],
  );

  const togglePlay = useCallback(() => {
    if (synthetic) {
      setPlaying((value) => !value);
      return;
    }
    const element = videoRef.current;
    if (!element) return;
    if (element.paused) {
      void element.play().catch(() => setVideoError(true));
    } else {
      element.pause();
    }
  }, [synthetic]);

  const stepFrame = useCallback(
    (direction: -1 | 1) => {
      if (!fps) return;
      const frame = Math.max(startFrame, Math.min(endFrame, currentFrame + direction));
      seek((frame - startFrame) / fps);
    },
    [currentFrame, endFrame, fps, seek, startFrame],
  );

  // Synthetic playback clock. Real playback is driven by the element's own
  // timeupdate so audio and picture stay together.
  useEffect(() => {
    if (!synthetic || !playing) return;
    let frameRequest = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const delta = (now - last) / 1000;
      last = now;
      setTime((current) => {
        const next = current + delta * rate;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      frameRequest = requestAnimationFrame(tick);
    };
    frameRequest = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRequest);
  }, [duration, playing, rate, synthetic]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate, videoKey]);

  useEffect(() => {
    // A different delivery is a different clip: reset the stage.
    setTime(0);
    setPlaying(false);
    setVideoError(false);
  }, [clipUrl, startFrame, endFrame]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (["INPUT", "SELECT", "BUTTON", "A", "TEXTAREA"].includes(target.tagName)) return;
    if (event.key === " " || event.key === "k") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      if (fps) stepFrame(-1);
      else seek(time - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      if (fps) stepFrame(1);
      else seek(time + 1);
    }
  };

  const nearestPoint = useMemo(() => {
    if (!synthetic) return null;
    return trajectory
      .filter((point) => point.frame <= currentFrame)
      .reduce<TrajectoryPoint | null>(
        (best, point) => (!best || point.frame > best.frame ? point : best),
        null,
      );
  }, [currentFrame, synthetic, trajectory]);

  const width = video.width ?? 1280;
  const height = video.height ?? 720;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className="overflow-hidden rounded-3xl border border-border gradient-card shadow-elegant focus-visible:ring-2 focus-visible:ring-ring"
      aria-label={`${title} clip player`}
      data-testid="clip-stage"
    >
      <div
        className="relative w-full overflow-hidden bg-black"
        style={{ aspectRatio: `${width} / ${height}` }}
      >
        {clipUrl ? (
          <video
            key={`${clipUrl}-${videoKey}`}
            ref={videoRef}
            src={clipUrl}
            className="h-full w-full object-contain"
            playsInline
            muted={muted}
            preload="metadata"
            onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => {
              setPlaying(false);
              setTime(duration);
            }}
            onError={() => {
              setVideoError(true);
              setPlaying(false);
            }}
            aria-label={`${title} delivery clip`}
          />
        ) : synthetic ? (
          <SyntheticStage
            width={width}
            height={height}
            currentFrame={currentFrame}
            showOverlay={showOverlay}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-white/80">
            <Film className="h-6 w-6" aria-hidden="true" />
            <p className="font-semibold">No clip media for this delivery</p>
            <p className="max-w-sm text-xs text-white/60">
              The result document carries no playable clip. Trajectory and pitch data are still shown
              below where the backend recorded them.
            </p>
          </div>
        )}

        {showOverlay && !videoError && (
          <>
            <TrajectoryOverlay
              points={trajectory}
              bounceFrame={bounceFrame}
              width={width}
              height={height}
              frameLimit={synthetic ? currentFrame : null}
            />
            <PitchOverlay pitch={pitch} width={width} height={height} />
            {synthetic && nearestPoint && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <svg
                  className="absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${width} ${height}`}
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  <circle
                    cx={nearestPoint.x}
                    cy={nearestPoint.y}
                    r={Math.max(5, Math.min(width, height) * 0.012)}
                    fill="#fef08a"
                    stroke="#0b1220"
                    strokeWidth="2"
                  />
                </svg>
              </div>
            )}
          </>
        )}

        <div className="absolute left-4 top-4 flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-white/20 bg-black/65 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
            {title}
          </span>
          {synthetic && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-warning/40 bg-warning/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
              <FlaskConical className="h-3 w-3" aria-hidden="true" /> SYNTHETIC PREVIEW
            </span>
          )}
        </div>

        {videoError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/80 px-6 text-center text-sm text-white">
            <Film className="h-6 w-6" aria-hidden="true" />
            <p className="font-semibold">This clip could not be played</p>
            <p className="max-w-sm text-xs text-white/70">
              The file may be unavailable or the browser may not support its codec. The analysis
              data is unaffected.
            </p>
            <Button
              size="sm"
              variant="outline"
              className="mt-2 border-white/30 bg-white/10 text-white hover:bg-white/20"
              onClick={() => {
                setVideoError(false);
                setVideoKey((key) => key + 1);
              }}
            >
              Try again
            </Button>
          </div>
        )}
      </div>

      <div className="border-t border-border bg-background/70 p-3 backdrop-blur md:p-4">
        <div className="flex items-center gap-2 md:gap-3">
          <Button
            variant="default"
            size="icon"
            onClick={togglePlay}
            aria-label={playing ? "Pause clip" : "Play clip"}
            data-testid="clip-play"
            disabled={Boolean(videoError) || (!clipUrl && !synthetic)}
          >
            {playing ? <Pause /> : <Play />}
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => {
              setPlaying(false);
              seek(0);
            }}
            aria-label="Restart clip"
          >
            <RotateCcw />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => stepFrame(-1)}
            disabled={!fps}
            aria-label="Previous frame"
            title={fps ? "Previous frame" : "Frame rate unavailable"}
          >
            <SkipBack />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => stepFrame(1)}
            disabled={!fps}
            aria-label="Next frame"
            title={fps ? "Next frame" : "Frame rate unavailable"}
          >
            <SkipForward />
          </Button>

          <div className="min-w-0 flex-1">
            <label htmlFor="clip-timeline" className="sr-only">
              Clip timeline
            </label>
            <input
              id="clip-timeline"
              data-testid="clip-timeline"
              type="range"
              min={0}
              max={Math.max(duration, 0.001)}
              step={fps ? 1 / fps : 0.01}
              value={Math.min(time, duration)}
              onChange={(event) => seek(Number(event.target.value))}
              disabled={duration <= 0}
              aria-valuetext={`${formatClock(time)} of ${formatClock(duration)}`}
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-primary"
            />
            <div className="mt-1.5 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
              <span className="font-mono tabular-nums">
                {formatClock(time)} / {formatClock(duration)}
              </span>
              <span className="hidden font-mono tabular-nums sm:inline">
                frame {currentFrame} · {Math.round(progress * 100)}%
              </span>
            </div>
          </div>

          <div className="hidden items-center gap-2 sm:flex">
            <label htmlFor="clip-rate" className="sr-only">
              Playback speed
            </label>
            <select
              id="clip-rate"
              value={rate}
              onChange={(event) => setRate(Number(event.target.value))}
              className="h-8 rounded-lg border border-border bg-background px-2 text-xs font-medium"
            >
              {RATES.map((value) => (
                <option key={value} value={value}>
                  {value}×
                </option>
              ))}
            </select>
            {clipUrl && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => setMuted((value) => !value)}
                aria-label={muted ? "Unmute clip" : "Mute clip"}
                aria-pressed={muted}
              >
                {muted ? <VolumeX /> : <Volume2 />}
              </Button>
            )}
            {clipUrl && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => {
                  const stage = containerRef.current;
                  if (!stage) return;
                  if (document.fullscreenElement) void document.exitFullscreen();
                  else void stage.requestFullscreen?.().catch(() => undefined);
                }}
                aria-label="Toggle fullscreen"
              >
                <Maximize />
              </Button>
            )}
          </div>

          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setShowOverlay((value) => !value)}
            aria-label={showOverlay ? "Hide overlays" : "Show overlays"}
            aria-pressed={showOverlay}
            title={showOverlay ? "Hide overlays" : "Show overlays"}
          >
            {showOverlay ? <Eye /> : <EyeOff />}
          </Button>
        </div>

        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
          <span>
            {synthetic
              ? "No clip media in demo mode — this preview animates the synthetic trajectory only."
              : `${formatNumber(video.width, 0)}×${formatNumber(video.height, 0)} · ${
                  fps ? `${fps} fps` : "frame rate unavailable"
                }`}
          </span>
          <span className="hidden md:inline">Space play/pause · ←/→ step one frame</span>
        </div>
      </div>
    </div>
  );
};

/** A broadcast-shaped field backdrop. Explicitly not footage. */
function SyntheticStage({
  width,
  height,
  currentFrame,
  showOverlay,
}: {
  width: number;
  height: number;
  currentFrame: number;
  showOverlay: boolean;
}) {
  return (
    <svg
      className="h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid slice"
      role="img"
      aria-label="Synthetic pitch preview — not real footage"
    >
      <defs>
        <linearGradient id="demo-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#050b16" />
          <stop offset="55%" stopColor="#0a1a2b" />
          <stop offset="100%" stopColor="#0e2a24" />
        </linearGradient>
        <radialGradient id="demo-light" cx="50%" cy="0%" r="80%">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="demo-strip" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7a5a36" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#b58a52" stopOpacity="0.75" />
        </linearGradient>
      </defs>
      <rect width={width} height={height} fill="url(#demo-sky)" />
      <rect width={width} height={height} fill="url(#demo-light)" />
      <ellipse cx={width / 2} cy={height * 0.78} rx={width * 0.52} ry={height * 0.34} fill="#0f3b2d" />
      <ellipse
        cx={width / 2}
        cy={height * 0.78}
        rx={width * 0.45}
        ry={height * 0.28}
        fill="none"
        stroke="#34d399"
        strokeOpacity="0.16"
        strokeWidth="2"
      />
      <polygon
        points={`${width * 0.39},${height * 0.43} ${width * 0.46},${height * 0.43} ${width * 0.62},${height * 0.97} ${width * 0.22},${height * 0.97}`}
        fill="url(#demo-strip)"
        stroke="#f5d0a9"
        strokeOpacity="0.18"
        strokeWidth="2"
      />
      {[0.72, 0.82, 0.92].map((y) => (
        <line
          key={y}
          x1={width * (0.49 - (y - 0.43) * 0.12)}
          y1={height * y}
          x2={width * (0.49 + (y - 0.43) * 0.12)}
          y2={height * y}
          stroke="#fef3c7"
          strokeOpacity="0.16"
          strokeWidth="1.5"
        />
      ))}
      <text
        x={width / 2}
        y={height * 0.16}
        textAnchor="middle"
        fill="#94a3b8"
        fillOpacity="0.7"
        fontSize={Math.max(12, width * 0.016)}
        fontWeight="700"
        letterSpacing="4"
      >
        CRICKETVISION · SYNTHETIC STAGE
      </text>
      <text
        x={width / 2}
        y={height * 0.91}
        textAnchor="middle"
        fill="#fef3c7"
        fillOpacity="0.55"
        fontSize={Math.max(10, width * 0.012)}
        fontWeight="600"
      >
        FRAME {currentFrame}
        {showOverlay ? " · TRAJECTORY REVEALED" : ""}
      </text>
    </svg>
  );
}

export default ClipStage;
