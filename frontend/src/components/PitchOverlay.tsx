/**
 * PitchOverlay.tsx — the pitch the model found, drawn over the delivery clip.
 *
 * calibrated          fresh quad, green
 * temporary_loss      last known quad, amber and dashed — stale is visible
 * no_calibration      chip only: there is no quad to draw
 * calibration_expired chip only: too old to trust
 *
 * Three keypoints are enough for a calibration but not for a quad, so only the
 * keypoints are drawn in that case. A fourth corner is never invented.
 */

import type { PitchBlock } from "@/lib/api";

interface Props {
  pitch?: PitchBlock | null;
  width?: number | null;
  height?: number | null;
}

const CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"];

const STATE_TEXT: Record<string, string> = {
  calibrated: "PITCH DETECTED",
  temporary_loss: "PREVIOUS CALIBRATION",
  no_calibration: "PITCH NOT DETECTED",
  calibration_expired: "CALIBRATION EXPIRED",
};

const STATE_COLOR: Record<string, string> = {
  calibrated: "#34d399",
  temporary_loss: "#fbbf24",
  no_calibration: "#94a3b8",
  calibration_expired: "#94a3b8",
};

const PitchOverlay = ({ pitch, width, height }: Props) => {
  if (!width || !height) return null;

  const state = pitch?.state ?? "no_calibration";
  const color = STATE_COLOR[state] ?? STATE_COLOR.no_calibration;
  const label = STATE_TEXT[state] ?? STATE_TEXT.no_calibration;
  const calibrated = Boolean(pitch?.calibrated) && (state === "calibrated" || state === "temporary_loss");
  const stale = state === "temporary_loss";

  const corners = CORNER_ORDER.map((name) => pitch?.corners?.[name]).filter(
    (point): point is number[] =>
      Array.isArray(point) && point.length === 2 && point.every(Number.isFinite),
  );
  const keypoints = Object.entries(pitch?.keypoints ?? {}).filter(
    ([, point]) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite),
  );
  const center = pitch?.center;

  const unit = Math.max(2, Math.min(width, height) * 0.006);
  const font = Math.max(10, Math.min(width, height) * 0.028);
  const confidence =
    typeof pitch?.confidence === "number" && Number.isFinite(pitch.confidence)
      ? ` · ${Math.round(pitch.confidence * 100)}%`
      : "";

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {calibrated && corners.length === 4 && (
          <g>
            <polygon
              points={corners.map((point) => `${point[0]},${point[1]}`).join(" ")}
              fill={color}
              fillOpacity={stale ? 0.08 : 0.14}
              stroke={color}
              strokeWidth={Math.max(2, unit * 0.9)}
              strokeDasharray={stale ? `${unit * 4} ${unit * 3}` : undefined}
              strokeOpacity={0.95}
            />
            <text
              x={(corners[0][0] + corners[2][0]) / 2}
              y={(corners[0][1] + corners[2][1]) / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={font}
              fontWeight={700}
              fill={color}
              fillOpacity={0.55}
            >
              {stale ? "STALE" : "PITCH"}
            </text>
          </g>
        )}

        {calibrated &&
          keypoints.map(([name, [x, y]]) => {
            const isCorner = CORNER_ORDER.includes(name);
            return (
              <g key={name}>
                <circle
                  cx={x}
                  cy={y}
                  r={unit * (isCorner ? 1.6 : 1.2)}
                  fill={color}
                  fillOpacity={0.95}
                  stroke="#0b1220"
                  strokeWidth={Math.max(1, unit * 0.4)}
                />
                <text
                  x={x + unit * 2.4}
                  y={y - unit * 1.8}
                  fontSize={font * 0.72}
                  fontWeight={600}
                  fill={color}
                  fillOpacity={0.9}
                >
                  {name.replace(/_/g, " ")}
                </text>
              </g>
            );
          })}

        {calibrated && Array.isArray(center) && center.length === 2 && (
          <g stroke={color} strokeWidth={Math.max(1.5, unit * 0.6)} strokeOpacity={0.95}>
            <line x1={center[0] - unit * 2.5} y1={center[1]} x2={center[0] + unit * 2.5} y2={center[1]} />
            <line x1={center[0]} y1={center[1] - unit * 2.5} x2={center[0]} y2={center[1] + unit * 2.5} />
          </g>
        )}
      </svg>

      <div className="absolute right-3 top-3">
        <span
          className="inline-flex items-center gap-1.5 rounded-full border border-white/20 bg-black/65 px-3 py-1 text-xs font-semibold text-white backdrop-blur"
          style={{ borderColor: `${color}66` }}
        >
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
          {label}
          {calibrated && confidence}
        </span>
      </div>
    </div>
  );
};

export default PitchOverlay;
