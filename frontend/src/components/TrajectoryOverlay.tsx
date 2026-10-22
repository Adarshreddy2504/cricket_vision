/**
 * TrajectoryOverlay.tsx — the tracked ball path drawn over the clip.
 *
 * Points are in SOURCE pixel coordinates and the SVG uses the video's own
 * viewBox, so the drawing lands on the pixels it describes. Nothing is
 * interpolated: the measured part is solid, and the tracker's predicted tail
 * is dashed so it never reads as a detection. The bounce marker is the point
 * at the frame the backend named, not a recomputed lowest point.
 */

import type { TrajectoryPoint } from "@/lib/api";

interface Props {
  points?: TrajectoryPoint[] | null;
  bounceFrame?: number | null;
  width?: number | null;
  height?: number | null;
  /** Synthetic preview only: reveal points up to this source frame. */
  frameLimit?: number | null;
  className?: string;
}

const DETECTED = "#38bdf8";
const PREDICTED = "#94a3b8";
const BOUNCE = "#f472b6";

const TrajectoryOverlay = ({ points, bounceFrame, width, height, frameLimit, className }: Props) => {
  if (!width || !height || !points || points.length === 0) return null;

  const all = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const visible =
    frameLimit == null ? all : all.filter((point) => Number.isFinite(point.frame) && point.frame <= frameLimit);
  if (visible.length === 0) return null;

  const unit = Math.max(2, Math.min(width, height) * 0.006);
  const firstPredicted = visible.findIndex((point) => point.source === "predicted");
  const measured = firstPredicted === -1 ? visible : visible.slice(0, Math.max(2, firstPredicted));
  const tail = firstPredicted === -1 ? [] : visible.slice(Math.max(0, firstPredicted - 1));

  const bouncePoint =
    bounceFrame != null ? visible.find((point) => point.frame === bounceFrame) : undefined;

  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className ?? ""}`}>
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {measured.length >= 2 && (
          <polyline
            points={measured.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke={DETECTED}
            strokeWidth={Math.max(2, unit * 0.8)}
            strokeOpacity={0.9}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {tail.length >= 2 && (
          <polyline
            points={tail.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke={PREDICTED}
            strokeWidth={Math.max(2, unit * 0.7)}
            strokeOpacity={0.85}
            strokeDasharray={`${unit * 3} ${unit * 2.5}`}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {visible.map((point, index) => (
          <circle
            key={`${point.frame}-${index}`}
            cx={point.x}
            cy={point.y}
            r={unit * (point.source === "predicted" ? 0.9 : 1.2)}
            fill={point.source === "predicted" ? PREDICTED : DETECTED}
            fillOpacity={0.9}
            stroke="#0b1220"
            strokeWidth={Math.max(1, unit * 0.35)}
          />
        ))}
        {bouncePoint && (
          <g>
            <circle
              cx={bouncePoint.x}
              cy={bouncePoint.y}
              r={unit * 2.2}
              fill="none"
              stroke={BOUNCE}
              strokeWidth={Math.max(2, unit * 0.8)}
              strokeOpacity={0.95}
            />
            <circle
              cx={bouncePoint.x}
              cy={bouncePoint.y}
              r={unit * 1.1}
              fill={BOUNCE}
              fillOpacity={0.95}
            />
          </g>
        )}
      </svg>
    </div>
  );
};

export default TrajectoryOverlay;
