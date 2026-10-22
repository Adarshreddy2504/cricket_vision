import * as React from "react";
import { cn } from "@/lib/utils";
import { clampProgress } from "@/lib/analysis";
import { isFiniteNumber } from "@/lib/format";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number | null;
  label?: string;
  indicatorClassName?: string;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, label, indicatorClassName, ...props }, ref) => {
    const indeterminate = !isFiniteNumber(value);
    const safe = clampProgress(value);
    return (
      <div
        ref={ref}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(safe)}
        data-indeterminate={indeterminate ? "true" : undefined}
        className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
        {...props}
      >
        <div
          className={cn(
            "h-full rounded-full gradient-primary transition-[width] duration-300",
            indeterminate && "w-1/3 animate-shimmer",
            indicatorClassName,
          )}
          style={indeterminate ? undefined : { width: `${safe}%` }}
        />
      </div>
    );
  },
);
Progress.displayName = "Progress";

export { Progress };
