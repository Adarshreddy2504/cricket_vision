import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
  className?: string;
  compact?: boolean;
}

const EmptyState = ({ icon: Icon, title, description, action, className, compact }: EmptyStateProps) => (
  <div
    className={cn(
      "flex flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-card/40 px-6 text-center",
      compact ? "py-8" : "py-14",
      className,
    )}
  >
    <span className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary/80 text-muted-foreground">
      <Icon className="h-6 w-6" aria-hidden="true" />
    </span>
    <h3 className="text-lg font-bold">{title}</h3>
    <p className="mt-2 max-w-md text-sm text-muted-foreground">{description}</p>
    {action && <div className="mt-5">{action}</div>}
  </div>
);

export default EmptyState;
