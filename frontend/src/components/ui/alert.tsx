import * as React from "react";
import { AlertCircle, CheckCircle2, Info, TriangleAlert } from "lucide-react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const alertVariants = cva("relative w-full rounded-2xl border p-4 text-sm", {
  variants: {
    variant: {
      info: "border-primary/25 bg-primary/10 text-foreground",
      warning: "border-warning/30 bg-warning/10 text-foreground",
      destructive: "border-destructive/30 bg-destructive/10 text-foreground",
      success: "border-success/30 bg-success/10 text-foreground",
    },
  },
  defaultVariants: { variant: "info" },
});

const ICONS = {
  info: Info,
  warning: TriangleAlert,
  destructive: AlertCircle,
  success: CheckCircle2,
} as const;

const ICON_COLOR = {
  info: "text-primary",
  warning: "text-warning",
  destructive: "text-destructive",
  success: "text-success",
} as const;

export interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {}

function Alert({ className, variant = "info", children, ...props }: AlertProps) {
  const Icon = ICONS[variant ?? "info"];
  return (
    <div
      role={variant === "destructive" ? "alert" : "status"}
      className={cn(alertVariants({ variant }), className)}
      {...props}
    >
      <div className="flex gap-3">
        <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", ICON_COLOR[variant ?? "info"])} aria-hidden="true" />
        <div className="min-w-0 flex-1 space-y-1">{children}</div>
      </div>
    </div>
  );
}

function AlertTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h4 className={cn("font-semibold leading-tight", className)} {...props} />;
}

function AlertDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm text-muted-foreground", className)} {...props} />;
}

export { Alert, AlertTitle, AlertDescription };
