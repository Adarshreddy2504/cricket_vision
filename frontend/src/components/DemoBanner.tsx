import { Link } from "react-router-dom";
import { FlaskConical, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DEMO_NOTICE } from "@/lib/demo";

/**
 * DemoBanner is always visible in demo mode and cannot be dismissed. A demo
 * result is useful for reviewing the interface, but it must never be mistaken
 * for footage that a model actually analysed.
 */
const DemoBanner = () => (
  <div
    role="status"
    data-testid="demo-banner"
    className="relative overflow-hidden rounded-2xl border border-warning/40 bg-warning/10 p-4 md:p-5"
  >
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-warning/20 text-warning">
          <FlaskConical className="h-5 w-5" aria-hidden="true" />
        </span>
        <div>
          <p className="font-semibold">
            Demo data — not a real analysis
          </p>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{DEMO_NOTICE}</p>
        </div>
      </div>
      <Button asChild size="sm" variant="outline" className="shrink-0">
        <Link to="/upload">
          <Upload /> Analyse a real video
        </Link>
      </Button>
    </div>
  </div>
);

export default DemoBanner;
