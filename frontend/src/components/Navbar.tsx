import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Activity, Menu, Moon, Sun, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "./ThemeProvider";
import { cn } from "@/lib/utils";
import { isDemoAnalysisId } from "@/lib/demo";
import { getStoredAnalysisId } from "@/lib/storage";

const NAV = [
  { to: "/", label: "Dashboard" },
  { to: "/upload", label: "Analyse" },
  { to: "/results", label: "Results" },
];

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const storedId = getStoredAnalysisId();
  const hasAnalysis = Boolean(storedId);

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open]);

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container flex h-16 items-center justify-between gap-4">
        <Link to="/" className="group flex items-center gap-2.5" aria-label="CricketVision home">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl gradient-primary shadow-glow transition-smooth group-hover:scale-105">
            <Activity className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-base font-extrabold tracking-tight">
              Cricket<span className="text-gradient">Vision</span>
            </span>
            <span className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Ball intelligence
            </span>
          </span>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {NAV.map((item) => {
            const active =
              item.to === "/" ? location.pathname === "/" : location.pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground",
                )}
              >
                {item.label}
                {item.to === "/results" && hasAnalysis && (
                  <span
                    className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-success align-middle"
                    aria-label={isDemoAnalysisId(storedId) ? "demo result available" : "result available"}
                  />
                )}
              </Link>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            {theme === "dark" ? <Sun /> : <Moon />}
          </Button>
          <Button asChild size="sm" className="hidden sm:inline-flex">
            <Link to="/upload">New analysis</Link>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <X /> : <Menu />}
          </Button>
        </div>
      </div>

      {open && (
        <div id="mobile-menu" className="border-t border-border/50 bg-background/95 md:hidden">
          <div className="container flex flex-col gap-1 py-3">
            {NAV.map((item) => {
              const active =
                item.to === "/" ? location.pathname === "/" : location.pathname.startsWith(item.to);
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "rounded-lg px-3 py-2.5 text-sm font-medium",
                    active ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-secondary/70",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
            <Link
              to="/upload"
              className="mt-1 rounded-lg gradient-primary px-3 py-2.5 text-center text-sm font-semibold text-primary-foreground"
            >
              New analysis
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
