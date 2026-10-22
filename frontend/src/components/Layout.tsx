import { Outlet } from "react-router-dom";
import Navbar from "@/components/Navbar";

const Layout = () => (
  <div className="flex min-h-screen flex-col bg-background">
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
    >
      Skip to content
    </a>
    <Navbar />
    <main id="main-content" className="flex-1">
      <Outlet />
    </main>
    <footer className="border-t border-border/50 py-8">
      <div className="container flex flex-col items-start justify-between gap-3 text-xs text-muted-foreground sm:flex-row sm:items-center">
        <p>
          CricketVision · frontend contract{" "}
          <span className="font-mono text-foreground/80">schema 2.1</span> · analysis runs locally
        </p>
        <p>
          Measurements are shown only when the backend produced them. Absent values read{" "}
          <span className="font-mono text-foreground/80">—</span>.
        </p>
      </div>
    </footer>
  </div>
);

export default Layout;
