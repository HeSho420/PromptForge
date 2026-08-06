import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, usePolling } from "./api";
import { Behind } from "./pages/Behind";
import { Gallery, Models, Queue, Settings } from "./pages/other";
import { Workspace } from "./pages/Workspace";

function GpuMeter() {
  const { data } = usePolling(api.system, 3000);
  const gpu = data?.gpu;
  if (!gpu) return null;
  const vramPct = Math.round((gpu.vram_used_mb / gpu.vram_total_mb) * 100);
  return (
    <div className="gpu-meter" title={gpu.name}>
      <div className="gpu-line">
        <span>GPU</span>
        <div className="gpu-bar">
          <div style={{ width: `${gpu.util_pct}%` }} />
        </div>
        <span className="mono">{gpu.util_pct}%</span>
      </div>
      <div className="gpu-line">
        <span>VRAM</span>
        <div className="gpu-bar">
          <div style={{ width: `${vramPct}%` }}
               className={vramPct > 88 ? "hot" : ""} />
        </div>
        <span className="mono">
          {(gpu.vram_used_mb / 1024).toFixed(1)}G
        </span>
      </div>
    </div>
  );
}

/* Small stroke icons, drawn to match the darkroom aesthetic. */
const ic = (d: string) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d={d} />
  </svg>
);

const PAGES: Record<string, { label: string; icon: ReactNode; el: ReactNode }> = {
  studio: {
    label: "Studio",
    icon: ic("M12 20h9 M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"),
    el: <Workspace />,
  },
  queue: {
    label: "Queue",
    icon: ic("M3 6h18 M3 12h13 M3 18h9"),
    el: <Queue />,
  },
  behind: {
    label: "Behind the Scenes",
    icon: ic("M4 5h16v14H4z M7.5 9.5l3 2.5-3 2.5 M12.5 14.5H17"),
    el: <Behind />,
  },
  gallery: {
    label: "Gallery",
    icon: ic("M3 5h18v14H3z M3 15l5-5 4 4 3-3 6 6"),
    el: <Gallery />,
  },
  models: {
    label: "Models",
    icon: ic("M12 2 3 7v10l9 5 9-5V7l-9-5 M3 7l9 5 9-5 M12 12v10"),
    el: <Models />,
  },
  settings: {
    label: "Settings",
    icon: ic("M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 2v2.5 M12 19.5V22 M4.5 12H2 M22 12h-2.5 M5 5l1.8 1.8 M17.2 17.2 19 19 M19 5l-1.8 1.8 M6.8 17.2 5 19"),
    el: <Settings />,
  },
};

type PageKey = keyof typeof PAGES;

export default function App() {
  const [page, setPage] = useState<PageKey>("studio");
  const { data: health, error: healthError } = usePolling(api.health, 10000);

  return (
    <div className="app">
      <nav className="rail" aria-label="Main">
        <div className="wordmark">
          Prompt<em>Forge</em>
        </div>
        {(Object.keys(PAGES) as PageKey[]).map((key) => (
          <button
            key={key}
            type="button"
            className={page === key ? "active" : ""}
            onClick={() => {
              setPage(key);
              // One scroller is shared by every page, so without this a long
              // page you had scrolled down leaves the next one starting
              // halfway through itself.
              document.querySelector(".main")?.scrollTo({ top: 0 });
            }}
            aria-current={page === key ? "page" : undefined}
          >
            {PAGES[key].icon}
            {PAGES[key].label}
          </button>
        ))}
        <div className="spacer" />
        <StaleBuild />
        <GpuMeter />
        <div className="rail-status" title={healthError ?? "Backend healthy"}>
          <span
            className={`dot ${healthError ? "bad" : health ? "good" : ""}`}
            aria-hidden
          />
          {healthError
            ? "backend offline"
            : health
              ? `${health.inpaint_adapter}${health.inpaint_is_mock ? " · mock" : ""}`
              : "connecting…"}
        </div>
      </nav>
      {/* All pages stay mounted; switching tabs only toggles visibility so
          uploads, prompts and results are never lost by navigating around. */}
      <main className="main">
        {(Object.keys(PAGES) as PageKey[]).map((key) => (
          <div
            key={key}
            className={`page-host${page === key ? "" : " is-hidden"}`}
          >
            {PAGES[key].el}
          </div>
        ))}
      </main>
    </div>
  );
}

/**
 * Tells you when the page you are looking at is not the app that is installed.
 *
 * This is a real papercut, not a hypothetical one: a fix would land, the build
 * would be replaced, and the browser would keep running the bundle it loaded
 * hours earlier — so an already-fixed bug looked alive and got reported again.
 * index.html is served no-cache, so re-fetching it reveals the current bundle
 * name; if it is not the one this page is running, say so.
 */
function StaleBuild() {
  const [stale, setStale] = useState(false);

  useEffect(() => {
    // Whatever script tag loaded us is this page's identity.
    const mine = [...document.querySelectorAll("script[src]")]
      .map((s) => (s as HTMLScriptElement).src)
      .find((src) => src.includes("/assets/"));
    if (!mine) return;
    let alive = true;
    const check = async () => {
      try {
        const html = await (await fetch("/", { cache: "reload" })).text();
        const served = html.match(/assets\/[A-Za-z0-9._-]+\.js/)?.[0];
        if (alive && served && !mine.endsWith(served)) setStale(true);
      } catch {
        /* offline is the health dot's job, not this one's */
      }
    };
    void check();
    const timer = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  if (!stale) return null;
  return (
    <button
      type="button"
      className="rail-stale"
      onClick={() => window.location.reload()}
      title="The app was rebuilt since this page loaded — click to load the new version"
    >
      ● update ready — reload
    </button>
  );
}
