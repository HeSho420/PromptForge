import { useEffect, useRef, useState } from "react";
import type { Job } from "../types";

/** Labels for known stage markers ("[stage] key — detail" in job logs). */
const LABELS: Record<string, string> = {
  models: "Models",
  plan: "Plan",
  render: "Render",
  check: "Realism check",
  retry: "Strategy change",
  save: "Save",
  segment: "Segment",
  coverage: "Coverage",
  angles: "Synthesize angles",
  consent: "Consent",
  reference: "Face reference",
  animate: "Animate",
  hardware: "Hardware",
  analyze: "Read request",
  inspect: "Seam inspection",
  score: "Quality scoring",
  verify: "Final validation",
  prepare: "Preparing memory",
  mask: "Mask",
  face: "Face restore",
  understand: "Scene analysis",
  research: "Research",
  install: "Install",
};

/* Tiny stroke icons per stage, in the same hand as the nav rail's. A step
   with a picture is scannable from across the room; the label stays for
   anyone closer. */
const ICONS: Record<string, string> = {
  models: "M12 2 3 7v10l9 5 9-5V7l-9-5 M3 7l9 5 9-5 M12 12v10",
  plan: "M4 5h6v6H4z M14 13h6v6h-6z M10 8h7v5",
  render: "M12 4v2.5 M12 17.5V20 M4 12h2.5 M17.5 12H20 M6.3 6.3l1.8 1.8 M15.9 15.9l1.8 1.8 M17.7 6.3l-1.8 1.8 M8.1 15.9l-1.8 1.8 M12 9.4a2.6 2.6 0 1 0 0 5.2 2.6 2.6 0 0 0 0-5.2",
  check: "M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13 M15.5 15.5 20 20",
  inspect: "M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13 M15.5 15.5 20 20",
  score: "M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13 M15.5 15.5 20 20",
  verify: "M4 12l5 5L20 6",
  retry: "M20 11a8 8 0 1 0-2.5 5.9 M20 5v6h-6",
  save: "M5 4h11l3 3v13H5z M8 4v5h7V4 M8 13h8v6H8z",
  segment: "M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z",
  mask: "M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z",
  coverage: "M12 9.6a2.4 2.4 0 1 0 0 4.8 2.4 2.4 0 0 0 0-4.8 M2.5 12c0 2.2 4.3 4 9.5 4s9.5-1.8 9.5-4-4.3-4-9.5-4-9.5 1.8-9.5 4Z",
  angles: "M12 9.6a2.4 2.4 0 1 0 0 4.8 2.4 2.4 0 0 0 0-4.8 M2.5 12c0 2.2 4.3 4 9.5 4s9.5-1.8 9.5-4-4.3-4-9.5-4-9.5 1.8-9.5 4Z",
  consent: "M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6Z",
  reference: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8 M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5",
  face: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8 M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5",
  animate: "M7 4l13 8-13 8Z",
  hardware: "M7 7h10v10H7z M4 10h3 M4 14h3 M17 10h3 M17 14h3 M10 4v3 M14 4v3 M10 17v3 M14 17v3",
  prepare: "M7 7h10v10H7z M4 10h3 M4 14h3 M17 10h3 M17 14h3 M10 4v3 M14 4v3 M10 17v3 M14 17v3",
  analyze: "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
  understand: "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
  research: "M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13 M15.5 15.5 20 20",
  install: "M12 3v12 M7 10l5 5 5-5 M4 20h16",
};

const stageIcon = (key: string) =>
  ICONS[key] ? (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
         strokeLinejoin="round" aria-hidden>
      <path d={ICONS[key]} />
    </svg>
  ) : null;

const RUNNING = ["pending", "running", "retrying"];
const PROGRESS_RE = /Downloading (.+?): (\d{1,3})%/;

/** Live console of everything the job does — stage changes, downloads,
    ComfyUI traffic and, highlighted, what the LLM is thinking ([llm] lines). */
function LogFeed({ job }: { job: Job }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const stick = useRef(true);

  // No dep array on purpose: pages stay mounted while hidden (display:none,
  // scrollHeight 0), so re-stick on every render once we're visible again.
  useEffect(() => {
    const el = boxRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  return (
    <div
      ref={boxRef}
      className="logfeed"
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {job.logs.map((entry, i) => {
        const llm = entry.msg.startsWith("[llm]");
        const stage = entry.msg.startsWith("[stage]");
        const search = entry.msg.startsWith("[search]");
        const eta = entry.msg.startsWith("[eta");
        return (
          <div
            key={i}
            className={
              "logline" +
              (llm ? " llm" : "") +
              (stage ? " stage" : "") +
              (search ? " search" : "") +
              (eta ? " eta" : "") +
              (entry.level === "error" ? " err" : "")
            }
          >
            <span className="log-t">{entry.t.slice(11, 19)}</span>
            {llm && <span className="log-brain" aria-hidden>✦</span>}
            {search && <span className="log-brain" aria-hidden>🔍</span>}
            {eta && <span className="log-brain" aria-hidden>⏱</span>}
            {entry.msg.replace(/^\[(llm|stage|search|mask|preview|eta(?::\d+)?)\]\s*/, "")}
          </div>
        );
      })}
      {job.logs.length === 0 && <div className="logline dim">Waiting…</div>}
    </div>
  );
}

/** Live view of what the program is doing: parses "[stage] x" job logs,
    shows download progress bars, and offers the full behind-the-scenes feed. */
export function Pipeline({ job }: { job: Job }) {
  const [showFeed, setShowFeed] = useState(false);
  const seen: string[] = [];
  for (const entry of job.logs) {
    const m = /^\[stage\] (\w+)/.exec(entry.msg);
    if (m && seen[seen.length - 1] !== m[1]) seen.push(m[1]);
  }
  const stages = [...new Set(seen)];
  const current = seen[seen.length - 1];
  const busy = RUNNING.includes(job.state);

  // live detail: last non-stage log line, plus download % when present
  const lastLog = job.logs.length ? job.logs[job.logs.length - 1].msg : "";
  const prog = PROGRESS_RE.exec(lastLog);
  // the program is out on the internet right now — show it
  const searching = busy && lastLog.startsWith("[search]");
  const startedMs = Date.parse(job.created_at); // ISO with +00:00 offset
  const elapsed = Number.isNaN(startedMs)
    ? null
    : Math.max(0, Math.round((Date.now() - startedMs) / 1000));
  // Estimated time remaining: [eta:<seconds>] marker minus elapsed time.
  const etaLine = [...job.logs].reverse().find((l) => l.msg.startsWith("[eta"));
  const etaMatch = etaLine ? /^\[eta:(\d+)\]/.exec(etaLine.msg) : null;
  const etaTotal = etaMatch ? Number(etaMatch[1]) : null;
  const remaining =
    etaTotal !== null && elapsed !== null ? etaTotal - elapsed : null;
  const remainingLabel =
    remaining === null
      ? null
      : remaining > 90
        ? `${Math.round(remaining / 60)} min`
        : remaining > 5
          ? `${Math.round(remaining)}s`
          : "any moment";

  if (stages.length === 0 && !busy && job.logs.length === 0) return null;

  return (
    <div className="pipeline" aria-label="Render pipeline">
      <div className="pipe-row">
        {stages.map((key) => {
          const isCurrent = key === current && busy;
          const isDone = !isCurrent || job.state === "completed";
          return (
            <div
              key={key}
              className={
                "pipe-step" +
                (isCurrent ? " current" : "") +
                (isDone && !isCurrent ? " done" : "")
              }
            >
              <span className="pipe-dot" aria-hidden>
                {isDone && !isCurrent ? "✓" : isCurrent ? "" : "·"}
              </span>
              {stageIcon(key)}
              {LABELS[key] ?? key}
            </div>
          );
        })}
        {busy && stages.length === 0 && (
          <div className="pipe-step current">
            <span className="pipe-dot" aria-hidden />
            Starting…
          </div>
        )}
        {busy && elapsed !== null && (
          <span className="pipe-elapsed mono">
            {elapsed >= 60 ? `${Math.floor(elapsed / 60)}m ${elapsed % 60}s` : `${elapsed}s`}
          </span>
        )}
        {busy && remainingLabel && (
          <span className="pipe-eta" title="Estimated time remaining">
            ⏱ Estimated time remaining: ~{remainingLabel}
          </span>
        )}
        <button
          type="button"
          className="btn ghost small pipe-toggle"
          onClick={() => setShowFeed((s) => !s)}
        >
          {showFeed ? "Hide details" : "Behind the scenes"}
        </button>
      </div>
      {busy && prog && (
        <div className="progress" role="progressbar"
             aria-valuenow={Number(prog[2])} aria-valuemin={0} aria-valuemax={100}>
          <div className="progress-fill" style={{ width: `${prog[2]}%` }} />
          <span className="progress-label">
            {prog[1]} — {prog[2]}%
          </span>
        </div>
      )}
      {searching && (
        <div className="searchbar" role="status">
          <span className="search-globe" aria-hidden>🌐</span>
          <span className="search-sweep" aria-hidden />
          Searching the web — {lastLog.replace(/^\[search\]\s*/, "")}
        </div>
      )}
      {busy && !prog && !searching && !showFeed && lastLog &&
        !lastLog.startsWith("[stage]") && (
        <div className="pipe-detail mono">{lastLog}</div>
      )}
      {showFeed && <LogFeed job={job} />}
    </div>
  );
}
