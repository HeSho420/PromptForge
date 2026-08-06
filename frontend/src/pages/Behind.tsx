import { useRef, useState } from "react";
import { useEffect } from "react";
import { api, usePolling } from "../api";

/** Friendly labels for "[stage] key" execution markers. */
const STAGE_LABELS: Record<string, string> = {
  models: "Selecting / checking models",
  plan: "Building the workflow",
  render: "Rendering",
  check: "Realism check",
  retry: "Strategy change",
  save: "Saving result",
  segment: "SAM segmentation",
  coverage: "Classifying view angles",
  angles: "Synthesizing views",
  consent: "Consent check",
  reference: "Preparing face reference",
  animate: "Animating",
  hardware: "Profiling hardware",
  prepare: "Preparing memory",
  discover: "Discovering workflows",
  mask: "Mask update",
  analyze: "Scene analysis",
  inspect: "Seam inspection",
  score: "Quality scoring",
  verify: "Final validation",
};

const PREFIX_RE = /^\[(stage|search|mask|preview|eta(?::\d+)?)\]\s*/;

/** Behind the Scenes: a live, timestamped execution log of everything the
    app is doing — stages, models, masks, downloads, renders, saves and
    service health — across all jobs. (Model reasoning is not shown here;
    this is the application's own activity.) */
export function Behind() {
  const { data, refresh } = usePolling(() => api.events(400), 1500);
  const [filter, setFilter] = useState("");
  const [clearing, setClearing] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [statusIsError, setStatusIsError] = useState(false);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const stick = useRef(true);

  const clearLog = async () => {
    if (
      !window.confirm(
        "Delete the Behind-the-Scenes log? A job that is still running " +
          "keeps its log lines until it finishes. This cannot be undone.",
      )
    )
      return;
    setClearing(true);
    setStatus(null);
    try {
      const r = await api.clearEvents();
      setStatus(
        `Log deleted (${r.events_cleared} system entries, logs of ${r.jobs_stripped} jobs).`,
      );
      setStatusIsError(false);
      refresh(); // show the emptied feed right away, not on the next poll
    } catch (e) {
      setStatus(`Delete failed: ${(e as Error).message}`);
      setStatusIsError(true);
    } finally {
      setClearing(false);
    }
  };

  // Stick to the bottom while new lines stream in; stop when the user
  // scrolls up to read (so scrolling during a render always works).
  useEffect(() => {
    const el = boxRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  const events = (data ?? []).filter(
    (e) =>
      !filter ||
      (e.msg + " " + e.source).toLowerCase().includes(filter.toLowerCase()),
  );

  return (
    <>
      <h1>Behind the Scenes</h1>
      <p className="sub">
        A real-time execution log: which workflow and models were selected,
        SAM and mask processing, preprocessing, rendering progress,
        post-processing, save locations, and service health — with timestamps.
      </p>
      <div className="row" style={{ marginBottom: 10 }}>
        <input
          type="text"
          placeholder="Filter the log (e.g. video, mask, download)…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ flex: 1, maxWidth: 380 }}
          aria-label="Filter execution log"
        />
        <span className="dim" style={{ fontSize: 12, alignSelf: "center" }}>
          {events.length} entries · updates live
        </span>
        <button
          type="button"
          className="btn danger small"
          disabled={clearing}
          onClick={() => void clearLog()}
          title="Delete the log (system events + stored job logs; a running job keeps its lines until it finishes)"
        >
          {clearing ? "Deleting…" : "Delete log"}
        </button>
        <span
          role="status"
          aria-live="polite"
          className={statusIsError ? "" : "dim"}
          style={{
            fontSize: 12,
            alignSelf: "center",
            ...(statusIsError ? { color: "var(--err)" } : {}),
          }}
        >
          {status}
        </span>
      </div>
      <div
        ref={boxRef}
        className="logfeed behind-feed"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {events.map((e, i) => {
          const stage = /^\[stage\] (\w+)/.exec(e.msg);
          const search = e.msg.startsWith("[search]");
          const maskUpd = e.msg.startsWith("[mask]");
          const text = e.msg.replace(PREFIX_RE, "");
          return (
            <div
              key={`${e.t}-${i}`}
              className={
                "logline" +
                (stage ? " stage" : "") +
                (search ? " search" : "") +
                (maskUpd ? " eta" : "") +
                (e.level === "error" ? " err" : "")
              }
            >
              <span className="log-t">{e.t.slice(11, 19)}</span>
              <span className="log-src">{e.source}</span>
              {search && <span aria-hidden>🔍 </span>}
              {stage ? (
                <>
                  <strong>{STAGE_LABELS[stage[1]] ?? stage[1]}</strong>
                  {text.includes("—") ? " — " + text.split("—").slice(1).join("—").trim() : ""}
                </>
              ) : (
                text
              )}
            </div>
          );
        })}
        {events.length === 0 && (
          <div className="logline dim">
            Nothing yet — run a render and watch it live here.
          </div>
        )}
      </div>
    </>
  );
}
