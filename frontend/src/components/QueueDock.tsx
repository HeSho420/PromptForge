import { useState } from "react";
import { api, usePolling } from "../api";
import type { Job } from "../types";

/**
 * The queue, always in reach.
 *
 * The Studio's modes each run ONE job and watch it — but the backend queue
 * happily holds more. This slim strip makes that real for the user: it
 * shows what is running (with its stage and clock), what waits behind it,
 * and gives a one-line way to queue another task without leaving the page
 * or interrupting anything ("cue up the next idea while this one cooks").
 *
 * It deliberately reuses the Queue page's data source (the jobs poll) so
 * the two can never disagree; the Queue page stays the place for history,
 * logs and bulk actions.
 */

const ACTIVE = new Set(["pending", "running", "retrying"]);

/** Last "[stage] …" log line — what the pipeline says it is doing. */
function lastStage(job: Job): string | null {
  for (let i = job.logs.length - 1; i >= 0; i--) {
    const msg = job.logs[i]?.msg ?? "";
    if (msg.startsWith("[stage] ")) return msg.slice("[stage] ".length);
  }
  return null;
}

/** WHERE the job renders, read from the delegation log lines — covers
 *  auto-delegation too, which payload.device never shows. A manual
 *  Retry is a boundary: lines from before it belong to a previous run
 *  (a retried job may run locally and write no new [peer] line at all). */
function renderSite(job: Job): string | null {
  for (let i = job.logs.length - 1; i >= 0; i--) {
    const m = job.logs[i]?.msg ?? "";
    if (m === "Manually re-queued") return null;
    if (!m.startsWith("[peer] ")) continue;
    if (/stopped answering|continuing on this machine|rendering locally/.test(m))
      return null;
    const hand = m.match(/^\[peer\] rendering on '([^']+)'/);
    if (hand) return hand[1];
    const auto = m.match(/^\[peer\] this machine is busy and '([^']+)'/);
    if (auto) return auto[1];
  }
  return null;
}

function elapsed(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const ms = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(ms)) return null;
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
}

/** Human name for a job type — the queue speaks the user's words. */
const TYPE_LABEL: Record<string, string> = {
  image_edit: "photo edit",
  workflow: "creation",
  video: "animation",
  motion_transfer: "motion copy",
  avatar: "avatar build",
  avatar_render: "avatar render",
  model_download: "model download",
  model_research: "model research",
  node_pack: "node pack install",
  setup: "first-run setup",
  discover: "workflow discovery",
  update: "app update",
};

const label = (t: string) => TYPE_LABEL[t] ?? t.replace(/_/g, " ");

export function QueueDock() {
  const { data: jobs } = usePolling(api.jobs, 2500);
  const { data: qstate } = usePolling(api.queueState, 3000);
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const all = jobs ?? [];
  const running = all.filter((j) => j.state === "running" || j.state === "retrying");
  const order = qstate?.order ?? [];
  const pending = all
    .filter((j) => j.state === "pending")
    .sort((a, b) => {
      const ia = order.indexOf(a.id);
      const ib = order.indexOf(b.id);
      return (ia === -1 ? 1e9 : ia) - (ib === -1 ? 1e9 : ib);
    });
  const active = [...running, ...pending];
  const recent = all.filter((j) => !ACTIVE.has(j.state)).slice(0, 3);
  const paused = qstate?.paused ?? false;

  const act = async (fn: () => Promise<unknown>) => {
    setNote(null);
    try {
      await fn();
    } catch (e) {
      setNote((e as Error).message);
    }
  };

  const addTask = async () => {
    const text = prompt.trim();
    if (!text) return;
    setBusy(true);
    setNote(null);
    try {
      // A pure text-to-image creation: the one task that needs no photo
      // attached, so it can be queued from anywhere. The render-device
      // picker in the rail applies to it like to every other job.
      await api.runWorkflow("generate", text);
      setPrompt("");
      setOpen(true);
      setNote("Queued — it runs as soon as the machine (or an idle peer) is free.");
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const summary = () => {
    const r = running[0];
    if (r) {
      const stage = lastStage(r);
      const time = elapsed(r.started_at ?? r.created_at);
      return (
        `${label(r.type)}${r.state === "retrying" ? " (retrying)" : ""}` +
        (stage ? ` — ${stage.length > 48 ? `${stage.slice(0, 48)}…` : stage}` : "") +
        (time ? ` · ${time}` : "") +
        (pending.length ? ` · ${pending.length} waiting` : "")
      );
    }
    if (pending.length) {
      return paused
        ? `${pending.length} waiting — queue is paused`
        : `${pending.length} waiting to start`;
    }
    return "Queue is empty — new tasks start right away";
  };

  return (
    <section className="qdock" aria-label="Job queue">
      <div className="qdock-bar">
        <span
          className={`dot ${running.length ? "good" : ""}${paused ? " bad" : ""}`}
          aria-hidden
        />
        <button
          type="button"
          className="qdock-summary"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          title={open ? "Collapse the queue" : "Show every queued task"}
        >
          {summary()}
          {active.length > 0 && (
            <span className="dim" style={{ marginLeft: 6 }}>
              {open ? "▴" : "▾"}
            </span>
          )}
        </button>
        {paused && (
          <span className="dim" style={{ fontSize: 11.5 }}>
            paused
          </span>
        )}
        {(active.length > 0 || paused) && (
          <button
            type="button"
            className="btn ghost small"
            onClick={() => void act(paused ? api.resumeQueue : api.pauseQueue)}
          >
            {paused ? "▶ Resume" : "⏸ Pause"}
          </button>
        )}
        <button
          type="button"
          className="btn ghost small"
          aria-expanded={adding}
          onClick={() => setAdding((a) => !a)}
        >
          {adding ? "× Close" : "+ Queue a task"}
        </button>
      </div>

      {adding && (
        <div className="qdock-add">
          <input
            type="text"
            value={prompt}
            placeholder="Describe something new to make — it queues behind whatever is running"
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void addTask();
            }}
            aria-label="Describe a new creation to queue"
          />
          <button
            type="button"
            className="btn primary small"
            disabled={busy || !prompt.trim()}
            onClick={() => void addTask()}
          >
            {busy ? "Queueing…" : "Queue it"}
          </button>
        </div>
      )}

      {note && (
        <div className="dim qdock-note" role="status" aria-live="polite">
          {note}
        </div>
      )}

      {open && active.length > 0 && (
        <ul className="qdock-list">
          {active.map((j, i) => {
            const stage = lastStage(j);
            const pinned = typeof j.payload?.device === "string" ? j.payload.device : null;
            const isRunning = j.state === "running" || j.state === "retrying";
            // Pinned target if set; otherwise where delegation actually
            // sent it (auto-delegated jobs carry no device field).
            const device =
              pinned && pinned !== "auto"
                ? pinned
                : isRunning
                  ? renderSite(j)
                  : null;
            return (
              <li key={j.id} className="qdock-row">
                <span
                  className={`dot ${isRunning ? "good" : ""}`}
                  aria-hidden
                />
                <span className="qdock-type">{label(j.type)}</span>
                <span className="dim qdock-stage">
                  {isRunning
                    ? (j.state === "retrying" ? "retrying — " : "") +
                      (stage ?? "starting…")
                    : `waiting (#${i - running.length + 1})`}
                </span>
                {isRunning && (
                  <span className="mono dim" style={{ fontSize: 11 }}>
                    {elapsed(j.started_at ?? j.created_at)}
                  </span>
                )}
                {device && (
                  <span className="qdock-device" title="Where this job renders">
                    → {device === "local" ? "this PC" : device}
                  </span>
                )}
                <span style={{ flex: 1 }} />
                {j.state === "pending" && i - running.length > 0 && (
                  <button
                    type="button"
                    className="btn ghost small"
                    title="Run this one next"
                    onClick={() => void act(() => api.moveJob(j.id, "top"))}
                  >
                    ↑ Next
                  </button>
                )}
                <button
                  type="button"
                  className="btn danger small"
                  onClick={() => void act(() => api.cancelJob(j.id))}
                >
                  Cancel
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {open && recent.length > 0 && (
        <ul className="qdock-list qdock-recent">
          {recent.map((j) => (
            <li key={j.id} className="qdock-row">
              <span
                className={`dot ${j.state === "completed" ? "good" : j.state === "failed" ? "bad" : ""}`}
                aria-hidden
              />
              <span className="qdock-type dim">{label(j.type)}</span>
              <span className="dim qdock-stage">
                {j.state === "failed" && j.error
                  ? j.error.length > 90
                    ? `${j.error.slice(0, 90)}…`
                    : j.error
                  : j.state}
              </span>
              <span style={{ flex: 1 }} />
              {(j.state === "failed" || j.state === "cancelled") && (
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => void act(() => api.retryJob(j.id))}
                >
                  Retry
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
