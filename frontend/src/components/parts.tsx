import { useEffect, useRef, useState, type CSSProperties } from "react";
import { api } from "../api";
import type { Job } from "../types";

/**
 * Reveal an image once it has actually decoded.
 *
 * Use as `<img ref={revealOnLoad} …>`. Fading on mount looks wrong for lazy
 * images: the fade plays against an empty box and the picture then pops in at
 * full opacity afterwards. A ref callback (rather than onLoad) is what makes
 * this correct for CACHED images too — those can finish loading before React
 * attaches a handler, which would leave them invisible forever.
 */
export const revealOnLoad = (el: HTMLImageElement | null) => {
  if (!el) return;
  const show = () => el.classList.add("is-loaded");
  if (el.complete && el.naturalWidth > 0) show();
  else {
    el.addEventListener("load", show, { once: true });
    // A broken image must not stay invisible — show the browser's own
    // placeholder rather than a silent gap.
    el.addEventListener("error", show, { once: true });
  }
};

/* ---------------- JobControls: Stop button + queue position ---------------- */

export function JobControls({
  job,
  onUpdate,
}: {
  job: Job;
  onUpdate: (j: Job) => void;
}) {
  const [queuePos, setQueuePos] = useState<number | null>(null);
  const [stopping, setStopping] = useState(false);
  const active = ["pending", "running", "retrying"].includes(job.state);

  // While the job waits behind others, show its live queue position.
  useEffect(() => {
    if (job.state !== "pending") {
      setQueuePos(null);
      return;
    }
    let alive = true;
    const check = async () => {
      try {
        const qs = await api.queueState();
        const idx = qs.order.indexOf(job.id);
        if (alive) setQueuePos(idx >= 0 ? idx + 1 : null);
      } catch {
        /* advisory — the badge just hides */
      }
    };
    void check();
    const t = setInterval(() => void check(), 2000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [job.id, job.state]);

  if (!active) return null;
  return (
    <span className="row" style={{ gap: 8, marginLeft: "auto" }}>
      {queuePos != null && queuePos > 1 && (
        <span
          className="badge pending"
          title="Waiting for earlier jobs — reorder or clear them on the Jobs page"
        >
          #{queuePos} in queue
        </span>
      )}
      <button
        type="button"
        className="btn danger small"
        disabled={stopping}
        title="Stop this render — a running ComfyUI render is interrupted immediately"
        onClick={() => {
          setStopping(true);
          void api
            .cancelJob(job.id)
            .then(onUpdate)
            .catch(() => {
              /* already finished — the regular poll shows the final state */
            })
            .finally(() => setStopping(false));
        }}
      >
        {stopping ? "Stopping…" : "■ Stop"}
      </button>
    </span>
  );
}

/* ---------------- UploadArea ---------------- */

export function UploadArea({
  onFile,
  busy,
  video = false,
  label,
  hint,
}: {
  onFile: (f: File) => void;
  busy: boolean;
  /** Accept video files too — used by the motion-transfer driving clip. */
  video?: boolean;
  label?: string;
  hint?: string;
}) {
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const accept = video
    ? ".mp4,.mov,.webm,.mkv"
    : ".png,.jpg,.jpeg,.webp,.bmp";

  return (
    <div
      className={`drop ${over ? "over" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files[0];
        if (file) onFile(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
      {busy ? (
        <p>Uploading…</p>
      ) : (
        <>
          <p style={{ margin: 0, fontWeight: 600, color: "var(--text)" }}>
            {label ??
              (video
                ? "Drop a video here, or click to choose"
                : "Drop an image here, or click to choose")}
          </p>
          <p style={{ margin: "6px 0 0", fontSize: 12.5 }}>
            {hint ??
              (video
                ? "MP4, MOV, WEBM or MKV · up to 64 MB, 60 seconds"
                : "PNG, JPG, WEBP or BMP · up to 64 MB")}
          </p>
        </>
      )}
    </div>
  );
}

/* ---------------- BeforeAfter ---------------- */

export function BeforeAfter({
  beforeUrl,
  afterUrl,
}: {
  beforeUrl: string;
  afterUrl: string;
}) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  // Natural sizes of both images. The RESULT sets the frame — an outpaint
  // genuinely grows the canvas, and forcing it into the original's box would
  // stretch/crop it. The original is drawn inside that frame at the SAME
  // scale, centered, so outpainted margins show as real added canvas.
  const [beforeW, setBeforeW] = useState<number | null>(null);
  const [afterW, setAfterW] = useState<number | null>(null);

  const move = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };

  // Same-size edits resolve to width 100% = exactly the old full overlay.
  const beforeStyle: CSSProperties =
    beforeW && afterW
      ? {
          position: "absolute",
          top: "50%",
          left: "50%",
          width: `${(beforeW / afterW) * 100}%`,
          height: "auto",
          transform: "translate(-50%, -50%)",
        }
      : { position: "absolute", inset: 0, width: "100%", height: "100%" };

  return (
    <div
      className="compare"
      ref={ref}
      onPointerMove={(e) => {
        if (e.buttons === 1) move(e.clientX);
      }}
      onPointerDown={(e) => move(e.clientX)}
    >
      <img
        ref={revealOnLoad}
        src={afterUrl}
        alt="After the edit"
        draggable={false}
        onLoad={(e) => setAfterW(e.currentTarget.naturalWidth || null)}
      />
      <div className="overlay" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img
          ref={revealOnLoad}
          src={beforeUrl}
          alt="Before the edit"
          draggable={false}
          style={beforeStyle}
          onLoad={(e) => setBeforeW(e.currentTarget.naturalWidth || null)}
        />
      </div>
      <div
        className="handle"
        style={{ left: `${pos}%` }}
        role="slider"
        aria-label="Before and after comparison"
        aria-valuenow={Math.round(pos)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 4));
          if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 4));
        }}
      />
    </div>
  );
}

/* ---------------- JobList ---------------- */

function jobSummary(job: Job): string {
  if (job.payload.prompt) return String(job.payload.prompt);
  if (job.payload.model) return `download ${String(job.payload.model)}`;
  return job.type;
}

export function JobList({
  jobs,
  onChanged,
  pendingOrder,
}: {
  jobs: Job[];
  onChanged: () => void;
  /** Dispatch order of pending jobs — enables the reorder arrows. */
  pendingOrder?: string[];
}) {
  const [actionError, setActionError] = useState<string | null>(null);

  if (jobs.length === 0) {
    return <div className="empty">No jobs match. Run an edit from the Studio.</div>;
  }

  const act = async (fn: () => Promise<unknown>) => {
    setActionError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const order = pendingOrder ?? [];
  const queuePos = (id: string) => order.indexOf(id);

  return (
    <div className="stack">
      {actionError && <div className="notice">{actionError}</div>}
      {jobs.map((job) => {
        const pos = job.state === "pending" ? queuePos(job.id) : -1;
        const finished = ["completed", "failed", "cancelled"].includes(job.state);
        return (
          <details key={job.id} className="job">
            <summary>
              <span className={`badge ${job.state}`}>{job.state}</span>
              <span className="mono">{job.id}</span>
              <span className="mono dim">{job.type}</span>
              <span className="grow">{jobSummary(job)}</span>
              {pos >= 0 && (
                <span className="mono dim" title="Position in the queue">
                  #{pos + 1}
                </span>
              )}
              <span className="mono">try {job.attempts}</span>
              {pos > 0 && (
                <button
                  type="button"
                  className="btn ghost small"
                  title="Move up in the queue"
                  onClick={(e) => {
                    e.preventDefault();
                    void act(() => api.moveJob(job.id, "up"));
                  }}
                >
                  ↑
                </button>
              )}
              {pos >= 0 && pos < order.length - 1 && (
                <button
                  type="button"
                  className="btn ghost small"
                  title="Move down in the queue"
                  onClick={(e) => {
                    e.preventDefault();
                    void act(() => api.moveJob(job.id, "down"));
                  }}
                >
                  ↓
                </button>
              )}
              {(job.state === "pending" || job.state === "running" || job.state === "retrying") && (
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={(e) => {
                    e.preventDefault();
                    void act(() => api.cancelJob(job.id));
                  }}
                >
                  Cancel
                </button>
              )}
              {(job.state === "failed" || job.state === "cancelled") && (
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={(e) => {
                    e.preventDefault();
                    void act(() => api.retryJob(job.id));
                  }}
                >
                  Run again
                </button>
              )}
              {(finished || job.state === "pending") && (
                <button
                  type="button"
                  className="btn ghost small"
                  title="Delete this job and its log"
                  onClick={(e) => {
                    e.preventDefault();
                    if (window.confirm("Delete this job and its log?")) {
                      void act(() => api.deleteJob(job.id));
                    }
                  }}
                >
                  🗑
                </button>
              )}
            </summary>
            {job.error && <div className="notice" style={{ margin: 12 }}>{job.error}</div>}
            <pre className="logs">
              {job.logs.map((entry, i) => (
                <div key={i} className={entry.level}>
                  {entry.t.slice(11, 19)} {entry.msg}
                </div>
              ))}
            </pre>
          </details>
        );
      })}
    </div>
  );
}
