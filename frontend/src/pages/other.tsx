import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  getRenderDevice,
  onRenderDevice,
  pollJob,
  setRenderDevice,
  usePolling,
} from "../api";
import type { PeerStatus } from "../api";
import { BeforeAfter, JobList, revealOnLoad } from "../components/parts";
import { Pipeline } from "../components/Pipeline";
import type {
  CivitaiModel,
  GalleryEntry,
  Job,
  RepoCandidate,
  SafetyRule,
  VersionInfo,
  WeightFile,
} from "../types";

interface WorkflowCandidate {
  id: string;
  name: string;
  task: string;
  description: string;
  required_models: string[];
  nodes: number;
}

/* -------- Improve the LLM: discover + approve new workflows -------- */

function ImproveLLM() {
  const [job, setJob] = useState<Job | null>(null);
  const [approved, setApproved] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  useEffect(() => () => pollRef.current?.(), []);

  const discover = async () => {
    setError(null);
    setApproved({});
    try {
      const created = await api.discoverWorkflows();
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, setJob, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const approve = async (id: string) => {
    setBusyId(id);
    setError(null);
    try {
      const r = await api.approveWorkflow(id);
      setApproved((a) => ({ ...a, [id]: r.verified }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
    }
  };

  const running = !!job && ["pending", "running", "retrying"].includes(job.state);
  const candidates =
    job?.state === "completed"
      ? ((job.result?.candidates as WorkflowCandidate[] | undefined) ?? [])
      : [];

  return (
    <div className="panel stack" style={{ marginTop: 20 }}>
      <h2 style={{ margin: 0 }}>Improve the LLM</h2>
      <p className="dim" style={{ margin: 0, fontSize: 13, maxWidth: "64ch" }}>
        The model proposes advanced workflows the library doesn&rsquo;t have
        yet, built only from safe, allowlisted nodes. Each is validated (and
        live-tested when self-contained); approve one and it&rsquo;s saved into
        the library, so the planner keeps getting smarter.
      </p>
      <div className="row">
        <button
          type="button"
          className="btn primary"
          disabled={running}
          onClick={() => void discover()}
        >
          {running ? (
            <>
              <span className="spinner" aria-hidden /> Thinking…
            </>
          ) : (
            "Discover new workflows"
          )}
        </button>
      </div>
      {job && running && <Pipeline job={job} />}
      {error && <div className="notice">{error}</div>}
      {job?.state === "completed" && candidates.length === 0 && (
        <div className="empty">No new valid workflows this round.</div>
      )}
      {candidates.map((c) => (
        <div key={c.id} className="row" style={{ gap: 8, alignItems: "flex-start" }}>
          <div style={{ flex: 1 }}>
            <strong className="mono">{c.name}</strong>{" "}
            <span className="badge">{c.task}</span>{" "}
            <span className="dim" style={{ fontSize: 12 }}>
              {c.nodes} nodes
            </span>
            <div className="dim" style={{ fontSize: 12.5 }}>
              {c.description}
            </div>
          </div>
          {approved[c.id] ? (
            <span className="badge completed">saved · {approved[c.id]} ✓</span>
          ) : (
            <button
              type="button"
              className="btn ghost small"
              disabled={busyId === c.id}
              onClick={() => void approve(c.id)}
            >
              {busyId === c.id ? "Verifying…" : "Approve & save"}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

function formatBytes(n: number | null): string {
  if (!n) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.round(n / 1e3)} kB`;
}

/* ---------------- Queue ---------------- */

const QUEUE_FILTERS = [
  { key: "all", label: "All" },
  { key: "active", label: "Active" },
  { key: "pending", label: "Pending" },
  { key: "completed", label: "Completed" },
  { key: "failed", label: "Failed" },
] as const;

export function Queue() {
  const { data: jobs, error, refresh } = usePolling(api.jobs, 1500);
  const { data: qstate, refresh: refreshState } = usePolling(
    api.queueState, 2000);
  const [filter, setFilter] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);

  const act = async (fn: () => Promise<unknown>) => {
    setActionError(null);
    try {
      await fn();
      refresh();
      refreshState();
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const order = qstate?.order ?? [];
  const paused = qstate?.paused ?? false;
  const all = jobs ?? [];
  const visible = all.filter((j) => {
    if (filter === "active" && !["running", "retrying"].includes(j.state)) return false;
    if (filter === "pending" && j.state !== "pending") return false;
    if (filter === "completed" && j.state !== "completed") return false;
    if (filter === "failed" && !["failed", "cancelled"].includes(j.state)) return false;
    if (search.trim()) {
      const hay = `${j.id} ${j.type} ${String(j.payload.prompt ?? "")} `
        + `${String(j.payload.model ?? "")}`;
      if (!hay.toLowerCase().includes(search.trim().toLowerCase())) return false;
    }
    return true;
  });
  const counts = {
    completed: all.filter((j) => j.state === "completed").length,
    failed: all.filter((j) => ["failed", "cancelled"].includes(j.state)).length,
    pending: all.filter((j) => j.state === "pending").length,
  };

  const confirmClear = (scope: string, label: string, n: number) => {
    if (n === 0) return;
    if (window.confirm(`Delete ${n} ${label} job(s)? This removes their history.`)) {
      void act(() => api.clearJobs(scope));
    }
  };

  return (
    <>
      <h1>
        Job queue{" "}
        {paused && <span className="badge pending">⏸ paused</span>}
      </h1>
      <p className="sub">
        Every render runs as a job with retries. Open a job to read its full
        log; reorder pending jobs, pause the queue, or clean up history here.
      </p>
      {(error || actionError) && (
        <div className="notice">{error ?? actionError}</div>
      )}

      <div className="row" style={{ flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
        {/* A segmented control, not an ARIA tab widget: there are no
            tabpanels and no arrow-key roving, so role=tab would promise a
            keyboard model that isn't here. Toggle buttons in a labelled
            group say exactly what this is — Tab to reach, Enter to pick. */}
        <div className="seg" role="group" aria-label="Filter jobs">
          {QUEUE_FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              className={filter === f.key ? "on" : ""}
              onClick={() => setFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search jobs…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ flex: 1, minWidth: 160, maxWidth: 260 }}
          aria-label="Search jobs"
        />
        <button
          type="button"
          className="btn ghost small"
          onClick={() => void act(paused ? api.resumeQueue : api.pauseQueue)}
        >
          {paused ? "▶ Resume queue" : "⏸ Pause queue"}
        </button>
        <button
          type="button"
          className="btn ghost small"
          disabled={counts.completed === 0}
          onClick={() => confirmClear("completed", "completed", counts.completed)}
        >
          Clear completed ({counts.completed})
        </button>
        <button
          type="button"
          className="btn ghost small"
          disabled={counts.failed === 0}
          onClick={() => confirmClear("finished", "finished", counts.failed + counts.completed)}
        >
          Clear all finished
        </button>
      </div>

      <JobList
        jobs={visible}
        onChanged={() => {
          refresh();
          refreshState();
        }}
        pendingOrder={order}
      />
    </>
  );
}

/* ---------------- Gallery ---------------- */

function GalleryCard({ entry }: { entry: GalleryEntry }) {
  const [open, setOpen] = useState(false);
  const edits = entry.versions.filter((v) => v.label === "edit");
  const latest = edits[edits.length - 1];

  // A rendered clip is a video asset — an <img> would show a broken icon.
  // It previews on hover so the grid stays quiet until you look at one.
  const isVideo = entry.asset.kind === "video";
  // A generated mesh is an asset too — it lands here after every avatar
  // build. Rendering it through the <img> branch gives a broken-image card.
  const isModel = entry.asset.kind === "model";

  return (
    <div className="card">
      {isModel ? (
        <a
          className="card-model"
          href={api.assetFileUrl(entry.asset.id)}
          download
          title="3D mesh — download as GLB"
        >
          <span>3D mesh</span>
          <span className="dim">GLB</span>
        </a>
      ) : isVideo ? (
        <video
          src={api.assetFileUrl(entry.asset.id)}
          muted
          loop
          playsInline
          preload="metadata"
          onMouseEnter={(e) => void e.currentTarget.play().catch(() => {})}
          onMouseLeave={(e) => e.currentTarget.pause()}
        />
      ) : (
        <img
          ref={revealOnLoad}
          src={
            latest
              ? api.versionFileUrl(latest.id)
              : api.assetFileUrl(entry.asset.id)
          }
          alt={entry.asset.filename}
          loading="lazy"
        />
      )}
      <div className="meta">
        <strong>{entry.asset.filename}</strong>
        <div>
          {isVideo && <span className="badge">video</span>}
          {isModel && <span className="badge">3D</span>}
          {!isVideo && !isModel && <>{edits.length} edit{edits.length === 1 ? "" : "s"}</>}
          {latest?.meta.is_mock && (
            <>
              {" "}
              · <span className="badge mock">mock</span>
            </>
          )}
        </div>
        {latest && (
          <div style={{ marginTop: 6 }}>
            <button
              type="button"
              className="btn ghost small"
              onClick={() => setOpen((o) => !o)}
            >
              {open ? "Hide comparison" : "Compare before / after"}
            </button>
          </div>
        )}
        {open && latest && (
          <div style={{ marginTop: 10 }}>
            <BeforeAfter
              beforeUrl={api.assetFileUrl(entry.asset.id)}
              afterUrl={api.versionFileUrl(latest.id)}
            />
            {latest.prompt && (
              <p style={{ fontSize: 12 }}>&ldquo;{latest.prompt}&rdquo;</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function Gallery() {
  const { data, error, refresh } = usePolling(api.gallery, 4000);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [undo, setUndo] = useState<{ ids: string[]; timer: number } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const toggle = (id: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  /** Soft-delete ids, show an undo toast; purge from disk when it expires. */
  const deleteIds = async (ids: string[]) => {
    setActionError(null);
    try {
      for (const id of ids) await api.deleteAsset(id);
      if (undo) window.clearTimeout(undo.timer);
      const timer = window.setTimeout(() => {
        // Undo window over: reclaim the disk space for real.
        for (const id of ids) void api.purgeAsset(id).catch(() => undefined);
        setUndo(null);
      }, 8000);
      setUndo({ ids, timer });
      setSelected(new Set());
      setSelecting(false);
      refresh();
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const undoDelete = async () => {
    if (!undo) return;
    window.clearTimeout(undo.timer);
    for (const id of undo.ids) await api.restoreAsset(id).catch(() => undefined);
    setUndo(null);
    refresh();
  };

  const entries = data ?? [];

  return (
    <>
      <h1>Gallery</h1>
      <p className="sub">
        Every upload and every edit is kept as a separate version — originals
        are never overwritten. Deleted images can be undone for a few seconds,
        then their disk space is reclaimed automatically.
      </p>
      {(error || actionError) && (
        <div className="notice">{error ?? actionError}</div>
      )}

      <div className="row" style={{ flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
        <button
          type="button"
          className="btn ghost small"
          disabled={entries.length === 0}
          onClick={() => {
            setSelecting((s) => !s);
            setSelected(new Set());
          }}
        >
          {selecting ? "Done selecting" : "Select images"}
        </button>
        {selecting && (
          <>
            <button
              type="button"
              className="btn ghost small"
              onClick={() =>
                setSelected(new Set(entries.map((e) => e.asset.id)))
              }
            >
              Select all
            </button>
            <button
              type="button"
              className="btn primary small"
              disabled={selected.size === 0}
              onClick={() => {
                if (window.confirm(`Delete ${selected.size} selected image(s)?`)) {
                  void deleteIds([...selected]);
                }
              }}
            >
              Delete selected ({selected.size})
            </button>
          </>
        )}
        <button
          type="button"
          className="btn ghost small"
          disabled={entries.length === 0}
          onClick={() => {
            if (window.confirm(
              `Delete the ENTIRE gallery (${entries.length} images)? `
              + "You can undo for a few seconds afterwards.")) {
              void deleteIds(entries.map((e) => e.asset.id));
            }
          }}
        >
          Delete entire gallery
        </button>
      </div>

      {undo && (
        <div className="notice info row" style={{ gap: 10 }}>
          Deleted {undo.ids.length} image(s).
          <button type="button" className="btn ghost small" onClick={() => void undoDelete()}>
            Undo
          </button>
        </div>
      )}

      {entries.length === 0 ? (
        <div className="empty">Nothing here yet. Upload an image in the Studio.</div>
      ) : (
        <div className="grid">
          {entries.map((entry) => (
            <div
              key={entry.asset.id}
              className={
                "gallery-selectable" +
                (selecting ? " selecting" : "") +
                (selected.has(entry.asset.id) ? " selected" : "")
              }
              onClick={selecting ? () => toggle(entry.asset.id) : undefined}
            >
              {selecting && (
                <span className="gallery-check" aria-hidden>
                  {selected.has(entry.asset.id) ? "✓" : ""}
                </span>
              )}
              <GalleryCard entry={entry} />
            </div>
          ))}
        </div>
      )}
    </>
  );
}

/* ---------------- Models ---------------- */

export function Models() {
  const { data: models, error, refresh } = usePolling(api.models, 2000);
  const [actionError, setActionError] = useState<string | null>(null);
  // name + the status at queue time: cleared only when the status CHANGES,
  // so retrying a "failed" model doesn't instantly re-enable the button.
  const [queued, setQueued] = useState<{ name: string; status: string } | null>(
    null,
  );

  const download = async (name: string) => {
    setActionError(null);
    try {
      const before =
        (models ?? []).find((x) => x.name === name)?.status ?? "unknown";
      await api.downloadModel(name);
      setQueued({ name, status: before });
      refresh();
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  // Clear the "Queued ✓" marker once the job picks the download up (status
  // moved on) so the button becomes clickable again for retries.
  useEffect(() => {
    if (!queued) return;
    const m = (models ?? []).find((x) => x.name === queued.name);
    if (m && m.status !== queued.status) setQueued(null);
  }, [models, queued]);

  return (
    <>
      <h1>Models</h1>
      <p className="sub">
        The registry tracks every model the real backends need: license notes,
        source URL, checksum and VRAM estimate. Nothing downloads without an
        explicit action here, and files failing checksum are discarded.
      </p>
      {(error || actionError) && (
        <div className="notice">{error ?? actionError}</div>
      )}
      <div className="panel" style={{ padding: 0, overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Purpose</th>
              <th>License</th>
              <th>VRAM</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(models ?? []).map((m) => (
              <tr key={m.name}>
                <td className="mono">{m.name}</td>
                <td>{m.purpose}</td>
                <td>{m.license}</td>
                <td>{m.vram_gb ? `${m.vram_gb} GB` : "—"}</td>
                <td>
                  <span
                    className={`badge ${
                      m.status === "ready"
                        ? "completed"
                        : m.status.includes("failed")
                          ? "failed"
                          : m.status === "downloading"
                            ? "running"
                            : "pending"
                    }`}
                  >
                    {m.status === "downloading" && m.progress != null
                      ? `downloading ${m.progress}%`
                      : m.status.replace("_", " ")}
                  </span>
                  {m.status === "downloading" && m.progress != null && (
                    <div className="cell-progress" aria-hidden>
                      <div style={{ width: `${m.progress}%` }} />
                    </div>
                  )}
                  {m.note && m.status.includes("failed") && (
                    <div
                      className="dim"
                      style={{ fontSize: 11.5, maxWidth: 320, marginTop: 4 }}
                    >
                      {m.note}
                    </div>
                  )}
                </td>
                <td>
                  {/* Always offer the action unless the model is ready or a
                      download is *actively* streaming (live progress). A status
                      stuck at "downloading" with no progress is a crashed run —
                      show Resume so the user is never locked out. */}
                  {m.status !== "ready" &&
                    !(m.status === "downloading" && m.progress != null) && (
                      <button
                        type="button"
                        className="btn ghost small"
                        onClick={() => void download(m.name)}
                        disabled={!m.url || queued?.name === m.name}
                        title={
                          m.url
                            ? "Download with checksum validation (resumes if interrupted)"
                            : "No download URL configured — see README"
                        }
                      >
                        {queued?.name === m.name
                          ? "Queued ✓"
                          : m.status === "failed" ||
                              m.status === "checksum_failed"
                            ? "Retry"
                            : m.status === "downloading"
                              ? "Resume"
                              : "Download"}
                      </button>
                    )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <NodePacks />
      <FindModelsOnline onProposed={refresh} />
      <ImproveLLM />
    </>
  );
}

/* -------- Node packs: curated ComfyUI extensions with probed status -------- */

type NodePackRow = {
  name: string;
  title: string;
  purpose: string;
  repo: string;
  status: "absent" | "installed" | "active" | "broken";
  unlocks: string;
  note: string;
};

function NodePacks() {
  const { data: packs, error, refresh } = usePolling(
    () => api.nodePacks(), 5000);
  const [msg, setMsg] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const install = async (name: string) => {
    setMsg(null);
    setActionError(null);
    try {
      const job = await api.installNodePack(name);
      setMsg(`Install queued (job ${job.id}) — ComfyUI restarts when it ` +
             "finishes; watch the Jobs page.");
      refresh();
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const badge = (s: NodePackRow["status"]) =>
    s === "active" ? "completed" : s === "broken" ? "failed" : "pending";

  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>ComfyUI node packs</h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5 }}>
        Curated extensions that unlock extra workflows (face refinement,
        pose guidance, frame interpolation, GGUF models…). Installing
        downloads third-party code from its official repo, installs its
        requirements into ComfyUI&rsquo;s environment and restarts ComfyUI —
        the status is verified against the live node list, never assumed.
      </p>
      {(error || actionError) && (
        <div className="notice">{error ?? actionError}</div>
      )}
      {msg && (
        <p className="dim" role="status" style={{ margin: 0, fontSize: 12.5 }}>
          {msg}
        </p>
      )}
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Pack</th>
              <th>Unlocks</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {((packs ?? []) as NodePackRow[]).map((p) => (
              <tr key={p.name}>
                <td>
                  <strong>{p.title}</strong>
                  <div className="dim" style={{ fontSize: 11.5, maxWidth: 340 }}>
                    {p.purpose}
                    {p.note ? ` ${p.note}` : ""}
                  </div>
                </td>
                <td style={{ fontSize: 12.5 }}>{p.unlocks}</td>
                <td>
                  <span className={`badge ${badge(p.status)}`}>{p.status}</span>
                </td>
                <td>
                  {p.status !== "active" && (
                    <button
                      type="button"
                      className="btn ghost small"
                      onClick={() => void install(p.name)}
                      title={`Installs ${p.repo} into ComfyUI's custom_nodes`}
                    >
                      {p.status === "absent" ? "Install" : "Reinstall"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* -------- Find models online: source tabs (Hugging Face | Civitai) -------- */

function FindModelsOnline({ onProposed }: { onProposed: () => void }) {
  const [source, setSource] = useState<"civitai" | "hf">("civitai");
  return (
    <div className="panel stack" style={{ marginTop: 20 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Find models online</h2>
        <div className="seg" role="group" aria-label="Model source">
          <button
            type="button"
            aria-pressed={source === "civitai"}
            className={source === "civitai" ? "on" : ""}
            onClick={() => setSource("civitai")}
          >
            Civitai
          </button>
          <button
            type="button"
            aria-pressed={source === "hf"}
            className={source === "hf" ? "on" : ""}
            onClick={() => setSource("hf")}
          >
            Hugging Face
          </button>
        </div>
      </div>
      <p className="dim" style={{ margin: 0, fontSize: 13 }}>
        Staging records a file&rsquo;s published checksum in the registry;
        nothing downloads until you click <em>Download</em> above. Checkpoints,
        LoRAs, ControlNets, embeddings, VAEs, upscalers and workflows are
        searchable; the popular-models index refreshes itself periodically.
      </p>
      {source === "civitai" ? (
        <CivitaiSearch onProposed={onProposed} />
      ) : (
        <ModelSearch onProposed={onProposed} />
      )}
    </div>
  );
}

/* -------- Civitai search (all model types, rich cards) -------- */

const CIVITAI_TYPES = [
  { key: "checkpoint", label: "Checkpoints" },
  { key: "lora", label: "LoRAs" },
  { key: "controlnet", label: "ControlNets" },
  { key: "embedding", label: "Embeddings" },
  { key: "vae", label: "VAEs" },
  { key: "upscaler", label: "Upscalers" },
  { key: "workflow", label: "Workflows" },
] as const;

function CivitaiSearch({ onProposed }: { onProposed: () => void }) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState<string>("checkpoint");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CivitaiModel[] | null>(null);
  const [fetchedAt, setFetchedAt] = useState<number | null>(null);
  const [staging, setStaging] = useState<string | null>(null);

  // With no query, show the periodically-refreshed popular index for the
  // type; with a query, live-search civitai.
  const load = useCallback(async (q: string, t: string) => {
    setError(null);
    setBusy(true);
    try {
      if (q.trim()) {
        setResults(await api.civitaiSearch(q, t));
        setFetchedAt(null);
      } else {
        const idx = await api.modelIndexOnline(t);
        setResults(idx.entries);
        setFetchedAt(idx.fetched_at);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void load("", type);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type]);

  const stage = async (m: CivitaiModel) => {
    const suggested = (m.filename ?? m.name)
      .replace(/\.[^.]+$/, "")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .slice(0, 40);
    const name = window.prompt("Registry name for this model:", suggested);
    if (!name) return;
    setStaging(m.name);
    setError(null);
    try {
      await api.proposeCivitai(m, name, `${m.type}: ${m.name}`);
      onProposed();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStaging(null);
    }
  };

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          aria-label="Model type"
        >
          {CIVITAI_TYPES.map((t) => (
            <option key={t.key} value={t.key}>{t.label}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Search civitai… (empty shows the most-downloaded)"
          value={query}
          style={{ flex: 1, minWidth: 200 }}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void load(query, type);
          }}
          aria-label="Civitai search query"
        />
        <button
          type="button"
          className="btn ghost"
          onClick={() => void load(query, type)}
          disabled={busy}
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </div>
      {fetchedAt !== null && fetchedAt > 0 && (
        <span className="dim" style={{ fontSize: 11.5 }}>
          Popular {type}s — index refreshed{" "}
          {new Date(fetchedAt * 1000).toLocaleString()} (auto-updates periodically)
        </span>
      )}
      {error && <div className="notice">{error}</div>}
      {results && results.length === 0 && (
        <div className="empty">No models matched.</div>
      )}
      <div className="civitai-grid">
        {(results ?? []).map((m, i) => (
          <div key={`${m.name}-${i}`} className="civitai-card">
            {m.preview_url ? (
              <img ref={revealOnLoad} src={m.preview_url} alt="" loading="lazy" />
            ) : (
              <div className="civitai-noimg" aria-hidden>◇</div>
            )}
            <div className="civitai-body">
              <strong>{m.name}</strong>
              <div className="dim" style={{ fontSize: 11.5 }}>
                {m.type} · by {m.creator}
                {m.version ? ` · ${m.version}` : ""}
                {m.base_model ? ` · base: ${m.base_model}` : ""}
                {" · "}⬇ {m.downloads.toLocaleString()}
              </div>
              {m.description && (
                <p className="dim" style={{ fontSize: 12, margin: "4px 0" }}>
                  {m.description}
                </p>
              )}
              {m.trigger_words.length > 0 && (
                <div className="row" style={{ flexWrap: "wrap", gap: 4 }}>
                  {m.trigger_words.map((w) => (
                    <span key={w} className="badge" title="Trigger word">
                      {w}
                    </span>
                  ))}
                </div>
              )}
              <div className="row" style={{ marginTop: 6 }}>
                {m.stageable ? (
                  <button
                    type="button"
                    className="btn ghost small"
                    disabled={staging === m.name}
                    onClick={() => void stage(m)}
                  >
                    {staging === m.name ? "Staging…" : "Stage for download"}
                  </button>
                ) : (
                  <span className="dim" style={{ fontSize: 11.5 }}>
                    {m.type === "workflow"
                      ? "workflow — view on civitai.com"
                      : "no verified safetensors file"}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* -------- Model search (Hugging Face) -------- */

function ModelSearch({ onProposed }: { onProposed: () => void }) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [repos, setRepos] = useState<RepoCandidate[] | null>(null);
  const [openRepo, setOpenRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<WeightFile[]>([]);
  const [proposing, setProposing] = useState<string | null>(null);

  const search = async () => {
    setError(null);
    setBusy(true);
    setOpenRepo(null);
    try {
      setRepos(await api.searchModels(query));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const inspect = async (repo: string) => {
    setError(null);
    if (openRepo === repo) {
      setOpenRepo(null);
      return;
    }
    try {
      setFiles(await api.listModelFiles(repo));
      setOpenRepo(repo);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const propose = async (repo: string, file: string) => {
    const suggested = file.split("/").pop()?.replace(/\.[^.]+$/, "") ?? file;
    const name = window.prompt("Registry name for this model:", suggested);
    if (!name) return;
    const purpose = window.prompt(
      "What is this model for? (shown in the registry)",
      "",
    );
    if (purpose === null) return;
    setProposing(file);
    setError(null);
    try {
      await api.proposeModel(repo, file, name, purpose || "(no description)");
      onProposed();
      setOpenRepo(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setProposing(null);
    }
  };

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row">
        <input
          type="text"
          placeholder='e.g. "sd 1.5 inpainting safetensors"'
          value={query}
          style={{ flex: 1, minWidth: 220 }}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && query.trim()) void search();
          }}
          aria-label="Model search query"
        />
        <button
          type="button"
          className="btn ghost"
          onClick={() => void search()}
          disabled={busy || !query.trim()}
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </div>
      {error && <div className="notice">{error}</div>}
      {repos && repos.length === 0 && (
        <div className="empty">No repositories matched.</div>
      )}
      {repos && repos.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Repository</th>
              <th>Downloads</th>
              <th>Type</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {repos.map((r) => (
              <Fragment key={r.repo_id}>
                <tr>
                  <td className="mono">
                    {r.repo_id}
                    {r.gated && (
                      <span className="badge" style={{ marginLeft: 8 }}>
                        gated
                      </span>
                    )}
                  </td>
                  <td>{r.downloads.toLocaleString()}</td>
                  <td className="dim">{r.pipeline_tag ?? "—"}</td>
                  <td>
                    <button
                      type="button"
                      className="btn ghost small"
                      onClick={() => void inspect(r.repo_id)}
                    >
                      {openRepo === r.repo_id ? "Hide files" : "Files"}
                    </button>
                  </td>
                </tr>
                {openRepo === r.repo_id && (
                  <tr>
                    <td colSpan={4} style={{ background: "var(--ink)" }}>
                      {files.length === 0 ? (
                        <span className="dim">No weight files found.</span>
                      ) : (
                        <div className="stack" style={{ gap: 6 }}>
                          {files.map((f) => (
                            <div className="row" key={f.filename}>
                              <span className="mono" style={{ flex: 1 }}>
                                {f.filename}
                              </span>
                              <span className="dim">{formatBytes(f.size_bytes)}</span>
                              {f.sha256 ? (
                                <span className="badge completed" title={f.sha256}>
                                  sha256 ✓
                                </span>
                              ) : (
                                <span className="badge failed">no checksum</span>
                              )}
                              <button
                                type="button"
                                className="btn ghost small"
                                disabled={proposing === f.filename}
                                onClick={() => void propose(r.repo_id, f.filename)}
                              >
                                {proposing === f.filename ? "Staging…" : "Stage"}
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* -------- Safety rules (add/delete custom; built-ins locked) -------- */

function SafetyRules() {
  const [builtin, setBuiltin] = useState<
    { category: string; description: string; locked: boolean }[]
  >([]);
  const [custom, setCustom] = useState<SafetyRule[]>([]);
  const [pattern, setPattern] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Always re-fetch from the server (no-store) — never trust a cached copy.
  const load = useCallback(async () => {
    try {
      const data = await api.safetyRules();
      setBuiltin(data.builtin);
      setCustom(data.custom);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const add = async () => {
    setError(null);
    setBusy(true);
    try {
      await api.addSafetyRule(pattern.trim(), reason.trim());
      setPattern("");
      setReason("");
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: number) => {
    setError(null);
    try {
      await api.deleteSafetyRule(id);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>Content safety rules</h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5, maxWidth: "64ch" }}>
        Rules block a prompt before any render. Your custom rules are stored in
        the database and read live on every check — nothing is cached, so an
        add or delete applies to the very next prompt. The built-in protections
        below are always on and cannot be removed.
      </p>

      <div className="stack" style={{ gap: 4 }}>
        <strong style={{ fontSize: 13 }}>Built-in · locked</strong>
        {builtin.map((b) => (
          <div key={b.category} className="row" style={{ gap: 8, fontSize: 12.5 }}>
            <span className="badge failed" title="Always enforced">
              🔒 {b.category}
            </span>
            <span className="dim">{b.description}</span>
          </div>
        ))}
      </div>

      <div className="stack" style={{ gap: 6 }}>
        <strong style={{ fontSize: 13 }}>Your rules</strong>
        {custom.length === 0 && (
          <span className="dim" style={{ fontSize: 12.5 }}>
            No custom rules yet.
          </span>
        )}
        {custom.map((r) => (
          <div key={r.id} className="row" style={{ gap: 8, fontSize: 12.5 }}>
            <span className="badge">{r.category}</span>
            <span className="mono" style={{ flex: 1 }}>
              {r.pattern}
            </span>
            <span className="dim" style={{ maxWidth: "26ch" }}>
              {r.reason}
            </span>
            <button
              type="button"
              className="btn ghost small"
              onClick={() => void remove(r.id)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <input
          type="text"
          placeholder="keyword or phrase to block (e.g. celebrity name)"
          value={pattern}
          onChange={(e) => setPattern(e.target.value)}
          style={{ flex: 2, minWidth: 200 }}
          aria-label="Rule pattern"
          onKeyDown={(e) => {
            if (e.key === "Enter" && pattern.trim()) void add();
          }}
        />
        <input
          type="text"
          placeholder="reason (optional)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          style={{ flex: 1, minWidth: 140 }}
          aria-label="Rule reason"
        />
        <button
          type="button"
          className="btn primary"
          disabled={busy || !pattern.trim()}
          onClick={() => void add()}
        >
          {busy ? "Adding…" : "Add rule"}
        </button>
      </div>
      <p className="dim" style={{ margin: 0, fontSize: 11.5 }}>
        A plain word/phrase matches whole words (case-insensitive). Advanced:
        anything with regex characters is used as a raw pattern.
      </p>
      {error && <div className="notice">{error}</div>}
    </div>
  );
}

/* -------- Network: peers on the LAN (model transfer + delegation) -------- */

/** "2m 10s" since an ISO timestamp; clamped at zero when clocks disagree. */
function elapsedSince(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const ms = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(ms)) return null;
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function fmtUptime(secs: number | null | undefined): string | null {
  if (secs == null) return null;
  const m = Math.floor(secs / 60);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (d > 0) return `${d}d ${h % 24}h`;
  if (h > 0) return `${h}h ${m % 60}m`;
  return `${Math.max(1, m)}m`;
}

/** One line saying what the peer is DOING right now, plus how urgent
 *  that is to look at. */
function peerActivity(p: PeerStatus): {
  text: string;
  tone: "ok" | "warn" | "err";
} {
  if (!p.reachable) {
    return {
      text:
        `offline — last answered ${Math.round(p.seen_ago_s)}s ago` +
        (p.last_error ? ` (${p.last_error})` : ""),
      tone: "err",
    };
  }
  const q = p.queue;
  const r = q?.running;
  if (r) {
    const bits = [
      r.type,
      r.stage ? `${r.stage} stage` : null,
      elapsedSince(r.started_at),
    ].filter(Boolean);
    const wait = q && q.pending > 0 ? ` · ${q.pending} waiting` : "";
    return { text: `working: ${bits.join(" · ")}${wait}`, tone: "ok" };
  }
  if (q?.paused) return { text: "its queue is paused", tone: "warn" };
  if (q && q.pending > 0)
    return { text: `${q.pending} job(s) queued`, tone: "ok" };
  if (p.comfy && !p.comfy.up)
    return {
      text: "idle — but its ComfyUI is not running, so it cannot render",
      tone: "warn",
    };
  return { text: "idle — ready to take renders", tone: "ok" };
}

/** How the peer's version relates to this install's, in plain words. */
function versionNote(
  mine: VersionInfo | null | undefined,
  p: PeerStatus,
  myAutoUpdate: boolean,
): { text: string; tone: "ok" | "warn" } | null {
  const theirs = p.version;
  if (!theirs) return null;
  if (!mine) return { text: `runs ${theirs.commit}`, tone: "ok" };
  if (theirs.commit === mine.commit)
    return { text: `same version (${mine.commit})`, tone: "ok" };
  if (theirs.ts === mine.ts)
    // Same commit time, different commits: an amend/rebase twin. The
    // backend (correctly) refuses to pick a direction, so saying
    // "catches up automatically" would be a standing lie.
    return {
      text:
        `runs different code of the same age (${theirs.commit}) — ` +
        "auto-update cannot pick a direction here; update by hand if " +
        "they should match",
      tone: "warn",
    };
  if (theirs.ts > mine.ts)
    return {
      text:
        `runs newer code (${theirs.commit}) — ` +
        (myAutoUpdate
          ? "this PC updates itself when idle (once that version is " +
            "published to the update source)"
          : "auto-update is off here; install it from Settings → Updates"),
      tone: "warn",
    };
  return {
    text:
      `runs older code (${theirs.commit}) — ` +
      (p.auto_update === false
        ? "its auto-update is off; update it from its own Settings page"
        : "it catches up automatically when idle"),
    tone: "warn",
  };
}

/** Labelled utilisation bar, the rail's gpu-bar reused at card size. */
function MiniBar({
  label,
  pct,
  text,
  hot,
}: {
  label: string;
  pct: number;
  text: string;
  hot?: boolean;
}) {
  const clamped = Math.max(0, Math.min(100, Math.round(pct)));
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 170 }}
    >
      <span className="dim" style={{ fontSize: 11, width: 36 }}>
        {label}
      </span>
      <div className="gpu-bar" style={{ flex: 1 }}>
        <div style={{ width: `${clamped}%` }} className={hot ? "hot" : ""} />
      </div>
      <span
        className="mono"
        style={{ fontSize: 11, minWidth: 62, textAlign: "right" }}
      >
        {text}
      </span>
    </div>
  );
}

/** GPU-utilisation history as a small polyline — "how is it performing"
 *  at a glance, accumulated from this page's own polls. */
function Sparkline({ points }: { points: number[] }) {
  if (points.length < 2) return null;
  const w = 120;
  const h = 26;
  const step = w / (points.length - 1);
  const line = points
    .map((p, i) => {
      const y = h - (Math.max(0, Math.min(100, p)) / 100) * (h - 2) - 1;
      return `${(i * step).toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      width={w}
      height={h}
      aria-hidden
      style={{ opacity: 0.85, flexShrink: 0 }}
    >
      <polyline
        points={line}
        fill="none"
        stroke="var(--ok)"
        strokeWidth="1.5"
      />
    </svg>
  );
}

/** Read a peer's whitelisted install/crash logs from HERE — the machine
 *  that works helps debug the one that does not, without walking over.
 *  Stays mounted through reachability blips: the peer restarting is
 *  exactly when its crash log is being read. */
function PeerLogReader({
  host,
  name,
  reachable,
}: {
  host: string;
  name: string;
  reachable: boolean;
}) {
  const LOGS = [
    "comfyui-err.log",
    "comfyui.log",
    "backend-live-err.log",
    "doctor-report.txt",
    "launch.log",
  ];
  const [picked, setPicked] = useState(LOGS[0]);
  // The fetched text stays labeled with the log it came from, so
  // changing the dropdown cannot mislabel what is on screen.
  const [shown, setShown] = useState<{ log: string; text: string } | null>(
    null,
  );
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const read = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await api.peerLog(host, picked);
      setShown({
        log: picked,
        text: r.text.trim() || "(the file exists but is empty)",
      });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <details style={{ fontSize: 12 }}>
      <summary className="dim" style={{ cursor: "pointer" }}>
        Read its logs (diagnose {name} from here)
      </summary>
      <div className="row" style={{ gap: 8, marginTop: 6 }}>
        <select
          aria-label="Which log to read"
          value={picked}
          onChange={(e) => setPicked(e.target.value)}
          style={{ fontSize: 12 }}
        >
          {LOGS.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn small ghost"
          disabled={loading || !reachable}
          title={
            reachable ? undefined : "The machine is not answering right now"
          }
          onClick={() => void read()}
        >
          {loading ? "Reading…" : "Read"}
        </button>
      </div>
      {err && (
        <div className="dim" style={{ marginTop: 6 }}>
          {err}
        </div>
      )}
      {shown !== null && (
        <div style={{ marginTop: 6 }}>
          <div className="dim mono" style={{ fontSize: 10.5 }}>
            {shown.log} · last 32 KB
          </div>
          <pre
            style={{
              marginTop: 4,
              maxHeight: 240,
              overflow: "auto",
              whiteSpace: "pre-wrap",
              fontSize: 11,
              background: "var(--panel-2, rgba(0,0,0,0.2))",
              padding: 8,
              borderRadius: 6,
            }}
          >
            {shown.text}
          </pre>
        </div>
      )}
    </details>
  );
}

function NetworkPanel() {
  const { data, refresh } = usePolling(api.peers, 5000);
  const [host, setHost] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [device, setDevice] = useState(getRenderDevice());
  // The rail's picker and this page's "Render here" buttons are the same
  // setting — follow changes made anywhere.
  useEffect(() => onRenderDevice(setDevice), []);
  // GPU-utilisation history per peer host, fed by the poll above.
  const histRef = useRef<Map<string, number[]>>(new Map());

  useEffect(() => {
    for (const p of data?.peers ?? []) {
      const util = p.stats?.gpu_util_pct;
      if (util == null || !p.reachable) continue;
      const hist = histRef.current.get(p.host) ?? [];
      hist.push(util);
      histRef.current.set(p.host, hist.slice(-40));
    }
  }, [data]);

  const connect = async () => {
    const target = host.trim();
    if (!target) return;
    setBusy(true);
    setNote(null);
    try {
      const [h, p] = target.split(":");
      const r = await api.probePeer(h, p ? Number(p) : undefined);
      setNote(`Connected to '${r.name}' — models now copy between the two
        machines, and renders delegate when one is busy and the other idle.`);
      setHost("");
      refresh();
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pushModels = async (p: PeerStatus) => {
    if (
      !window.confirm(
        `Send every model on this machine to '${p.name}'? It downloads ` +
          "what it is missing over your network (checksum-verified) — " +
          "that can be many gigabytes of disk on the other machine.",
      )
    )
      return;
    setBusy(true);
    setNote(null);
    try {
      const r = await api.pushModels(p.host, p.port);
      setNote(
        `Offered ${r.offered} model(s) to ${p.name}: it queued ` +
          `${r.queued.length} download(s) (watch its Queue page) and ` +
          `already had ${r.already.length}.`,
      );
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const fetchModels = async (p: PeerStatus) => {
    if (
      !window.confirm(
        `Ask '${p.name}' for its models? THIS machine downloads every ` +
          "model it is missing over your network (checksum-verified) — " +
          "that can be many gigabytes of disk here.",
      )
    )
      return;
    setBusy(true);
    setNote(null);
    try {
      const r = await api.fetchModels(p.host, p.port);
      setNote(
        `${p.name} offered ${r.offered} model(s): queued ` +
          `${r.queued.length} download(s) here (watch the Queue page); ` +
          `${r.already.length} already on this machine.`,
      );
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pickDevice = (h: string) => {
    const next = device === h ? "auto" : h;
    setDevice(next);
    setRenderDevice(next);
  };

  const peers = data?.peers ?? [];
  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>Network</h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5, maxWidth: "64ch" }}>
        PromptForge machines on your network help each other automatically:
        model downloads copy from a peer that already has the file
        (checksum-verified), renders delegate to whichever machine is
        idle, and when one install updates, the others catch up by
        themselves. Only the model library is ever shared — never photos,
        prompts or projects.
      </p>
      {data && (
        <div className="dim" style={{ fontSize: 12.5 }}>
          This machine ({data.name}): sharing {data.share ? "on" : "off"},
          accepting renders {data.render ? "on" : "off"}, auto-update{" "}
          {data.auto_update ? "on" : "off"}, port {data.port}
          {data.version ? ` · version ${data.version.commit}` : ""}.
        </div>
      )}
      {peers.length > 0 ? (
        <div className="stack" style={{ gap: 10 }}>
          {peers.map((p) => {
            const act = peerActivity(p);
            const ver = versionNote(
              data?.version,
              p,
              data?.auto_update ?? true,
            );
            const hist = histRef.current.get(p.host) ?? [];
            const stats = p.stats;
            const vramPct =
              stats?.vram_used_mb != null && stats?.vram_total_mb
                ? (stats.vram_used_mb / stats.vram_total_mb) * 100
                : null;
            const ramPct =
              stats?.ram_used_gb != null && stats?.ram_total_gb
                ? (stats.ram_used_gb / stats.ram_total_gb) * 100
                : null;
            return (
              <div
                key={`${p.host}:${p.port}`}
                className="panel stack"
                style={{ gap: 6, padding: 12, opacity: p.reachable ? 1 : 0.75 }}
              >
                <div className="row" style={{ gap: 8 }}>
                  <span
                    className={`dot ${p.reachable ? "good" : "bad"}`}
                    aria-hidden
                  />
                  <strong>{p.name}</strong>
                  <span className="dim mono" style={{ fontSize: 12 }}>
                    {p.host}:{p.port}
                  </span>
                  {p.latency_ms != null && p.reachable && (
                    <span className="dim mono" style={{ fontSize: 11 }}>
                      {p.latency_ms} ms
                    </span>
                  )}
                  {p.static && (
                    <span className="dim" style={{ fontSize: 11 }}>
                      · pinned
                    </span>
                  )}
                  {fmtUptime(p.uptime_s) && p.reachable && (
                    <span className="dim" style={{ fontSize: 11 }}>
                      · up {fmtUptime(p.uptime_s)}
                    </span>
                  )}
                  <span style={{ flex: 1 }} />
                  <button
                    type="button"
                    className={`btn small${device === p.host ? " primary" : " ghost"}`}
                    disabled={!p.reachable}
                    onClick={() => pickDevice(p.host)}
                    title="Send every new render to this machine (click again to go back to Auto)"
                  >
                    {device === p.host ? "★ Renders here" : "Render here"}
                  </button>
                </div>
                <div
                  style={{
                    fontSize: 12.5,
                    color:
                      act.tone === "err"
                        ? "var(--err)"
                        : act.tone === "warn"
                          ? "var(--safelight, var(--text))"
                          : "var(--text)",
                  }}
                >
                  {act.text}
                </div>
                {p.reachable && stats?.gpu_name && (
                  <div className="row" style={{ gap: 12, alignItems: "center" }}>
                    <div className="stack" style={{ gap: 4, flex: 1, minWidth: 200 }}>
                      {stats.gpu_util_pct != null && (
                        <MiniBar
                          label="GPU"
                          pct={stats.gpu_util_pct}
                          text={`${stats.gpu_util_pct}%`}
                        />
                      )}
                      {vramPct != null ? (
                        <MiniBar
                          label="VRAM"
                          pct={vramPct}
                          hot={vramPct > 88}
                          text={`${((stats.vram_used_mb ?? 0) / 1024).toFixed(1)}/${Math.round((stats.vram_total_mb ?? 0) / 1024)}G`}
                        />
                      ) : stats?.vram_total_mb ? (
                        // AMD/Intel peers report the total only (live use is
                        // an NVIDIA-tool thing) — still better than silence.
                        <MiniBar
                          label="VRAM"
                          pct={0}
                          hot={false}
                          text={`${Math.round((stats.vram_total_mb ?? 0) / 1024)}G`}
                        />
                      ) : null}
                      {ramPct != null && (
                        <MiniBar
                          label="RAM"
                          pct={ramPct}
                          hot={ramPct > 90}
                          text={`${Math.round(stats.ram_used_gb ?? 0)}/${Math.round(stats.ram_total_gb ?? 0)}G`}
                        />
                      )}
                    </div>
                    <Sparkline points={hist} />
                  </div>
                )}
                <div className="dim" style={{ fontSize: 12 }}>
                  {stats?.gpu_name ? `${stats.gpu_name} · ` : ""}
                  {p.comfy?.up
                    ? `ComfyUI up (${p.comfy.device ?? "?"})`
                    : "ComfyUI down"}
                  {p.comfy?.up && p.comfy.device === "cpu"
                    ? " ⚠ CPU-only rendering"
                    : ""}
                  {p.comfy_env?.python
                    ? ` · env: Python ${p.comfy_env.python}, ${
                        p.comfy_env.torch
                          ? `torch ${p.comfy_env.torch}${
                              p.comfy_env.gpu_visible === false
                                ? " (GPU not visible!)"
                                : ""
                            }`
                          : "no torch installed"
                      }`
                    : ""}
                </div>
                {ver && (
                  <div
                    className={ver.tone === "warn" ? "" : "dim"}
                    style={{ fontSize: 12 }}
                  >
                    {ver.text}
                  </div>
                )}
                <div className="row" style={{ gap: 8 }}>
                  <button
                    type="button"
                    className="btn small ghost"
                    disabled={busy || !p.reachable}
                    onClick={() => void pushModels(p)}
                  >
                    Send all models
                  </button>
                  <button
                    type="button"
                    className="btn small ghost"
                    disabled={busy || !p.reachable}
                    onClick={() => void fetchModels(p)}
                  >
                    Ask for its models
                  </button>
                </div>
                <PeerLogReader
                  host={p.host}
                  name={p.name}
                  reachable={p.reachable}
                />
              </div>
            );
          })}
        </div>
      ) : (
        <div className="notice info" style={{ fontSize: 12.5 }}>
          No other PromptForge found yet. On BOTH machines: run{" "}
          <code>allow-lan.ps1</code> (in the PromptForge folder) once as
          administrator — Windows blocks the discovery ports until then —
          and make sure the network profile is <em>Private</em>, the other
          machine has pulled the latest version, and its app is running.
          You can also connect directly by address below.
        </div>
      )}
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <input
          type="text"
          placeholder="Other PC's IP, e.g. 192.168.1.50"
          value={host}
          onChange={(e) => setHost(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void connect();
          }}
          style={{ minWidth: 220 }}
        />
        <button
          type="button"
          className="btn"
          disabled={busy || !host.trim()}
          onClick={() => void connect()}
        >
          {busy ? "Connecting…" : "Connect to address"}
        </button>
      </div>
      {note && (
        <div className="dim" role="status" aria-live="polite" style={{ fontSize: 12.5 }}>
          {note}
        </div>
      )}
    </div>
  );
}

/* -------- Updates: pull what was pushed through git -------- */

function UpdatePanel() {
  const [status, setStatus] = useState<Awaited<
    ReturnType<typeof api.updateStatus>
  > | null>(null);
  const [checking, setChecking] = useState(false);
  const [applying, setApplying] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const check = async (fetchRemote: boolean) => {
    setChecking(true);
    setNote(null);
    try {
      setStatus(await api.updateStatus(fetchRemote));
    } catch (e) {
      setNote(`Could not check: ${(e as Error).message}`);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    void check(false); // instant answer from the last fetch; no network
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apply = async () => {
    if (
      !window.confirm(
        "Install the update now? The app restarts itself and is back in " +
          "about 15 seconds. If the new version fails to start, the " +
          "previous one is restored automatically.",
      )
    )
      return;
    setApplying(true);
    setNote("Updating… the app will restart itself.");
    try {
      await api.applyUpdate();
      // Poll health until the restarted backend answers, then reload.
      const until = Date.now() + 120_000;
      const probe = async () => {
        try {
          await api.health();
          window.location.reload();
        } catch {
          if (Date.now() < until) setTimeout(() => void probe(), 2000);
          else setNote("The app did not come back — check the console.");
        }
      };
      setTimeout(() => void probe(), 8000);
    } catch (e) {
      setApplying(false);
      setNote(`Update failed: ${(e as Error).message}`);
    }
  };

  const behind = status?.behind ?? 0;
  const dirty = status?.dirty ?? [];
  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>Updates</h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5, maxWidth: "64ch" }}>
        Updates arrive through git: push to the repository and every install
        can pull them. Your data folder (photos, models, database) is never
        touched by an update.
      </p>
      {status?.error && <div className="notice">{status.error}</div>}
      {status && !status.error && (
        <div className="dim" style={{ fontSize: 12.5 }}>
          Version {status.commit} on {status.branch}
          {behind > 0
            ? ` — ${behind} update${behind === 1 ? "" : "s"} available`
            : " — up to date at last check"}
        </div>
      )}
      {behind > 0 && (
        <ul className="dim" style={{ margin: 0, fontSize: 12, paddingLeft: 18 }}>
          {(status?.incoming ?? []).slice(0, 6).map((c) => (
            <li key={c.sha}>{c.subject}</li>
          ))}
        </ul>
      )}
      {dirty.length > 0 && (
        <div className="notice">
          Locally edited files block updates: {dirty.join(", ")}
        </div>
      )}
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <button
          type="button"
          className="btn"
          disabled={checking || applying}
          onClick={() => void check(true)}
        >
          {checking ? "Checking…" : "Check for updates"}
        </button>
        {behind > 0 && dirty.length === 0 && (
          <button
            type="button"
            className="btn primary"
            disabled={applying}
            onClick={() => void apply()}
          >
            {applying ? "Updating…" : `Install ${behind} update${behind === 1 ? "" : "s"} & restart`}
          </button>
        )}
      </div>
      {note && (
        <div className="dim" role="status" aria-live="polite" style={{ fontSize: 12.5 }}>
          {note}
        </div>
      )}
    </div>
  );
}

/* -------- Privacy: delete Behind-the-Scenes logs / prompt history -------- */

function PrivacyHistory() {
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (
    kind: string,
    confirmText: string,
    action: () => Promise<string>,
  ) => {
    if (!window.confirm(confirmText)) return;
    setBusy(kind);
    setMsg(null);
    setError(null);
    try {
      setMsg(await action());
    } catch (e) {
      setError(`Delete failed: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>Privacy &amp; history</h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5, maxWidth: "64ch" }}>
        Everything below is stored locally on this machine. Deleting is
        immediate and cannot be undone.
      </p>
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <button
          type="button"
          className="btn danger"
          disabled={busy !== null}
          onClick={() =>
            void run(
              "events",
              "Delete the entire Behind-the-Scenes log? This cannot be undone.",
              async () => {
                const r = await api.clearEvents();
                return `Behind-the-Scenes log deleted (${r.events_cleared} system entries, logs of ${r.jobs_stripped} jobs).`;
              },
            )
          }
        >
          {busy === "events" ? "Deleting…" : "Delete Behind-the-Scenes logs"}
        </button>
        <button
          type="button"
          className="btn danger"
          disabled={busy !== null}
          onClick={() =>
            void run(
              "prompts",
              "Delete the prompt history? Every finished job record — including its prompt — is removed, and prompts kept in the workflow-learning memory are blanked. This cannot be undone.",
              async () => {
                const r = await api.clearPromptHistory();
                return `Prompt history deleted (${r.cleared} job records, ${r.prompts_scrubbed} learning-memory prompts blanked).`;
              },
            )
          }
        >
          {busy === "prompts" ? "Deleting…" : "Delete prompt history"}
        </button>
      </div>
      <p className="dim" style={{ margin: 0, fontSize: 12 }}>
        The log option clears the Behind-the-Scenes stream (system events +
        stored job logs); a job that is still running keeps its log lines
        until it finishes. The prompt option removes finished jobs and their
        prompts from the Queue history and blanks the prompts kept in the
        workflow-learning memory; queued and running jobs, and prompts
        already saved into gallery recipe cards, are not touched.
      </p>
      <div role="status" aria-live="polite">
        {msg && <div className="notice info">{msg}</div>}
        {error && <div className="notice">{error}</div>}
      </div>
    </div>
  );
}

/* -------- Civitai API token (durable; no restart/env needed) -------- */

function CivitaiToken() {
  const { data, refresh } = usePolling(api.getSettings, 15000);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const configured = data?.civitai_token_set ?? false;

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      await api.setSettings({ civitai_token: token.trim() });
      setToken("");
      setMsg("Saved — applies immediately, no restart needed.");
      refresh();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel stack" style={{ marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>
        Civitai API token{" "}
        {configured && <span className="badge completed">configured ✓</span>}
      </h2>
      <p className="dim" style={{ margin: 0, fontSize: 12.5, maxWidth: "64ch" }}>
        Many Civitai downloads return 403 without an account token. Create a
        free one at civitai.com → Account settings → API Keys and paste it
        here. It&rsquo;s stored locally, applied to downloads right away, and
        never shown back or sent anywhere but Civitai.
      </p>
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <input
          type="password"
          placeholder={configured ? "•••••••• (set) — paste to replace" : "paste your Civitai API key"}
          value={token}
          onChange={(e) => setToken(e.target.value)}
          style={{ flex: 1, minWidth: 240 }}
          aria-label="Civitai API token"
          autoComplete="off"
        />
        <button
          type="button"
          className="btn primary"
          disabled={busy || !token.trim()}
          onClick={() => void save()}
        >
          {busy ? "Saving…" : "Save token"}
        </button>
        {configured && (
          <button
            type="button"
            className="btn ghost small"
            onClick={() => void (async () => {
              await api.setSettings({ civitai_token: "" });
              refresh();
            })()}
          >
            Clear
          </button>
        )}
      </div>
      {msg && <div className="notice info">{msg}</div>}
    </div>
  );
}

/* ---------------- Settings ---------------- */

export function Settings() {
  const { data: health, error } = usePolling(api.health, 5000);

  return (
    <>
      <h1>Settings</h1>
      <p className="sub">
        Backends are adapters. The active ones are shown below; switch them
        with environment variables when starting the server (see README).
      </p>
      {error && <div className="notice">Backend unreachable: {error}</div>}
      {health && (
        <div className="panel stack">
          <div className="row">
            <strong>Inpainting</strong>
            <span className="mono" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {health.inpaint_adapter}
            </span>
            {health.inpaint_is_mock && <span className="badge mock">mock</span>}
          </div>
          <div className="row">
            <strong>Segmentation</strong>
            <span className="mono" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {health.segmentation_adapter}
            </span>
            {health.segmentation_is_mock && (
              <span className="badge mock">mock</span>
            )}
          </div>
          <div className="row">
            <strong>Workflow LLM</strong>
            <span className="mono" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {health.llm_local}
            </span>
            <span className="badge prov-local">local</span>
          </div>
          <div className="row">
            <strong>API fallback</strong>
            <span className="mono" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {health.llm_api_fallback ?? "disabled"}
            </span>
            {health.llm_api_fallback && (
              <span className="badge prov-api">only when local fails</span>
            )}
          </div>
          {(health.inpaint_is_mock || health.segmentation_is_mock) && (
            <div className="notice info">
              Mock adapters demonstrate the pipeline without model downloads.
              Set <code>PROMPTFORGE_INPAINT_BACKEND=comfyui</code> and download
              the required models to render with ComfyUI.
            </div>
          )}
        </div>
      )}
      <NetworkPanel />
      <UpdatePanel />
      <CivitaiToken />
      <PrivacyHistory />
      <SafetyRules />
    </>
  );
}
