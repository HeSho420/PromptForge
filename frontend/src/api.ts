import { useEffect, useRef, useState } from "react";
import type {
  Asset,
  AvatarProfile,
  CivitaiModel,
  EventEntry,
  GalleryEntry,
  GeneratedWorkflow,
  Health,
  Job,
  MaskPreview,
  ModelInfo,
  QueueSnapshot,
  QueueState,
  RepoCandidate,
  SafetyRule,
  VersionInfo,
  WeightFile,
} from "./types";

/** One discovered PromptForge machine, as /api/peers reports it. */
export interface PeerStatus {
  name: string;
  host: string;
  port: number;
  static: boolean;
  seen_ago_s: number;
  reachable: boolean;
  latency_ms: number | null;
  last_error: string | null;
  info_age_s: number | null;
  idle?: boolean | null;
  stats?: {
    gpu_name?: string;
    gpu_util_pct?: number;
    vram_used_mb?: number;
    vram_total_mb?: number;
    ram_used_gb?: number;
    ram_total_gb?: number;
  } | null;
  comfy?: {
    up: boolean;
    device?: string | null;
    gpu?: string | null;
  } | null;
  comfy_env?: {
    python?: string;
    torch?: string | null;
    gpu_visible?: boolean;
  } | null;
  version?: VersionInfo | null;
  queue?: QueueSnapshot | null;
  uptime_s?: number | null;
  auto_update?: boolean;
}

/** An HTTP error that keeps the backend's machine-readable code, so the
    UI can react to specific conditions (e.g. persona_consent_required
    opens the consent confirmation) instead of string-matching prose. */
export class ApiError extends Error {
  code?: string;
  constructor(message: string, code?: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) {
    const err =
      body && typeof body === "object" && "error" in body
        ? (body as { error: { code?: string; message: string } }).error
        : null;
    throw new ApiError(
      err?.message ?? `Request failed (${resp.status})`,
      err?.code,
    );
  }
  return body as T;
}

const json = (data: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

/* The render device picked in the rail: "auto" (default), "local", or a
   peer's host. Injected into every job-launching request body, so one
   picker governs Studio, Forge, video, motion and avatars alike. Multiple
   surfaces can change it (the rail select, a peer card's "Render here"),
   so changes are broadcast — every picker shows the same truth. */
let renderDevice = localStorage.getItem("pf-device") || "auto";
const deviceListeners = new Set<(d: string) => void>();
export const setRenderDevice = (d: string) => {
  renderDevice = d;
  localStorage.setItem("pf-device", d);
  deviceListeners.forEach((fn) => fn(d));
};
export const getRenderDevice = () => renderDevice;
/** Subscribe to render-device changes; returns the unsubscribe. */
export const onRenderDevice = (fn: (d: string) => void): (() => void) => {
  deviceListeners.add(fn);
  return () => {
    deviceListeners.delete(fn);
  };
};
const jsonJob = (data: Record<string, unknown>): RequestInit =>
  json(renderDevice === "auto" ? data : { ...data, device: renderDevice });

export const api = {
  health: () => request<Health>("/api/health"),

  uploadAsset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Asset>("/api/assets", { method: "POST", body: form });
  },
  assetFileUrl: (assetId: string) => `/api/assets/${assetId}/file`,
  versionFileUrl: (versionId: string) => `/api/versions/${versionId}/file`,
  promoteVersion: (versionId: string) =>
    request<Asset>(`/api/versions/${versionId}/promote`, { method: "POST" }),

  maskPreview: (assetId: string, prompt: string) =>
    request<MaskPreview>(
      "/api/masks/preview",
      json({ asset_id: assetId, prompt }),
    ),

  /** Compile the request into its step plan without rendering — the program
   *  is visible BEFORE minutes are spent, so a dropped half of a compound
   *  request is a visible wrong plan, not a silent wrong picture. */
  previewEditPlan: (prompt: string, hasMask: boolean) =>
    request<{
      planned: boolean;
      steps: {
        step: number;
        task: string;
        operation: string;
        target: string;
        instruction: string;
      }[];
    }>("/api/edits/plan", json({ prompt, has_mask: hasMask })),

  createEdit: (
    assetId: string,
    prompt: string,
    maskB64?: string,
    referenceAssetIds?: string[],
    // "make a persona from this image" needs an explicit consent
    // attestation; the backend answers persona_consent_required until
    // the user confirms and this rides the retry.
    personaConsent?: boolean,
  ) =>
    request<Job>(
      "/api/edits",
      jsonJob({
        asset_id: assetId,
        prompt,
        mask_b64: maskB64,
        // Attaching a second image is what turns an edit into a combine —
        // the backend routes on their presence.
        reference_asset_ids: referenceAssetIds?.length
          ? referenceAssetIds
          : undefined,
        consent: personaConsent || undefined,
      }),
    ),

  motionTransfer: (opts: {
    referenceAssetId: string;
    drivingAssetId: string;
    prompt?: string;
    preserveBackground?: boolean;
    maxFrames?: number;
    fast?: boolean;
  }) =>
    request<Job>(
      "/api/motion_transfer",
      jsonJob({
        reference_asset_id: opts.referenceAssetId,
        driving_asset_id: opts.drivingAssetId,
        prompt: opts.prompt ?? "",
        preserve_background: opts.preserveBackground,
        max_frames: opts.maxFrames,
        fast: opts.fast,
      }),
    ),

  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  updateStatus: (fetchRemote = true) =>
    request<{
      repo: boolean;
      error?: string;
      branch?: string;
      commit?: string;
      behind?: number;
      ahead?: number;
      dirty?: string[];
      incoming?: { sha: string; subject: string }[];
    }>(`/api/update?fetch=${fetchRemote ? 1 : 0}`),
  applyUpdate: () => request<Job>("/api/update/apply", { method: "POST" }),
  peers: () =>
    request<{
      share: boolean;
      render: boolean;
      port: number | null;
      auto_update: boolean;
      version: VersionInfo | null;
      name: string;
      queue: QueueSnapshot;
      peers: PeerStatus[];
    }>("/api/peers"),
  probePeer: (host: string, port?: number) =>
    request<{
      connected: boolean;
      name?: string;
      idle?: boolean;
    }>("/api/peers/probe", json({ host, port })),
  peerLog: (host: string, name: string) =>
    request<{ name: string; text: string }>(
      `/api/peers/log?host=${encodeURIComponent(host)}&name=${encodeURIComponent(name)}`,
    ),
  pushModels: (host: string, port?: number) =>
    request<{
      offered: number;
      queued: string[];
      already: string[];
      skipped_no_checksum: string[];
    }>("/api/peers/push-models", json({ host, port })),
  fetchModels: (host: string, port?: number) =>
    request<{
      peer: string;
      offered: number;
      queued: string[];
      already: string[];
      skipped_no_checksum: string[];
    }>("/api/peers/fetch-models", json({ host, port })),
  cancelJob: (id: string) =>
    request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) =>
    request<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  deleteJob: (id: string) =>
    request<{ deleted: string }>(`/api/jobs/${id}`, { method: "DELETE" }),
  clearJobs: (scope: string) =>
    request<{ cleared: number }>("/api/jobs/clear", json({ scope })),
  queueState: () => request<QueueState>("/api/queue/state"),
  nodePacks: () => request<Record<string, unknown>[]>("/api/nodepacks"),
  installNodePack: (name: string) =>
    request<Job>(`/api/nodepacks/${name}/install`, { method: "POST" }),
  pauseQueue: () =>
    request<{ paused: boolean }>("/api/queue/pause", { method: "POST" }),
  resumeQueue: () =>
    request<{ paused: boolean }>("/api/queue/resume", { method: "POST" }),
  moveJob: (id: string, to: "up" | "down" | "top") =>
    request<{ order: string[] }>(`/api/jobs/${id}/move`, json({ to })),

  events: (limit = 400) => request<EventEntry[]>(`/api/events?limit=${limit}`),
  clearEvents: () =>
    request<{ events_cleared: number; jobs_stripped: number }>("/api/events", {
      method: "DELETE",
    }),
  clearPromptHistory: () =>
    request<{ cleared: number; prompts_scrubbed: number }>(
      "/api/history/prompts",
      { method: "DELETE" },
    ),
  samStatus: () =>
    request<{ loaded: boolean; adapter: string }>("/api/masks/status"),

  gallery: () => request<GalleryEntry[]>("/api/gallery"),
  deleteAsset: (id: string) =>
    request<{ deleted: string }>(`/api/assets/${id}`, { method: "DELETE" }),
  purgeAsset: (id: string) =>
    request<{ purged: string }>(`/api/assets/${id}?hard=1`, { method: "DELETE" }),
  restoreAsset: (id: string) =>
    request<{ restored: string }>(`/api/assets/${id}/restore`, { method: "POST" }),
  deleteGallery: () =>
    request<{ deleted: string[] }>("/api/gallery", { method: "DELETE" }),

  civitaiSearch: (q: string, type: string) =>
    request<CivitaiModel[]>(
      `/api/models/civitai?q=${encodeURIComponent(q)}&type=${encodeURIComponent(type)}`),
  modelIndexOnline: (type: string) =>
    request<{ type: string; fetched_at: number; entries: CivitaiModel[] }>(
      `/api/models/index?type=${encodeURIComponent(type)}`),
  proposeCivitai: (candidate: CivitaiModel, name: string, purpose: string) =>
    request<ModelInfo>("/api/models/propose-civitai",
      json({ candidate, name, purpose })),

  models: () => request<ModelInfo[]>("/api/models"),
  downloadModel: (name: string) =>
    request<Job>(`/api/models/${name}/download`, { method: "POST" }),

  generateWorkflow: (task: string, prompt: string) =>
    request<GeneratedWorkflow>("/api/workflows/generate", json({ task, prompt })),
  discoverWorkflows: () =>
    request<Job>("/api/workflows/discover", { method: "POST" }),
  approveWorkflow: (id: string, liveTest = true) =>
    request<{ saved: string; verified: string; path: string }>(
      "/api/workflows/approve", json({ id, live_test: liveTest })),
  runWorkflow: (task: string, prompt: string, assetId?: string,
                draft?: boolean) =>
    request<Job>("/api/workflows/run",
      jsonJob({ task, prompt, asset_id: assetId,
                draft: draft || undefined })),
  createVideo: (assetId: string, prompt: string, length?: number) =>
    request<Job>("/api/video", jsonJob({ asset_id: assetId, prompt, length })),

  maskPoint: (assetId: string, x: number, y: number) =>
    request<MaskPreview>("/api/masks/point", json({ asset_id: assetId, x, y })),
  createAvatar: (
    assetIds: string[],
    consent: boolean,
    name?: string,
    opts?: { texture?: boolean; completeBody?: boolean },
  ) =>
    request<Job>(
      "/api/avatar",
      jsonJob({
        asset_ids: assetIds,
        consent,
        name,
        texture: opts?.texture ?? true,
        complete_body: opts?.completeBody ?? true,
      }),
    ),
  avatars: () => request<AvatarProfile[]>("/api/avatars"),
  // Personas: 2D digital people for consistent image generation. Same
  // profile shape as avatars (they share a store), different product —
  // no 3D build, created in about a minute.
  personas: () => request<AvatarProfile[]>("/api/personas"),
  createPersona: (assetIds: string[], consent: boolean, name?: string) =>
    request<Job>(
      "/api/personas",
      jsonJob({ asset_ids: assetIds, consent, name }),
    ),
  deletePersona: (id: string) =>
    request<{ deleted: string }>(`/api/personas/${id}`, {
      method: "DELETE",
    }),
  renderPersona: (personaId: string, prompt: string) =>
    request<Job>(`/api/personas/${personaId}/render`, jsonJob({ prompt })),
  deleteAvatar: (id: string, frames = true) =>
    request<{ deleted: string; frames_removed: number }>(
      `/api/avatars/${id}${frames ? "?frames=1" : ""}`, { method: "DELETE" }),
  renderAvatar: (
    avatarId: string,
    prompt: string,
    video: boolean,
    length?: number,
  ) =>
    request<Job>(
      `/api/avatars/${avatarId}/render`,
      jsonJob({ prompt, video, length }),
    ),
  system: () =>
    request<{ gpu: { name: string; util_pct: number; vram_used_mb: number;
                     vram_total_mb: number } | null }>("/api/system"),
  getSettings: () =>
    request<{ civitai_token_set: boolean; lan_combine: boolean }>(
      "/api/settings", { cache: "no-store" }),
  setSettings: (s: { civitai_token?: string; lan_combine?: boolean }) =>
    request<{ civitai_token_set: boolean; lan_combine: boolean }>(
      "/api/settings", json(s)),

  // Safety rules — always fetched fresh (no-store) so the UI never shows a
  // stale ruleset; the backend also sends Cache-Control: no-store.
  safetyRules: () =>
    request<{
      builtin: { category: string; description: string; locked: boolean }[];
      custom: SafetyRule[];
    }>("/api/safety/rules", { cache: "no-store" }),
  addSafetyRule: (pattern: string, reason: string, category?: string) =>
    request<SafetyRule>("/api/safety/rules",
      json({ pattern, reason, category })),
  deleteSafetyRule: (id: number) =>
    request<{ deleted: number }>(`/api/safety/rules/${id}`, {
      method: "DELETE",
    }),

  searchModels: (q: string) =>
    request<RepoCandidate[]>(`/api/models/search?q=${encodeURIComponent(q)}`),
  listModelFiles: (repo: string) =>
    request<WeightFile[]>(`/api/models/files?repo=${encodeURIComponent(repo)}`),
  proposeModel: (repo: string, file: string, name: string, purpose: string) =>
    request<ModelInfo>("/api/models/propose", json({ repo, file, name, purpose })),
};

const TERMINAL = ["completed", "failed", "cancelled"];

/** Poll one job until it reaches a terminal state. Stops itself after 20
    consecutive errors (e.g. the backend restarted and the job is gone) so a
    kept-mounted page can never hammer the API forever. Returns a stop fn. */
export function pollJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  intervalMs = 1200,
): () => void {
  let errors = 0;
  const stop = () => window.clearInterval(timer);
  const timer = window.setInterval(async () => {
    if (document.hidden) return;
    try {
      const j = await api.job(jobId);
      errors = 0;
      onUpdate(j);
      if (TERMINAL.includes(j.state)) stop();
    } catch {
      if (++errors >= 20) stop();
    }
  }, intervalMs);
  return stop;
}

/** Poll a fetcher on an interval; pauses while the tab is hidden. */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
): { data: T | null; error: string | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const run = () => {
      if (document.hidden) return;
      fetcherRef.current()
        .then((d) => {
          if (!cancelled) {
            setData(d);
            setError(null);
          }
        })
        .catch((e: Error) => {
          if (!cancelled) setError(e.message);
        });
    };
    run();
    const id = window.setInterval(run, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs, tick]);

  return { data, error, refresh: () => setTick((t) => t + 1) };
}
