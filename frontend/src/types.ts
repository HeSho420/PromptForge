export type JobState =
  | "pending"
  | "running"
  | "retrying"
  | "failed"
  | "completed"
  | "cancelled";

export interface LogEntry {
  t: string;
  level: "info" | "error" | "debug";
  msg: string;
}

export interface Job {
  id: string;
  type: string;
  state: JobState;
  attempts: number;
  payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  logs: LogEntry[];
  created_at: string;
  updated_at: string;
  /** When the current attempt started RUNNING (not when it was queued) —
   *  the honest base for an elapsed-time display. */
  started_at?: string | null;
}

/** Prompt-free live picture of a queue — this machine's own, or what a
 *  peer shares about itself over the LAN. */
export interface QueueSnapshot {
  pending: number;
  paused: boolean;
  running: {
    id?: string;
    type: string;
    attempts: number;
    started_at: string | null;
    stage: string | null;
  } | null;
}

/** A PromptForge install's version identity (git commit + commit time). */
export interface VersionInfo {
  commit: string;
  ts: number;
}

export interface Asset {
  id: string;
  kind: "image" | "video" | "model";
  filename: string;
  created_at: string;
  meta: Record<string, unknown>;
}

export interface Version {
  id: string;
  asset_id: string;
  label: "original" | "edit";
  prompt: string | null;
  adapter: string | null;
  created_at: string;
  meta: { is_mock?: boolean; [k: string]: unknown };
}

export interface GalleryEntry {
  asset: Asset;
  versions: Version[];
}

export interface ModelInfo {
  name: string;
  purpose: string;
  license: string;
  url: string | null;
  path: string | null;
  sha256: string | null;
  status:
    | "not_downloaded"
    | "downloading"
    | "ready"
    | "failed"
    | "checksum_failed";
  vram_gb: number | null;
  /** live download percentage while status is "downloading" */
  progress?: number | null;
  /** last failure reason, e.g. a civitai token hint */
  note?: string | null;
}

export interface AvatarProfile {
  id: string;
  name: string;
  created_at: string;
  source_assets: string[];
  frames: string[];
  face_asset: string | null;
  meta: Record<string, unknown>;
}

export interface MaskPreview {
  mask_b64: string;
  adapter: string;
  is_mock: boolean;
  width: number;
  height: number;
  /** How the region was chosen: named-part | text | sam | background |
   *  whole-frame | point. "sam" means shape-and-position only — a guess. */
  source?: string;
  /** Backend warnings about the region ("chosen by shape and position, not
   *  by your words…"). These were being silently dropped at this interface
   *  (D10) — render them beside the overlay. */
  notes?: string[];
}

export interface Health {
  status: string;
  inpaint_adapter: string;
  inpaint_is_mock: boolean;
  segmentation_adapter: string;
  segmentation_is_mock: boolean;
  llm_local: string;
  llm_api_fallback: string | null;
}

export interface WorkflowNode {
  class_type: string;
  inputs: Record<string, unknown>;
}

export interface GeneratedWorkflow {
  task: string;
  graph: Record<string, WorkflowNode>;
  provenance: { source: "local" | "api"; model: string; attempts: number };
}

export interface RepoCandidate {
  repo_id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  gated: boolean;
}

export interface WeightFile {
  filename: string;
  size_bytes: number | null;
  sha256: string | null;
}

export interface GenerationRecipe {
  task: string;
  prompt: string;
  workflow: string;
  planned_by: string;
  model_choice: string;
  nodes: number;
  repairs: number;
  strategy_rounds: number;
  realism: number | null;
  draft?: boolean;
  checkpoint?: string;
  sampler?: string;
  scheduler?: string;
  steps?: number;
  cfg?: number;
  seed?: number;
  denoise?: number;
  resolution?: string;
  loras?: number;
  controlnets?: number;
  trail: { t: string; step: string; detail: string }[];
}

export interface EventEntry {
  t: string;
  level: "info" | "error" | "debug";
  source: string;
  msg: string;
}

export interface QueueState {
  paused: boolean;
  order: string[];
}

export interface CivitaiModel {
  name: string;
  type: string;
  creator: string;
  downloads: number;
  rating?: number | null;
  description: string;
  trigger_words: string[];
  base_model?: string | null;
  version?: string | null;
  preview_url?: string | null;
  nsfw: boolean;
  folder?: string | null;
  stageable: boolean;
  filename?: string;
  sha256?: string;
  size_bytes?: number;
  url?: string;
}

export interface SafetyRule {
  id: number;
  category: string;
  pattern: string;
  reason: string;
  created_at?: string;
}

export interface ApiError {
  error: { code: string; message: string };
}
