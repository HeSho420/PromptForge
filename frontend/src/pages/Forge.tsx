import { useEffect, useRef, useState } from "react";
import { api, pollJob } from "../api";
import { Pipeline } from "../components/Pipeline";
import { currentStage, pickFx, ProcessFX } from "../components/ProcessFX";
import { JobControls, revealOnLoad, UploadArea } from "../components/parts";
import type { Asset, GeneratedWorkflow, GenerationRecipe, Job } from "../types";
import type { PanelProps } from "./Studio";

const STEP_NAMES: Record<string, string> = {
  models: "checked models",
  plan: "planned workflow",
  render: "rendered",
  check: "judged realism",
  retry: "changed strategy",
  save: "saved",
  prepare: "freed memory",
};

/** Tiny "how this image was made" log shown under a forged result. */
function RecipeCard({ recipe }: { recipe: GenerationRecipe }) {
  const params = [
    recipe.checkpoint,
    recipe.resolution,
    recipe.sampler &&
      `${recipe.sampler}/${recipe.scheduler ?? "?"} · ${recipe.steps ?? "?"} steps · cfg ${recipe.cfg ?? "?"}`,
    recipe.denoise != null && `denoise ${recipe.denoise}`,
    recipe.seed != null && `seed ${recipe.seed}`,
    recipe.loras ? `${recipe.loras} LoRA(s)` : null,
    recipe.controlnets ? `${recipe.controlnets} ControlNet(s)` : null,
  ].filter(Boolean);
  return (
    <details className="recipe">
      <summary>🧾 How this image was made</summary>
      <div className="recipe-body mono">
        <div>
          <span className="dim">workflow&nbsp;&nbsp;</span>
          {recipe.workflow} · {recipe.nodes} nodes
          {recipe.repairs > 0 ? ` · ${recipe.repairs} repair(s)` : ""}
          {recipe.strategy_rounds > 0
            ? ` · ${recipe.strategy_rounds} strategy round(s)`
            : ""}
        </div>
        <div>
          <span className="dim">planner&nbsp;&nbsp;&nbsp;</span>
          {recipe.planned_by}
        </div>
        <div>
          <span className="dim">model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
          {recipe.model_choice}
        </div>
        {params.length > 0 && (
          <div>
            <span className="dim">settings&nbsp;&nbsp;</span>
            {params.join(" · ")}
          </div>
        )}
        <div>
          <span className="dim">steps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
          {recipe.trail
            .map((s) => STEP_NAMES[s.step] ?? s.step)
            .join(" → ")}
        </div>
        {recipe.trail.length > 0 && (
          <div className="dim" style={{ fontSize: 10.5 }}>
            {recipe.trail[0].t}–{recipe.trail[recipe.trail.length - 1].t}
            {typeof recipe.realism === "number"
              ? ` · realism ${recipe.realism}/10`
              : ""}
          </div>
        )}
      </div>
    </details>
  );
}

const TASKS = [
  { key: "generate", label: "Generate" },
  { key: "img2img", label: "Img → img" },
  { key: "inpaint", label: "Inpaint" },
  { key: "outpaint", label: "Outpaint" },
  { key: "upscale", label: "Upscale" },
  { key: "video", label: "Animate (img → video)" },
] as const;

/** Tasks that transform an uploaded image (the backend requires one). */
const IMAGE_TASKS = new Set(["img2img", "inpaint", "outpaint", "upscale"]);

/** Badge showing which brain produced the plan — local model or cloud API. */
export function ProvenanceBadge({
  provenance,
}: {
  provenance: GeneratedWorkflow["provenance"];
}) {
  return (
    <span
      className={`badge ${provenance.source === "local" ? "prov-local" : "prov-api"}`}
      title={
        provenance.source === "local"
          ? "Planned by the local model — the prompt never left this machine."
          : "The local model was unavailable; planned via the Anthropic API."
      }
    >
      {provenance.source === "local" ? "local" : "cloud API"} ·{" "}
      {provenance.model}
      {provenance.attempts > 1 ? ` · ${provenance.attempts} attempts` : ""}
    </span>
  );
}

function NodeTable({ graph }: { graph: GeneratedWorkflow["graph"] }) {
  return (
    <table>
      <thead>
        <tr>
          <th style={{ width: 46 }}>#</th>
          <th>Node</th>
          <th>Inputs</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(graph).map(([id, node]) => (
          <tr key={id}>
            <td className="mono">{id}</td>
            <td className="mono">{node.class_type}</td>
            <td className="mono dim">
              {Object.entries(node.inputs)
                .map(([k, v]) =>
                  Array.isArray(v) ? `${k}←#${String(v[0])}` : `${k}=${String(v)}`,
                )
                .join("  ")}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const RUNNING_STATES = ["pending", "running", "retrying"];

export function Forge({ onBusy }: PanelProps = {}) {
  const [task, setTask] = useState<string>("generate");
  const [prompt, setPrompt] = useState("");
  const [source, setSource] = useState<Asset | null>(null);
  const [uploading, setUploading] = useState(false);
  const [videoLength, setVideoLength] = useState(49);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<GeneratedWorkflow | null>(null);
  const [showJson, setShowJson] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [batchCount, setBatchCount] = useState<number | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      pollRef.current?.();
    },
    [],
  );

  const isVideo = task === "video";
  const needsImage = isVideo || IMAGE_TASKS.has(task);

  const uploadSource = async (file: File) => {
    setError(null);
    setUploading(true);
    try {
      setSource(await api.uploadAsset(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const generate = async () => {
    setError(null);
    setPlan(null);
    setPlanning(true);
    try {
      setPlan(await api.generateWorkflow(task, prompt));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPlanning(false);
    }
  };

  const trackJob = (created: Job) => {
    setJob(created);
    pollRef.current?.();
    pollRef.current = pollJob(created.id, setJob, 1200);
  };

  const run = async () => {
    setError(null);
    setJob(null);
    setBatchCount(null);
    try {
      if (isVideo) {
        if (!source) {
          setError("Upload a start image first.");
          return;
        }
        trackJob(await api.createVideo(source.id, prompt, videoLength));
      } else {
        if (needsImage && !source) {
          setError("Upload an image for this task first.");
          return;
        }
        // Only image-transform tasks send the source: a stale upload must
        // never silently condition a pure text-to-image request.
        const created = await api.runWorkflow(
          task, prompt, needsImage ? source?.id : undefined);
        const batch = (created as Job & { batch_count?: number }).batch_count;
        if (batch && batch > 1) setBatchCount(batch);
        trackJob(created);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = !!job && RUNNING_STATES.includes(job.state);
  useEffect(() => onBusy?.(busy), [busy]);
  // The live picture of the work: generation shows noise crystallising,
  // an upscale shows the grid refining, the check stage shows the loupe.
  const liveFx = pickFx({
    stage: currentStage(job).stage,
    task: isVideo ? "video" : task,
  });
  const showSourceFx = needsImage && !!source;
  const resultAssetId =
    job?.state === "completed" ? (job.result?.asset_id as string) : null;
  const runProvenance =
    job?.state === "completed"
      ? (job.result?.provenance as GeneratedWorkflow["provenance"] | undefined)
      : undefined;
  const realism =
    job?.state === "completed" ? (job.result?.realism as number | null) : null;

  return (
    <>
      <h1 className="ws-hide">Forge</h1>
      <p className="sub ws-hide">
        Describe what you want. The local LLM picks the best model for the
        prompt (downloading a better one when needed), plans the ComfyUI
        workflow, renders it, judges the realism of the result, and changes
        strategy when it isn&rsquo;t convincing.
      </p>

      <div className="panel stack" style={{ maxWidth: 760 }}>
        {/* Toggle-button group, not an ARIA tab widget (no tabpanels, no
            arrow-key roving) — aria-pressed matches how it truly works. */}
        <div className="seg" role="group" aria-label="Task type">
          {TASKS.map((t) => (
            <button
              key={t.key}
              type="button"
              aria-pressed={task === t.key}
              className={task === t.key ? "on" : ""}
              onClick={() => setTask(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {needsImage &&
          (source ? (
            <div className="row">
              <img
                ref={revealOnLoad}
                src={api.assetFileUrl(source.id)}
                alt={source.filename}
                style={{ height: 64, borderRadius: 8, border: "1px solid var(--line)" }}
              />
              <span className="dim">{source.filename}</span>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => setSource(null)}
              >
                Change
              </button>
              {isVideo && (
                <label className="dim" style={{ marginLeft: "auto", fontSize: 12.5 }}>
                  Length{" "}
                  <input
                    type="range"
                    min={17}
                    max={81}
                    step={8}
                    value={videoLength}
                    onChange={(e) => setVideoLength(Number(e.target.value))}
                  />{" "}
                  {(videoLength / 24).toFixed(1)}s
                </label>
              )}
            </div>
          ) : (
            <UploadArea onFile={(f) => void uploadSource(f)} busy={uploading} />
          ))}

        <textarea
          rows={3}
          placeholder={
            isVideo
              ? 'e.g. "gentle camera push-in, waves rolling, cinematic"'
              : 'e.g. "a lighthouse at dusk, heavy fog, 35mm film look"'
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          aria-label="Prompt"
        />

        <div className="row">
          <button
            type="button"
            className="btn primary"
            onClick={() => void run()}
            disabled={busy || planning || !prompt.trim() || (needsImage && !source)}
          >
            {busy ? (
              <>
                <span className="spinner" aria-hidden /> Working…
              </>
            ) : isVideo ? (
              "Animate it"
            ) : (
              "Forge it"
            )}
          </button>
          {!isVideo && (
            <button
              type="button"
              className="btn ghost"
              onClick={() => void generate()}
              disabled={busy || planning || !prompt.trim()}
            >
              {planning ? "Planning…" : "Preview plan only"}
            </button>
          )}
          {isVideo && (
            <span className="dim" style={{ fontSize: 12.5 }}>
              WAN 2.2 · first run downloads ~18 GB of models
            </span>
          )}
        </div>

        {error && <div className="notice">{error}</div>}

        {job && (
          <div className="stack" style={{ gap: 8 }}>
            <div className="row">
              <span className={`badge ${job.state}`}>{job.state}</span>
              <span className="mono dim" style={{ fontSize: 12 }}>
                job {job.id} · attempt {job.attempts}
              </span>
              <JobControls job={job} onUpdate={setJob} />
            </div>
            {batchCount != null && (
              <p className="dim" style={{ fontSize: 12.5, margin: 0 }} role="status">
                🖼 {batchCount} images queued — they render one after another
                and appear in the Gallery as each finishes.
              </p>
            )}
            {busy && (
              <div className={`fx-stage${showSourceFx ? "" : " empty"}`}>
                {showSourceFx && source && (
                  <img
                    className="fx-stage-img"
                    src={api.assetFileUrl(source.id)}
                    alt=""
                    aria-hidden
                  />
                )}
                <ProcessFX
                  active
                  effect={liveFx.effect}
                  label={liveFx.label}
                  standalone={!showSourceFx}
                />
              </div>
            )}
            <Pipeline job={job} />
            {job.error && <div className="notice">{job.error}</div>}
          </div>
        )}
      </div>

      {resultAssetId && (
        <div className="panel stack" style={{ maxWidth: 760, marginTop: 18 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>Result</h2>
            <span className="row" style={{ gap: 8 }}>
              {typeof realism === "number" && (
                <span
                  className={`badge ${realism >= 6 ? "completed" : "pending"}`}
                  title="Judged by the local vision model"
                >
                  realism {realism}/10
                </span>
              )}
              {runProvenance && <ProvenanceBadge provenance={runProvenance} />}
            </span>
          </div>
          <img
            ref={revealOnLoad}
            src={api.assetFileUrl(resultAssetId)}
            alt={prompt}
            style={{ width: "100%", borderRadius: 10, border: "1px solid var(--line)" }}
          />
          <span className="dim" style={{ fontSize: 12.5 }}>
            Saved to the Gallery
            {typeof job?.result?.repairs === "number" &&
            (job.result.repairs as number) > 0
              ? ` · plan repaired ${String(job.result.repairs)}× after ComfyUI errors`
              : ""}
          </span>
          {job?.result?.recipe != null && (
            <RecipeCard recipe={job.result.recipe as GenerationRecipe} />
          )}
        </div>
      )}

      {plan && (
        <div className="panel stack" style={{ maxWidth: 760, marginTop: 18 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>
              Plan · <span className="mono dim">{plan.task}</span>
            </h2>
            <ProvenanceBadge provenance={plan.provenance} />
          </div>
          <NodeTable graph={plan.graph} />
          <div className="row">
            <button
              type="button"
              className="btn ghost small"
              onClick={() => setShowJson((s) => !s)}
            >
              {showJson ? "Hide JSON" : "Show JSON"}
            </button>
            <button
              type="button"
              className="btn ghost small"
              onClick={() =>
                void navigator.clipboard.writeText(
                  JSON.stringify(plan.graph, null, 2),
                )
              }
            >
              Copy JSON
            </button>
          </div>
          {showJson && (
            <pre className="logs" style={{ borderTop: "none", padding: 12 }}>
              {JSON.stringify(plan.graph, null, 2)}
            </pre>
          )}
        </div>
      )}
    </>
  );
}
