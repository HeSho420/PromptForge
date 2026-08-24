import { useEffect, useRef, useState } from "react";
import { api, pollJob, usePolling } from "../api";
import { Pipeline } from "../components/Pipeline";
import { currentStage, pickFx, ProcessFX } from "../components/ProcessFX";
import { revealOnLoad } from "../components/parts";
import type { Asset, AvatarProfile, Job } from "../types";
import type { PanelProps } from "./Studio";

const RUNNING = ["pending", "running", "retrying"];

/** Prompt renders of a saved persona: consistent 2D images of the same
    person in any scene, pose or outfit — likeness measured per render. */
function PersonaRenderPanel({ persona }: { persona: AvatarProfile }) {
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      pollRef.current?.();
    },
    [],
  );

  const run = async () => {
    setError(null);
    setJob(null);
    try {
      const created = await api.renderPersona(persona.id, prompt);
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, setJob, 1200);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = !!job && RUNNING.includes(job.state);
  const imageId =
    job?.state === "completed" ? (job.result?.asset_id as string) : null;
  const realism =
    job?.state === "completed" ? (job.result?.realism as number | null) : null;
  const identityMatch =
    job?.state === "completed"
      ? (job.result?.identity_match as number | null)
      : null;

  return (
    <div className="stack" style={{ gap: 10 }}>
      <h2 style={{ margin: 0 }}>Render {persona.name}</h2>
      <p className="dim" style={{ margin: 0, fontSize: 13 }}>
        Describe any scene, pose or outfit — the same person every time,
        with the likeness measured (ArcFace) on every render. Tip: in the
        edit box, &quot;use persona &apos;{persona.name}&apos;: …&quot; does
        the same thing.
      </p>
      <textarea
        rows={2}
        placeholder='e.g. "reading in a cozy cafe, warm light" or "full body, hiking at sunrise"'
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        aria-label="Persona render prompt"
      />
      <div className="row">
        <button
          type="button"
          className="btn primary"
          disabled={busy || !prompt.trim()}
          onClick={() => void run()}
        >
          {busy ? (
            <>
              <span className="spinner" aria-hidden /> Rendering…
            </>
          ) : (
            "Render image"
          )}
        </button>
      </div>
      {error && <div className="notice">{error}</div>}
      {job && (
        <div className="stack" style={{ gap: 8 }}>
          <div className="row">
            <span className={`badge ${job.state}`}>{job.state}</span>
            <span className="mono dim" style={{ fontSize: 12 }}>
              job {job.id}
            </span>
          </div>
          {busy &&
            (() => {
              const fx = pickFx({
                stage: currentStage(job).stage,
                fallback: {
                  effect: "generate",
                  label: "Rendering this person into the scene",
                },
              });
              return (
                <div className="fx-stage empty">
                  <ProcessFX active effect={fx.effect} label={fx.label} standalone />
                </div>
              );
            })()}
          <Pipeline job={job} />
          {job.error && <div className="notice">{job.error}</div>}
        </div>
      )}
      {imageId && (
        <div className="stack" style={{ gap: 8 }}>
          <div className="row">
            <h2 style={{ margin: 0 }}>Result</h2>
            {typeof realism === "number" && (
              <span className={`badge ${realism >= 6 ? "completed" : "pending"}`}>
                realism {realism}/10
              </span>
            )}
            {typeof identityMatch === "number" && (
              <span
                className={`badge ${
                  identityMatch >= 0.5
                    ? "completed"
                    : identityMatch >= 0.35
                      ? "pending"
                      : "failed"
                }`}
                title="ArcFace likeness vs the reference photo — ≥0.50 is the same person"
              >
                identity {identityMatch.toFixed(2)}
              </span>
            )}
          </div>
          <img
            ref={revealOnLoad}
            src={api.assetFileUrl(imageId)}
            alt={prompt}
            style={{
              width: "100%",
              maxWidth: 520,
              borderRadius: 10,
              border: "1px solid var(--line)",
            }}
          />
          <span className="dim" style={{ fontSize: 12.5 }}>
            Saved to the Gallery
          </span>
        </div>
      )}
    </div>
  );
}

/** Personas: 2D digital people for consistent image generation. An avatar
    is a 3D rigged character; a persona is the character card — created in
    about a minute from one or more photos, then rendered anywhere. */
export function Personas({ onBusy }: PanelProps = {}) {
  const [photos, setPhotos] = useState<Asset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [consent, setConsent] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  const { data: personaData, refresh } = usePolling(api.personas, 8000);
  const personas = personaData ?? [];

  useEffect(() => {
    if (selected === null && personas.length > 0) setSelected(personas[0].id);
  }, [personas, selected]);

  useEffect(
    () => () => {
      pollRef.current?.();
    },
    [],
  );

  const busy = !!job && RUNNING.includes(job.state);
  useEffect(() => onBusy?.(busy || uploading), [busy, uploading, onBusy]);

  const addFiles = async (files: FileList | File[]) => {
    setError(null);
    setUploading(true);
    try {
      for (const f of Array.from(files)) {
        const asset = await api.uploadAsset(f);
        setPhotos((p) => (p.some((x) => x.id === asset.id) ? p : [...p, asset]));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const create = async () => {
    setError(null);
    setJob(null);
    try {
      const created = await api.createPersona(
        photos.map((p) => p.id),
        consent,
        name.trim() || undefined,
      );
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, (j) => {
        setJob(j);
        if (j.state === "completed") {
          setPhotos([]);
          setName("");
          refresh();
        }
      }, 900);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const remove = async (id: string) => {
    setError(null);
    try {
      await api.deletePersona(id);
      if (selected === id) setSelected(null);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const active = personas.find((p) => p.id === selected) ?? null;

  return (
    <>
      <h1 className="ws-hide">Personas</h1>
      <p className="sub ws-hide">
        A persona is a 2D digital person: saved once from a photo or a few,
        then rendered into any scene, pose or outfit — the same person every
        time, likeness measured on every render. (For a 3D rigged character,
        build an avatar instead.)
      </p>

      <div className="panel stack" style={{ maxWidth: 760 }}>
        <h2 style={{ margin: 0 }}>New persona</h2>
        <label className="drop" style={{ padding: 18 }}>
          <input
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => e.target.files && void addFiles(e.target.files)}
          />
          {uploading
            ? "Uploading…"
            : photos.length
              ? `${photos.length} photo(s) ready — add more or create`
              : "Drop or pick one or more photos of the person"}
        </label>
        {photos.length > 0 && (
          <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
            {photos.map((p) => (
              <img
                key={p.id}
                src={api.assetFileUrl(p.id)}
                alt=""
                style={{
                  width: 64,
                  height: 64,
                  objectFit: "cover",
                  borderRadius: 8,
                  border: "1px solid var(--line)",
                }}
              />
            ))}
          </div>
        )}
        <input
          type="text"
          placeholder="Name (e.g. Mira)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          aria-label="Persona name"
        />
        <label className="row" style={{ gap: 8, fontSize: 13.5 }}>
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => setConsent(e.target.checked)}
          />
          I confirm the person depicted has given explicit consent for
          their likeness to be saved and rendered (or the image is not of
          a real person).
        </label>
        <div className="row">
          <button
            type="button"
            className="btn primary"
            disabled={busy || uploading || photos.length === 0 || !consent}
            onClick={() => void create()}
          >
            {busy ? (
              <>
                <span className="spinner" aria-hidden /> Creating…
              </>
            ) : (
              "Create persona"
            )}
          </button>
        </div>
        {error && <div className="notice">{error}</div>}
        {job && (
          <div className="stack" style={{ gap: 8 }}>
            <div className="row">
              <span className={`badge ${job.state}`}>{job.state}</span>
              <span className="mono dim" style={{ fontSize: 12 }}>
                job {job.id}
              </span>
            </div>
            <Pipeline job={job} />
            {job.error && <div className="notice">{job.error}</div>}
          </div>
        )}
      </div>

      {personas.length > 0 && (
        <div className="panel stack" style={{ maxWidth: 760 }}>
          <h2 style={{ margin: 0 }}>Saved personas</h2>
          <div className="row" style={{ flexWrap: "wrap", gap: 10 }}>
            {personas.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`btn${selected === p.id ? " primary" : ""}`}
                style={{ display: "flex", alignItems: "center", gap: 8 }}
                onClick={() => setSelected(p.id)}
              >
                {p.face_asset && (
                  <img
                    src={api.assetFileUrl(p.face_asset)}
                    alt=""
                    style={{
                      width: 36,
                      height: 36,
                      objectFit: "cover",
                      borderRadius: "50%",
                    }}
                  />
                )}
                {p.name}
              </button>
            ))}
          </div>
          {active && (
            <div className="row" style={{ gap: 8 }}>
              <span className="dim" style={{ fontSize: 12.5 }}>
                created {active.created_at.slice(0, 10)} ·{" "}
                {active.source_assets.length} source photo(s)
              </span>
              <button
                type="button"
                className="btn"
                onClick={() => void remove(active.id)}
              >
                Delete
              </button>
            </div>
          )}
          {active && (
            <div key={active.id}>
              <PersonaRenderPanel persona={active} />
            </div>
          )}
        </div>
      )}
    </>
  );
}
