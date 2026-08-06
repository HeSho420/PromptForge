import { useEffect, useRef, useState } from "react";
import { api, pollJob, usePolling } from "../api";
import { Pipeline } from "../components/Pipeline";
import { currentStage, pickFx, ProcessFX } from "../components/ProcessFX";
import type { Asset, AvatarProfile, Job } from "../types";
import type { PanelProps } from "./Studio";
import { revealOnLoad } from "../components/parts";
import { ResultView } from "../components/ResultView";

const BINS = ["front", "front-right", "right", "back-right",
              "back", "back-left", "left", "front-left"] as const;

const RUNNING = ["pending", "running", "retrying"];

/** Interactive 3D-environment viewer: the SV3D orbital frames are placed as a
    view-dependent billboard on a ground plane in a perspective scene, so the
    subject genuinely rotates in 3D as you orbit the camera. Drag to orbit +
    tilt, wheel to zoom, ▶ to auto-rotate. Dependency-free (CSS 3D). */
const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function OrbitViewer({ frames }: { frames: string[] }) {
  const [azimuth, setAzimuth] = useState(0); // degrees, continuous
  const [tilt, setTilt] = useState(12); // camera elevation
  const [zoom, setZoom] = useState(1);
  // Respect the OS "reduce motion" setting: start paused when it's on.
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const drag = useRef<{ x: number; y: number; az: number; tl: number } | null>(
    null,
  );
  const sceneRef = useRef<HTMLDivElement | null>(null);
  const n = frames.length;

  useEffect(() => {
    for (const id of frames) {
      const img = new Image();
      img.src = api.assetFileUrl(id);
    }
  }, [frames]);

  // Auto-rotate loop. Skips work when the tab is hidden OR the viewer is
  // route-hidden (display:none → offsetParent is null), so it never spins an
  // off-screen component or drains the battery in the background.
  useEffect(() => {
    if (!playing || n === 0) return;
    const t = window.setInterval(() => {
      if (document.hidden || sceneRef.current?.offsetParent == null) return;
      setAzimuth((a) => a + 1.4);
    }, 40);
    return () => window.clearInterval(t);
  }, [playing, n]);

  if (n === 0) return null;
  const az = ((azimuth % 360) + 360) % 360;
  const shown = Math.round((az / 360) * n) % n; // view-dependent frame

  return (
    <div className="stack" style={{ gap: 8 }}>
      <div
        ref={sceneRef}
        className="scene3d"
        role="slider"
        aria-label="Rotate avatar in 3D"
        aria-valuemin={0}
        aria-valuemax={359}
        aria-valuenow={Math.round(az)}
        tabIndex={0}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, az: azimuth, tl: tilt };
          setPlaying(false);
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          setAzimuth(drag.current.az + (e.clientX - drag.current.x) * 0.7);
          setTilt(
            Math.max(-4, Math.min(46, drag.current.tl - (e.clientY - drag.current.y) * 0.3)),
          );
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onWheel={(e) => setZoom((z) => Math.max(0.6, Math.min(2.2, z - e.deltaY * 0.0016)))}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") { setPlaying(false); setAzimuth((a) => a - 6); }
          if (e.key === "ArrowRight") { setPlaying(false); setAzimuth((a) => a + 6); }
        }}
      >
        <div
          className="scene3d-world"
          style={{ transform: `rotateX(${tilt}deg) scale(${zoom})` }}
        >
          <div className="scene3d-ground" />
          <div className="scene3d-shadow" style={{ transform: `translate(-50%, -50%) rotateX(90deg) scale(${1 + (46 - tilt) / 90})` }} />
          <img
            ref={revealOnLoad}
            className="scene3d-subject"
            src={api.assetFileUrl(frames[shown])}
            alt={`avatar view at ${Math.round(az)}°`}
            style={{ transform: `translateX(-50%) rotateX(${-tilt}deg)` }}
            draggable={false}
          />
        </div>
        <span className="orbit-angle">{Math.round(az)}°</span>
        <span className="orbit-hint">drag to orbit · wheel to zoom</span>
      </div>
      <div className="row" style={{ gap: 8 }}>
        <button
          type="button"
          className="btn ghost small"
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? "⏸ Pause" : "▶ Auto-rotate"}
        </button>
        <button
          type="button"
          className="btn ghost small"
          onClick={() => { setTilt(12); setZoom(1); setAzimuth(0); }}
        >
          Reset view
        </button>
        <span className="dim" style={{ fontSize: 11.5, alignSelf: "center" }}>
          {n} orbital views · move around the subject in 3D
        </span>
      </div>
    </div>
  );
}

/** Prompt-based renders with a saved avatar: photoreal images and videos. */
function AvatarRenderPanel({ avatar }: { avatar: AvatarProfile }) {
  const [prompt, setPrompt] = useState("");
  const [video, setVideo] = useState(false);
  const [length, setLength] = useState(49);
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
      const created = await api.renderAvatar(avatar.id, prompt, video, length);
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
  const videoId =
    job?.state === "completed"
      ? (job.result?.video_asset_id as string | undefined)
      : null;
  const realism =
    job?.state === "completed" ? (job.result?.realism as number | null) : null;

  return (
    <div className="stack" style={{ gap: 10 }}>
      <h2 style={{ margin: 0 }}>Render with this avatar</h2>
      <p className="dim" style={{ margin: 0, fontSize: 13 }}>
        Describe any scene — the identity pipeline (PhotoMaker + SDXL) renders
        this person into it photoreal, and can animate the result (WAN).
      </p>
      <textarea
        rows={2}
        placeholder='e.g. "hiking on a mountain ridge at golden hour, 35mm photo"'
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        aria-label="Avatar render prompt"
      />
      <div className="row" style={{ flexWrap: "wrap" }}>
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
          ) : video ? (
            "Render + animate"
          ) : (
            "Render image"
          )}
        </button>
        <label className="row" style={{ gap: 6, fontSize: 13 }}>
          <input
            type="checkbox"
            checked={video}
            onChange={(e) => setVideo(e.target.checked)}
          />
          also make a video
        </label>
        {video && (
          <label className="dim" style={{ fontSize: 12.5 }}>
            Length{" "}
            <input
              type="range"
              min={17}
              max={81}
              step={8}
              value={length}
              onChange={(e) => setLength(Number(e.target.value))}
            />{" "}
            {(length / 24).toFixed(1)}s
          </label>
        )}
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
                  label: video
                    ? "Rendering this person into the scene, then animating"
                    : "Rendering this person into the scene",
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
          </div>
          <img
            ref={revealOnLoad}
            src={api.assetFileUrl(imageId)}
            alt={prompt}
            style={{ width: "100%", maxWidth: 520, borderRadius: 10,
                     border: "1px solid var(--line)" }}
          />
          {videoId && (
        <>
          <h2 style={{ margin: 0 }}>Animated</h2>
          {/* ResultView, not <img>: this asset is an MP4 and an <img> tag
              renders it as a broken-image icon. */}
          <ResultView kind="video" url={api.assetFileUrl(videoId)} />
        </>
      )}
          <span className="dim" style={{ fontSize: 12.5 }}>
            Saved to the Gallery
          </span>
        </div>
      )}
    </div>
  );
}

export function Avatar({ incoming, onConsumed, onBusy }: PanelProps = {}) {
  const [photos, setPhotos] = useState<Asset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [consent, setConsent] = useState(false);
  // Both default on: they are what make the mesh look like the person, and
  // like a whole person. Off is a deliberate choice, not a fallback.
  const [texture, setTexture] = useState(true);
  const [completeBody, setCompleteBody] = useState(true);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  // Polled (not fetched once) so the list appears/recovers even when the
  // backend comes up after the app loaded.
  const { data: avatarData, refresh: refreshAvatars } = usePolling(
    api.avatars,
    8000,
  );
  const avatars = avatarData ?? [];

  useEffect(() => {
    if (selected === null && avatars.length > 0) setSelected(avatars[0].id);
  }, [avatars, selected]);

  useEffect(
    () => () => {
      pollRef.current?.();
    },
    [],
  );

  // A photo the shell uploaded joins the dataset rather than replacing it —
  // an avatar is built from a SET, so dropping another photo should add to it.
  useEffect(() => {
    if (!incoming?.length) return;
    setPhotos((prev) => {
      const seen = new Set(prev.map((p) => p.id));
      return [...prev, ...incoming.filter((a) => !seen.has(a.id))];
    });
    onConsumed?.();
  }, [incoming]);

  const addFiles = async (files: FileList | null) => {
    if (!files) return;
    setError(null);
    setUploading(true);
    try {
      for (const f of Array.from(files).slice(0, 50)) {
        const a = await api.uploadAsset(f);
        setPhotos((prev) => [...prev, a]);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const run = async () => {
    setError(null);
    setJob(null);
    try {
      const created = await api.createAvatar(
        photos.map((p) => p.id),
        consent,
        name.trim() || undefined,
        { texture, completeBody },
      );
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, (j) => {
        setJob(j);
        if (j.state === "completed") {
          refreshAvatars();
          const id = j.result?.avatar_id as string | undefined;
          if (id) setSelected(id);
        }
      }, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = !!job && RUNNING.includes(job.state);
  useEffect(() => onBusy?.(busy), [busy]);
  const result = job?.state === "completed"
    ? (job.result as {
        coverage: Record<string, string[]>;
        missing: string[];
        synthetic_assets: string[];
        next: string;
      } | null)
    : null;
  const active = avatars.find((a) => a.id === selected) ?? null;

  return (
    <>
      <h1 className="ws-hide">Avatar</h1>
      <p className="sub ws-hide">
        Build a consented digital human from photos: the pipeline isolates the
        subject (SAM), checks which angles you covered, synthesizes the missing
        ones (SV3D), and saves a movable avatar you can rotate and render into
        any prompted image or video — photoreal, identity-preserving.
      </p>

      <div className="panel stack" style={{ maxWidth: 760 }}>
        <label className="drop" style={{ padding: 22 }}>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => void addFiles(e.target.files)}
          />
          {uploading
            ? "Uploading…"
            : photos.length
              ? `${photos.length} photo(s) — click to add more (8+ recommended, all sides)`
              : "Click to add photos of the person (8+ recommended, all sides)"}
        </label>

        {photos.length > 0 && (
          <div className="thumb-row">
            {photos.map((p) => (
              <img ref={revealOnLoad} key={p.id} src={api.assetFileUrl(p.id)} alt={p.filename} />
            ))}
          </div>
        )}

        <div className="row">
          <input
            type="text"
            placeholder="Avatar name (optional)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ flex: 1, maxWidth: 260 }}
            aria-label="Avatar name"
          />
        </div>

        <div className="stack" style={{ gap: 6 }}>
          <label className="row" style={{ gap: 8, fontSize: 13.5 }}>
            <input
              type="checkbox"
              checked={texture}
              onChange={(e) => setTexture(e.target.checked)}
            />
            <span>
              Texture the mesh from your photos
              <span className="dim" style={{ fontSize: 12 }}>
                {" "}
                — colour projected onto real UVs. Turn off for bare geometry,
                which is easier to paint or sculpt on yourself.
              </span>
            </span>
          </label>
          <label className="row" style={{ gap: 8, fontSize: 13.5 }}>
            <input
              type="checkbox"
              checked={completeBody}
              onChange={(e) => setCompleteBody(e.target.checked)}
            />
            <span>
              Complete the body when the photo cuts it off
              <span className="dim" style={{ fontSize: 12 }}>
                {" "}
                — a photo cropped at the thigh reconstructs as a person
                cropped at the thigh, so the rest is generated first. The
                added part is invented, not photographed.
              </span>
            </span>
          </label>
        </div>

        <label className="row" style={{ gap: 8, fontSize: 13.5 }}>
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => setConsent(e.target.checked)}
          />
          I confirm the person depicted has given explicit consent for
          creating a digital avatar from these photos.
        </label>

        <div className="row">
          <button
            type="button"
            className="btn primary"
            disabled={busy || uploading || photos.length === 0 || !consent}
            onClick={() => void run()}
          >
            {busy ? (
              <>
                <span className="spinner" aria-hidden /> Processing…
              </>
            ) : (
              "Build avatar"
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
                // The build, made visible: a camera orbits the subject's
                // photo, angle bins lighting up as coverage is checked and
                // the missing views are synthesized.
                const fx = pickFx({
                  stage: currentStage(job).stage,
                  fallback: { effect: "orbit", label: "Building the avatar" },
                });
                return (
                  <div className={`fx-stage${photos.length ? "" : " empty"}`}>
                    {photos[0] && (
                      <img
                        className="fx-stage-img"
                        src={api.assetFileUrl(photos[0].id)}
                        alt=""
                        aria-hidden
                      />
                    )}
                    <ProcessFX
                      active
                      effect={fx.effect}
                      label={fx.label}
                      standalone={!photos.length}
                    />
                  </div>
                );
              })()}
            <Pipeline job={job} />
            {job.error && <div className="notice">{job.error}</div>}
          </div>
        )}
      </div>

      {result && (
        <div className="panel stack" style={{ maxWidth: 760, marginTop: 18 }}>
          <h2 style={{ margin: 0 }}>Angle coverage</h2>
          <div className="coverage">
            {BINS.map((b) => (
              <div
                key={b}
                className={
                  "coverage-bin" +
                  (result.coverage[b]?.length ? " have" : "") +
                  (result.missing.includes(b) ? " miss" : "")
                }
              >
                <strong>{result.coverage[b]?.length ?? 0}</strong>
                {b}
              </div>
            ))}
          </div>
          <p className="dim" style={{ margin: 0, fontSize: 13 }}>
            {result.next}
          </p>
        </div>
      )}

      {avatars.length > 0 && (
        <div className="panel stack" style={{ maxWidth: 760, marginTop: 18 }}>
          <h2 style={{ margin: 0 }}>Your avatars</h2>
          <div className="avatar-list">
            {avatars.map((a) => (
              <button
                key={a.id}
                type="button"
                className={"avatar-chip" + (a.id === selected ? " on" : "")}
                onClick={() => setSelected(a.id)}
              >
                {a.face_asset && (
                  <img ref={revealOnLoad} src={api.assetFileUrl(a.face_asset)} alt="" />
                )}
                {a.name}
              </button>
            ))}
          </div>

          {active && (
            <div className="stack" style={{ gap: 14 }}>
              {(() => {
                // A real mesh beats the frame-swapping billboard whenever the
                // build produced one; the billboard stays as the fallback so
                // avatars built before (or without) a mesh tier still work.
                const mesh = (active.meta?.mesh ?? {}) as {
                  mesh_asset?: string;
                  tier_name?: string;
                  textured?: boolean;
                  note?: string;
                };
                if (mesh.mesh_asset) {
                  return (
                    <div className="stack" style={{ gap: 8 }}>
                      <ResultView
                        kind="model"
                        url={api.assetFileUrl(mesh.mesh_asset)}
                      />
                      {mesh.note && (
                        <span className="dim" style={{ fontSize: 11.5 }}>
                          {mesh.note}
                        </span>
                      )}
                      {active.frames.length > 0 && (
                        <details>
                          <summary className="dim" style={{ fontSize: 12 }}>
                            Also show the photo orbit
                          </summary>
                          <OrbitViewer frames={active.frames} />
                        </details>
                      )}
                    </div>
                  );
                }
                return active.frames.length > 0 ? (
                  <OrbitViewer frames={active.frames} />
                ) : (
                  <p className="dim" style={{ margin: 0, fontSize: 13 }}>
                    No orbit frames yet — rebuild the avatar with ComfyUI
                    running (SV3D) to make it movable.
                  </p>
                );
              })()}
              {/* keyed so prompt/job state never leaks across avatars */}
              <AvatarRenderPanel key={active.id} avatar={active} />
              <div className="row">
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => {
                    if (window.confirm(
                      `Delete avatar "${active.name}"? Its synthetic orbit `
                      + "frames are removed too; your original photos stay "
                      + "in the gallery.")) {
                      void (async () => {
                        await api.deleteAvatar(active.id);
                        setSelected(null);
                        refreshAvatars();
                      })();
                    }
                  }}
                >
                  🗑 Delete avatar
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
