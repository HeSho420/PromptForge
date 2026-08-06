import { useEffect, useRef, useState } from "react";
import { api, pollJob } from "../api";
import { JobControls, revealOnLoad, UploadArea } from "../components/parts";
import { Pipeline } from "../components/Pipeline";
import { currentStage, pickFx, ProcessFX } from "../components/ProcessFX";
import type { Asset, Job } from "../types";
import type { PanelProps } from "./Studio";

type VideoMeta = { frames?: number; fps?: number; duration_s?: number };

/**
 * Motion transfer: a person from a photo performs a video's motion.
 *
 * Two inputs, deliberately: a photo of the person, and the clip whose motion
 * should be copied. Everything else has a working default, because the
 * settings that matter here are the ones this machine forces (length and
 * resolution) and the pipeline decides those from measured memory rather than
 * asking the user to guess.
 */
export function Motion({ incoming, onConsumed, onBusy }: PanelProps = {}) {
  const [person, setPerson] = useState<Asset | null>(null);
  const [clip, setClip] = useState<Asset | null>(null);
  const [prompt, setPrompt] = useState("");
  const [keepScene, setKeepScene] = useState(true);
  const [busyPerson, setBusyPerson] = useState(false);
  const [busyClip, setBusyClip] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<(() => void) | null>(null);

  useEffect(() => () => pollRef.current?.(), []);

  // The shell routes a dropped file here only when it is a video, so it can
  // only be the driving clip — a photo would have stayed in the edit mode.
  useEffect(() => {
    if (!incoming?.length) return;
    // Sorted by role, not by drop order: the video is always the motion to
    // copy and a photo is always the person performing it.
    const clipAsset = incoming.find((a) => a.kind === "video");
    const personAsset = incoming.find((a) => a.kind !== "video");
    if (clipAsset) setClip(clipAsset);
    if (personAsset) setPerson(personAsset);
    onConsumed?.();
  }, [incoming]);

  const meta = (clip?.meta?.video ?? {}) as VideoMeta;
  const running = !!job && ["pending", "running", "retrying"].includes(job.state);
  useEffect(() => onBusy?.(running), [running]);
  const resultId =
    job?.state === "completed" ? (job.result?.asset_id as string) : null;

  const run = async () => {
    if (!person || !clip) return;
    setError(null);
    setJob(null);
    try {
      const created = await api.motionTransfer({
        referenceAssetId: person.id,
        drivingAssetId: clip.id,
        prompt,
        preserveBackground: keepScene,
      });
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, setJob, 900);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const upload = (
    file: File,
    setBusy: (b: boolean) => void,
    set: (a: Asset) => void,
  ) => {
    setError(null);
    setBusy(true);
    void api
      .uploadAsset(file)
      .then(set)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <>
      <h1 className="ws-hide">Motion</h1>
      <p className="sub ws-hide">
        Give it a photo of a person and a clip of someone moving, and the
        person from the photo performs that motion. Longer clips are rendered
        in overlapping parts and joined — this machine cannot hold a long one
        in memory at once.
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="slots">
        <div className="panel">
          <h2>The person</h2>
          {person ? (
            <div className="ref-slot">
              <img
                ref={revealOnLoad}
                src={api.assetFileUrl(person.id)}
                alt={person.filename}
              />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="ellipsis">{person.filename}</div>
                <div className="hint">Their look is carried into the clip</div>
              </div>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => setPerson(null)}
              >
                Change
              </button>
            </div>
          ) : (
            <UploadArea
              busy={busyPerson}
              label="Drop a photo of the person"
              hint="A clear, well-lit shot works best"
              onFile={(f) => upload(f, setBusyPerson, setPerson)}
            />
          )}
        </div>

        <div className="panel">
          <h2>The motion</h2>
          {clip ? (
            <div className="ref-slot">
              <video
                src={api.assetFileUrl(clip.id)}
                muted
                loop
                autoPlay
                playsInline
              />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="ellipsis">{clip.filename}</div>
                <div className="hint">
                  {meta.frames != null
                    ? `${meta.frames} frames · ${Math.round(meta.fps ?? 0)} fps · ${(meta.duration_s ?? 0).toFixed(1)}s`
                    : "video"}
                </div>
              </div>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => setClip(null)}
              >
                Change
              </button>
            </div>
          ) : (
            <UploadArea
              busy={busyClip}
              video
              label="Drop the clip to copy"
              hint="MP4, MOV, WEBM or MKV · up to 60 seconds"
              onFile={(f) => upload(f, setBusyClip, setClip)}
            />
          )}
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <label className="field" htmlFor="motion-prompt">
          Describe the person (optional)
        </label>
        <textarea
          id="motion-prompt"
          rows={2}
          placeholder='e.g. "a woman with ginger hair in a green top, dancing"'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <p className="hint" style={{ marginTop: 6 }}>
          The reference photo carries appearance — hair, clothing, build — far
          more reliably than a face. Expect a family resemblance rather than a
          match.
        </p>

        <label className="row" style={{ marginTop: 12, gap: 8 }}>
          <input
            type="checkbox"
            checked={keepScene}
            onChange={(e) => setKeepScene(e.target.checked)}
          />
          <span>
            Keep the clip&rsquo;s background
            <span className="hint">
              {" "}
              — only the person is replaced. Turn off to rebuild the whole
              frame.
            </span>
          </span>
        </label>

        <div className="row" style={{ marginTop: 14 }}>
          <button
            type="button"
            className="btn primary"
            onClick={() => void run()}
            disabled={running || !person || !clip}
            title={
              !person || !clip
                ? "Add both a photo and a clip first"
                : "Render the motion transfer"
            }
          >
            {running ? "Rendering…" : "Transfer the motion"}
          </button>
          {job && <JobControls job={job} onUpdate={setJob} />}
        </div>
      </div>

      {running && (
        // The performance being transferred, live on the person's photo:
        // the driving motion streams across as a walking skeleton while the
        // stage captions narrate (finding the person, rendering part 2/3…).
        (() => {
          const fx = pickFx({
            stage: currentStage(job).stage,
            fallback: { effect: "motion", label: "Transferring the motion" },
          });
          return (
            <div
              className={`fx-stage${person ? "" : " empty"}`}
              style={{ marginTop: 16 }}
            >
              {person && (
                <img
                  className="fx-stage-img"
                  src={api.assetFileUrl(person.id)}
                  alt=""
                  aria-hidden
                />
              )}
              <ProcessFX
                active
                effect={fx.effect}
                label={fx.label}
                standalone={!person}
              />
            </div>
          );
        })()
      )}

      {job && (
        <div style={{ marginTop: 16 }}>
          <Pipeline job={job} />
        </div>
      )}

      {resultId && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h2>
            Result{" "}
            <span className="badge completed">
              {String(job?.result?.frames ?? "?")} frames ·{" "}
              {String(job?.result?.resolution ?? "")}
            </span>
            {job?.result?.parts != null && (job.result.parts as number) > 1 && (
              <span className="badge" title="Rendered in overlapping parts and cross-faded">
                {String(job.result.parts)} parts joined
              </span>
            )}
            {job?.result?.preserved_background === true && (
              <span className="badge">background kept</span>
            )}
          </h2>
          <video
            className="result-video"
            src={api.assetFileUrl(resultId)}
            controls
            loop
            autoPlay
            muted
            playsInline
          />
        </div>
      )}
    </>
  );
}
