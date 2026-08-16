import { useEffect, useRef, useState } from "react";
import { api, pollJob } from "../api";
import { MaskEditor } from "../components/MaskEditor";
import { BeforeAfter, JobControls, revealOnLoad } from "../components/parts";
import { Pipeline } from "../components/Pipeline";
import { currentStage, pickFx } from "../components/ProcessFX";
import { ResultView } from "../components/ResultView";
import type { Asset, Job } from "../types";

/** Props the Workspace shell passes in; all optional so the panel still
 *  stands alone. `incoming` is a file the shell already uploaded on this
 *  mode's behalf, `onBusy` reports a running job so the mode chip can show a
 *  dot while you are looking at a different mode. */
export type PanelProps = {
  incoming?: Asset[] | null;
  onConsumed?: () => void;
  onBusy?: (busy: boolean) => void;
};

export function Studio({ incoming, onConsumed, onBusy }: PanelProps = {}) {
  const [asset, setAsset] = useState<Asset | null>(null);
  const [prompt, setPrompt] = useState("");
  const [proposing, setProposing] = useState(false);
  const [autoMask, setAutoMask] = useState<string | null>(null);
  const [maskIsMock, setMaskIsMock] = useState(false);
  // What the backend said about HOW the region was chosen. The API always
  // sent these; the old interface dropped them on arrival, which is what
  // turned a face-mask guess from a caught problem into a silent one (D10).
  const [maskSource, setMaskSource] = useState<string | null>(null);
  const [maskNotes, setMaskNotes] = useState<string[]>([]);
  // The compiled program for the edit that is currently rendering.
  const [pendingPlan, setPendingPlan] = useState<
    | { step: number; task: string; operation: string; target: string;
        instruction: string }[]
    | null
  >(null);
  const [userMask, setUserMask] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  // A second photo to take a subject FROM. Its presence turns the edit into
  // a combine — that routing lives in the backend, not here.
  const [reference, setReference] = useState<Asset | null>(null);
  // ONE image on screen: the editor canvas becomes the result in place.
  const [view, setView] = useState<"edit" | "result">("edit");
  // Set after "Continue from this result": the asset file URL then serves the
  // promoted version, and the ?v= param forces the browser past its cache.
  const [imgRev, setImgRev] = useState<string | null>(null);
  const pollRef = useRef<(() => void) | null>(null);
  const workingUrl = (id: string) =>
    imgRev ? `${api.assetFileUrl(id)}?v=${imgRev}` : api.assetFileUrl(id);

  // Auto-switch the single image to the finished result.
  //
  // …and let the mask go with it. A mask is an instruction for ONE edit, and
  // it was outliving the edit it belonged to: the next prompt silently
  // inherited it, and a drawn mask pins step 1 to a regional inpaint — so a
  // follow-up that should have gone to the background, pose or 3D engine
  // was quietly repainted inside the old rectangle instead. Clearing costs a
  // redraw; not clearing costs a wrong answer with no way to see why.
  useEffect(() => {
    if (job?.state !== "completed") return;
    const vid = job.result?.version_id as string | undefined;
    if (vid) {
      // PRELOAD the result before swapping the view: switching first made
      // the panel appear with an image that had no bytes yet (collapsed
      // frame, then a jump when it landed) — the intermittent completion
      // glitch. The editor stays on screen until the pixels exist; if the
      // fetch stalls, a short fallback still swaps so the UI can't hang.
      let done = false;
      const swap = () => {
        if (!done) {
          done = true;
          setView("result");
        }
      };
      const img = new Image();
      img.onload = swap;
      img.onerror = swap;
      img.src = api.versionFileUrl(vid);
      const fallback = window.setTimeout(swap, 2000);
      setUserMask(null);
      setAutoMask(null);
      setMaskIsMock(false);
      setMaskSource(null);
      setMaskNotes([]);
      return () => {
        done = true;
        window.clearTimeout(fallback);
      };
    }
    setUserMask(null);
    setAutoMask(null);
    setMaskIsMock(false);
    setMaskSource(null);
    setMaskNotes([]);
  }, [job?.state, job?.result]);

  useEffect(
    () => () => {
      pollRef.current?.();
    },
    [],
  );

  // A file the shell uploaded for us. Adopt it exactly as a local upload
  // would, then tell the shell so it cannot be adopted twice.
  useEffect(() => {
    if (!incoming?.length) return;
    // First image is what gets edited; a second becomes the reference to take
    // a subject or a face from.
    // Stop any poll still running for the PREVIOUS image. Without this the
    // orphan keeps ticking and re-attaches the old job to the new photo -
    // the before/after slider then compares two unrelated pictures.
    pollRef.current?.();
    pollRef.current = null;
    setAsset(incoming[0]);
    setReference(incoming[1] ?? null);
    setAutoMask(null);
    setUserMask(null);
    setMaskSource(null);
    setMaskNotes([]);
    setJob(null);
    setImgRev(null);
    setView("edit");
    onConsumed?.();
  }, [incoming]);


  const proposeMask = async () => {
    if (!asset) return;
    setError(null);
    setProposing(true);
    try {
      const preview = await api.maskPreview(asset.id, prompt);
      setAutoMask(preview.mask_b64);
      setMaskIsMock(preview.is_mock);
      setMaskSource(preview.source ?? null);
      setMaskNotes(preview.notes ?? []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setProposing(false);
    }
  };

  const runEdit = async () => {
    if (!asset) return;
    setError(null);
    setJob(null);
    setPendingPlan(null);
    setView("edit"); // watch the steps happen live on the image
    // Show the compiled program while the render runs — fire-and-forget so
    // a slow planner never delays the actual job (Step 11: the plan used to
    // be visible only in the job log, after the minutes were spent).
    void api
      .previewEditPlan(prompt, !!userMask)
      .then((p) => setPendingPlan(p.planned ? p.steps : null))
      .catch(() => setPendingPlan(null));
    try {
      const created = await api.createEdit(
        asset.id,
        prompt,
        userMask ?? undefined,
        reference ? [reference.id] : undefined,
      );
      setJob(created);
      pollRef.current?.();
      pollRef.current = pollJob(created.id, setJob, 700);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const resultVersionId =
    job?.state === "completed" ? (job.result?.version_id as string) : null;

  // "Continue from this result": the rendered version becomes the working
  // image — the next edit, mask or video builds on it, not the original.
  const continueFromResult = async () => {
    if (!resultVersionId) return;
    setError(null);
    try {
      await api.promoteVersion(resultVersionId);
      setImgRev(resultVersionId);
      setAutoMask(null);
      setUserMask(null);
      setJob(null);
      setPrompt("");
      setView("edit");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const busy = !!job && ["pending", "running", "retrying"].includes(job.state);
  useEffect(() => onBusy?.(busy), [busy]);
  const latestLogVersion = (re: RegExp) => {
    for (const l of [...(job?.logs ?? [])].reverse()) {
      const m = re.exec(l.msg);
      if (m) return m[1];
    }
    return null;
  };
  // Masks the pipeline created/changed (auto-generated or refined) and
  // intermediate step results — both mirrored live on THE image.
  const refinedVersionId = latestLogVersion(/\[mask\] .*— version (\w+)/);
  const previewVersionId = latestLogVersion(/\[preview\] .*— version (\w+)/);

  // What is happening to the image RIGHT NOW, in the planner's own words:
  // the last [stage] marker names the phase, and during `render` the step
  // counter picks the operation out of the compiled plan — so the canvas
  // can animate "replacing the background" as background replacement, not
  // as a generic sweep.
  const stageInfo = currentStage(job);
  const planStep = pendingPlan?.length
    ? pendingPlan[Math.min(
        Math.max((stageInfo.step ?? 1) - 1, 0),
        pendingPlan.length - 1,
      )]
    : null;
  const liveFx = pickFx({
    stage: stageInfo.stage,
    operation:
      planStep?.operation ??
      // Without a plan yet, a background-sourced mask still says what the
      // edit is — the one operation the region itself identifies.
      (maskSource === "background" ? "REPLACE_BACKGROUND" : null),
    target: planStep?.target ?? null,
    task: planStep?.task ?? null,
  });
  // The region the animation should live in: the user's own mask first,
  // then the proposal, then whatever the pipeline refined mid-run.
  const fxMaskUrl =
    userMask ??
    autoMask ??
    (refinedVersionId ? api.versionFileUrl(refinedVersionId) : null);
  const plan = (job?.result?.plan ?? null) as
    | { step: number; task: string; instruction: string; workflow: string;
        model: string }[]
    | null;

  return (
    <>
      {!asset ? (
        <p className="dim" style={{ margin: 0, fontSize: 13 }}>
          Add a photo above to start. Describe the change, review the proposed
          mask — painted red, like rubylith film — correct it if needed, then
          run the edit.
        </p>
      ) : (
        <div className="studio">
          <div className="stack">
            {view === "result" && resultVersionId ? (
              <div className="panel">
                <h2>
                  Result{" "}
                  {job?.result?.is_mock === true && (
                    <span className="badge mock">mock adapter</span>
                  )}
                  {typeof job?.result?.overall === "number" && (
                    <span
                      className={`badge ${(job.result.passed as boolean) ? "completed" : "pending"}`}
                      title="Overall quality score (0-100)"
                    >
                      quality {String(job.result.overall)}/100
                    </span>
                  )}
                </h2>
                {plan && plan.length > 0 && (
                  <div className="row" style={{ flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                    {plan.map((s) => (
                      <span
                        key={s.step}
                        className="badge"
                        title={`"${s.instruction}" — ${s.workflow}, model: ${s.model}`}
                      >
                        {s.step}. {s.task} · {s.model}
                      </span>
                    ))}
                  </div>
                )}
                {job?.result?.scores != null && (
                  <div className="row" style={{ flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                    {Object.entries(job.result.scores as Record<string, number>).map(
                      ([k, v]) => (
                        <span
                          key={k}
                          className={`badge ${v >= 95 ? "completed" : v >= 80 ? "pending" : "failed"}`}
                          style={{ fontSize: 10.5 }}
                        >
                          {k.replace(/_/g, " ")} {v}
                        </span>
                      ),
                    )}
                  </div>
                )}
                {/* A 3D scene is not a version of the photo — it is a
                    separate thing you move around in, so it gets the walk
                    viewer rather than the before/after slider. */}
                {typeof job?.result?.scene_asset === "string" ? (
                  <div className="stack" style={{ gap: 8 }}>
                    <ResultView
                      kind="model"
                      mode="walk"
                      url={api.assetFileUrl(job.result.scene_asset as string)}
                      height={420}
                    />
                    <span className="dim" style={{ fontSize: 11.5 }}>
                      {Number(job.result.layers) > 1
                        ? "Two layers: the photograph, and a reconstruction of what stood behind its foreground so moving sideways does not open black gaps. The second layer is a guess, not a photograph."
                        : "One camera frustum — anything the lens could not see is absent, so stepping sideways opens gaps."}
                    </span>
                  </div>
                ) : (
                  <BeforeAfter
                    beforeUrl={workingUrl(asset.id)}
                    afterUrl={api.versionFileUrl(resultVersionId)}
                  />
                )}
                <div className="row" style={{ marginTop: 10 }}>
                  <button
                    type="button"
                    className="btn primary small"
                    onClick={() => void continueFromResult()}
                    title="The result becomes the working image — your next edit builds on it"
                  >
                    Continue from this result
                  </button>
                  <button
                    type="button"
                    className="btn ghost small"
                    onClick={() => {
                      // Going back is a fresh start, not a resumption: the
                      // mask that produced the result must not be waiting.
                      setUserMask(null);
                      setAutoMask(null);
                      setMaskIsMock(false);
                      setMaskSource(null);
                      setMaskNotes([]);
                      setView("edit");
                    }}
                    title="Back to the editor with the image you started from — the mask is cleared"
                  >
                    ← Back to the original
                  </button>
                </div>
              </div>
            ) : (
              <>
                <MaskEditor
                  imageUrl={workingUrl(asset.id)}
                  assetId={asset.id}
                  autoMask={autoMask}
                  refinedMaskUrl={
                    refinedVersionId
                      ? api.versionFileUrl(refinedVersionId)
                      : null
                  }
                  previewUrl={
                    previewVersionId
                      ? api.versionFileUrl(previewVersionId)
                      : null
                  }
                  onMaskChange={setUserMask}
                  rendering={busy}
                  fx={busy ? { ...liveFx, maskUrl: fxMaskUrl } : null}
                />
                {autoMask && maskNotes.length > 0 && (
                  <div
                    role="note"
                    style={{
                      fontSize: 12.5,
                      margin: "6px 0 0",
                      padding: "6px 10px",
                      borderRadius: 6,
                      border: "1px solid",
                      borderColor:
                        maskSource === "sam" ? "#c98a2b" : "#3a4a5a",
                      color: maskSource === "sam" ? "#e8b45a" : "#9ab0c4",
                    }}
                  >
                    {maskSource === "sam" && <strong>⚠ Guessed region — </strong>}
                    {maskNotes.join(" · ")}
                  </div>
                )}
                {refinedVersionId && busy && (
                  <p className="dim" style={{ fontSize: 12.5, margin: "6px 0 0" }}>
                    ✨ The pipeline generated/refined the mask automatically —
                    shown live above.
                  </p>
                )}
                {resultVersionId && (
                  <div className="row">
                    <button
                      type="button"
                      className="btn ghost small"
                      onClick={() => setView("result")}
                    >
                      Show result →
                    </button>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="panel">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <strong>{asset.filename}</strong>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => {
                  // Same orphan-poll problem as adopting a new photo: without
                  // stopping it the interval keeps hitting /api/jobs forever
                  // with no UI to show it.
                  pollRef.current?.();
                  pollRef.current = null;
                  setAsset(null);
                  setReference(null);
                  setJob(null);
                }}
              >
                Change image
              </button>
            </div>

            <label className="field" htmlFor="prompt">
              Describe the edit
            </label>
            <textarea
              id="prompt"
              rows={3}
              placeholder='e.g. "remove the chair" or "change the sky to sunset"'
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />

            {/* Second image. Attaching one is the whole instruction: the
                subject of THAT photo gets brought into THIS one, cut from
                its own background, and blended so the light matches. */}
            <label className="field" style={{ marginTop: 14 }}>
              Combine with a second photo
            </label>
            {reference ? (
              <div className="ref-slot">
                <img
                  ref={revealOnLoad}
                  src={api.assetFileUrl(reference.id)}
                  alt={reference.filename}
                />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="ellipsis">{reference.filename}</div>
                  <div style={{ fontSize: 12, color: "var(--muted)" }}>
                    Its subject will be placed into your image
                  </div>
                </div>
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => setReference(null)}
                >
                  Remove
                </button>
              </div>
            ) : (
              <p className="dim" style={{ margin: 0, fontSize: 12.5 }}>
                Drop <strong>two</strong> photos at the top to combine them —
                the second one&rsquo;s subject (or face) is placed into the
                first.
              </p>
            )}
            {reference && (
              <p className="hint" style={{ marginTop: 8 }}>
                Say where it goes in the prompt (&ldquo;put her on the bench on
                the left&rdquo;), or paint the spot on the image — either works.
              </p>
            )}

            <div className="row" style={{ marginTop: 12 }}>
              <button
                type="button"
                className="btn ghost"
                onClick={() => void proposeMask()}
                disabled={proposing || !prompt.trim()}
              >
                {proposing ? "Proposing…" : "Propose mask"}
              </button>
              <button
                type="button"
                className="btn primary"
                onClick={() => void runEdit()}
                disabled={busy || !prompt.trim()}
              >
                {busy ? "Rendering…" : "Run edit"}
              </button>
            </div>

            {maskIsMock && autoMask && (
              <p style={{ fontSize: 12.5, color: "var(--muted)", marginBottom: 0 }}>
                Mask proposed by the <span className="badge mock">mock</span>{" "}
                segmentation adapter — a keyword heuristic, not a real model.
                Correct it with the brush before running.
              </p>
            )}

            {error && (
              <div className="notice" style={{ marginTop: 12 }}>
                {error}
              </div>
            )}

            {job && (
              <div className="stack" style={{ marginTop: 14 }}>
                <div className="row">
                  <span className={`badge ${job.state}`}>{job.state}</span>
                  <span className="mono" style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--muted)" }}>
                    job {job.id} · attempt {job.attempts}
                  </span>
                  <JobControls job={job} onUpdate={setJob} />
                </div>
                {busy && pendingPlan && pendingPlan.length > 0 && (
                  <div
                    className="row"
                    style={{ flexWrap: "wrap", gap: 6 }}
                    title="The program your request compiled to — every step listed here will run; anything missing was not understood"
                  >
                    <span className="dim" style={{ fontSize: 11.5 }}>
                      plan:
                    </span>
                    {pendingPlan.map((s) => (
                      <span
                        key={s.step}
                        className="badge"
                        title={`"${s.instruction}"`}
                      >
                        {s.step}. {s.operation || s.task}
                        {s.target ? `(${s.target})` : ""}
                      </span>
                    ))}
                  </div>
                )}
                <Pipeline job={job} />
                {job.error && <div className="notice">{job.error}</div>}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
