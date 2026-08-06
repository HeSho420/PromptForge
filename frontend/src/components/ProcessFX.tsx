import { useEffect, useRef } from "react";
import type { Job } from "../types";

/**
 * ProcessFX — the live picture of what is being DONE to the picture.
 *
 * While a job runs, this canvas sits over the image and animates the actual
 * operation in progress: a background replacement shows the background
 * dissolving and a new wash sweeping in behind the subject, a removal shows
 * the region breaking into particles, an upscale shows the pixel grid
 * subdividing, a generation shows noise crystallising into an image. The
 * effect is chosen from the SAME vocabulary the backend plans with
 * (REPLACE_BACKGROUND, REMOVE_OBJECT, …) and, when a mask exists, the
 * animation happens exactly where the edit will happen — so the overlay is
 * information, not decoration: it tells you what is being changed and WHERE.
 *
 * Engineering constraints, in the spirit of the rest of this codebase:
 *  - Everything is one composited canvas; no per-frame DOM or layout work.
 *  - The loop stops when the tab is hidden or the element is display:none
 *    (offsetParent null) — a hidden mode must not burn the GPU the render
 *    needs.
 *  - prefers-reduced-motion renders ONE static frame: the caption still says
 *    what is happening, nothing moves.
 *  - The device-pixel ratio is capped: this overlay runs at exactly the
 *    moment the machine is busiest.
 */

export type FxEffect =
  | "analyze"
  | "plan"
  | "mask"
  | "background"
  | "remove"
  | "add"
  | "attribute"
  | "style"
  | "relight"
  | "camera"
  | "compose"
  | "faceswap"
  | "outpaint"
  | "upscale"
  | "animate"
  | "pose"
  | "scene3d"
  | "generate"
  | "inspect"
  | "save"
  | "retry"
  | "motion"
  | "orbit"
  | "generic";

/* ---- palette (mirrors styles.css tokens; canvas cannot read CSS vars
        cheaply per-frame, so the values are duplicated here on purpose) ---- */
const AMBER = "232, 163, 61";
const RUBY = "224, 68, 44";
const COOL = "122, 162, 255";
const INK = "22, 24, 29";

/* =========================================================================
   Effect selection — one place that speaks the backend's vocabulary.
   ========================================================================= */

/** Atomic operation (the planner's word) → effect. */
const OPERATION_FX: Record<string, FxEffect> = {
  REPLACE_BACKGROUND: "background",
  REMOVE_OBJECT: "remove",
  ADD_OBJECT: "add",
  REPLACE_OBJECT: "add",
  CHANGE_ATTRIBUTE: "attribute",
  CHANGE_TEXT: "attribute",
  RESTORE: "attribute",
  CHANGE_STYLE: "style",
  CHANGE_LIGHTING: "relight",
  CHANGE_CAMERA: "camera",
  MULTI_VIEW: "camera",
  COMPOSE: "compose",
  SWAP_FACE: "faceswap",
  OUTPAINT: "outpaint",
  UPSCALE: "upscale",
  ANIMATE: "animate",
  CHANGE_POSE: "pose",
  SCENE_3D: "scene3d",
};

/** Engine task (forge tab, background router) → effect. */
const TASK_FX: Record<string, FxEffect> = {
  generate: "generate",
  img2img: "style",
  inpaint: "attribute",
  outpaint: "outpaint",
  upscale: "upscale",
  video: "animate",
  relight: "relight",
  angles: "camera",
  compose: "compose",
  faceswap: "faceswap",
  background: "background",
  pose: "pose",
  scene3d: "scene3d",
};

const OPERATION_LABEL: Record<string, string> = {
  REPLACE_BACKGROUND: "Replacing the background",
  REMOVE_OBJECT: "Removing",
  ADD_OBJECT: "Adding",
  REPLACE_OBJECT: "Swapping",
  CHANGE_ATTRIBUTE: "Restyling",
  CHANGE_TEXT: "Rewriting the text",
  RESTORE: "Restoring the photo",
  CHANGE_STYLE: "Repainting the whole image",
  CHANGE_LIGHTING: "Moving the light",
  CHANGE_CAMERA: "Moving the camera",
  MULTI_VIEW: "Orbiting the subject",
  COMPOSE: "Blending in the second photo",
  SWAP_FACE: "Swapping the face",
  OUTPAINT: "Extending the canvas",
  UPSCALE: "Adding resolution",
  ANIMATE: "Bringing it to motion",
  CHANGE_POSE: "Moving the body",
  SCENE_3D: "Building the 3D scene",
};

/** Stages that mean the same thing whatever the operation is. The render
    stage falls through to the operation's own effect. */
const STAGE_FX: Record<string, { effect: FxEffect; label: string }> = {
  analyze: { effect: "analyze", label: "Reading the request" },
  understand: { effect: "analyze", label: "Analyzing the scene" },
  research: { effect: "analyze", label: "Researching the subject" },
  plan: { effect: "plan", label: "Planning the steps" },
  models: { effect: "plan", label: "Fetching models" },
  install: { effect: "plan", label: "Installing components" },
  prepare: { effect: "plan", label: "Preparing memory" },
  hardware: { effect: "plan", label: "Measuring the hardware" },
  mask: { effect: "mask", label: "Finding the region" },
  segment: { effect: "mask", label: "Isolating the subject" },
  inspect: { effect: "inspect", label: "Inspecting the seams" },
  score: { effect: "inspect", label: "Grading the result" },
  check: { effect: "inspect", label: "Judging realism" },
  verify: { effect: "inspect", label: "Verifying the result" },
  save: { effect: "save", label: "Saving" },
  retry: { effect: "retry", label: "Changing strategy" },
  face: { effect: "faceswap", label: "Restoring the face" },
  coverage: { effect: "orbit", label: "Checking angle coverage" },
  angles: { effect: "orbit", label: "Synthesizing missing angles" },
  animate: { effect: "animate", label: "Animating" },
  consent: { effect: "analyze", label: "Checking consent" },
};

export interface FxState {
  effect: FxEffect;
  label: string;
}

/** Last "[stage] key — …" marker in the job log, with the "step i/n" counter
    when the stage line carries one. */
export function currentStage(job: Job | null): {
  stage: string | null;
  step: number | null;
} {
  if (!job) return { stage: null, step: null };
  for (let i = job.logs.length - 1; i >= 0; i--) {
    const m = /^\[stage\] (\w+)(?: — step (\d+)\/\d+)?/.exec(job.logs[i].msg);
    if (m) return { stage: m[1], step: m[2] ? Number(m[2]) : null };
  }
  return { stage: null, step: null };
}

/**
 * Pick effect + caption from what is known right now. Preference order:
 * a meaningful stage (analyzing, masking, inspecting…) wins; during `render`
 * the operation says what is being done to the pixels; the task is the
 * fallback vocabulary when no per-step operation is known.
 */
export function pickFx(opts: {
  stage?: string | null;
  operation?: string | null;
  target?: string | null;
  task?: string | null;
  /** what "rendering" means for this workflow (motion, orbit, generate…) */
  fallback?: FxState;
}): FxState {
  const { stage, operation, target, task } = opts;
  if (stage && STAGE_FX[stage]) return STAGE_FX[stage];
  const op = operation?.toUpperCase() ?? null;
  if (op && OPERATION_FX[op]) {
    let label = OPERATION_LABEL[op] ?? "Rendering";
    // "Removing" + target → "Removing the chair" — the caption names the
    // thing so the animation and the words point at the same object.
    if (target && ["REMOVE_OBJECT", "ADD_OBJECT", "REPLACE_OBJECT", "CHANGE_ATTRIBUTE"].includes(op)) {
      label = `${label} ${/^(the|a|an)\s/i.test(target) ? "" : "the "}${target}`;
    }
    return { effect: OPERATION_FX[op], label };
  }
  if (task && TASK_FX[task]) {
    const labels: Record<string, string> = {
      generate: "Forging from noise",
      img2img: "Reimagining the image",
      inpaint: "Repainting the region",
      outpaint: "Extending the canvas",
      upscale: "Adding resolution",
      video: "Bringing it to motion",
    };
    return { effect: TASK_FX[task], label: labels[task] ?? "Rendering" };
  }
  return opts.fallback ?? { effect: "generic", label: "Rendering" };
}

/* =========================================================================
   Canvas machinery
   ========================================================================= */

type Part = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  max: number;
  seed: number;
};

interface MaskInfo {
  /** white-with-alpha region canvas, low-res; scaled when composited */
  alpha: HTMLCanvasElement;
  /** ~2px edge ring of the region, same space as `alpha` */
  ring: HTMLCanvasElement;
  /** normalized (0..1) points inside the region, for particle emission */
  pts: { x: number; y: number }[];
  cx: number;
  cy: number;
}

interface Env {
  rand: () => number;
  parts: Part[];
  mask: MaskInfo | null;
  eff: HTMLCanvasElement;
}

/** Deterministic rand — effects look the same every run, and nothing here
    may call Math.random in a hot loop anyway. */
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Load a white-on-black (or white-on-transparent) mask into region data. */
function loadMask(url: string, onReady: (m: MaskInfo) => void) {
  const img = new Image();
  img.onload = () => {
    const W = 224;
    const H = Math.max(8, Math.round((img.naturalHeight / img.naturalWidth) * W) || W);
    const alpha = document.createElement("canvas");
    alpha.width = W;
    alpha.height = H;
    const actx = alpha.getContext("2d");
    if (!actx) return;
    actx.drawImage(img, 0, 0, W, H);
    const im = actx.getImageData(0, 0, W, H);
    const d = im.data;
    const pts: { x: number; y: number }[] = [];
    let sx = 0;
    let sy = 0;
    const rnd = mulberry(7);
    for (let i = 0; i < d.length; i += 4) {
      const on = d[i] > 127 && d[i + 3] > 127;
      d[i] = d[i + 1] = d[i + 2] = 255;
      d[i + 3] = on ? 255 : 0;
      if (on) {
        const px = (i / 4) % W;
        const py = Math.floor(i / 4 / W);
        sx += px;
        sy += py;
        // subsample: ~1 in 6 region pixels becomes an emission point
        if (rnd() < 0.16) pts.push({ x: px / W, y: py / H });
      }
    }
    if (pts.length === 0) return; // empty mask — caller keeps the fallback
    actx.putImageData(im, 0, 0);
    // Edge ring: dilate the region by drawing it 8 times slightly offset,
    // then punch the original back out. Cheap, and reads as an outline.
    const ring = document.createElement("canvas");
    ring.width = W;
    ring.height = H;
    const rctx = ring.getContext("2d");
    if (!rctx) return;
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2;
      rctx.drawImage(alpha, Math.cos(ang) * 2, Math.sin(ang) * 2);
    }
    rctx.globalCompositeOperation = "destination-out";
    rctx.drawImage(alpha, 0, 0);
    const n = pts.length * 6; // undo the subsample for the centroid
    onReady({ alpha, ring, pts, cx: sx / n / W, cy: sy / n / H });
  };
  img.src = url;
}

/** Draw `paint` into the scratch canvas, keep only what falls inside the
    mask region, then stamp it onto the main canvas. */
function clipToRegion(
  ctx: CanvasRenderingContext2D,
  env: Env,
  w: number,
  h: number,
  paint: (ectx: CanvasRenderingContext2D) => void,
) {
  const eff = env.eff;
  if (eff.width !== ctx.canvas.width || eff.height !== ctx.canvas.height) {
    eff.width = ctx.canvas.width;
    eff.height = ctx.canvas.height;
  }
  const ectx = eff.getContext("2d");
  if (!ectx) return;
  ectx.setTransform(ctx.getTransform());
  ectx.clearRect(0, 0, w, h);
  paint(ectx);
  if (env.mask) {
    ectx.globalCompositeOperation = "destination-in";
    ectx.drawImage(env.mask.alpha, 0, 0, w, h);
    ectx.globalCompositeOperation = "source-over";
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(eff, 0, 0);
  ctx.restore();
}

/** A point inside the edit region (or a sensible default without a mask). */
function regionPoint(env: Env, w: number, h: number): { x: number; y: number } {
  if (env.mask && env.mask.pts.length) {
    const p = env.mask.pts[Math.floor(env.rand() * env.mask.pts.length)];
    return { x: p.x * w, y: p.y * h };
  }
  return { x: env.rand() * w, y: env.rand() * h };
}

function regionCenter(env: Env, w: number, h: number) {
  return env.mask
    ? { x: env.mask.cx * w, y: env.mask.cy * h }
    : { x: w / 2, y: h / 2 };
}

/* ---- shared flourishes ---- */

function corners(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, color: string) {
  const m = 10;
  const len = 16 + Math.sin(t * 2.2) * 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  for (const [cx, cy, dx, dy] of [
    [m, m, 1, 1],
    [w - m, m, -1, 1],
    [m, h - m, 1, -1],
    [w - m, h - m, -1, -1],
  ] as const) {
    ctx.beginPath();
    ctx.moveTo(cx + dx * len, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * len);
    ctx.stroke();
  }
}

function regionRing(ctx: CanvasRenderingContext2D, env: Env, w: number, h: number, alpha: number, color: string) {
  if (!env.mask) return;
  const eff = env.eff;
  const ectx = eff.getContext("2d");
  if (!ectx) return;
  if (eff.width !== ctx.canvas.width || eff.height !== ctx.canvas.height) {
    eff.width = ctx.canvas.width;
    eff.height = ctx.canvas.height;
  }
  ectx.setTransform(ctx.getTransform());
  ectx.clearRect(0, 0, w, h);
  ectx.drawImage(env.mask.ring, 0, 0, w, h);
  ectx.globalCompositeOperation = "source-in";
  ectx.fillStyle = color;
  ectx.fillRect(0, 0, w, h);
  ectx.globalCompositeOperation = "source-over";
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = alpha;
  ctx.drawImage(eff, 0, 0);
  ctx.restore();
  ctx.globalAlpha = 1;
}

/* =========================================================================
   The effects. Each draws one frame: (ctx, w, h, t seconds, env).
   ========================================================================= */

type EffectFn = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number, env: Env) => void;

const fxAnalyze: EffectFn = (ctx, w, h, t) => {
  corners(ctx, w, h, t, `rgba(${AMBER}, 0.85)`);
  // scanline sweeping down, with a faint measurement grid behind it
  const y = ((t * 0.24) % 1.25) * h - h * 0.1;
  const g = ctx.createLinearGradient(0, y - 46, 0, y + 8);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(${AMBER}, 0.16)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, y - 46, w, 54);
  ctx.fillStyle = `rgba(${AMBER}, 0.75)`;
  ctx.fillRect(0, y, w, 1.3);
  ctx.strokeStyle = `rgba(${AMBER}, 0.12)`;
  ctx.lineWidth = 1;
  for (let gy = 0; gy < h; gy += 36) {
    if (Math.abs(gy - y) < 70) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }
  }
  // ticks riding the line
  ctx.fillStyle = `rgba(${AMBER}, 0.9)`;
  for (let i = 0; i < 5; i++) {
    const tx = ((i * 0.23 + 0.08 + Math.sin(t * 0.7 + i * 9) * 0.02) % 1) * w;
    ctx.fillRect(tx, y - 4, 1.4, 8);
  }
};

const fxPlan: EffectFn = (ctx, w, h, t) => {
  // blueprint: seeded nodes joined by lines that draw themselves in
  const r = mulberry(11);
  const nodes = Array.from({ length: 6 }, () => ({
    x: (0.14 + r() * 0.72) * w,
    y: (0.16 + r() * 0.68) * h,
  }));
  ctx.strokeStyle = `rgba(${COOL}, 0.4)`;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([5, 7]);
  ctx.lineDashOffset = -t * 26;
  ctx.beginPath();
  nodes.forEach((n, i) => (i ? ctx.lineTo(n.x, n.y) : ctx.moveTo(n.x, n.y)));
  ctx.stroke();
  ctx.setLineDash([]);
  nodes.forEach((n, i) => {
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.4 - i * 0.9);
    ctx.fillStyle = `rgba(${COOL}, ${0.35 + pulse * 0.5})`;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 2.5 + pulse * 1.5, 0, Math.PI * 2);
    ctx.fill();
  });
  corners(ctx, w, h, t, `rgba(${COOL}, 0.45)`);
};

const fxMask: EffectFn = (ctx, w, h, t, env) => {
  if (env.mask) {
    // region breathes in rubylith while marching stripes ride its edge
    clipToRegion(ctx, env, w, h, (e) => {
      e.fillStyle = `rgba(${RUBY}, ${0.1 + 0.07 * Math.sin(t * 2.6)})`;
      e.fillRect(0, 0, w, h);
      // diagonal stripes scrolling — the "ants"
      e.strokeStyle = `rgba(${RUBY}, 0.2)`;
      e.lineWidth = 1;
      const off = (t * 22) % 14;
      for (let x = -h; x < w + h; x += 14) {
        e.beginPath();
        e.moveTo(x + off, 0);
        e.lineTo(x + off + h, h);
        e.stroke();
      }
    });
    regionRing(ctx, env, w, h, 0.65 + 0.3 * Math.sin(t * 3), `rgba(${RUBY}, 1)`);
  } else {
    // no region yet: a crosshair searching for one
    const cx = w / 2 + Math.sin(t * 0.9) * w * 0.22;
    const cy = h / 2 + Math.cos(t * 0.7) * h * 0.18;
    ctx.strokeStyle = `rgba(${RUBY}, 0.7)`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, 26, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - 38, cy);
    ctx.lineTo(cx - 14, cy);
    ctx.moveTo(cx + 14, cy);
    ctx.lineTo(cx + 38, cy);
    ctx.moveTo(cx, cy - 38);
    ctx.lineTo(cx, cy - 14);
    ctx.moveTo(cx, cy + 14);
    ctx.lineTo(cx, cy + 38);
    ctx.stroke();
  }
};

const fxBackground: EffectFn = (ctx, w, h, t, env) => {
  // The old background lifts away as drifting motes while a wash of new
  // colour sweeps through the region behind the subject.
  if (env.parts.length === 0) {
    for (let i = 0; i < 110; i++) {
      const p = env.mask
        ? regionPoint(env, w, h)
        : { x: env.rand() * w, y: env.rand() * h };
      env.parts.push({
        x: p.x,
        y: p.y,
        vx: (env.rand() - 0.5) * 8,
        vy: -14 - env.rand() * 26,
        life: env.rand() * 3,
        max: 2.4 + env.rand() * 2.2,
        seed: env.rand(),
      });
    }
  }
  clipToRegion(ctx, env, w, h, (e) => {
    // the wash: a wide gradient band gliding across, hue easing amber→cool
    const mix = 0.5 + 0.5 * Math.sin(t * 0.5);
    const cA = `rgba(${AMBER}, ${0.10 + mix * 0.05})`;
    const cB = `rgba(${COOL}, ${0.06 + (1 - mix) * 0.06})`;
    const x = ((t * 0.16) % 1.4) * (w * 1.6) - w * 0.5;
    const g = e.createLinearGradient(x - w * 0.4, 0, x + w * 0.4, h * 0.2);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(0.45, cA);
    g.addColorStop(0.55, cB);
    g.addColorStop(1, "rgba(0,0,0,0)");
    e.fillStyle = g;
    e.fillRect(0, 0, w, h);
    // the lift: old-background motes rising out of the region
    for (const p of env.parts) {
      p.life += 1 / 60;
      if (p.life > p.max) {
        const np = regionPoint(env, w, h);
        p.x = np.x;
        p.y = np.y;
        p.life = 0;
      }
      const k = p.life / p.max;
      const px = p.x + p.vx * p.life + Math.sin(t * 1.8 + p.seed * 9) * 5;
      const py = p.y + p.vy * p.life;
      const a = Math.sin(Math.PI * k) * 0.55;
      e.fillStyle = p.seed > 0.72 ? `rgba(${COOL}, ${a})` : `rgba(${AMBER}, ${a})`;
      const s = 1 + p.seed * 2.2;
      e.fillRect(px - s / 2, py - s / 2, s, s);
    }
  });
  regionRing(ctx, env, w, h, 0.35 + 0.2 * Math.sin(t * 2.1), `rgba(${AMBER}, 1)`);
};

const fxRemove: EffectFn = (ctx, w, h, t, env) => {
  // the region breaks apart: fragments accelerate away from its centre
  const c = regionCenter(env, w, h);
  if (env.parts.length === 0) {
    for (let i = 0; i < 90; i++) {
      const p = regionPoint(env, w, h);
      const dx = p.x - c.x;
      const dy = p.y - c.y;
      const d = Math.hypot(dx, dy) || 1;
      env.parts.push({
        x: p.x,
        y: p.y,
        vx: (dx / d) * (26 + env.rand() * 40),
        vy: (dy / d) * (26 + env.rand() * 40) - 8,
        life: env.rand() * 2,
        max: 1.6 + env.rand() * 1.6,
        seed: env.rand(),
      });
    }
  }
  clipToRegion(ctx, env, w, h, (e) => {
    e.fillStyle = `rgba(${INK}, ${0.16 + 0.1 * Math.sin(t * 2.2)})`;
    e.fillRect(0, 0, w, h);
  });
  for (const p of env.parts) {
    p.life += 1 / 60;
    if (p.life > p.max) {
      const np = regionPoint(env, w, h);
      p.x = np.x;
      p.y = np.y;
      p.life = 0;
    }
    const k = p.life / p.max;
    const px = p.x + p.vx * p.life;
    const py = p.y + p.vy * p.life + 14 * p.life * p.life;
    const a = (1 - k) * 0.6;
    ctx.fillStyle = p.seed > 0.6 ? `rgba(${RUBY}, ${a})` : `rgba(${AMBER}, ${a * 0.8})`;
    const s = (1 - k) * (1.6 + p.seed * 2.4);
    ctx.fillRect(px - s / 2, py - s / 2, s, s);
  }
  regionRing(ctx, env, w, h, 0.3 + 0.2 * Math.sin(t * 2.8), `rgba(${RUBY}, 1)`);
};

const fxAdd: EffectFn = (ctx, w, h, t, env) => {
  // reverse of remove: material streams IN from the edges and assembles
  const c = regionCenter(env, w, h);
  if (env.parts.length === 0) {
    for (let i = 0; i < 90; i++) {
      const edge = env.rand();
      const sx = edge < 0.25 ? 0 : edge < 0.5 ? w : env.rand() * w;
      const sy = edge < 0.5 ? env.rand() * h : edge < 0.75 ? 0 : h;
      env.parts.push({
        x: sx,
        y: sy,
        vx: 0,
        vy: 0,
        life: env.rand() * 2,
        max: 1.4 + env.rand() * 1.4,
        seed: env.rand(),
      });
    }
  }
  for (const p of env.parts) {
    p.life += 1 / 60;
    if (p.life > p.max) {
      const edge = env.rand();
      p.x = edge < 0.25 ? 0 : edge < 0.5 ? w : env.rand() * w;
      p.y = edge < 0.5 ? env.rand() * h : edge < 0.75 ? 0 : h;
      p.life = 0;
      p.seed = env.rand();
    }
    // seed picks a STABLE target point — resampling each frame would make
    // the destination (and so the whole stream) jitter
    const mk = env.mask;
    const tgt = mk
      ? { x: mk.pts[Math.floor(p.seed * mk.pts.length)].x * w,
          y: mk.pts[Math.floor(p.seed * mk.pts.length)].y * h }
      : { x: c.x + (p.seed - 0.5) * 60, y: c.y + ((p.seed * 13 % 1) - 0.5) * 60 };
    const k = p.life / p.max;
    const ease = k * k * (3 - 2 * k);
    const px = p.x + (tgt.x - p.x) * ease;
    const py = p.y + (tgt.y - p.y) * ease;
    const a = Math.sin(Math.PI * k) * 0.65;
    ctx.fillStyle = p.seed > 0.7 ? `rgba(255,255,255, ${a * 0.7})` : `rgba(${AMBER}, ${a})`;
    const s = 1 + (1 - k) * 2;
    ctx.fillRect(px - s / 2, py - s / 2, s, s);
  }
  clipToRegion(ctx, env, w, h, (e) => {
    e.fillStyle = `rgba(${AMBER}, ${0.07 + 0.05 * Math.sin(t * 2.4)})`;
    e.fillRect(0, 0, w, h);
  });
  regionRing(ctx, env, w, h, 0.35 + 0.25 * Math.sin(t * 2.4), `rgba(${AMBER}, 1)`);
};

const fxAttribute: EffectFn = (ctx, w, h, t, env) => {
  // glowing brush strokes being laid down inside the region
  clipToRegion(ctx, env, w, h, (e) => {
    const r = mulberry(23);
    for (let i = 0; i < 7; i++) {
      const phase = (t * 0.55 + i * 0.31) % 1;
      const p0 = env.mask
        ? env.mask.pts[Math.floor(r() * env.mask.pts.length)]
        : { x: 0.2 + r() * 0.6, y: 0.2 + r() * 0.6 };
      const x0 = p0.x * w;
      const y0 = p0.y * h;
      const len = 30 + r() * 50;
      const ang = r() * Math.PI * 2;
      const cxq = x0 + Math.cos(ang) * len * 0.5 + (r() - 0.5) * 24;
      const cyq = y0 + Math.sin(ang) * len * 0.5 + (r() - 0.5) * 24;
      const x1 = x0 + Math.cos(ang) * len;
      const y1 = y0 + Math.sin(ang) * len;
      const a = Math.sin(Math.PI * phase) * 0.55;
      e.strokeStyle = i % 3 === 0 ? `rgba(${COOL}, ${a})` : `rgba(${AMBER}, ${a})`;
      e.lineWidth = 2.5 + Math.sin(phase * Math.PI) * 2;
      e.lineCap = "round";
      // draw the stroke up to its current progress
      e.setLineDash([len * 1.2, len * 1.2]);
      e.lineDashOffset = len * 1.2 * (1 - phase);
      e.beginPath();
      e.moveTo(x0, y0);
      e.quadraticCurveTo(cxq, cyq, x1, y1);
      e.stroke();
      e.setLineDash([]);
    }
  });
  regionRing(ctx, env, w, h, 0.3 + 0.18 * Math.sin(t * 2.2), `rgba(${AMBER}, 1)`);
};

const fxStyle: EffectFn = (ctx, w, h, t) => {
  // whole-image repaint: an angled wave front crosses the frame, colour
  // shifting in its wake
  const k = (t * 0.22) % 1.35;
  const x = k * (w + h) - h * 0.6;
  ctx.save();
  ctx.translate(x, 0);
  ctx.rotate(-0.32);
  const g = ctx.createLinearGradient(-90, 0, 60, 0);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(0.6, `rgba(${AMBER}, 0.14)`);
  g.addColorStop(0.85, `rgba(${COOL}, 0.18)`);
  g.addColorStop(1, "rgba(255,255,255,0.10)");
  ctx.fillStyle = g;
  ctx.fillRect(-90, -h, 150, h * 3);
  ctx.restore();
  // sparkles along the front
  const r = mulberry(31);
  for (let i = 0; i < 14; i++) {
    const sy = r() * h;
    const sx = x - sy * Math.tan(0.32) + (r() - 0.5) * 30;
    const tw = 0.5 + 0.5 * Math.sin(t * 5 + i * 7);
    if (sx > 0 && sx < w) {
      ctx.fillStyle = `rgba(255, 255, 255, ${0.25 * tw})`;
      ctx.fillRect(sx, sy, 1.6, 1.6);
    }
  }
};

const fxRelight: EffectFn = (ctx, w, h, t) => {
  // a light source orbiting the frame; the glow and the opposite shadow move
  const ang = t * 0.7;
  const lx = w / 2 + Math.cos(ang) * w * 0.42;
  const ly = h / 2 + Math.sin(ang) * h * 0.36;
  const g = ctx.createRadialGradient(lx, ly, 4, lx, ly, Math.max(w, h) * 0.55);
  g.addColorStop(0, `rgba(255, 236, 200, 0.30)`);
  g.addColorStop(0.25, `rgba(${AMBER}, 0.12)`);
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  // shadow gathers on the far side
  const sg = ctx.createRadialGradient(w - lx, h - ly, 10, w - lx, h - ly, Math.max(w, h) * 0.6);
  sg.addColorStop(0, "rgba(0,0,0,0.20)");
  sg.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = sg;
  ctx.fillRect(0, 0, w, h);
  // the source itself
  ctx.fillStyle = "rgba(255, 244, 214, 0.9)";
  ctx.beginPath();
  ctx.arc(lx, ly, 3.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = `rgba(${AMBER}, 0.5)`;
  ctx.lineWidth = 1;
  for (let i = 0; i < 8; i++) {
    const ra = (i / 8) * Math.PI * 2 + t * 0.4;
    ctx.beginPath();
    ctx.moveTo(lx + Math.cos(ra) * 7, ly + Math.sin(ra) * 7);
    ctx.lineTo(lx + Math.cos(ra) * (11 + Math.sin(t * 3 + i) * 2), ly + Math.sin(ra) * (11 + Math.sin(t * 3 + i) * 2));
    ctx.stroke();
  }
};

const fxCamera: EffectFn = (ctx, w, h, t) => {
  // a photo-card swinging in perspective: the camera is moving around it
  const cx = w / 2;
  const cy = h / 2;
  const yaw = Math.sin(t * 0.8) * 0.9;
  const cw = Math.min(w, h) * 0.42;
  const ch = cw * 0.72;
  const persp = (x: number, y: number) => {
    const rx = x * Math.cos(yaw);
    const z = 1 + x * Math.sin(yaw) * 0.0016;
    return { x: cx + rx / z, y: cy + y / z };
  };
  const pts = [
    persp(-cw, -ch),
    persp(cw, -ch),
    persp(cw, ch),
    persp(-cw, ch),
  ];
  ctx.strokeStyle = `rgba(${COOL}, 0.75)`;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.stroke();
  // inner thirds — reads as a viewfinder
  ctx.strokeStyle = `rgba(${COOL}, 0.25)`;
  for (const f of [-1 / 3, 1 / 3]) {
    const a = persp(cw * f, -ch);
    const b = persp(cw * f, ch);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  // orbit path + moving camera dot
  ctx.strokeStyle = `rgba(${COOL}, 0.3)`;
  ctx.setLineDash([3, 6]);
  ctx.beginPath();
  ctx.ellipse(cx, cy + ch * 1.15, cw * 1.25, ch * 0.32, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  const ca = t * 0.8 + Math.PI / 2;
  ctx.fillStyle = `rgba(${COOL}, 0.95)`;
  ctx.beginPath();
  ctx.arc(cx + Math.cos(ca) * cw * 1.25, cy + ch * 1.15 + Math.sin(ca) * ch * 0.32, 3, 0, Math.PI * 2);
  ctx.fill();
};

const fxCompose: EffectFn = (ctx, w, h, t, env) => {
  // a second frame glides in and pours its subject into the region
  const k = 0.5 + 0.5 * Math.sin(t * 0.9 - Math.PI / 2); // 0→1 breathing
  const fw = w * 0.2;
  const fh = fw * 0.75;
  const fx0 = w * 0.06 + k * w * 0.05;
  const fy0 = h * 0.08;
  ctx.strokeStyle = `rgba(${COOL}, 0.8)`;
  ctx.lineWidth = 1.4;
  ctx.strokeRect(fx0, fy0, fw, fh);
  ctx.fillStyle = `rgba(${COOL}, 0.08)`;
  ctx.fillRect(fx0, fy0, fw, fh);
  const c = regionCenter(env, w, h);
  if (env.parts.length === 0) {
    for (let i = 0; i < 46; i++) {
      env.parts.push({
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        life: env.rand() * 1.8,
        max: 1.2 + env.rand() * 1.2,
        seed: env.rand(),
      });
    }
  }
  for (const p of env.parts) {
    p.life += 1 / 60;
    if (p.life > p.max) {
      p.life = 0;
      p.seed = env.rand();
    }
    const kk = p.life / p.max;
    const ease = kk * kk * (3 - 2 * kk);
    const sx = fx0 + fw * (0.2 + p.seed * 0.6);
    const sy = fy0 + fh * (0.2 + (p.seed * 7 % 1) * 0.6);
    const tx = env.mask ? c.x + (p.seed - 0.5) * 70 : c.x + (p.seed - 0.5) * w * 0.3;
    const ty = env.mask ? c.y + ((p.seed * 13 % 1) - 0.5) * 70 : c.y + ((p.seed * 13 % 1) - 0.5) * h * 0.3;
    // arc: control point lifts the path so the stream reads as a pour
    const mx = (sx + tx) / 2;
    const my = Math.min(sy, ty) - 40;
    const ix = (1 - ease) * (1 - ease) * sx + 2 * (1 - ease) * ease * mx + ease * ease * tx;
    const iy = (1 - ease) * (1 - ease) * sy + 2 * (1 - ease) * ease * my + ease * ease * ty;
    const a = Math.sin(Math.PI * kk) * 0.7;
    ctx.fillStyle = p.seed > 0.75 ? `rgba(255,255,255,${a * 0.6})` : `rgba(${COOL}, ${a})`;
    ctx.beginPath();
    ctx.arc(ix, iy, 1 + (1 - kk) * 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
  regionRing(ctx, env, w, h, 0.35 + 0.2 * Math.sin(t * 2.2), `rgba(${COOL}, 1)`);
};

const fxFaceswap: EffectFn = (ctx, w, h, t, env) => {
  // an oval target where the face goes; shimmer crossfades inside it
  const c = env.mask ? regionCenter(env, w, h) : { x: w / 2, y: h * 0.32 };
  const rx = Math.min(w, h) * 0.16;
  const ry = rx * 1.3;
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(c.x, c.y, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();
  const x = ((t * 0.5) % 1.4) * rx * 4 - rx * 2;
  const g = ctx.createLinearGradient(c.x - rx + x - 30, 0, c.x - rx + x + 30, 0);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(0.5, `rgba(${AMBER}, 0.22)`);
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(c.x - rx * 2, c.y - ry * 2, rx * 4, ry * 4);
  ctx.restore();
  ctx.setLineDash([6, 5]);
  ctx.lineDashOffset = -t * 16;
  ctx.strokeStyle = `rgba(${AMBER}, 0.8)`;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.ellipse(c.x, c.y, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  // alignment ticks
  ctx.strokeStyle = `rgba(${AMBER}, 0.5)`;
  for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]] as const) {
    ctx.beginPath();
    ctx.moveTo(c.x + dx * (rx + 4), c.y + dy * (ry + 4));
    ctx.lineTo(c.x + dx * (rx + 12), c.y + dy * (ry + 12));
    ctx.stroke();
  }
};

const fxOutpaint: EffectFn = (ctx, w, h, t) => {
  // the original frame sits inside; new canvas crystallises beyond it
  const inset = 0.16;
  const ix = w * inset;
  const iy = h * inset;
  ctx.setLineDash([7, 6]);
  ctx.strokeStyle = `rgba(${AMBER}, 0.55)`;
  ctx.lineWidth = 1.2;
  ctx.strokeRect(ix, iy, w - ix * 2, h - iy * 2);
  ctx.setLineDash([]);
  // flickering blocks in the margin, densest near the original frame,
  // "growing" outward with the sweep
  const r = mulberry(41);
  const sweep = (t * 0.45) % 1.2;
  for (let i = 0; i < 130; i++) {
    const px = r() * w;
    const py = r() * h;
    const inside = px > ix && px < w - ix && py > iy && py < h - iy;
    if (inside) continue;
    // distance from the inner frame, normalised to the margin width
    const dx = px < ix ? (ix - px) / ix : px > w - ix ? (px - (w - ix)) / ix : 0;
    const dy = py < iy ? (iy - py) / iy : py > h - iy ? (py - (h - iy)) / iy : 0;
    const d = Math.max(dx, dy);
    const on = d < sweep ? 0.5 : 0.12;
    const tw = 0.5 + 0.5 * Math.sin(t * 4 + i * 13);
    ctx.fillStyle = i % 4 === 0 ? `rgba(${COOL}, ${on * tw})` : `rgba(${AMBER}, ${on * tw})`;
    const s = 2 + r() * 3;
    ctx.fillRect(px - s / 2, py - s / 2, s, s);
  }
  // arrows pushing outward at the edge midpoints
  ctx.strokeStyle = `rgba(${AMBER}, ${0.5 + 0.3 * Math.sin(t * 2.6)})`;
  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  const push = 4 + Math.sin(t * 2.6) * 3;
  for (const [ax, ay, dx, dy] of [
    [w / 2, iy * 0.55, 0, -1],
    [w / 2, h - iy * 0.55, 0, 1],
    [ix * 0.55, h / 2, -1, 0],
    [w - ix * 0.55, h / 2, 1, 0],
  ] as const) {
    const tipx = ax + dx * push;
    const tipy = ay + dy * push;
    ctx.beginPath();
    ctx.moveTo(tipx - dx * 10 - dy * 5, tipy - dy * 10 - dx * 5);
    ctx.lineTo(tipx, tipy);
    ctx.lineTo(tipx - dx * 10 + dy * 5, tipy - dy * 10 + dx * 5);
    ctx.stroke();
  }
};

const fxUpscale: EffectFn = (ctx, w, h, t) => {
  // a refine front crosses the image: coarse pixels ahead, fine grid behind
  const x = ((t * 0.28) % 1.25) * (w * 1.2) - w * 0.1;
  ctx.strokeStyle = `rgba(${COOL}, 0.10)`;
  ctx.lineWidth = 1;
  const coarse = 34;
  for (let gx = 0; gx < w; gx += coarse) {
    if (gx > x) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
      ctx.stroke();
    }
  }
  for (let gy = 0; gy < h; gy += coarse) {
    ctx.beginPath();
    ctx.moveTo(Math.max(x, 0), gy);
    ctx.lineTo(w, gy);
    ctx.stroke();
  }
  ctx.strokeStyle = `rgba(${COOL}, 0.14)`;
  const fine = coarse / 4;
  for (let gx = 0; gx < Math.min(x, w); gx += fine) {
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, h);
    ctx.stroke();
  }
  for (let gy = 0; gy < h; gy += fine) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(Math.min(x, w), gy);
    ctx.stroke();
  }
  // the front itself + sparkles where detail is being minted
  const g = ctx.createLinearGradient(x - 40, 0, x + 4, 0);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(${COOL}, 0.25)`);
  ctx.fillStyle = g;
  ctx.fillRect(x - 40, 0, 44, h);
  ctx.fillStyle = `rgba(${COOL}, 0.9)`;
  ctx.fillRect(x, 0, 1.4, h);
  const r = mulberry(53);
  for (let i = 0; i < 10; i++) {
    const sy = r() * h;
    const tw = 0.5 + 0.5 * Math.sin(t * 6 + i * 5);
    ctx.fillStyle = `rgba(255,255,255,${0.5 * tw})`;
    ctx.fillRect(x - r() * 26, sy, 1.6, 1.6);
  }
};

const fxAnimate: EffectFn = (ctx, w, h, t) => {
  // a film strip ticking along the bottom; motion streaks cross the frame
  const sh = Math.max(26, h * 0.09);
  const y0 = h - sh - 8;
  const cw = sh * 1.5;
  const off = (t * 40) % (cw + 6);
  ctx.fillStyle = "rgba(10, 12, 16, 0.55)";
  ctx.fillRect(0, y0 - 4, w, sh + 12);
  for (let x = -off; x < w; x += cw + 6) {
    ctx.strokeStyle = `rgba(${AMBER}, 0.55)`;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(x, y0, cw, sh);
    ctx.fillStyle = `rgba(${AMBER}, 0.25)`;
    for (let px = x + 3; px < x + cw - 3; px += 7) {
      ctx.fillRect(px, y0 - 3, 3, 2);
      ctx.fillRect(px, y0 + sh + 1, 3, 2);
    }
  }
  // streaks
  const r = mulberry(61);
  for (let i = 0; i < 5; i++) {
    const sy = (0.15 + r() * 0.5) * h;
    const len = 60 + r() * 90;
    const sx = ((t * (70 + r() * 60) + r() * w) % (w + len)) - len;
    const g = ctx.createLinearGradient(sx, 0, sx + len, 0);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, `rgba(${COOL}, 0.35)`);
    ctx.fillStyle = g;
    ctx.fillRect(sx, sy, len, 1.6);
  }
  // play glyph breathing at centre
  const a = 0.25 + 0.15 * Math.sin(t * 2.2);
  ctx.fillStyle = `rgba(255,255,255,${a})`;
  const s = Math.min(w, h) * 0.06;
  ctx.beginPath();
  ctx.moveTo(w / 2 - s * 0.5, h / 2 - s * 0.8);
  ctx.lineTo(w / 2 + s, h / 2);
  ctx.lineTo(w / 2 - s * 0.5, h / 2 + s * 0.8);
  ctx.closePath();
  ctx.fill();
};

/** Walk-cycle stick figure — shared by pose and motion-transfer. */
function skeleton(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number, t: number, color: string) {
  const swing = Math.sin(t * 3.4);
  const swing2 = Math.sin(t * 3.4 + Math.PI);
  const bob = Math.abs(Math.sin(t * 3.4)) * s * 0.04;
  const hip = { x: cx, y: cy + s * 0.05 - bob };
  const neck = { x: cx, y: hip.y - s * 0.5 };
  const head = { x: cx, y: neck.y - s * 0.17 };
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.lineCap = "round";
  const seg = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  };
  seg(hip, neck);
  ctx.beginPath();
  ctx.arc(head.x, head.y, s * 0.12, 0, Math.PI * 2);
  ctx.stroke();
  // arms
  const sh = { x: neck.x, y: neck.y + s * 0.06 };
  for (const [dir, sw] of [[1, swing], [-1, swing2]] as const) {
    const elbow = { x: sh.x + dir * s * 0.16 + sw * s * 0.1, y: sh.y + s * 0.2 };
    const hand = { x: elbow.x + sw * s * 0.16, y: elbow.y + s * 0.18 };
    seg(sh, elbow);
    seg(elbow, hand);
  }
  // legs
  for (const [dir, sw] of [[1, swing2], [-1, swing]] as const) {
    const knee = { x: hip.x + dir * s * 0.07 + sw * s * 0.12, y: hip.y + s * 0.26 };
    const foot = { x: knee.x + sw * s * 0.14, y: knee.y + s * 0.26 };
    seg(hip, knee);
    seg(knee, foot);
  }
  // joints
  ctx.fillStyle = color;
  for (const j of [hip, neck, sh]) {
    ctx.beginPath();
    ctx.arc(j.x, j.y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

const fxPose: EffectFn = (ctx, w, h, t, env) => {
  const c = regionCenter(env, w, h);
  const s = Math.min(w, h) * 0.5;
  // ghost of the previous pose, then the live one
  ctx.globalAlpha = 0.3;
  skeleton(ctx, c.x, c.y, s, t - 0.16, `rgba(${AMBER}, 0.5)`);
  ctx.globalAlpha = 1;
  skeleton(ctx, c.x, c.y, s, t, `rgba(${AMBER}, 0.85)`);
  regionRing(ctx, env, w, h, 0.25 + 0.15 * Math.sin(t * 2), `rgba(${AMBER}, 1)`);
};

const fxMotion: EffectFn = (ctx, w, h, t, env) => {
  // the driving motion flowing onto the person: skeleton + streaming trails
  const cx = w * 0.5;
  const cy = h * 0.52;
  const s = Math.min(w, h) * 0.55;
  if (env.parts.length === 0) {
    for (let i = 0; i < 40; i++) {
      env.parts.push({
        x: env.rand() * w,
        y: (0.2 + env.rand() * 0.6) * h,
        vx: 60 + env.rand() * 80,
        vy: 0,
        life: env.rand() * 2,
        max: 1.4 + env.rand() * 1.2,
        seed: env.rand(),
      });
    }
  }
  for (const p of env.parts) {
    p.life += 1 / 60;
    if (p.life > p.max) {
      p.x = -20;
      p.y = (0.2 + env.rand() * 0.6) * h;
      p.life = 0;
      p.seed = env.rand();
    }
    const px = p.x + p.vx * p.life;
    const py = p.y + Math.sin(t * 2 + p.seed * 12) * 8;
    const a = Math.sin(Math.PI * (p.life / p.max)) * 0.4;
    const g = ctx.createLinearGradient(px - 26, 0, px, 0);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, `rgba(${COOL}, ${a})`);
    ctx.fillStyle = g;
    ctx.fillRect(px - 26, py, 26, 1.4);
  }
  ctx.globalAlpha = 0.25;
  skeleton(ctx, cx - s * 0.05, cy, s, t - 0.2, `rgba(${COOL}, 0.6)`);
  ctx.globalAlpha = 1;
  skeleton(ctx, cx, cy, s, t, `rgba(${COOL}, 0.9)`);
};

const fxScene3d: EffectFn = (ctx, w, h, t) => {
  // one-point perspective: the photograph becomes a place with depth
  const vx = w / 2;
  const vy = h * 0.42;
  ctx.strokeStyle = `rgba(${COOL}, 0.30)`;
  ctx.lineWidth = 1;
  for (let i = 0; i < 10; i++) {
    const ang = (i / 10) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    ctx.lineTo(vx + Math.cos(ang) * w, vy + Math.sin(ang) * w);
    ctx.stroke();
  }
  // depth rings racing toward the viewer
  for (let i = 0; i < 5; i++) {
    const k = ((t * 0.32 + i / 5) % 1);
    const rw = k * k * w * 0.75;
    const rh = k * k * h * 0.62;
    ctx.strokeStyle = `rgba(${COOL}, ${0.35 * (1 - k)})`;
    ctx.strokeRect(vx - rw, vy - rh, rw * 2, rh * 2);
  }
  ctx.fillStyle = `rgba(${COOL}, 0.9)`;
  ctx.beginPath();
  ctx.arc(vx, vy, 2.4, 0, Math.PI * 2);
  ctx.fill();
};

const fxGenerate: EffectFn = (ctx, w, h, t) => {
  // the diffusion picture: a field of noise, and a refine pass that leaves
  // smoothness behind it — noise crystallising into an image
  const cell = Math.max(8, Math.round(w / 42));
  const cols = Math.ceil(w / cell);
  const rows = Math.ceil(h / cell);
  const sweepY = ((t * 0.21) % 1.3) * (h * 1.25) - h * 0.1;
  const r = mulberry(71);
  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const base = r(); // stable per cell (deterministic sequence)
      const y = gy * cell;
      const settled = y < sweepY;
      const flick = Math.sin(t * (3 + base * 5) + base * 40);
      const a = settled
        ? 0.05 + base * 0.05 + flick * 0.015
        : 0.10 + base * 0.2 + flick * 0.09;
      const warm = (gx + gy) % 3 === 0;
      ctx.fillStyle = settled
        ? `rgba(${warm ? AMBER : COOL}, ${Math.max(0, a)})`
        : `rgba(255,255,255,${Math.max(0, a) * 0.5})`;
      const pad = settled ? 0 : 1;
      ctx.fillRect(gx * cell + pad, y + pad, cell - pad * 2, cell - pad * 2);
    }
  }
  // the refine line
  const g = ctx.createLinearGradient(0, sweepY - 30, 0, sweepY + 2);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(${AMBER}, 0.30)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, sweepY - 30, w, 32);
  ctx.fillStyle = `rgba(${AMBER}, 0.8)`;
  ctx.fillRect(0, sweepY, w, 1.4);
  // a slow bloom in the middle: the image "arriving"
  const bloom = 0.5 + 0.5 * Math.sin(t * 0.6 - 1);
  const bg = ctx.createRadialGradient(w / 2, h / 2, 8, w / 2, h / 2, Math.min(w, h) * 0.5);
  bg.addColorStop(0, `rgba(${AMBER}, ${0.10 * bloom})`);
  bg.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
};

const fxInspect: EffectFn = (ctx, w, h, t) => {
  // a loupe drifting over the result, ticking off what it has checked
  const lx = w / 2 + Math.sin(t * 0.7) * w * 0.3;
  const ly = h / 2 + Math.sin(t * 1.1 + 1.3) * h * 0.24;
  const rr = Math.min(w, h) * 0.17;
  const g = ctx.createRadialGradient(lx, ly, rr * 0.2, lx, ly, rr);
  g.addColorStop(0, "rgba(255, 255, 255, 0.10)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(lx, ly, rr, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = `rgba(${AMBER}, 0.85)`;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(lx, ly, rr, 0, Math.PI * 2);
  ctx.stroke();
  // graduations
  ctx.strokeStyle = `rgba(${AMBER}, 0.5)`;
  ctx.lineWidth = 1;
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2 + t * 0.5;
    ctx.beginPath();
    ctx.moveTo(lx + Math.cos(a) * (rr - 5), ly + Math.sin(a) * (rr - 5));
    ctx.lineTo(lx + Math.cos(a) * rr, ly + Math.sin(a) * rr);
    ctx.stroke();
  }
  // crosshair
  ctx.beginPath();
  ctx.moveTo(lx - 8, ly);
  ctx.lineTo(lx + 8, ly);
  ctx.moveTo(lx, ly - 8);
  ctx.lineTo(lx, ly + 8);
  ctx.stroke();
  // handle
  ctx.strokeStyle = `rgba(${AMBER}, 0.7)`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(lx + rr * 0.72, ly + rr * 0.72);
  ctx.lineTo(lx + rr * 1.1, ly + rr * 1.1);
  ctx.stroke();
};

const fxSave: EffectFn = (ctx, w, h, t) => {
  // one bright line wipes down and leaves stillness behind it
  const k = Math.min(1, (t % 2.4) / 1.1);
  const y = k * h;
  const g = ctx.createLinearGradient(0, y - 40, 0, y);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(${AMBER}, 0.20)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, y - 40, w, 40);
  ctx.fillStyle = `rgba(${AMBER}, ${0.8 * (1 - k * 0.6)})`;
  ctx.fillRect(0, y, w, 1.4);
  corners(ctx, w, h, t, `rgba(${AMBER}, ${0.5 * (1 - k)})`);
};

const fxRetry: EffectFn = (ctx, w, h, t) => {
  // a strategy turn: an arc rewinding counter-clockwise, amber giving way
  // to ruby and back
  const cx = w / 2;
  const cy = h / 2;
  const rr = Math.min(w, h) * 0.16;
  const mix = 0.5 + 0.5 * Math.sin(t * 2);
  ctx.strokeStyle = `rgba(${mix > 0.5 ? RUBY : AMBER}, 0.8)`;
  ctx.lineWidth = 2.4;
  ctx.lineCap = "round";
  const start = -t * 2.6;
  ctx.beginPath();
  ctx.arc(cx, cy, rr, start, start + Math.PI * 1.4);
  ctx.stroke();
  // arrowhead at the leading end
  const ax = cx + Math.cos(start) * rr;
  const ay = cy + Math.sin(start) * rr;
  const dir = start - Math.PI / 2;
  ctx.beginPath();
  ctx.moveTo(ax + Math.cos(dir + 0.5) * 9, ay + Math.sin(dir + 0.5) * 9);
  ctx.lineTo(ax, ay);
  ctx.lineTo(ax + Math.cos(dir - 0.5) * 9, ay + Math.sin(dir - 0.5) * 9);
  ctx.stroke();
};

const fxOrbit: EffectFn = (ctx, w, h, t) => {
  // building a person you can walk around: a camera circles the subject,
  // lighting up each angle bin as it passes
  const cx = w / 2;
  const cy = h * 0.55;
  const rx = Math.min(w, h) * 0.36;
  const ry = rx * 0.32;
  ctx.strokeStyle = `rgba(${COOL}, 0.35)`;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  const sweep = t * 1.1;
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const px = cx + Math.cos(a) * rx;
    const py = cy + Math.sin(a) * ry;
    // a bin glows when the sweep has most recently passed it
    const d = ((sweep - a) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
    const glow = Math.max(0, 1 - d / 1.6);
    ctx.fillStyle = `rgba(${COOL}, ${0.25 + glow * 0.7})`;
    ctx.beginPath();
    ctx.arc(px, py, 2.4 + glow * 2, 0, Math.PI * 2);
    ctx.fill();
  }
  // the camera itself
  const camx = cx + Math.cos(sweep) * rx;
  const camy = cy + Math.sin(sweep) * ry;
  ctx.fillStyle = `rgba(${COOL}, 0.95)`;
  ctx.fillRect(camx - 4, camy - 3, 8, 6);
  // the lens looks at the subject
  const la = Math.atan2(cy - camy, cx - camx);
  ctx.strokeStyle = `rgba(${COOL}, 0.35)`;
  ctx.beginPath();
  ctx.moveTo(camx + Math.cos(la) * 6, camy + Math.sin(la) * 6);
  ctx.lineTo(camx + Math.cos(la) * 22, camy + Math.sin(la) * 22);
  ctx.stroke();
  // vertical axis through the subject
  ctx.strokeStyle = `rgba(${COOL}, 0.2)`;
  ctx.beginPath();
  ctx.moveTo(cx, cy - h * 0.4);
  ctx.lineTo(cx, cy + ry + 8);
  ctx.stroke();
};

const fxGeneric: EffectFn = (ctx, w, h, t) => {
  // the original render sweep, kept as the fallback for unknown work
  const x = ((t * 0.4) % 1.4) * (w * 1.5) - w * 0.4;
  const g = ctx.createLinearGradient(x - 60, 0, x + 60, 0);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(0.5, `rgba(${COOL}, 0.14)`);
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(x - 60, 0, 120, h);
};

const EFFECTS: Record<FxEffect, EffectFn> = {
  analyze: fxAnalyze,
  plan: fxPlan,
  mask: fxMask,
  background: fxBackground,
  remove: fxRemove,
  add: fxAdd,
  attribute: fxAttribute,
  style: fxStyle,
  relight: fxRelight,
  camera: fxCamera,
  compose: fxCompose,
  faceswap: fxFaceswap,
  outpaint: fxOutpaint,
  upscale: fxUpscale,
  animate: fxAnimate,
  pose: fxPose,
  scene3d: fxScene3d,
  generate: fxGenerate,
  inspect: fxInspect,
  save: fxSave,
  retry: fxRetry,
  motion: fxMotion,
  orbit: fxOrbit,
  generic: fxGeneric,
};

/* =========================================================================
   The component
   ========================================================================= */

interface Props {
  active: boolean;
  effect: FxEffect;
  /** caption under the animation; null hides it */
  label?: string | null;
  /** white-on-black mask URL/data-URL — regionalises the effect */
  maskUrl?: string | null;
  /** paint an opaque backdrop (there is no image under the overlay) */
  standalone?: boolean;
}

export function ProcessFX({ active, effect, label, maskUrl, standalone }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<MaskInfo | null>(null);
  const effectRef = useRef(effect);

  // Particles and the scratch canvas are reset when the effect (or region)
  // changes, so one effect's debris never leaks into the next.
  const envRef = useRef<Env | null>(null);

  useEffect(() => {
    maskRef.current = null;
    if (maskUrl) loadMask(maskUrl, (m) => {
      maskRef.current = m;
      if (envRef.current) envRef.current.mask = m;
      if (envRef.current) envRef.current.parts = [];
    });
  }, [maskUrl]);

  useEffect(() => {
    if (effectRef.current !== effect && envRef.current) {
      envRef.current.parts = [];
    }
    effectRef.current = effect;
  }, [effect]);

  useEffect(() => {
    if (!active) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const env: Env = {
      rand: mulberry(97),
      parts: [],
      mask: maskRef.current,
      eff: document.createElement("canvas"),
    };
    envRef.current = env;

    // Capped DPR: this overlay runs while the GPU is the busiest thing in
    // the machine — crispness loses to not competing with the render.
    const dpr = Math.min(1.5, window.devicePixelRatio || 1);
    const fit = () => {
      const r = canvas.parentElement?.getBoundingClientRect();
      if (!r || r.width === 0) return;
      const bw = Math.round(r.width * dpr);
      const bh = Math.round(r.height * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
    };
    fit();
    const ro = new ResizeObserver(fit);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const t0 = performance.now();
    let raf = 0;
    let drewStatic = false;

    const frame = () => {
      raf = requestAnimationFrame(frame);
      // A hidden mode (display:none) or a hidden tab draws nothing.
      if (document.hidden || canvas.offsetParent === null) return;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      if (w === 0 || h === 0) return;
      // Reduced motion: one legible still, then stop drawing entirely.
      if (reduced && drewStatic) return;
      const t = reduced ? 0.9 : (performance.now() - t0) / 1000;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (standalone) {
        const g = ctx.createLinearGradient(0, 0, 0, h);
        g.addColorStop(0, "#1a1d24");
        g.addColorStop(1, "#14161b");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
      }
      (EFFECTS[effectRef.current] ?? fxGeneric)(ctx, w, h, t, env);
      drewStatic = true;
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      envRef.current = null;
    };
  }, [active, standalone]);

  if (!active) return null;
  return (
    <div className={`fx-layer${standalone ? " standalone" : ""}`}>
      <canvas ref={canvasRef} aria-hidden />
      {label && (
        <div className="fx-caption" role="status" aria-live="polite">
          <span className="fx-caption-dot" aria-hidden />
          {label}
          <span className="fx-caption-ellipsis" aria-hidden />
        </div>
      )}
    </div>
  );
}
