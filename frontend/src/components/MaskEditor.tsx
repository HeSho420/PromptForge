import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ProcessFX, type FxState } from "./ProcessFX";

interface Props {
  imageUrl: string;
  /** Asset id — enables the SAM click-to-select tool when provided. */
  assetId?: string;
  /** data-URL of an auto-proposed mask to load, or null to start blank */
  autoMask: string | null;
  /** URL of a server-refined mask (auto-grown during a retry): loaded as the
      current mask with the newly-added area highlighted, so the preview
      always matches what actually renders. */
  refinedMaskUrl?: string | null;
  /** URL of an intermediate render (multi-step pipelines): fades onto THE
      image so every step is visible in place, like a live automation. */
  previewUrl?: string | null;
  /** Called with a PNG data-URL of the current mask (white = edit region). */
  onMaskChange: (maskB64: string | null) => void;
  /** True while a render job is working on this image — animates the canvas
      so the image visibly "works". */
  rendering?: boolean;
  /** What is being done to the image right now (effect + caption + region);
      drives the operation-specific animation on the canvas. */
  fx?: (FxState & { maskUrl: string | null }) | null;
}

type Tool = "paint" | "erase" | "smart";

/**
 * Paints the edit mask over the image. The overlay renders in rubylith red —
 * painted pixels are the region the backend is allowed to change.
 * Internally the mask canvas is white-on-transparent; export converts it to
 * the white-on-black L-mask the API expects.
 */
const SAM_STAGES_COLD = ["Loading SAM model…", "Preparing segmentation…",
                         "Detecting objects…"];
const SAM_STAGES_WARM = ["Preparing segmentation…", "Detecting objects…"];

export function MaskEditor({
  imageUrl, assetId, autoMask, refinedMaskUrl, previewUrl, onMaskChange,
  rendering, fx,
}: Props) {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null); // offscreen truth
  const viewRef = useRef<HTMLCanvasElement>(null); // composited display
  const highlightRef = useRef<HTMLCanvasElement | null>(null); // added area
  const highlightOn = useRef(false);
  const [tool, setTool] = useState<Tool>("paint");
  const [brush, setBrush] = useState(36);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [selectStage, setSelectStage] = useState<string | null>(null);
  const [selectError, setSelectError] = useState<string | null>(null);
  const drawing = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const appliedRefined = useRef<string | null>(null);
  const appliedPreview = useRef<string | null>(null);
  const [flash, setFlash] = useState(false);

  const redraw = useCallback(() => {
    const base = baseRef.current;
    const mask = maskRef.current;
    const view = viewRef.current;
    if (!base || !mask || !view) return;
    const ctx = view.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, view.width, view.height);
    ctx.drawImage(base, 0, 0);
    // rubylith: tint painted (white) mask pixels red, semi-transparent
    const tint = document.createElement("canvas");
    tint.width = view.width;
    tint.height = view.height;
    const tctx = tint.getContext("2d");
    if (!tctx) return;
    tctx.drawImage(mask, 0, 0);
    tctx.globalCompositeOperation = "source-in";
    tctx.fillStyle = "#e0442c";
    tctx.fillRect(0, 0, tint.width, tint.height);
    ctx.globalAlpha = 0.5;
    ctx.drawImage(tint, 0, 0);
    ctx.globalAlpha = 1;
    // Newly-added mask area (from a server refine): drawn amber on top so
    // the change is immediately visible against the original red region.
    if (highlightOn.current && highlightRef.current) {
      ctx.globalAlpha = 0.75;
      ctx.drawImage(highlightRef.current, 0, 0);
      ctx.globalAlpha = 1;
    }
  }, []);

  const exportMask = useCallback(() => {
    const mask = maskRef.current;
    if (!mask) return;
    // Convert white-on-transparent to white-on-black for the API.
    const out = document.createElement("canvas");
    out.width = mask.width;
    out.height = mask.height;
    const ctx = out.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(mask, 0, 0);
    // Empty masks are reported as null so the caller falls back to auto.
    const data = ctx.getImageData(0, 0, out.width, out.height).data;
    let any = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 8) {
        any = true;
        break;
      }
    }
    onMaskChange(any ? out.toDataURL("image/png") : null);
  }, [onMaskChange]);

  // load base image (and reset mask) whenever the asset changes
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      setSize({ w, h });
      for (const ref of [baseRef, maskRef, viewRef]) {
        if (ref.current) {
          ref.current.width = w;
          ref.current.height = h;
        }
      }
      baseRef.current?.getContext("2d")?.drawImage(img, 0, 0);
      maskRef.current?.getContext("2d")?.clearRect(0, 0, w, h);
      redraw();
      onMaskChange(null);
    };
    img.src = imageUrl;
  }, [imageUrl, redraw, onMaskChange]);

  /** Composite a white-on-black API mask onto the mask canvas.
      replace=true clears first; false unions with what's painted. */
  const applyApiMask = useCallback(
    (dataUrl: string, replace: boolean) => {
      if (!size) return;
      const img = new Image();
      img.onload = () => {
        const mask = maskRef.current;
        const ctx = mask?.getContext("2d");
        if (!mask || !ctx) return;
        const tmp = document.createElement("canvas");
        tmp.width = size.w;
        tmp.height = size.h;
        const tctx = tmp.getContext("2d");
        if (!tctx) return;
        tctx.drawImage(img, 0, 0, size.w, size.h);
        const im = tctx.getImageData(0, 0, size.w, size.h);
        const d = im.data;
        for (let i = 0; i < d.length; i += 4) {
          const v = d[i];
          d[i] = d[i + 1] = d[i + 2] = 255;
          d[i + 3] = v; // luminance becomes alpha
        }
        tctx.putImageData(im, 0, 0);
        if (replace) ctx.clearRect(0, 0, size.w, size.h);
        ctx.drawImage(tmp, 0, 0);
        redraw();
        exportMask();
      };
      img.src = dataUrl;
    },
    [size, redraw, exportMask],
  );

  // load auto-proposed mask when it arrives
  useEffect(() => {
    if (autoMask) applyApiMask(autoMask, true);
  }, [autoMask, applyApiMask]);

  // A server-refined mask arrived (the app auto-grew the mask during a
  // retry): snapshot the old mask, load the refined one as the truth, and
  // highlight exactly the pixels that were added.
  useEffect(() => {
    if (!refinedMaskUrl || !size) return;
    if (appliedRefined.current === refinedMaskUrl) return;
    appliedRefined.current = refinedMaskUrl;
    const old = document.createElement("canvas");
    old.width = size.w;
    old.height = size.h;
    const octx = old.getContext("2d");
    const mask = maskRef.current;
    if (!octx || !mask) return;
    octx.drawImage(mask, 0, 0);
    const img = new Image();
    img.onload = () => {
      const mctx = mask.getContext("2d");
      if (!mctx) return;
      // Refined mask (white-on-black) becomes the new truth…
      const tmp = document.createElement("canvas");
      tmp.width = size.w;
      tmp.height = size.h;
      const tctx = tmp.getContext("2d");
      if (!tctx) return;
      tctx.drawImage(img, 0, 0, size.w, size.h);
      const ref = tctx.getImageData(0, 0, size.w, size.h);
      const oldData = octx.getImageData(0, 0, size.w, size.h);
      const hi = document.createElement("canvas");
      hi.width = size.w;
      hi.height = size.h;
      const hctx = hi.getContext("2d");
      if (!hctx) return;
      const hiImg = hctx.createImageData(size.w, size.h);
      const d = ref.data;
      for (let i = 0; i < d.length; i += 4) {
        const on = d[i] > 127;
        const wasOn = oldData.data[i + 3] > 32; // old mask stores alpha
        // new truth: white with luminance as alpha
        d[i] = d[i + 1] = d[i + 2] = 255;
        d[i + 3] = on ? 255 : 0;
        if (on && !wasOn) { // …and the ADDED area glows amber
          hiImg.data[i] = 255;
          hiImg.data[i + 1] = 176;
          hiImg.data[i + 2] = 32;
          hiImg.data[i + 3] = 255;
        }
      }
      tctx.putImageData(ref, 0, 0);
      mctx.clearRect(0, 0, size.w, size.h);
      mctx.drawImage(tmp, 0, 0);
      hctx.putImageData(hiImg, 0, 0);
      highlightRef.current = hi;
      // Blink the added area a few times, then leave it visible.
      let ticks = 0;
      const timer = window.setInterval(() => {
        highlightOn.current = !highlightOn.current;
        redraw();
        if (++ticks >= 6) {
          window.clearInterval(timer);
          highlightOn.current = true;
          redraw();
        }
      }, 260);
      exportMask(); // keep the exported mask in sync with the render truth
    };
    img.src = refinedMaskUrl;
  }, [refinedMaskUrl, size, redraw, exportMask]);

  // An intermediate step result arrived: fade it onto THE image so the
  // pipeline is visible in place, and clear the old mask (a new step begins).
  useEffect(() => {
    if (!previewUrl || !size) return;
    if (appliedPreview.current === previewUrl) return;
    appliedPreview.current = previewUrl;
    const img = new Image();
    img.onload = () => {
      const base = baseRef.current;
      const bctx = base?.getContext("2d");
      if (!base || !bctx) return;
      bctx.drawImage(img, 0, 0, size.w, size.h);
      maskRef.current?.getContext("2d")?.clearRect(0, 0, size.w, size.h);
      highlightRef.current = null;
      highlightOn.current = false;
      redraw();
      onMaskChange(null);
      setFlash(true);
      window.setTimeout(() => setFlash(false), 650);
    };
    img.src = previewUrl;
  }, [previewUrl, size, redraw, onMaskChange]);

  /** SAM click-to-select: segment whatever is under the cursor, add it. */
  const smartSelect = async (p: { x: number; y: number }) => {
    if (!assetId || selecting) return;
    setSelecting(true);
    setSelectError(null);
    // Staged status so the UI never looks frozen while SAM cold-loads.
    let stages = SAM_STAGES_WARM;
    try {
      const s = await api.samStatus();
      if (!s.loaded) stages = SAM_STAGES_COLD;
    } catch {
      stages = SAM_STAGES_COLD;
    }
    let stageIdx = 0;
    setSelectStage(stages[0]);
    const stageTimer = window.setInterval(() => {
      stageIdx = Math.min(stageIdx + 1, stages.length - 1);
      setSelectStage(stages[stageIdx]);
    }, 2200);
    try {
      const m = await api.maskPoint(assetId, Math.round(p.x), Math.round(p.y));
      applyApiMask(m.mask_b64, false);
    } catch (e) {
      setSelectError((e as Error).message);
    } finally {
      window.clearInterval(stageTimer);
      setSelectStage(null);
      setSelecting(false);
    }
  };

  const canvasPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const view = viewRef.current;
    if (!view) return null;
    const rect = view.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * view.width,
      y: ((e.clientY - rect.top) / rect.height) * view.height,
    };
  };

  const stroke = (to: { x: number; y: number }) => {
    const ctx = maskRef.current?.getContext("2d");
    if (!ctx) return;
    const from = lastPoint.current ?? to;
    ctx.globalCompositeOperation =
      tool === "paint" ? "source-over" : "destination-out";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    lastPoint.current = to;
    redraw();
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = canvasPoint(e);
    if (!p) return;
    if (tool === "smart") {
      void smartSelect(p);
      return;
    }
    drawing.current = true;
    lastPoint.current = p;
    e.currentTarget.setPointerCapture(e.pointerId);
    stroke(p);
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = canvasPoint(e);
    if (p) stroke(p);
  };
  const onUp = () => {
    if (!drawing.current) return;
    drawing.current = false;
    lastPoint.current = null;
    exportMask();
  };

  const clear = () => {
    const mask = maskRef.current;
    mask?.getContext("2d")?.clearRect(0, 0, mask.width, mask.height);
    redraw();
    onMaskChange(null);
  };

  return (
    <div>
      <div
        className={`frame${rendering ? " render-live" : ""}`}
        style={{ position: "relative" }}
      >
        <canvas ref={baseRef} style={{ display: "none" }} />
        <canvas ref={maskRef} style={{ display: "none" }} />
        <canvas
          ref={viewRef}
          className={flash ? "canvas-flash" : undefined}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={onUp}
          aria-label="Mask editor. Painted red areas will be edited."
        />
        {rendering && (
          <ProcessFX
            active
            effect={fx?.effect ?? "generic"}
            label={fx?.label ?? "Rendering"}
            maskUrl={fx?.maskUrl ?? null}
          />
        )}
        {selecting && (
          <div className="sam-overlay" role="status" aria-live="polite">
            <span className="spinner" aria-hidden />
            {selectStage ?? "Working…"}
          </div>
        )}
      </div>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Brush tool">
          {assetId && (
            <button
              type="button"
              className={tool === "smart" ? "on" : ""}
              onClick={() => setTool("smart")}
              title="Click any object — SAM segments exactly what's under the cursor"
            >
              {selecting ? "Selecting…" : "✦ Select object"}
            </button>
          )}
          <button
            type="button"
            className={tool === "paint" ? "on" : ""}
            onClick={() => setTool("paint")}
          >
            Paint mask
          </button>
          <button
            type="button"
            className={tool === "erase" ? "on" : ""}
            onClick={() => setTool("erase")}
          >
            Erase
          </button>
        </div>
        {selectError && (
          <span style={{ fontSize: 12, color: "var(--err)" }}>{selectError}</span>
        )}
        <label className="row" style={{ gap: 6 }}>
          <span style={{ fontSize: 12.5, color: "var(--muted)" }}>Brush</span>
          <input
            type="range"
            min={6}
            max={140}
            value={brush}
            onChange={(e) => setBrush(Number(e.target.value))}
          />
        </label>
        <button type="button" className="btn ghost small" onClick={clear}>
          Clear mask
        </button>
      </div>
    </div>
  );
}
