# Roadmap

The vision: a local, prompt-based media studio. Describe what you want; the
program plans a ComfyUI workflow for it (using a local LLM, checking itself at
every step), finds and stages any models it is missing, renders images and
video through ComfyUI, and — from a consented photo set — builds a 3D avatar
of a person that can then appear in generated photos and videos. Everything
runs locally; a cloud API model (Claude Fable 5) is used only when the local
LLM fails, and its use is always visible in the output's provenance.

Phases are ordered so each one replaces a seam that already exists — no
rewrites, no fake output in the meantime.

## Phase 1 — Real image AI (replace the mock adapters)

1. **ComfyUI end-to-end.** The adapter, templates, validation and registry
   checks exist and are unit-tested. ✅ Done since: mask loading switched to
   `LoadImageMask` (channel=red, values used directly — avoids LoadImage's
   inverted-alpha MASK convention), model registry pre-seeded with vetted
   URLs + published sha256 for `sd15-inpaint` and `sam-vit-b`, and an opt-in
   live integration test added (`PROMPTFORGE_LIVE_COMFYUI=1`,
   `tests/test_live_comfyui.py`). Remaining: run that test against a live
   ComfyUI and tune the template (denoise, grow_mask_by) in `inpaint_v2.json`
   — templates are versioned files, never edited in place once released.
2. **Real segmentation.** ✅ Done since: `SamSegmentationAdapter`
   (`adapters/sam.py`, `PROMPTFORGE_SEGMENT_BACKEND=sam`) runs Segment
   Anything's automatic mask generation and grounds the prompt by scoring
   candidates with spatial/size priors; selection logic is pure and
   unit-tested offline, torch imports stay lazy, and the registry gate
   (`sam-vit-b`) raises ModelMissingError → HTTP 409 instead of falling back
   silently. Remaining: exercise against a real checkpoint; optionally
   replace heuristic `select_candidate` with CLIP ranking behind the same
   signature.
3. **Safety stage 2.** Add an image-aware checker (e.g. person/NSFW detection
   on upload) returning the same `SafetyVerdict`, run alongside the prompt
   rules. Keep the rule filter as the fast first gate.

## Phase 2 — Intelligence layer (LLM-planned workflows + model discovery)

The engine exists and is unit-tested offline (built 2026-07):

- **LLM chain** (`core/llm.py`): local OpenAI-compatible endpoint
  (Ollama/LM Studio, `PROMPTFORGE_LLM_URL/_MODEL`) first; Claude Fable 5 via
  the Anthropic API only when the local model fails
  (`PROMPTFORGE_LLM_API_MODEL`, empty = fully local). Every reply carries
  `source` + exact model — cloud output can never pass as local.
- **Workflow generation** (`core/workflow_ai.py`, `POST
  /api/workflows/generate`): the LLM emits a ComfyUI graph for the prompt;
  every candidate must pass the same structural allowlist validation that
  gates hand-written templates (node types, link integrity, output node,
  size cap). Failures are fed back verbatim for bounded repair attempts;
  `repair()` accepts ComfyUI runtime errors the same way. Trust is
  structural, not model-based.
- **Model discovery** (`core/model_search.py`, `GET /api/models/search`,
  `POST /api/models/propose`): searches the Hugging Face hub, extracts each
  file's published LFS sha256, and stages candidates in the registry.
  Downloads remain explicit + checksum-verified; missing checksums stay
  visible.

End-to-end execution shipped (2026-07-08):

1. **Execution integration.** ✅ The `workflow` job type submits a generated
   graph to ComfyUI (`ComfyUIClient.run_graph`), extracts ComfyUI's own node
   errors (submit-time 400 bodies + history status messages), feeds them back
   through `repair()` with bounded retries, logs every step, and saves the
   result to the gallery with provenance.
2. **Inventory awareness.** ✅ Plans are grounded in live `/object_info`
   inventory — the LLM is told exactly which checkpoints ComfyUI can load.
3. **Model-gap closure.** ✅ (first stage) If ComfyUI has no usable
   checkpoint, the job auto-installs one from the registry
   (checksum-verified; `PROMPTFORGE_AUTO_INSTALL=0` disables). Remaining:
   LLM-driven search-term suggestion for arbitrary missing models (LoRAs,
   VAEs) via model search.
4. **Forge UI.** ✅ Plan preview + one-click run with provenance badges.
   Remaining: history of past runs, cancel button.

## Phase 3 — Video editing and generation (through ComfyUI)

Infrastructure that already exists and will be reused: job queue with retries
and per-frame logging, asset store (video kind + extension allowlist), safety
gate, adapters, LLM workflow generation. New work:

1. Frame extraction service (ffmpeg is a hard dependency; `imageio-ffmpeg`
   fallback) producing a frame manifest.
2. `video_edit` job type: keyframe selection, per-frame inpainting through the
   *same* InpaintingAdapter, temporal consistency pass (start naive: locked
   seed + shared mask; later: optical-flow-guided mask propagation).
3. Video *generation* templates for ComfyUI (AnimateDiff / SVD / Wan-style
   nodes). Requires deliberately extending `ALLOWED_NODE_TYPES` per vetted
   node pack — the allowlist stays the gate for generated workflows too.
4. Failed-frame ledger inside the job payload; retry re-renders only the
   failed frames (the queue's TransientError path already handles backoff).
5. Re-encode with the original audio track muxed back (`ffmpeg -i edited.mp4
   -i original.mp4 -map 0:v -map 1:a? -c:a copy`), logging when the source has
   no audio.

## Phase 4 — Consent-safe 3D avatar (ComfyUI cannot do this part)

ComfyUI renders images/video but has no real photogrammetry/reconstruction
path, so 3D uses dedicated local tools behind a `ReconstructionAdapter` —
same seam pattern as the other backends. Phased deliberately; each step ships
alone and never invents unseen geometry.

1. **Dataset intake + consent record.** Import a photo set; store an explicit
   consent attestation in asset metadata; refuse datasets without it.
2. **Coverage analysis.** Estimate camera poses (COLMAP feature matching) and
   report angular coverage as a sphere heatmap. If coverage is insufficient,
   the output is a *report of missing angles* — e.g. "no views of the back
   between 120° and 240°" — never a hallucinated completion of hidden body
   details.
3. **Reconstruction backends behind `ReconstructionAdapter`:**
   photogrammetry first (COLMAP + openMVS/Meshroom meshing, GLB/GLTF export),
   Gaussian splatting second (gsplat/OpenSplat — better fidelity, needs a
   GPU; splats export to .ply/.splat).
4. **Preview environment.** three.js viewer page (pose/turntable, simple rig
   for humanoids where a mesh + skeleton exists).
5. **Bridge back to ComfyUI for "photos and videos of the avatar":** render
   the reconstructed model from chosen poses/cameras, then feed those renders
   through ComfyUI img2img/ControlNet (depth/pose) to generate styled photos
   and video frames. The avatar provides real geometry; ComfyUI provides the
   look — no identity is ever transferred onto other people's footage.
6. Safety: the existing prompt filter applies to any texturing/edit prompts;
   identity-transfer requests stay blocked (deepfake category); consent
   attestation is checked before any avatar render job.

## Infrastructure

- **FastAPI migration** when async/streaming matters: rewrite `api/routes.py`
  only; `core/` and `adapters/` are framework-free by construction.
- **Queue upgrade** to Redis + RQ/Celery for multi-worker rendering: keep the
  `JobQueue` public surface (enqueue/get/cancel/retry/logs) as the interface.
- **Storage** to S3-compatible object store; `AssetStore` is the seam.
- **Auth** before any non-localhost deployment (the MVP is single-user local).
- Progress streaming (SSE/WebSocket) to replace UI polling.
