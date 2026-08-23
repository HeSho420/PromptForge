# PromptForge — Continuous Improvement State

Working memory for the autonomous improvement loop. Updated after every
meaningful cycle. (User docs live in docs/PromptForge-Documentation.pdf —
this file is engineering state, not documentation.)

## ACTIVE MANDATE (2026-08-18, user-issued; extended to INDEFINITE)

Run the loop **indefinitely** (user: "keep continueing indefinetely"),
focused on: **output quality**, **efficiency**, and **maximization of
processing power**. REFOCUSED 2026-08-18 (user): "some images still have
weird artefacts with inpainting and outpainting, focus on improving
this... keep working untill you are done and the software is perfect" —
inpaint/outpaint output quality leads the queue until measured clean.
Every cycle:
measure → change → verify (tests+lint) → measure again → commit → update
this file. Cycle counter continues from 6. Candidate ledger (adapt as
measurements dictate): render-speed flags on Ada (--fast/fp8, VAE dtype),
gallery/assets latency, FaceDetailer wiring for portraits, batched
count-requests, hires-fix routing, Ollama reload-latency policy, draft-
mode defaults, GPU-util telemetry, generate-model tiering (Flux-GGUF).

## Architecture summary (2026-08-17)

Fully-local AI image/video/avatar studio. Flask backend (port 8000, venv
`backend/.venv`) drives ComfyUI (8188) + Ollama (11434); React/Vite
frontend served from `frontend/dist` by Flask. SQLite (WAL). Job queue
with four lanes: main worker, peer-delegation worker pool (combine mode),
download lane, monitor thread. LAN fabric on 8765/8766: discovery,
model transfer, render+LLM delegation (whole-job), remote logs, git-based
auto-update propagation. Self-healing layers: missing models (registry +
scout + LAN pull), missing Ollama models (autopull), missing node packs
(install-or-reroute at ComfyUI submit), Kontext/DirectML reroutes, video
capability gating, miopen tiled-VAE retry, ComfyUI/Ollama crash revival.
Machines: HerlockLaptop2 (RTX 4060 8GB, CUDA) and HerlockGame (RX 6700 XT
12GB, native ROCm since 2026-08-16, intermittently powered on).

## Baseline (2026-08-17, commit 43c4dd2)

- Tests: 905 passed, 1 skipped (full suite ~257 s, green with live
  Ollama + ComfyUI + LAN peer up — that is the required bar)
- Lint (ruff, app+tests, safety.py excluded): clean
- Backend health: OK on the required venv interpreter
- Ollama 0.32.13; ComfyUI local v0.28.0 (CUDA), peer v0.30.2 (ROCm)
- Known-good live E2E: image edit, background route, delegation with
  peer-side LLM, model/pack self-heal reroute

## Known problems

1. Critic quality: llava-7B hallucinates (measured: 100% checklist on
   1/10 garbage; 40% realism on a blank gradient). Wreckage-veto is a
   band-aid, not a fix.
2. Planner JSON: format:"json" only — schema shape enforced by prompt
   begging + salvage parsers; 7B models still misformat under pressure.
3. HerlockGame video: WAN sampling works on native ROCm; VAE decode hit
   miopenStatusUnknownError. Tiled retry shipped (43c4dd2) but NOT yet
   live-proven (machine powered off). Also pending: missing-node E2E.
4. sqlite: WAL on, but no busy_timeout — concurrent writers (combine
   mode) can hit "database is locked" instead of waiting.
5. Peer endpoints have no shared-secret auth (home-LAN trust model —
   acceptable for now, revisit before any commercial positioning).
6. ~~routes.py em-dash mojibake~~ FIXED (cycle 4): 18 double-encoded
   em-dashes repaired in user-facing API messages.

## Improvement backlog (scored: impact/user value/difficulty/risk)

- [NOW] Ollama structured outputs (JSON schema per call): 9/8/3/2 —
  verified live on Ollama 0.32; kills the misformat failure class.
- [NOW] Critic upgrade llava → qwen2.5-vl (tiered, fallback chain):
  9/9/4/3 — registry confirmed; A/B on known hallucination cases.
- [NOW] sqlite busy_timeout: 5/4/1/1.
- Vision judging with schema-constrained replies (rides on both above).
- Peer pairing secret (auth for /pf-peer/*): 6/5/6/5 — design carefully,
  must not break existing fleet on rolling update.
- Startup/latency profiling pass (unmeasured): 5/5/3/2.
- ~~Frontend bundle split~~ VERIFIED ALREADY DONE: Viewer3D is
  React.lazy-loaded (ResultView.tsx) — the 631 kB chunk only loads when
  a 3D result renders. Closed without change.
- Adherence: GroundingDINO text-grounded masks landed earlier; extend
  mask_verdict telemetry into Behind-the-Scenes: 5/5/4/2.
- routes.py mojibake cleanup: 2/2/1/1.

## Implemented improvements (this loop)

- **Cycle 1 (00832db)** Structured outputs: planner replies are now a
  server-enforced grammar (Ollama JSON schema, verified live on 0.32).
  complete_with_schema inspects client signatures so every test fake
  keeps working. Measured 6-prompt live A/B: 6/6 → 6/6 (no regression;
  the win is the guaranteed tail + the OPERATION_TASK enum). 910 tests.
  Learned: the engineered plan prompt was already strong on happy paths —
  future schema claims must measure tails, not means.
- **Cycle 2 (9a94a02)** Vision judge llava → qwen2.5-vl: "auto" default
  resolves by hardware (7B ≥6 GB VRAM or ≥12 GB RAM, else 3B), explicit
  names honored, CriticChain keeps llava as live fallback while autopull
  fetches the upgrade at startup — the fleet migrates itself.
  /api/health reports the resolved judge. MEASURED A/B: noise 5→1,
  blank gradient 8→1, real photo 8→9. llava had ZERO separation between
  a gradient and a photo; qwen2.5-vl separates by 8 points. 913 tests.
  Process lesson (recorded the hard way): the suite's source-pin tests
  read line numbers from DISK — never edit the tree while it runs (a
  mid-run edit produced 35 phantom failures).
- **Cycle 3 (this commit)** Schema-constrained vision judging:
  ImageCritic.ask(schema=) + ask_with_schema (signature-inspecting twin
  of complete_with_schema) at all five vision sites — critique
  (score/issues), scene graph (full shape, live-verified richer than
  llava era: 6 concrete objects + valid enums), placement (cell 1-9 +
  size enum), view classifier (bin enum + "unknown" escape hatch),
  checklist probe (key-only schema; the answer stays free text so the
  anti-rubber-stamp design survives). Live lesson encoded in the scene
  schema: grammar-constrained models OMIT non-required fields — require
  everything you want answered. 914 tests, lint clean.
- **Cycle 4 (2dbc947)** 18 double-encoded em-dashes repaired in
  user-facing API messages; Viewer3D confirmed already lazy-loaded and
  sqlite busy-handling confirmed already covered — two backlog items
  closed by verification instead of code.
- **Cycle 5 (measurement, no code churn)** Judge re-calibration DONE.
  Controlled matrix, REAL SD renders for style/photoreal rows, both
  judges (llava | qwen2.5vl): cartoon 7|8, watercolor 9|8, barn render
  6|9, pure noise 1|1, corrupted render 1|1, real photo 8|9. VERDICT:
  the wreckage-veto band (≤2 wreck / ≥3 style) HOLDS under the new
  judge — the critique's "for the request" clause makes it grade
  contextually, so proper cartoons score 8, nowhere near the floor.
  Separation WIDENED (worst-legit 8 vs best-wreck 1; llava managed
  6 vs 1) and legit renders score honestly higher → retry pressure
  drops with zero tuning. Thresholds left untouched on measured
  evidence; the veto comment now carries the 2026-08-17 calibration.

## Rejected / deferred

- Rewrite of peer HTTP layer to FastAPI/asyncio: working, tested,
  no measured bottleneck — rejected (rewrite risk > value).
- Speculative micro-optimizations without measurements — deferred on
  principle.
- Quality-threshold re-tune after the judge swap — MEASURED as
  unnecessary (cycle 5) and closed without churn.

- **Cycle 6 (this commit)** Profiling baseline + the whale it found.
  Cold start → healthy: 3.65 s. Warm medians: health 25 ms, models
  75 ms, peers 45 ms, events 45 ms, assets 215 ms, gallery ~495 ms,
  jobs LIST **4.25 MB / poll** — 3.7 MB of it payload.mask_b64
  (historical hand-drawn masks) plus finished jobs' full logs, polled
  every few seconds by the UI. Fix: the LIST elides mask_b64 and caps
  finished jobs' logs to the tail (live jobs keep full logs — the
  queue dock reads status from them); /api/jobs/<id> and ?full=1 stay
  complete. MEASURED: 4,253,567 → 139,867 bytes (30x), 50 → 8.5 ms.
  915 tests OK. Remaining latency candidates: gallery ~495 ms and
  assets ~215 ms over 464 assets (per-file existence checks suspected
  — measure before touching).

- **Cycle 7 (9275ab8)** Backend singleton (pid-file lock, Windows-true
  liveness via OpenProcess/GetExitCodeProcess), updater rollback now
  kills the slow instance before starting the old one, monitor revive
  clears stray ComfyUI processes first. KEY PROCESS-MODEL LESSON: a
  Python 3.13 venv python.exe is a launcher SHIM whose child is the
  real interpreter — every backend/ComfyUI is a two-process PAIR with
  one fate. Count pairs, kill by cmdline (both members), never
  "keep the listener".
- **Cycle 8 (measured, rejected)** ComfyUI --fast on the RTX 4060:
  baseline 512²@22st median 8.20 s, 768² 10.23 s; with --fast 8.21 s /
  10.28 s (5-sample tiebreak). ZERO gain — fp16 SD15 with sage
  attention already saturates Ada; flag not adopted (it can also shift
  outputs slightly, and quality outranks a null win).
- **Cycle 9 (this commit)** Delegated renders no longer unload the
  LOCAL Ollama: _free_vram returns early when the render is bound to a
  peer proxy (the peer's own render proxy unloads ITS Ollama). Saves
  up to a measured 18.8 s cold reload of qwen2.5:7b per delegated job
  — every helper-carried job in combine mode.

- **Cycle 10 (780ebfa)** hires_split_graph: oversized single-pass
  txt2img on SD1.5-class checkpoints auto-rewrites to 640-base +
  latent-upscale + 14-step denoise-0.55 refine. Measured trio at 1024²:
  single-pass broke (doubled irises, waxy skin) at 21.6 s; the split
  produced a clean image at 22.6 s. Wired at _apply_hardware_limits
  AFTER the VRAM clamp. Native-1024 families/img2img/small/two-pass
  graphs untouched.
- **Cycle 11 (a1def69)** Automatic judged face-refinement pass
  (FaceDetailer): facedetail_v1 template + _face_polish at the forge
  save; fail-open everywhere; judge discards a worse polish. Detector
  face_yolov8m sha-pinned in the registry (auto-downloads);
  ultralytics_bbox/segm folder keys added to the launcher yaml (newer
  Impact-Subpack ignores the legacy combined key — found live).
  Side-by-side verified: smeared face → clean eyes/smile. ~17 s per
  portrait pass, ~6 s passthrough without faces.

## Cycle ledger for 12-30 (mandate: quality, efficiency, hardware max)

Done: 7 singleton/updater/revive, 8 --fast rejected-by-measurement,
9 delegated-render planner stays warm, 10 hires split, 11 face polish.
DECIDED (cycle 12a): _face_polish stays FORGE-ONLY — edited photos
carry REAL people's faces and FaceDetailer REGENERATES what it touches;
identity preservation outranks detail polish (same doctrine as the
never-mask-the-face rule). Do not "extend" it to image_edit.
- **Cycle 12 (72787c9)** Draft-intent coercion SHIPPED and live-proven:
  quality.draft_intent (intent phrases only; "fast car"/"draft horse"
  tested negative) overrides triage to generate_draft when its models
  are ready. Live job used the 4-step template end to end.
- **Cycle 13 (measured, both rejected)** Gallery latency is really
  ~180 ms hot (495 was IWR overhead; storage 85 ms) — closed. Batched
  count-requests: max_batch=1 on mid tier = both real machines, and
  combine mode already parallelizes across the fleet — closed.
- **Cycle 14 (this commit)** Drafts skip the quality ladder + polish.
  Job-timeline measurement: ladder cost 62 s on a ~5 s draft render.
  Live after: same request 56 s total vs ~140 s steady-state (2.5x);
  warm repeats ≈ 10-15 s. recipe.draft records it; finals unchanged.
  OPEN MYSTERY (logged): the backend died hard once (~04:0x, no
  finally, lock left stale — singleton stole it correctly); restarts
  now capture stderr to data/logs/backend-session-err.log so a repeat
  leaves evidence.
- **Cycle 15 (dd52c1f)** Ready drafts skip the triage LLM call (its 33 s
  routing was overridden by the coercion anyway): 56 → 41 s cold. Warm
  floor ~40 s explained structurally: --disable-smart-memory reloads
  SDXL per render on 8 GB/16 GB (crash protection); ~10 s on 12/64 GB.
- **Cycle 16 (57e3e23)** Opt-in pairing secret (PROMPTFORGE_PEER_SECRET)
  guards /pf-peer/pull, /pf-peer/install-pack, /pf-peer/log/* via
  X-PF-Secret; discovery + render/LLM proxies stay open (rolling-safe).
  Loopback-tested all three states. Default off.
- **Cycle 17 (this commit)** Drafts render the user's words VERBATIM —
  the enhancement call was the draft path's last LLM touch (~19 s cold),
  and enhancing changes the wording a draft exists to preview. The
  draft path is now LLM-free: measured 24 s total (arc: 140 → 56 → 41
  → 24 s, 5.8x).
- **Cycle 18 (adb7b7e)** Edit jobs batch text-model calls (plan +
  per-step enhance) BEFORE the vision scene pass — on one 8 GB GPU the
  two models evict each other, so every swap was a full reload
  (measured 17.1 s mid-job). Verified same-prompt/same-asset: the
  separate enhance gap vanished from the timeline; 369.6 → 314.5 s
  total (retry-luck noisy). FUTURE candidate from the same timeline:
  the judging chain (scorecard=vision, checklist-build=text,
  probes=vision, disagreement-check=text) still ping-pongs — but its
  ordering is semantically constrained; needs careful design.
- **Cycle 19 (this commit)** Composite-back: every masked-edit template
  now ends ImageCompositeMasked(original, decoded, same feathered mask)
  → SaveImage. MEASURED CAUSE of the user-reported "weird artefacts":
  inpaint/outpaint returned the whole VAE-roundtripped frame, so EVERY
  edit shifted 13.5-27.7% of the pixels it had no business touching by
  >8/255 (mean 5-7.5, p99 27-32, max 163-189; PSNR 28-31 dB) — visible
  shimmer on hair/foliage/fabric, compounding per plan step, and even
  counted as mask leak by objective_report. Fixed in inpaint_v3,
  inpaint_universal_v2, outpaint_v2 (pad mask), remove_object_v1,
  replace_background_v1 (subject stays byte-identical), face_detail_v1;
  WORKFLOW_GUIDE teaches the pattern to LLM-authored graphs; blanket
  structural test pins every current+future inpaint/outpaint template.
  RE-MEASURED: outside-mask pixels byte-identical (max diff 0.0) on all
  three paths at no speed cost; seams clean (feather band blends).
  Live E2E through /api/edits verified. 938 tests.
  NOTED for later: outpaint still paints EXTRA PEOPLE in new margins at
  the model level (template negative + judge-retry is the only defence);
  video_inpaint/video_outpaint templates still roundtrip whole frames
  (chunked pipeline — needs its own measurement); kontext whole-image
  edits are inherently uncomposited.
  E2E postscript (fffa575 verified live): masked hires edit at
  1471×1828 (odd size) changed 4.64% of pixels, ALL inside the drawn
  region, 0 outside — through the full plan→mask→judge→retry→save
  pipeline. Learned en route: /api/assets/<id>/file serves the WORKING
  image (original until the user promotes a result) BY DESIGN — E2E
  before/after must fetch version files, not the asset file.
- **Cycle 20 (this commit)** Outpaint person guard. MEASURED first: 8
  production-style outpaints (2 photos × 4 seeds) → 1/8 grew a full
  standalone man in the new margin (plus the cycle-19 baseline hit =
  2/9 that day). Deterministic guard shipped: after each outpaint,
  every new margin + a 96px inland slab is matted with
  BiRefNet-portrait; CALIBRATED on the same data (15/15 clean margins
  matted 0.0%, the intruder 24.6% → floor 6%); mass continuing inland
  past the junction = legitimate subject completion (ceiling 2%),
  mass only in the margin = invented stranger → ONE fresh-seed
  re-render, keep the cleaner. Fail-open (pack off/errors/mock).
  Live-proven on the measured renders (pool_s44 → ['left'] at
  margin 31.9%/inner 0.9%; clean + completion patterns pass);
  E2E outpaint job through /api/edits on the new code. 945 tests.
  ALSO MEASURED (ledgered): caption/watermark bands in the SOURCE get
  continued into margins as garbled glyphs 4/4 seeds on the affected
  photo — retry cannot fix that class; needs source-edge text
  detection BEFORE padding (next artifact candidate).
- **Cycle 21 (this commit)** Canvas-growth verification is arithmetic.
  The cycle-20 E2E delivered 1471→1855 wide and the vision checklist
  still said "matches 0%; missing: extend the picture to the left and
  right" twice, burning a retry (~2.5 min) on a succeeded job. D7's
  format_delivered HAD fired — but about_format had no growth verbs, so
  the settled item was never retired. Fixed: about_format recognises
  growth phrasings; format_delivered gates on canvas-scoped
  _CANVAS_GROWTH instead of the loose _CANVAS_INTENT (bare "extend"
  meant "extend her dress to the floor" measured the unchanged canvas
  as a FAILED format request → phantom missing entry on content edits);
  a named direction pins the axis (upward request answered sideways
  returns False honestly — which also documents the real gap: the
  outpaint step always pads left+right; planner-driven directional
  padding is a ledgered candidate). Verified live on the identical
  request: settles by measurement, no retry. 949 tests.
- **Cycle 22 (b34b5d6)** Directional outpainting shipped. quality.
  outpaint_directions() maps the instruction's side words to explicit
  four-side pads (unnamed sides pinned to 0 — the template defaults
  left+right to 192, so "upward" must not also grow sideways; no
  direction → None → template default). Threaded through the template
  render (`extra` declared-slots pass-through), the person guard's
  margin geometry, _pad_mask (both assumed a CENTERED original —
  wrong for one-sided pads), and the ladder's outpaint retry.
  Live E2E: "extend the picture upward" → "Extending the canvas on
  the requested side: top" → 1471×1828 → 1471×2020 (top-only, +192;
  the commit message says 2212 — erratum, 2020 is correct), settled
  by the axis-pinned arithmetic, best of 1, no retry. 956 tests.
- **Cycle 26 (870e3e9)** Checklist built at PLAN time. The judging
  chain ping-ponged models: scorecard(vl) → checklist(TEXT, 17s reload
  that also evicts vl) → probes(vl again, 22s reload) ≈ 39s of swaps
  per judged edit (measured on the deglyph E2E). request_checklist
  reads only the instruction → moved into cycle-18's warm-text batch
  (s["_checklist"] on the last step; presence-checked consume with a
  live fallback; plan_report can't leak underscore keys). MEASURED
  after, same request: scores → checklist SAME SECOND, probes +3s on
  the still-warm vl — 39s → 3s. Glyph guard also fired again on this
  run (consistent). 959 tests.
- **Cycle 27 (f1b8666)** Outpaint junction exposure harmonization.
  Measured 3 independent renders of the same L+R outpaint: no sharp
  seam (feather works, col-deltas p57-p87), but the 16px strip step at
  the RIGHT junction sat p99.4-p100 of the interior distribution every
  time (14.9-16.5 vs median 2.7; margins ~2x brighter) — the recurring
  "lighting/colour mismatch" in inspections, visible as a tonal wall.
  quality.harmonize_margins: pin margin low-freq colour to the SOURCE's
  edge strip, 361-row smoothing, cap 28, smoothstep falloff to the
  outer edge, plus a 24px inland tail whose junction value is
  CONSTRUCTED as E−d (margin correction minus the junction's own raw
  step) so both sides land on the same tone by definition. FOUR
  parameter revisions, each killed by a measurement: k65/cap48 painted
  a visible stripe (eye); margin-only introduced a 1px edge (sharp
  metric p99.9); per-column inland source-diff MOVED the edge; 96px
  tail overdarkened correct interior (profile: tone reach ~16px though
  byte reach 64px). Wired post-deglyph in _guarded_outpaint + ladder
  outpaint retry, fail-safe, logs measured step/side. LIVE fresh seed:
  guard logged left 12/right 28; delivered file: left strip p2.3
  (0.31!), right p93.2 (8.79, halved; residual = deliberate cap),
  sharp p31.5/p72.1 (≤ raw baseline), deep interior byte-identical,
  junction invisible in crops; inspector dropped its across-the-seam
  complaint. Glyph guard fired on both runs (5 consecutive). 966 tests.
- **Cycle 28 (807279b)** Inspector judges only regions the edit could
  touch. The inspect zoom used the mask BBOX — an L+R outpaint's two
  bands make that the whole frame, so the vision model judged the
  byte-identical subject: on the CLEAN cycle-27 render 3/3 runs gave
  the identical four "bikini top" complaints (100% about provably
  unchanged pixels, 0 about the junctions), and on a render with a
  REAL stripe artifact (rev-1 params re-applied) the full view MISSED
  it 2/2 while repeating the same four complaints — blind to real
  defects AND a reliable false generator, feeding the ladder's
  "avoid:" clause. _mask_view_boxes: split on column/row projection
  gaps (proportional threshold), hollow ring (all-side outpaint) → its
  4 edge bands, single compact region → bbox zoom unchanged; one
  vision call per view, issues prefixed with the view's side. Per-band
  on the same evidence: subject complaints structurally impossible,
  stripe caught 2/2 both sides, remaining observations track measured
  residuals; band calls faster (273px vs 1855px views). LIVE: "Issues
  found: left region: Seam between the wall and the plant; right
  region: Color mismatch" in the same ~24s the full-frame call took.
  NOTE for later: scorecard still sees the whole frame and scored the
  STRIPED render artifact_free 97 — whole-frame scoring is insensitive
  to junction defects; candidate: feed per-band issues into scoring.
  972 tests.
- **Cycle 29 (0990e35)** Measured junction defects cap artifact_free.
  The cycle-28 finding closed: quality.junction_flaws is the
  deterministic verdict (cycle-21 doctrine — exact measurement outranks
  the judge). Two signals per junction, each DOUBLE-gated magnitude AND
  percentile vs the image's OWN interior (self-calibrating): hard edge
  (col-delta ≥6 at ≥p99.5; worst clean 3.4@p86.7, stripes 11.1+@p99.9)
  → cap 55; exposure wall (strip16 ≥12 at ≥p98; clean 8.8@p93.2, walls
  14.9+@p99.4+) → cap 70. Calibration 11/11 TOTAL separation: 3 raw
  walls→70, rev-1 stripe→55, harmonized raws→None, live renders→None,
  mild top-pad→None (~0.5s/check). _ground_scores wired at BOTH scoring
  sites (first pass + ladder), lower-only, honest log when it bites;
  last_outpaint carries pre_size for exact geometry. LIVE healthy
  render: no overrule line, artifact_free 97 stands, delivered file
  measures None — correct negative path (positive path = offline 11/11
  + unit tests). Harmonize guard 4th consecutive live fire (L16/R28);
  deglyph guard hit its keep-first branch live this run ("re-render no
  cleaner" — first live sighting). 976 tests.
- **Cycle 30 (a445145)** Draft mode surfaced in the UI + --bf16-vae
  closed by measurement. The backend's complete draft pipeline
  (generate_draft DMD2 template, LLM-free fast path, ladder skip,
  recipe.draft) had ZERO frontend presence. Now: "Quick draft"
  checkbox on the generate task (button relabels "Draft it"),
  routes pass draft:true → _workflow_inner ORs it with the words-based
  draft_intent (same deterministic path); wanted-but-not-ready drafts
  log honestly instead of silent full-quality. Result shows a dashed
  DRAFT badge + recipe line "DRAFT (4-step preview, judging skipped)".
  LIVE through the real UI: toggle → payload flag (prompt had NO draft
  words) → "Draft requested — skipping workflow triage" → 29s LLM-free
  render → recipe.draft true → badge + recipe line render. 978 tests,
  frontend built. --bf16-vae: NO-OP — comfyui-revive.log shows "VAE
  ... dtype: torch.bfloat16" already auto-selected by v0.28 on Ada;
  fp32 direction rejected (VRAM risk, no measured banding defect).
  Two quirks noted: (a) ProvenanceBadge shows "cloud API ·
  generate_draft" on template renders that touched no API — mislabel,
  candidate fix; (b) the Claude-Browser pane reports document.hidden
  even fronted, so the UI's deliberate hidden-tab poll pause needed a
  visibility override to observe — env quirk, not an app bug.
- **ENV MANDATE (d4de2f8, 2026-08-19)** Environment-aware background
  edits — new user brief: scene reconstruction, not background swap.
  NEW scene_geometry.py + scene_probe template (MoGe normals/depth/
  valid): SceneCard = contacts (matte bottom clusters), cut-at-bottom,
  posture (vl + aspect veto), ground plane, camera pitch (up-normal
  tilt), horizon (disparity-row fit, r²-gated; depth PNG proved
  DISPARITY-encoded live r²=0.9998; 8-bit-saturated rows dropped).
  environment_spec (warm-batch text LLM, conservative-without-facts
  rule pinned by test) + spatial_prompt compiler (ground contract,
  camera words, horizon band, lighting, solid-ground anti-flood
  negatives; retries RECOMPILE — first impl dropped it and every retry
  flooded). Validation = same measurements on the result + contact
  checks in two layers (up-normal window; region-scoped schema vl
  SUPPORT probe — normals can't tell water from deck: both up-facing,
  measured live; probe judges support-class only after it rejected
  "dry tiles" vs planned "wet tiles" twice). Misses cap
  scene_consistency 70 + feed avoid-clauses (candidates too). LIVE
  runs: card deterministic across runs (standing, 2 contacts, pitch
  8°, horizon 38% r²=1.00, ground 55%); validator caught real breaks
  (8°→21°, horizon 38%→1%/17%/22%/11%) and PASSED the good draws;
  final result: subject ON the deck with matching cast shadow. Also
  fixed: extra_model_paths.yaml never mapped geometry_estimation
  (scene3d silently broken since the ComfyUI upgrade); launch.ps1
  writes it now. Zero new models (MoGe was registered; 631MB MIT,
  auto-fetched once). 1006 tests (commit msg says 1004 — erratum, 2
  probe-pin tests added after drafting). REMAINING (Phase B queue):
  camera-pitch ADHERENCE needs conditioning, not words — depth/
  perspective guidance via controlnet-union-sdxl (registered, not
  fetched); horizon misses correctly detected but only ~half the
  draws comply; contact_ground_frac never fired in anger (vl probe
  covers water); DWPose keypoints (SavePoseKpsAsJsonFile exists) for
  sitting/lying contracts; occlusion (objects in front) undone;
  scene-extension mode still = outpaint route; retry budget shares
  the unreliable vl-verify circuit breaker (a static geometry miss
  burned the round budget). Env aux versions saved for debugging.
- **ENV Phase B1 (1cf96a8)** Depth-guided environment renders. Words
  held the measured horizon on 0/3 first draws; conditioning is the
  lever. guidance_depth: subject keeps measured disparity, below the
  measured horizon the ground plane's FITTED ramp (NOT raw pixels —
  raw ground disparity made the ControlNet repaint the plaza's tile
  grid as striped pavement, measured live, fixed), far/free above; no
  confident horizon → no guide. background_guided template = proven
  background graph + ControlNet v1.1 depth (strength 0.55, released
  last 20% of steps). NEW MODEL controlnet-sd15-depth (OpenRAIL-M,
  ~700MB fp16, auto-fetched with hub-verified checksum, MODEL_USAGE
  line: never prompt-routed, pipeline-attached). LIVE A/B same
  request/asset: guided geometry pass 4/4 draws vs 2/5 unguided;
  first-draw 3/3 vs 0/3; delivered image keeps the plaza's exact
  camera, horizon at measured 38%, subject on solid deck (sampler
  built a tiled platform to satisfy position+contract — eccentric but
  physically coherent, shadow+reflection present). 1008 tests.
  NEXT: vl-verify flakiness burns retries ("swimming pool background
  matched 0%" on images DOMINATED by pools — geometry+scores were
  fine; reproduce the adherence probe offline, likely needs the
  region-scoped treatment); DWPose keypoints for sitting/lying;
  subject softness flag traces to prompted depth-of-field (whole-frame
  sharpness compare) — explain or scope it.
- **ENV B2 (e4b2316 + 9f342cc)** Verify + routing rounds, all live-
  measured. (1) The checklist trap: "what is the new background?" was
  answered — correctly — "trees and mountains" on a pool-dominated
  scene (the far field IS the background); no honest answer could name
  the requirement, every run burned a ~2min retry (reproduced offline
  twice/run, tie-break incl.). Background steps now derive the check
  from the SPEC and ask WHERE the photo was taken ("Resort poolside"
  → True vs "swimming pool"). Live: verify passes, best-of-1. (2)
  quality.environment_intent: "put her in a nightclub" was planned
  CHANGE_STYLE → inpaint → honest mask failure; relocation phrasing
  (person object + place preposition, garment/pose guard) now coerces
  to the background pipeline (deterministic, beside the add/format
  coercions). (3) Trailing CHANGE_LIGHTING after a background step
  PRUNED: the standalone relight template re-ran full IC-Light on the
  already-lit scene — objective check measured 92% of face pixels
  moved; env's own _match_lighting is illumination-only by design.
  Nightclub rerun: single step, validate ✓, place-check ✓, identity
  90, real interior with the floor under her feet. 1012 tests.
  KNOWN LIMIT (documented, principled): extreme relight (daylight →
  dim club) shifts the subject only partway — the low-frequency
  transfer trades light completeness for identity; the full-redraw
  alternative is measured identity-destroying. Future: IC-Light fbc
  with detail-transfer strength tuning, or brightness/temperature
  histogram match on the subject as a post step.
- **PARITY 1 (b90b13b) — ChatGPT-editor parity mandate begins (2026-08-
  20): "remove the background" delivers TRANSPARENCY.** The class was
  excluded from the repaint route by design and handled by NOTHING (fell
  to generic inpaint). cutout_intent (direct-object background removal,
  transparent/no-background, cut me/her out, sticker, isolate; "remove
  the man in the background" stays inpaint) + default_edit_step entry
  (before background) + plan coercion (after env coercion so "remove bg
  and put her in a bar" keeps env routing). New cutout step: BiRefNet
  matte → alpha channel over ORIGINAL pixels → RGBA PNG version;
  early-return like angles (no judging — matte IS the quality);
  coverage gate 2-98% honest errors; matte model from scene subject
  knowledge (instruction alone picked lite over the exact portrait
  matte live). Mock E2E (route/adapter/RGBA on disk) + live sticker-
  grade figure checkerboard-verified. 1016 tests. Parity queue next:
  style transfer identity/structure (live test running — CHANGE_STYLE
  routes to img2img 0.6 denoise; Kontext excluded from style ops,
  suspect), text editing in images (CHANGE_TEXT/Kontext live check),
  multi-image compose live check, colorize B&W, iterative edit-the-
  result UX, product/white-background shots.
- **PARITY 2-6 (5339679…1c68962, 2026-08-20)** ChatGPT-editor parity
  sweep, all live-measured: STYLE (Kontext-eligible CHANGE_STYLE; art-
  frame scorecard after realism-20-on-a-delivered-watercolor burned 3
  renders; softness veto + face-drift deferred in style mode; live:
  1 attempt, identity 95). FORMATS (HEIC/TIFF→PNG at ingest via
  pillow-heif; EXIF orientation baked out — portrait phone JPEGs were
  edited SIDEWAYS before; real-encoder tests). PRODUCT SHOTS ("on a
  white background" was ADD_OBJECT → background engine). TEXT: render
  intent routes to zimage where it fits; HERE the Qwen3-4B encoder's
  24GB RAM floor gates it (menu-hidden — the router never "ignored"
  the rule) → honest lettering warning; text EDITING via Kontext
  CHANGE_TEXT works (live: CDOSED sign → crisp OPEN, accuracy 100,
  1 attempt). COLORIZE (Kontext routing was right; chroma settler
  0.0→67.8 overrules the verifier that called a perfect colorization
  missing twice). COMPOSE (scene-picked placement pasted the reference
  woman ON the subject's torso at 26% scale → placement_correction
  from the destination matte: person-size, standing line, slide-off;
  provenance stripped from checklists; count-question verify; live:
  two friends at natural scale, shared light, identities intact).
  1033 tests. REMAINING (honest): text GEN hardware-gated here (peer
  64GB machine capable); restore/sketch→photo ride Kontext untested
  live; identity-across-generations templates untested this session;
  verify vl still flaky on some classes (settlers cover format/colour/
  colorize/count; the static-verdict breaker catches the rest).
- **QUALITY 1 (289ed34, 2026-08-20)** Environments 3× sharper. Root
  cause of the perpetual 0.17–0.33x softness flags: the background
  engine painted ~1.8 MP scenes with SD15 far off-distribution. An
  environment IS scene continuation = outpaint-class work; juggernautXL
  is the measured winner. NEW background_guided_xl (juggernaut +
  controlnet-union-sdxl depth, pose_v2's proven wiring, CN after
  InpaintModelConditioning); guided ladder XL-first → SD15 fallback;
  unguided prefers the outpaint-class checkpoint. MEASURED, identical
  request: generated-scene gradient energy 157.9 → 462.7 (0.12x →
  0.36x source; remainder = intended DoF), geometry validated on the
  FIRST draw (union in auto mode accepted the depth canvas), correct
  cast shadow agreeing with the loungers', NO softness flag for the
  first time, single attempt — best environment render of the project.
  Remaining softness class = degraded SOURCE subjects (the draft-gen
  test asset), not the pipeline. 1033 tests.
- **QUALITY 2 (a7a62b3, 2026-08-20)** Compose harmonisation on the
  plain SDXL base (v1-5-pruned harmonised at ~1.8 MP native — same
  off-distribution class as the soft environments). Picker prefers a
  plain XL base; surprising-name filter intact; sd15-base = no-XL
  fallback; pins updated. Live A/B: crisp photoreal faces on both
  women, inserted identity clearly held, verify FIRST attempt via the
  count checklist — best composite of the project. NEXT quality
  candidates: subject-quality floor (draft-gen test assets degrade
  every edit view — use real photos for demos), size-aware inpaint
  checkpoint pick (large masks → XL, mirror of the env lesson, MEASURE
  first), Kontext q4 vs higher quant on 8 GB, IC-Light fbc strength
  for extreme relights.
- **QUALITY 3 (85474ab, 2026-08-20)** Region-verify rescue KEPT;
  large-mask XL swap MEASURED AND REVERTED. Kept: changed-region
  verify rescue (three classes of false "missing" in one day — a
  colorization, a second woman, a stone statue — each burned a ~4min
  re-render twice; still-missing items get one region-scoped look,
  overrule missing→met only; it honestly refused a truly absent
  statue); juggernaut-xl-inpaint vram 8.0→7.0 (escalation refused it
  on the card that runs it daily). Reverted: large-mask XL checkpoint
  swap — on a drawn mask overlapping the subject, juggernautXL
  full-denoise REDREW the person (different face, accuracy 20, no
  statue) where SD15 kept her at 0.60x sharpness. Sharper wrong
  content < softer right content. NEXT design for that class: protect
  the subject matte INSIDE drawn masks (trimap, as the background
  route does) before any checkpoint change; note the maskless Kontext
  route already delivered an excellent statue on the same request.
  Numeric floor to beat: masked repaint at 0.60x untouched sharpness.
2026-08-24 — QUALITY 4 (the ledgered trimap design, DELIVERED): subject
  shield for drawn masks. quality.subject_shield (pure geometry: drawn
  region minus the BiRefNet matte core eroded ~4px for a blending rim;
  stands down when <10% of the region touches the subject OR when <30%
  of the region survives — a mask that mostly IS the person is a
  deliberate person edit and the user outranks the shield) +
  Services._shield_subject (skips when about_the_subject(request);
  person-gated by ONE region-scoped critic question on the matte bbox
  since BiRefNet mattes whatever is salient and protecting a vase from
  "replace the vase" would block the very edit) + driver hook ONLY in
  the mask_source=="user" branch, after grow/shrink, before the variant
  choice. Measured on the exact reverted-swap scenario (fem.png, 0.443
  drawn mask, statue request): shield took 42% of the region off her;
  the smaller region then qualified for hi-res crop&stitch (render 38s,
  statue at native detail — the large-mask softness class fixed itself
  as a SIDE EFFECT, no checkpoint change); protected-core sharpness
  ratio 0.998 vs the 0.60 floor, mean|diff| 0.48 levels, 1.7% of her
  pixels moved >8 levels (the rim); judge: identity 98, accuracy 95,
  realism 90, verify 95/100, statue confirmed by eyeball. 1045 tests
  (+9: shield geometry, gating incl. vase-refusal + words-settle-it,
  user-branch-only source pin). NEW candidate from the run: the seam
  inspector complained about her BIKINI (source pixels at the new mask
  boundary) and burned 2 retries that keep-best discarded (both
  identity 100 — protection held through retries); inspection should
  not raise issues about content the shield excluded.
2026-08-24 — QUALITY 5 (retry containment): the keep_going ladder's
  "delivered + overall>=85 stops the rounds" rule applied only to
  checklist-sourced verdicts; the judged-adherence path fell through to
  bare meets_target and a 90-realism render chased the 95 target the
  judge never awards on real photos. Measured twice on the drawn-mask
  statue job: 2 extra rounds, both accuracy-0 garbage, both discarded
  by keep-best (the depressing complaint was SOURCE pixels — her bikini
  at the shield boundary — unfixable by re-render). Containment now
  source-independent. A/B same job: rounds 2→0, wall 4:34→2:30,
  identical quality (overall 94, identity 98, statue verified).
  1046 tests (+1 integration: 90-across-the-board render, one render
  only, no retry stage).
2026-08-24 — QUALITY 6 (the "partway relight" limit was a DROPPED
  match, not a weak one): measured on "put her in a nightclub", the
  delivered subject's luma shifted -0.1 while her scene dimmed 54
  levels — the lighting match ran on attempt 0, but the ladder retry
  re-rendered the background WITHOUT it and "Round 1 kept" (a +3
  visual_quality tiebreak) delivered the raw composite. Fix 1
  (load-bearing): the background retry path now runs the same
  _match_lighting. Fix 2 (principled, per the template's own
  lighting-only contract; magnitude not separately isolated):
  scene_geometry.lighting_prompt — conditioning leads with the spec's
  lighting_wish instead of the full env prompt + hard-coded "natural
  light"; matte hint now describes the subject. A/B: illumination gap
  +92.7 → +31.1, subject -46 luma into the amber/neon bar light,
  identity intact (judge 95), eyeball: dapple subdued, neon reads as a
  source on her. 1049 tests. Round also exposed, still OPEN: (1) the
  place-question verify false-missed "nightclub" on a plainly-rendered
  club in all 3 runs (examiner likely answers "bar"; match too
  strict) — burns one retry per env job (static-verdict breaker
  contains round 2); (2) validate's "nothing walkable under feet"
  fires when a bar counter legitimately occludes the feet. ALSO: a
  PowerShell -replace/Set-Content append silently failed AND
  mojibake'd test_scene_geometry.py (caught via git diff, restored) —
  machine landmine: never edit source via PS string ops, use the Edit
  tool.
Candidates, roughly ranked (artifact focus first per 2026-08-18 mandate):
- **Cycle 23 (0079083)** Outpaint glyph guard SHIPPED and live-proven.
  Detection calibrated on 16 real margins (per-row density of >40 grey
  gradients, bottom 12%, ≥3 rows over 0.08: all 4 real bands 8-14 hot
  rows incl. the faintest, all 12 clean bottoms 0; top-edge check
  excluded — its one calibration signal was branch texture). On hit:
  ONE re-render from a band-neutralized source copy (band rows filled
  with the stretched+blurred row above), keep the cleaner render,
  byte-restore the ORIGINAL band over the center. _margin_geometry
  extracted, shared with the person guard. Live E2E on the affected
  photo: guard fired, re-render clean, band restored, judging chain
  ran once (no ladder retry), visual check: margins glyph-free with
  the overlay intact. 959 tests. Scope note: L/R margins only;
  top/bottom-margin bands need their own calibration set.
- **Cycle 24 (this commit)** Drawn-mask replacements explain their
  misses. Scope collapsed on inspection: WITHOUT a drawn mask, object
  replacement already routes to Kontext (no mask at all; measured
  accuracy 100 on the same request) — the extent gap only bites
  USER-DRAWN masks, and silently enlarging a drawn region would break
  the "your mask is an instruction" doctrine. Shipped the honest
  alternative: when a REPLACE_OBJECT inpaint with a drawn mask still
  misses requirements, the verify log explains the drawn region covers
  the OLD object and suggests a roomier region or clearing the mask.
  drew_mask captured at payload time (user_mask_b64 is cleared on
  consumption and cannot answer this at verify time). Log-only.
- **Cycle 25 (measured, REJECTED)** Video background composite-back.
  Hypothesis: preserved-background motion transfer ships VACE's
  generative approximation of the scene (cycle-19 class, per-frame).
  MEASURED live (21 frames, 384², seed 777, dance.mp4 + real person
  matte): background outside the grown matte differs mean 3.4/255 with
  0.1% of pixels >8 and worst 20 — while the MP4 CODEC FLOOR alone
  (driving frames re-encoded, no render) measures mean 1.41, 0.34%>8,
  worst 48. The rendered background is AT the compression floor; VACE's
  mask-0 copy is faithful to within codec noise. The drafted per-frame
  composite (implemented during the render) was REVERTED unshipped: it
  would add silhouette-clipping risk for a gain buried under H.264
  noise. Same doctrine as --fast (cycle 8): rejected by measurement,
  closed without churn.
- Batched count-requests ("make 4 images" → batch_size on one graph,
  result plumbing for multiple assets per job) — throughput on VRAM
  headroom; render_budget.max_batch exists.
- Gallery/assets endpoint latency (495/215 ms @ 464 assets) — profile
  the per-file exists() suspicion first.
- Draft-mode default routing (DMD2/LCM speed LoRAs exist since T11 —
  are they ever picked automatically? verify like hires was).
- Ollama reload cost per job (~6-19 s) — measure the full job timeline
  first (instrument [eta] logs already exist).
- VAE dtype flags (--bf16-vae) — measure like --fast was.
- Peer pairing secret (security; rolling-migration design needed).
- HerlockGame E2Es when it appears (miopen tiled retry, missing-node
  heal, critic auto-migration, and now: does its combine-mode helper
  benefit from cycle 9's warm-planner skip).

## Next priorities

1. Live E2Es when HerlockGame reappears: miopen tiled-VAE retry (video),
   missing-node heal (pinned background edit), and confirm its critic
   auto-migrates to qwen2.5vl:7b (64 GB machine → 7B tier).
2. Gallery/assets endpoint latency (~495/215 ms over 464 assets) —
   profile the per-file checks before optimizing.
3. Peer pairing secret for /pf-peer/*: design the rolling migration
   first (old peers must not be locked out mid-fleet-update).
4. Backlog: appearance-question site (services ~8659) could take a loose
   schema; GroundingDINO mask telemetry into Behind-the-Scenes.

