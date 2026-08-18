# PromptForge — Continuous Improvement State

Working memory for the autonomous improvement loop. Updated after every
meaningful cycle. (User docs live in docs/PromptForge-Documentation.pdf —
this file is engineering state, not documentation.)

## ACTIVE MANDATE (2026-08-18, user-issued; extended to INDEFINITE)

Run the loop **indefinitely** (user: "keep continueing indefinetely"),
focused on: **output quality**, **efficiency**, and **maximization of
processing power**. Every cycle:
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
Candidates, roughly ranked:
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
