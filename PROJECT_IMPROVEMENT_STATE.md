# PromptForge — Continuous Improvement State

Working memory for the autonomous improvement loop. Updated after every
meaningful cycle. (User docs live in docs/PromptForge-Documentation.pdf —
this file is engineering state, not documentation.)

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
