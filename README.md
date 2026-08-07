# PromptForge

Local, prompt-based media editing with honest AI plumbing: describe an edit
("remove the chair", "change the sky to sunset"), review and correct the
proposed mask, run the render as a retryable job, and keep every before/after
version. The AI backends are adapters — a clearly-labeled mock adapter proves
the pipeline today, and a ComfyUI adapter (workflow templates, validation,
model registry) is wired in for real rendering.

## Architecture

```
frontend/  React + TypeScript + Vite      backend/  Python 3.12 + Flask
┌──────────────────────────┐              ┌────────────────────────────────────┐
│ Studio · Queue · Gallery │   /api ───▶  │ api/routes.py    thin HTTP layer   │
│ Models · Settings        │              │ core/services.py composition root  │
│ MaskEditor (rubylith)    │              │ ├─ safety.py    prompt filter      │
└──────────────────────────┘              │ ├─ jobs.py      queue + retries    │
                                          │ ├─ registry.py  models + checksums │
                                          │ ├─ storage.py   assets + versions  │
                                          │ └─ adapters/    mock · comfyui     │
                                          │ SQLite metadata · files on disk    │
                                          └────────────────────────────────────┘
```

Design rules the code follows:

- **The HTTP layer is thin.** `core/` and `adapters/` never import Flask, so
  swapping Flask for FastAPI is a one-file change in `api/`. (Flask was chosen
  for the MVP because it was the framework available in the offline build
  environment; the seam for FastAPI is deliberate.)
- **Backends are adapters.** `SegmentationAdapter` and `InpaintingAdapter`
  protocols in `adapters/base.py`. Every adapter declares `is_mock`; the API
  and UI surface that flag everywhere, so mock output can never pass as a real
  model result.
- **The queue is broker-shaped.** `JobQueue` (thread + queue.Queue) exposes
  enqueue/get/cancel/retry and per-job logs. Replacing it with Redis+RQ/Celery
  later means re-implementing that small surface, not touching call sites.
- **Originals are never overwritten.** Every edit is a new version row + file.

## Quick start from a clone

```powershell
git clone https://github.com/<you>/PromptForge.git
cd PromptForge
.\launch.ps1
```

**No models ship in this repository** — `data/` (models, your photos, the
database) is deliberately untracked. On a fresh clone the app rebuilds it:

- **First run**: a visible "setup" job profiles the machine (GPU, VRAM, RAM,
  disk) and pre-downloads a starter set of models sized to fit it.
- **On demand**: any job that needs a model the disk doesn't have downloads
  it first — every download is checksum-verified (SHA-256) and comes only
  from sources listed in the model registry (`core/services.py`
  DEFAULT_MODELS: Hugging Face and Civitai). Progress is visible in the
  Models tab and the job log.
- `launch.ps1` installs and self-repairs everything else: Python and
  Node.js (via winget) if the machine has neither, the backend venv with
  the right torch for your GPU brand (NVIDIA CUDA / AMD ROCm / CPU),
  the UI build, **ComfyUI itself** (downloaded into `tools\ComfyUI` when
  no install is found), and Ollama with a planner model sized to your
  hardware.

**Tuned to the machine, automatically.** The launcher prints a summary of
every decision: torch build by GPU brand, planner LLM size by VRAM/RAM
(3B/7B/14B), SageAttention installed and enabled on capable NVIDIA cards
(measured +11% render speed), checkpoint RAM caching on ≥20 GB machines
and disabled below (prevents OOM kills), and the backend scales the rest
per job — mesh detail, video resolution/length and model choices all
follow VRAM and RAM.

**Two PromptForge machines on one network help each other** (both default
on):

- *Model transfer* — a fresh install copies model weights from a peer's
  library instead of the internet (`PROMPTFORGE_LAN_SHARE=0` to disable).
  Only the model library is served, never photos or projects, and every
  copied file is verified against the registry's pinned SHA-256 before it
  is accepted.
- *Render delegation* — when this machine's queue is busy and a peer is
  idle, whole render jobs run on the peer's GPU through a proxy the peer
  controls; the peer refuses while it is doing its own work, and any
  failure falls back to rendering locally (`PROMPTFORGE_LAN_RENDER=0` to
  disable). Peers find each other automatically (UDP beacon on ports
  8766-8769, transfers on 8765) — allow PromptForge through the Windows
  firewall prompt on first run for this to work.

Expect roughly 2–10 GB of downloads for the starter set depending on your
GPU tier, and more as features are first used (the full library is ~100 GB
if you eventually use everything). Set `PROMPTFORGE_AUTO_INSTALL=0` to
forbid all automatic downloads, or `PROMPTFORGE_FIRST_RUN_SETUP=0` to skip
only the first-run pre-staging.

## Setup

**Windows (PowerShell)** — from the project folder (e.g. `C:\Users\you\Desktop\claude`):

```powershell
.\setup.ps1          # one-time setup, builds the UI, starts http://127.0.0.1:8000
.\setup.ps1 -Dev     # or: backend + hot-reloading Vite dev server on :5173
```

(If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, once.)

**Linux/macOS** — backend (Python 3.12+, only Flask and Pillow needed):

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                      # API on http://127.0.0.1:8000
```

Frontend (Node 20+):

```bash
cd frontend
npm install
npm run dev                        # UI on http://127.0.0.1:5173 (proxies /api)
```

`npm run build` produces `frontend/dist`, which the backend serves at `/`
automatically, so production use needs only `python run.py`.

## Running tests, lint, types

```bash
cd backend
python3 -m unittest discover -s tests -v   # 94 tests, no network needed
ruff check app tests                        # lint (pip install ruff)
mypy app                                    # types (pip install mypy)

cd frontend
npm run typecheck                           # strict tsc
```

Tests use the mock adapters and `file://` download sources, so the whole suite
runs offline in ~1 second. Nothing in the suite fakes a success state: the
mock adapters really execute, files are really written, checksums are really
verified.

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `PROMPTFORGE_DATA_DIR` | `./data` | assets, masks, models, SQLite DB |
| `PROMPTFORGE_INPAINT_BACKEND` | `comfyui` | real rendering by default; `mock` is opt-in for offline demos |
| `PROMPTFORGE_SEGMENT_BACKEND` | `sam` | real Segment Anything masks by default; `mock` is opt-in |
| `PROMPTFORGE_CRITIC_MODEL` | `llava` | Ollama vision model judging realism; `""` disables |
| `PROMPTFORGE_CRITIC_MIN` | `6` | realism score (1-10) below which strategy changes |
| `PROMPTFORGE_COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI server |
| `PROMPTFORGE_MAX_UPLOAD_MB` | `64` | upload size limit |
| `PROMPTFORGE_JOB_MAX_RETRIES` | `3` | transient-error retries |
| `PROMPTFORGE_JOB_BACKOFF_S` | `0.5` | base retry backoff (exponential) |
| `PROMPTFORGE_LLM_URL` | `http://127.0.0.1:11434/v1` | local LLM (OpenAI-compatible: Ollama/LM Studio) |
| `PROMPTFORGE_LLM_MODEL` | `qwen2.5:7b` | local model for workflow generation |
| `PROMPTFORGE_LLM_API_MODEL` | `claude-fable-5` | API fallback when the local LLM fails; `""` disables |
| `PROMPTFORGE_AUTO_INSTALL` | `1` | auto-download missing required models (registry sources, checksum-verified) |
| `PROMPTFORGE_WORKFLOW_REPAIRS` | `2` | LLM repair attempts for a failing workflow |
| `PROMPTFORGE_COMFYUI_PATH` | auto-detect | where the launcher looks for ComfyUI |

## Running it — one double-click

`launch.ps1` (or the **PromptForge** desktop shortcut) starts everything and
self-repairs what it can: it creates the Python env and builds the UI on
first run, starts Ollama and pulls the local model if missing (with retries),
finds and starts ComfyUI (repairing its Python environment automatically if
broken), frees port 8000 from stale instances, then serves
http://127.0.0.1:8000 and opens the browser. Everything it started is stopped
again when the window closes. Logs land in `data/logs/`.

## The Forge (prompt → planned → rendered → judged)

The Forge page turns a prompt into a rendered image end to end, and shows
each stage live (models → plan → render → realism check → save):

1. **Model scout**: an LLM picks the best-fitting installed checkpoint for
   the prompt — or searches Hugging Face and auto-downloads a better one
   (safetensors only, published checksum required, ≤ 8 GB).
2. The local LLM plans a ComfyUI graph, grounded in **live inventory** (the
   exact checkpoints ComfyUI reports having). Plans are validated against
   the node allowlist + per-node output indices before they may run.
3. **Missing models auto-install** (checksum-verified;
   `PROMPTFORGE_AUTO_INSTALL=0` disables). Models land in typed subfolders
   (`checkpoints/`, `diffusion_models/`, `vae/`, …) so ComfyUI only ever
   sees the right kind in each loader.
4. Before rendering, LLMs are unloaded from the shared GPU (Ollama
   `keep_alive:0`) so ComfyUI gets the full VRAM — on 8 GB cards this is the
   difference between rendering and crashing.
5. ComfyUI renders; node errors are fed back to the LLM for repair
   (`PROMPTFORGE_WORKFLOW_REPAIRS`).
6. **Realism check**: a local vision model (`PROMPTFORGE_CRITIC_MODEL`,
   default llava) scores the result 1-10. Below
   `PROMPTFORGE_CRITIC_MIN` (default 6) the pipeline *changes strategy* —
   different sampler/steps/cfg/prompt — and keeps the better result. The
   same gate runs on Studio inpaints.
7. Saved to the Gallery with provenance (planning model, attempts, repairs,
   realism score).

`POST /api/workflows/generate` returns the validated plan without running it;
`POST /api/workflows/run` does the full pipeline as a retryable job.

### It learns (workflow memory)

Every workflow run is recorded in the `workflow_memory` table: the graph,
success/failure, the realism score, repairs, and every ComfyUI error seen.
Future plans receive the best-scoring past graph for the same task as a
proven example plus recent errors as known pitfalls — the program gets
better with use, exactly as specified in the pipeline document.

### Click anything (SAM point select)

The Studio's mask editor has a **✦ Select object** tool: click any object
and SAM segments exactly what's under the cursor (`/api/masks/point`). The
image embedding is cached, so follow-up clicks answer in ~0.1 s. This is the
reliable way to select arbitrary objects; the prompt-based auto-mask remains
for hands-free flows.

### Avatar datasets (digital-human intake)

The **Avatar** page implements the intake stages of the digital-human
pipeline: upload 8+ consented photos → SAM isolates the subject → a local
vision model classifies each photo's viewing angle into 8 bins → missing
angles are synthesized with the SV3D multi-view template (labeled synthetic)
→ you get a coverage report. Consent is mandatory and enforced server-side.
3D reconstruction (Gaussian splatting + SMPL-X) builds on this dataset.

### Image → video (WAN 2.2)

The Forge's **Animate** tab turns an image into a short video using WAN 2.2
TI2V-5B through ComfyUI (chosen to fit an 8 GB GPU with RAM offload; output
is an animated WEBP saved to the Gallery). The three model files (~18 GB
total) auto-download on first use with checksums fetched from the hub's LFS
metadata. `POST /api/video {asset_id, prompt, length}`.

## AI workflow generation (local LLM, API fallback)

`POST /api/workflows/generate` `{task, prompt}` asks an LLM to plan a ComfyUI
workflow for the prompt and returns a validated graph plus provenance.

- **Local first.** The default backend is an OpenAI-compatible local server
  (Ollama or LM Studio at `PROMPTFORGE_LLM_URL`); prompts stay on this
  machine. `ollama pull qwen2.5:7b` (or set `PROMPTFORGE_LLM_MODEL`) to
  enable it.
- **API fallback, never silently.** Only when the local model fails, the
  Anthropic API (Claude Fable 5, with automatic re-serve via Claude Opus 4.8
  on a safety decline) is used — requires `pip install anthropic` and
  `ANTHROPIC_API_KEY`. Every reply is stamped `{"source": "local"|"api",
  "model": ...}` and that provenance is returned by the API. Set
  `PROMPTFORGE_LLM_API_MODEL=""` for fully local operation.
- **Checked every step of the way.** A generated graph must pass the same
  structural validation as the hand-written templates — node-type allowlist,
  link integrity, an output node, a size cap — before it is ever returned.
  Rejections are fed back to the LLM verbatim for a bounded number of repair
  attempts, and ComfyUI runtime errors can be repaired the same way. The
  allowlist is the trust boundary, not the model.

### Model search

`GET /api/models/search?q=...` searches the Hugging Face hub;
`GET /api/models/files?repo=...` lists a repo's weight files with their
published sha256; `POST /api/models/propose` stages one in the registry.
Nothing downloads until you click *Download* — the existing checksum-verified
downloader handles it, and files without a published checksum are flagged.

## Model configuration

The registry (Models page, `/api/models`) tracks name, purpose, **license
notes**, download URL, local path, sha256 and a VRAM estimate. Two entries are
pre-seeded with vetted sources and their published sha256 checksums — nothing
downloads until you click *Download* on the Models page:

| Model | Source | Size |
|---|---|---|
| `sd15-inpaint` | `huggingface.co/webui/stable-diffusion-inpainting` (safetensors mirror of the original runwayml weights) | 4.27 GB |
| `sam-vit-b` | `dl.fbaipublicfiles.com/segment_anything/` (official Meta checkpoint) | 375 MB |

To add or change a model, register it with `Services().registry.register(...)`
(see `app/core/services.py` `DEFAULT_MODELS` for the pattern). Review each
model's license before commercial use — the OpenRAIL-M license in particular
carries use restrictions.

### Connecting ComfyUI

The downloader stores files under `data/models/`. ComfyUI loads checkpoints
from its own `models/checkpoints/` folder, so either copy
`sd-v1-5-inpainting.safetensors` there, or point ComfyUI at PromptForge's
folder by adding it to ComfyUI's `extra_model_paths.yaml`. Then start the
backend with `PROMPTFORGE_INPAINT_BACKEND=comfyui`
(PowerShell: `$env:PROMPTFORGE_INPAINT_BACKEND="comfyui"` before `.\setup.ps1`).
A live end-to-end test exists at `tests/test_live_comfyui.py`
(`PROMPTFORGE_LIVE_COMFYUI=1` to enable); it verifies the mask convention and
that only masked pixels change.

### Real segmentation (SAM)

`PROMPTFORGE_SEGMENT_BACKEND=sam` switches mask proposals from the mock
heuristics to Segment Anything (ViT-B). It needs two things, both explicit:

1. `pip install -r backend/requirements-sam.txt` (torch + segment-anything —
   kept out of the core requirements because torch is large);
2. the `sam-vit-b` checkpoint downloaded from the Models page.

Missing either produces an actionable error (HTTP 409 on mask preview), never
a silent fallback to the mock. Honesty note: SAM proposes real object masks,
but SAM itself is not text-conditioned — the prompt selects among SAM's
candidates via transparent spatial/size priors (`select_candidate` in
`app/adapters/sam.py`, same keyword vocabulary as the mock). A learned
text-grounding stage (e.g. CLIP ranking) can later replace that function
behind the same signature. You always review/correct the proposed mask in the
editor before rendering. Without a GPU, expect roughly a minute per proposal
on CPU; images larger than 1024px are downscaled for segmentation and the
mask is scaled back.

Downloads stream to a temp file, hash while streaming, and verify the checksum
**before** the file may land in `models/`. A mismatch discards the file and
marks the model `checksum_failed`. Models without a published checksum are
allowed but the gap is visible in the registry — prefer sources that publish
hashes.

## Safety rules

- Every prompt passes `core/safety.py` **before** a job is created, for both
  edits and mask previews. Blocked categories: sexual/exposure edits,
  anything sexualizing minors, appearance edits of minors, non-consensual
  imagery, and deepfake-style identity manipulation (face swaps,
  impersonation). Blocked requests return HTTP 422 with the category code.
- The filter is rule-based (normalized text, word-boundary patterns, basic
  de-obfuscation) and fully unit-tested. It is a first gate, not a guarantee —
  the roadmap adds a classifier stage behind the same `SafetyVerdict`
  interface.
- Workflow generation refuses any task type not in `ALLOWED_TASKS`, and every
  workflow graph is structurally validated against an allowlist of node types
  before it may execute.
- The planned 3D avatar feature is consent-first by design: it will analyze
  dataset coverage and *report missing angles rather than hallucinating hidden
  body details*. See ROADMAP.md for the phased plan.

## Implemented vs prepared

| Area | Status |
|---|---|
| Image upload, prompt edit, auto-mask + manual mask, job queue, gallery, before/after, model registry + safe downloader, safety filter, logs | **Implemented and tested** |
| ComfyUI adapter | **Prepared**: templates, validation, registry checks and HTTP client are written and unit-tested; end-to-end rendering needs a running ComfyUI + downloaded models |
| SAM segmentation | **Prepared**: adapter, prompt grounding and registry gate are written and unit-tested; real proposals need `requirements-sam.txt` installed + the `sam-vit-b` checkpoint downloaded (see "Real segmentation (SAM)") |
| LLM workflow generation + model search | **Prepared**: generation/validation/repair engine, LLM fallback chain and HF search are written and unit-tested; live use needs a local LLM (Ollama) and/or `ANTHROPIC_API_KEY`; execution of generated graphs as jobs is roadmap Phase 2 |
| Video editing/generation | **Prepared**: job/queue/storage infrastructure is ready; frame pipeline + video templates are roadmap Phase 3 (ffmpeg + imageio are available) |
| 3D avatar | **Roadmap** (Phase 4) — ComfyUI cannot do 3D, so this uses COLMAP + photogrammetry/Gaussian splatting behind a `ReconstructionAdapter`, deliberately not stubbed with fake output |

## Troubleshooting

- **"Frontend not built" JSON at `/`** — run `npm run dev` (dev) or
  `npm run build` (serve from Flask).
- **Job stuck in `retrying` with "ComfyUI … unreachable"** — the `comfyui`
  backend is selected but no server is running; start ComfyUI or unset
  `PROMPTFORGE_INPAINT_BACKEND`.
- **`Required models are not downloaded`** — download them from the Models
  page after configuring URLs (above). This is a permanent job failure by
  design; use *Run again* after the download.
- **`Checksum mismatch … discarded`** — the source served a different file
  than the registry expects. Verify the URL/hash; never bypass the check.
- **Mask errors ("Mask is empty")** — paint a region or use *Propose mask*
  first; empty and size-mismatched masks are rejected before rendering.
- **Uploads rejected (415/413)** — check the extension allowlist and the
  `PROMPTFORGE_MAX_UPLOAD_MB` limit.
- Detailed per-job logs live in the Queue page; server logs go to stdout.
