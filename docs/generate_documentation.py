"""Generates docs/PromptForge-Documentation.pdf — the project's manual.

Run with the backend venv (reportlab ships with it):

    backend\\.venv\\Scripts\\python.exe docs\\generate_documentation.py

Everything in the PDF is grounded in the CODE as it exists today — the
facts below (routes, job types, templates, hardware numbers, ports) were
read out of the modules they describe, and the file paths named beside
each section say where to re-check them. Regenerate after changes that
move any of these facts.
"""
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from reportlab.graphics.shapes import (Drawing, Line, Polygon, Rect, String)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "PromptForge-Documentation.pdf"

INK = colors.HexColor("#1a2233")
ACCENT = colors.HexColor("#2455a4")
SOFT = colors.HexColor("#eef2f9")
EDGE = colors.HexColor("#8fa3c4")
GOOD = colors.HexColor("#2e7d4f")
WARN = colors.HexColor("#b25b12")

W = A4[0] - 30 * mm  # usable width with the margins below


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ------------------------------------------------------------- drawing kit --

def box(d, x, y, w, h, label, fill=SOFT, size=8.5, stroke=EDGE, bold=False):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke,
               strokeWidth=1, rx=3, ry=3))
    lines = label.split("\n")
    total = len(lines) * (size + 2)
    ty = y + h / 2 + total / 2 - size
    for line in lines:
        d.add(String(x + w / 2, ty, line, textAnchor="middle",
                     fontSize=size, fillColor=INK,
                     fontName="Helvetica-Bold" if bold else "Helvetica"))
        ty -= size + 2
    return (x, y, w, h)


def arrow(d, x1, y1, x2, y2, color=EDGE, width=1.2):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (0.42, -0.42):
        d.add(Line(x2, y2, x2 - 7 * math.cos(ang - da),
                   y2 - 7 * math.sin(ang - da),
                   strokeColor=color, strokeWidth=width))


def label(d, x, y, text, size=7.5, color=colors.HexColor("#5a6a85"),
          anchor="middle"):
    d.add(String(x, y, text, textAnchor=anchor, fontSize=size,
                 fillColor=color, fontName="Helvetica-Oblique"))


# ---------------------------------------------------------------- diagrams --

def d_architecture():
    d = Drawing(W, 250)
    box(d, 150, 218, 200, 28, "Browser UI  (React, served at :8000)",
        bold=True)
    arrow(d, 250, 218, 250, 196)
    box(d, 110, 164, 280, 32,
        "Flask API  backend/app/api/routes.py\n60 endpoints under /api/*",
        fill=colors.HexColor("#dfe9fa"), bold=True)
    arrow(d, 250, 164, 250, 142)
    box(d, 60, 88, 380, 54,
        "Services core  backend/app/core/services.py\n"
        "planning - masking - rendering - verification - jobs - models",
        fill=colors.HexColor("#d7e6d9"), bold=True)
    # adapters row
    box(d, 8, 26, 110, 44, "ComfyUI\n:8188\n(render engine)")
    box(d, 128, 26, 110, 44, "Ollama\n:11434\n(planner + critic LLM)")
    box(d, 248, 26, 110, 44, "SAM / CLIPSeg\n(in-process\nsegmentation)")
    box(d, 368, 26, 124, 44, "PeerService\n:8765 HTTP + :8766-69 UDP\n(LAN machines)")
    for cx in (63, 183, 303, 430):
        arrow(d, min(max(cx, 90), 410), 88, cx, 70)
    box(d, 8, 0, 484, 16,
        "data\\  -  models (typed folders), gallery, SQLite DB, logs - "
        "never touched by updates", size=7.5)
    return d


def d_launch_flow():
    d = Drawing(W, 120)
    steps = [
        ("git update\n+ re-exec", SOFT),
        ("Python venv\n(self-repair)", SOFT),
        ("UI build\n(if missing)", SOFT),
        ("Ollama up\n+ model pull", SOFT),
        ("ComfyUI find /\ninstall / repair", SOFT),
        ("VERIFY:\ntest render", colors.HexColor("#d7e6d9")),
        ("backend\n:8000", colors.HexColor("#dfe9fa")),
    ]
    x, bw, gap = 2, 66, 4
    for i, (txt, fill) in enumerate(steps):
        box(d, x, 62, bw, 40, txt, fill=fill, size=7.5, bold=(i >= 5))
        if i < len(steps) - 1:
            arrow(d, x + bw, 82, x + bw + gap, 82)
        x += bw + gap
    label(d, W / 2, 44,
          "every step self-repairs; a failed step degrades honestly instead "
          "of stopping the launch")
    box(d, 40, 4, 190, 26, "launch.bat / install.bat\nExecutionPolicy-proof entry",
        size=7.5)
    box(d, 270, 4, 190, 26, "PromptForge-Setup.bat\nGUI one-file installer",
        size=7.5)
    arrow(d, 135, 30, 105, 58)
    arrow(d, 365, 30, 395, 58)
    return d


def d_gpu_ladder():
    d = Drawing(W, 208)
    rungs = [
        ("NVIDIA (nvidia-smi answers, or adapter present)",
         "CUDA torch (cu126) + SageAttention + xformers", GOOD),
        ("AMD on the ROCm wheel list (RDNA3/4)",
         "AMD ROCm SDK + torch wheels  -  Python 3.12 only", GOOD),
        ("Intel Arc (discrete or Core Ultra)",
         "torch XPU wheels  -  checked before the AMD catch-all", GOOD),
        ("any other Radeon (RX 6000/5000, Ryzen iGPU)",
         "torch-directml 2.4.1 + ComfyUI pinned v0.30.2  -  Python 3.12",
         WARN),
        ("nothing usable",
         "CPU torch  -  slow renders, never the mock", WARN),
    ]
    y = 172
    for cond, act, tone in rungs:
        box(d, 0, y, 236, 30, cond, size=7.6)
        arrow(d, 236, y + 15, 252, y + 15)
        box(d, 252, y, 240, 30, act, size=7.6,
            fill=colors.Color(tone.red, tone.green, tone.blue, 0.12),
            stroke=tone)
        if y > 20:
            arrow(d, 118, y, 118, y - 8, width=0.8)
        y -= 38
    label(d, W / 2, 6,
          "first match wins - after install every stack is PROBED "
          "(torch.cuda / torch.xpu / torch_directml) and reported honestly")
    return d


def d_edit_pipeline():
    d = Drawing(W, 168)
    row1 = [("scene graph\n(one vision pass)", SOFT),
            ("plan: atomic\noperations (LLM)", SOFT),
            ("prune invented\nsteps", SOFT),
            ("per step:\nmask ladder", colors.HexColor("#dfe9fa")),
            ("render via\ntemplate/engine", colors.HexColor("#dfe9fa"))]
    x, bw, gap = 2, 92, 8
    for i, (txt, fill) in enumerate(row1):
        box(d, x, 118, bw, 40, txt, size=7.6, fill=fill)
        if i < len(row1) - 1:
            arrow(d, x + bw, 138, x + bw + gap, 138)
        x += bw + gap
    arrow(d, x - gap - bw / 2, 118, x - gap - bw / 2, 96)
    row2 = [("verify: seams,\nscorecard, adherence", colors.HexColor("#d7e6d9")),
            ("escalate: params ->\nmodel -> workflow", SOFT),
            ("keep BEST result\n+ recipe card", colors.HexColor("#d7e6d9"))]
    x = 96
    for i, (txt, fill) in enumerate(row2):
        box(d, x, 56, 118, 40, txt, size=7.6, fill=fill)
        if i < len(row2) - 1:
            arrow(d, x + 118, 76, x + 118 + 10, 76)
        x += 128
    label(d, W / 2, 38,
          "mask ladder: named body/clothing parts -> text-grounded (CLIPSeg "
          "worker) -> geometric SAM, each verified deterministically")
    label(d, W / 2, 24,
          "a mask that finds nothing is reported as 'not found here' - the "
          "app never silently edits the wrong pixels")
    label(d, W / 2, 10,
          "retries always re-run from the last step's input and can never "
          "undo the requested edit")
    return d


def d_queue_lanes():
    d = Drawing(W, 150)
    box(d, 150, 122, 200, 24, "one queue  (pause / move / cancel / retry)",
        bold=True)
    lanes = [
        ("MAIN worker", "renders, avatars, video,\nworkflows - one at a time",
         colors.HexColor("#dfe9fa")),
        ("PEER helper", "hands jobs to an idle LAN\nmachine when this one is busy;\nhonours hand-picked devices",
         SOFT),
        ("DOWNLOAD lane", "model fetches run BESIDE\nrenders and never count\nas busy anywhere",
         colors.HexColor("#d7e6d9")),
    ]
    x = 12
    for name, desc, fill in lanes:
        box(d, x, 58, 150, 22, name, fill=fill, bold=True, size=8)
        box(d, x, 8, 150, 46, desc, size=7.3, fill=colors.white)
        arrow(d, x + 75, 122, x + 75, 82)
        arrow(d, x + 75, 58, x + 75, 56)
        x += 162
    return d


def d_model_flow():
    d = Drawing(W, 120)
    box(d, 0, 84, 150, 30, "job needs a model\n(or a node pack)", size=7.6)
    arrow(d, 150, 99, 166, 99)
    box(d, 166, 84, 150, 30, "registry entry:\nSHA-256 pin + trusted host",
        size=7.6)
    arrow(d, 316, 99, 332, 99)
    box(d, 332, 84, 160, 30, "visible download job\n(own queue lane)",
        size=7.6, fill=colors.HexColor("#dfe9fa"))
    arrow(d, 412, 84, 412, 62)
    box(d, 260, 30, 110, 30, "1. LAN peer copy\n(same SHA only)", size=7.3,
        fill=colors.HexColor("#d7e6d9"))
    box(d, 382, 30, 110, 30, "2. internet source\n(HF / Civitai)", size=7.3)
    arrow(d, 370, 45, 382, 45)
    label(d, W / 2, 10,
          "bytes are checksum-verified either way; unknown or NSFW-flagged "
          "sources are never auto-installed (trust.py); missing Ollama "
          "models auto-pull; missing curated node packs auto-install")
    return d


def d_lan():
    d = Drawing(W, 158)
    box(d, 10, 96, 200, 52, "Machine A\nbackend :8000 (local only)\n"
        "peer listener :8765", bold=True)
    box(d, 290, 96, 200, 52, "Machine B\nbackend :8000 (local only)\n"
        "peer listener :8765", bold=True)
    arrow(d, 210, 132, 290, 132)
    arrow(d, 290, 118, 210, 118)
    label(d, 250, 138, "UDP beacons :8766-69 + subnet scan", size=7)
    label(d, 250, 106, "HTTP: models, renders, logs, version", size=7)
    rows = [
        "model copy: sha-pinned, LAN-first ('Send all models' / 'Ask for its models')",
        "render delegation: busy machine hands whole jobs to an idle peer; "
        "device picker pins a job to a chosen PC",
        "remote diagnosis: whitelisted logs + ComfyUI env readable from the "
        "healthy machine",
        "auto-update propagation: a peer running newer code triggers this "
        "machine's own update job when idle",
    ]
    y = 74
    for r in rows:
        box(d, 10, y, 480, 18, r, size=7.4, fill=colors.white)
        y -= 22
    return d


def d_updates():
    d = Drawing(W, 96)
    steps = [
        ("push to\nGitHub main", SOFT),
        ("launch: ff-only pull\n+ immediate re-exec", SOFT),
        ("Settings -> Updates\n(or peer-triggered job)", SOFT),
        ("deps/UI refresh\nonly when changed", SOFT),
        ("restart + health probe\nauto-ROLLBACK on failure",
         colors.HexColor("#d7e6d9")),
    ]
    x, bw, gap = 2, 94, 6
    for i, (txt, fill) in enumerate(steps):
        box(d, x, 40, bw, 40, txt, size=7.4, fill=fill)
        if i < len(steps) - 1:
            arrow(d, x + bw, 60, x + bw + gap, 60)
        x += bw + gap
    label(d, W / 2, 20,
          "data\\ is untracked - an update can never touch models, photos or "
          "the database; dirty local edits block updates by name")
    return d


# ---------------------------------------------------------------- document --

def build():
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=ACCENT,
                        spaceBefore=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK,
                        spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                          leading=13.5, textColor=INK)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11)
    src = ParagraphStyle("src", parent=small,
                         textColor=colors.HexColor("#5a6a85"),
                         fontName="Helvetica-Oblique", spaceBefore=2)

    def T(data, widths, header=True, fs=8):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), fs),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("GRID", (0, 0), (-1, -1), 0.4, EDGE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    story = []
    P = lambda txt, s=body: story.append(Paragraph(txt, s))  # noqa: E731
    S = lambda h=6: story.append(Spacer(1, h))  # noqa: E731

    # -- cover ---------------------------------------------------------------
    story.append(Spacer(1, 120))
    P("PromptForge", ParagraphStyle("t", parent=styles["Title"], fontSize=40,
                                    textColor=ACCENT))
    S(10)
    P("A fully local AI image, video and avatar studio for Windows",
      ParagraphStyle("st", parent=styles["Title"], fontSize=15,
                     textColor=INK))
    S(30)
    P(f"Complete documentation, generated from the code on {date.today()} "
      f"(commit {_commit()}).", small)
    P("Regenerate any time:  backend\\.venv\\Scripts\\python.exe "
      "docs\\generate_documentation.py", src)
    story.append(PageBreak())

    # -- 1 what it is --------------------------------------------------------
    P("1. What PromptForge is", h1)
    P("PromptForge turns plain-language requests — “change the shirt to "
      "a red jacket”, “make a video of this photo”, "
      "“build a 3D avatar from these pictures” — into real renders "
      "on your own hardware. A local LLM plans the work, a segmentation "
      "stack finds the exact pixels the words name, ComfyUI executes the "
      "render, and vision models judge the result and retry with a better "
      "strategy when it falls short. Nothing leaves the machine by default: "
      "prompts, photos, models and results all live in the local "
      "<b>data\\</b> folder.", body)
    S()
    P("Honesty is a design rule, enforced in code: mock output is always "
      "labelled, cloud LLM fallback is stamped on every reply that used it, "
      "quality scores come from measurements, and every generated image "
      "carries a recipe card recording exactly how it was made.", body)
    S()
    P("Capabilities (each is a routed pipeline, not a prompt trick): "
      "generation (SD1.5 / SDXL / Z-Image / Flux Kontext), region edits "
      "with automatic masks, background replacement, relighting, pose "
      "change, viewpoint change, style transfer, restoration, upscaling, "
      "outpainting, text-to-video and image-to-video (WAN 2.x), motion "
      "transfer, multi-image composition, photo-to-3D scenes, and full 3D "
      "avatars with photoreal texturing and rigging.", body)

    # -- 2 architecture ------------------------------------------------------
    P("2. System architecture", h1)
    story.append(d_architecture())
    P("Sources: backend/run.py, backend/app/api/routes.py, "
      "backend/app/core/*.py, frontend/src/", src)
    S()
    P("One Flask process serves both the built React UI and the JSON API on "
      "127.0.0.1:8000. The Services core owns all intelligence and talks to "
      "three engines: ComfyUI (rendering, auto-revived if it dies), Ollama "
      "(planning and image critique, auto-revived and auto-pulled), and an "
      "in-process segmentation stack (SAM plus a resident CLIPSeg worker "
      "running inside ComfyUI's Python). The PeerService is the only "
      "network-facing surface and exposes models, renders and logs to other "
      "PromptForge machines on the same LAN.", body)

    # -- 3 running it --------------------------------------------------------
    P("3. Installing and launching", h1)
    story.append(d_launch_flow())
    P("Sources: launch.ps1, launch.bat, install.bat, installer/*.ps1", src)
    S()
    P("Three equivalent entries, all safe on machines where PowerShell "
      "scripts are disabled (each runs with a per-process ExecutionPolicy "
      "bypass; nothing changes system settings):", body)
    story.append(T([
        ["Entry", "When to use it"],
        ["install.bat", "After unpacking a GitHub ZIP (connects the folder "
         "to the update channel, then launches) — or even as a lone file "
         "on an empty PC: it installs git (winget, or a direct Portable "
         "Git download), clones the repository and launches."],
        ["launch.bat", "Day-to-day start. Runs launch.ps1, which "
         "self-repairs everything on every start."],
        ["PromptForge-Setup.bat", "Single-file GUI installer built by "
         "installer\\build-installer.ps1 — pick a folder, press Install; "
         "no git or Node needed on the target, auto-starts when done."],
    ], [90, W - 90]))
    S()
    P("launch.ps1 installs anything missing: Python 3.12 (winget, falling "
      "back to python.org's own silent installer), Node for the UI build, "
      "Ollama (winget or ollama.com), git, ComfyUI itself (git clone first "
      "— its schannel TLS survives antivirus certificate interception — "
      "then a zip, with retries and full logs), and the right torch for "
      "the GPU. A script-level error net turns any remaining failure into "
      "one readable message plus the transcript path "
      "(data\\logs\\launch.log). The launcher then proves the install: a "
      "test graph must actually render before the backend is told ComfyUI "
      "is real — otherwise renders fall back to CPU mode, and only when "
      "even that fails does the clearly-labelled mock appear.", body)

    # -- 4 hardware ----------------------------------------------------------
    story.append(PageBreak())
    P("4. Hardware adaptation", h1)
    P("4.1 GPU decision ladder", h2)
    story.append(d_gpu_ladder())
    P("Sources: launch.ps1 Get-GpuMode / Repair-ComfyVenv, "
      "installer/installer.ps1, doctor.ps1", src)
    S()
    P("4.2 VRAM detection", h2)
    P("VRAM is read from nvidia-smi when it answers, otherwise from the "
      "display-class registry key the GPU driver itself writes "
      "(HardwareInformation.qwMemorySize) — so AMD and Intel cards report "
      "their true memory instead of 0 GB. The same probe runs in the "
      "launcher, the doctor and the backend (hardware.py), and feeds the "
      "planner-model choice, render budgets and the peer dashboard.", body)
    S()
    P("4.3 What the numbers decide", h2)
    story.append(T([
        ["Tier", "VRAM", "Max render", "Max steps", "Video frames",
         "Planner LLM"],
        ["high", ">= 16 GB", "1536 px", "60", "81 @ 768 px",
         "qwen2.5:14b (GPU or >= 32 GB RAM)"],
        ["mid", ">= 6 GB", "1280 px", "50", "81 @ 768 px",
         "qwen2.5:7b (with >= 12 GB RAM)"],
        ["low", "< 6 GB", "768 px", "40", "33 @ 512 px",
         "qwen2.5:3b (7b when RAM >= 32 GB)"],
    ], [45, 55, 65, 55, 75, W - 295]))
    P("Source: backend/app/core/hardware.py (tier, render_budget, "
      "llm_model_for) and the launcher's planner picker. Budgets both "
      "inform the planner and CLAMP generated graphs, so an "
      "over-ambitious workflow cannot hard-crash the GPU. Machines with "
      "20 GB RAM or less also drop ComfyUI's checkpoint cache between "
      "renders — measured as the difference between working video "
      "renders and OOM crashes on a 16 GB machine.", src)

    # -- 5 pipeline ----------------------------------------------------------
    P("5. The edit pipeline", h1)
    story.append(d_edit_pipeline())
    P("Sources: backend/app/core/services.py (_handle_image_edit), "
      "quality.py (plan_edit, mask_verdict, scorecard), scene_graph.py",
      src)
    S()
    P("Every request is compiled to atomic operations (ADD_OBJECT, "
      "REMOVE_OBJECT, REPLACE_OBJECT, CHANGE_ATTRIBUTE, CHANGE_STYLE, "
      "CHANGE_LIGHTING, CHANGE_POSE, CHANGE_CAMERA, CHANGE_TEXT, COMPOSE, "
      "OUTPAINT, UPSCALE, RESTORE, ANIMATE), each routed to its own "
      "engine. Compound requests are split so no half is silently "
      "dropped; steps the LLM invents are pruned deterministically. "
      "Masks are chosen by the best available evidence and validated "
      "with pure geometry (floor, ceiling, speckle, subject-leak against "
      "a BiRefNet matte) — the face is structurally protected. "
      "Verification measures seams and scores realism, prompt accuracy, "
      "identity, consistency and artifacts; failures escalate through "
      "parameters, then model, then workflow, and the best attempt is "
      "kept. Flux Kontext's silent refusals are detected by measuring "
      "whether the image changed at all, and rerouted to inpainting.",
      body)
    S()
    P("49 validated workflow templates back these routes "
      "(backend/app/workflows/): generation (draft/hires/XL/Z-Image/"
      "Kontext/portrait), img2img and style, inpaint (universal, hires, "
      "remove, replace-background), outpaint, upscale (model-based and "
      "creative), relight, pose, angles, compose, identity, face detail, "
      "restore, sketch-to-photo, video (t2v, i2v, inpaint, outpaint), "
      "motion transfer (standard and fast), 3D reconstruction and "
      "scene3d. The planner can also author new graphs, which are "
      "schema-checked against the LIVE ComfyUI node list before "
      "execution, with one self-repair round.", body)

    # -- 6 queue -------------------------------------------------------------
    story.append(PageBreak())
    P("6. Jobs and the three-lane queue", h1)
    story.append(d_queue_lanes())
    P("Source: backend/app/core/jobs.py", src)
    S()
    P("Twelve job types (image_edit, workflow, video, motion_transfer, "
      "avatar, avatar_render, model_download, model_research, node_pack, "
      "setup, discover, update) run through one visible queue with "
      "pause, reorder, cancel and retry. Failures retry with exponential "
      "backoff; transient engine deaths revive the engine and continue. "
      "Downloads live on their own lane so a multi-gigabyte fetch never "
      "delays a render — and a downloading machine still reports itself "
      "idle to peers. ETAs come from the median of past run times of the "
      "same kind.", body)

    # -- 7 models ------------------------------------------------------------
    P("7. Models: registry, search, trust", h1)
    story.append(d_model_flow())
    P("Sources: backend/app/core/registry.py, trust.py, model_search.py, "
      "model_intel.py, node_packs.py", src)
    S()
    P("35 curated registry entries cover checkpoints, video models, text "
      "encoders, VAEs, LoRAs, ControlNets, upscalers and segmentation "
      "weights, each with a SHA-256 pin and a typed destination folder "
      "that ComfyUI reads directly. When a prompt needs something new, "
      "the scout searches Hugging Face and Civitai, a trust layer "
      "vets the source (safetensors only, known hosts, pickle gate, "
      "adoption floor; NSFW-flagged sources are never auto-installed), "
      "and gated files self-heal to public mirrors only when the "
      "published hash matches byte-for-byte. A model knowledge file "
      "(data\\model_knowledge.json) records what each checkpoint is good "
      "at and steers model choice. Seven curated ComfyUI node packs "
      "(impact-pack, controlnet-aux, frame-interpolation, rmbg, "
      "ic-light, instantid, gguf) install as visible jobs — "
      "automatically when a request needs one — and are verified "
      "against the live node list after restart.", body)

    # -- 8 LAN ---------------------------------------------------------------
    P("8. Two or more machines: the LAN fabric", h1)
    story.append(d_lan())
    P("Source: backend/app/core/peers.py; firewall rules via "
      "allow-lan.ps1 (TCP 8765, UDP 8766-8769, private networks only)",
      src)
    S()
    P("Machines find each other with UDP beacons (multi-interface, "
      "directed broadcasts) plus an active subnet scan, and appear live "
      "in the UI rail with GPU, VRAM and RAM. The app itself stays on "
      "127.0.0.1 — only the peer listener is exposed. All transfers are "
      "checksum-verified against this machine's own pins; renders "
      "proxied to a busy peer wait politely instead of dying; a peer "
      "running newer code triggers this machine's own ordinary update "
      "job when idle.", body)

    # -- 9 updates -----------------------------------------------------------
    P("9. Updates", h1)
    story.append(d_updates())
    P("Sources: launch.ps1 section 0, backend/app/core/update.py", src)

    # -- 10 safety -----------------------------------------------------------
    P("10. Safety and privacy", h1)
    P("All policy lives in one user-owned module, "
      "backend/app/core/safety.py: it screens prompts, gates avatar "
      "creation behind explicit consent, and blocks NSFW-flagged model "
      "sources from auto-installing. Hard lines — minors, "
      "photo-undressing of real people, non-consensual imagery — are "
      "enforced regardless of any toggle. Settings offers a custom "
      "rule manager whose rules are read live on every check. "
      "Local-first is structural: the cloud LLM is only a fallback and "
      "every reply that used it is stamped as such.", body)

    # -- 11 API --------------------------------------------------------------
    story.append(PageBreak())
    P("11. API reference (all under /api on 127.0.0.1:8000)", h1)
    api_rows = [
        ["Area", "Endpoints"],
        ["Health & system", "GET /health - GET /system - GET /events - "
         "DELETE /events - DELETE /history/prompts"],
        ["Assets & gallery", "POST/GET /assets - GET /assets/{id}/file - "
         "DELETE /assets/{id} - POST /assets/{id}/restore - GET /gallery "
         "- DELETE /gallery - GET /versions/{id}/file - "
         "POST /versions/{id}/promote"],
        ["Editing", "POST /edits - POST /edits/plan - POST /masks/preview "
         "- POST /masks/point - GET /masks/status"],
        ["Generation & video", "POST /workflows/generate - "
         "POST /workflows/run - POST /video - POST /motion_transfer"],
        ["Avatars", "POST /avatar - GET /avatars - DELETE /avatars/{id} - "
         "POST /avatars/{id}/render"],
        ["Jobs & queue", "GET /jobs - GET /jobs/{id} - "
         "POST /jobs/{id}/cancel|retry|move - DELETE /jobs/{id} - "
         "POST /jobs/clear - GET /queue/state - POST /queue/pause|resume"],
        ["Models", "GET /models - POST /models/{name}/download - "
         "GET /models/search|files|index|civitai - "
         "POST /models/propose|propose-civitai - GET /nodepacks - "
         "POST /nodepacks/{name}/install"],
        ["LAN peers", "GET /peers - POST /peers/probe - "
         "POST /peers/push-models - POST /peers/fetch-models - "
         "GET /peers/log"],
        ["Updates & settings", "GET /update - POST /update/apply - "
         "GET/POST /settings - GET/POST /safety/rules - "
         "DELETE /safety/rules/{id}"],
        ["Workflow discovery", "POST /workflows/discover - "
         "POST /workflows/approve"],
    ]
    story.append(T([[Paragraph(c, small) for c in r] for r in api_rows],
                   [95, W - 95]))
    P("Source: backend/app/api/routes.py (60 routes)", src)

    # -- 12 configuration ----------------------------------------------------
    P("12. Configuration (environment variables)", h1)
    cfg = [
        ["Variable", "Meaning (default)"],
        ["PROMPTFORGE_DATA_DIR", "where models/photos/DB live (data\\)"],
        ["PROMPTFORGE_COMFYUI_DIR / _URL",
         "ComfyUI install dir (set by the launcher) / API url (:8188)"],
        ["PROMPTFORGE_LLM_URL / _MODEL / _API_MODEL",
         "local LLM endpoint (:11434), planner model, cloud fallback"],
        ["PROMPTFORGE_INPAINT_BACKEND / _SEGMENT_BACKEND",
         "comfyui|mock / sam|mock - mock means fully offline"],
        ["PROMPTFORGE_CRITIC_MODEL / _MIN / _RETRIES",
         "vision judge (llava), minimum score, retry count"],
        ["PROMPTFORGE_QUALITY_ROUNDS / _TARGET",
         "edit-loop iterations (2) and target score (95)"],
        ["PROMPTFORGE_ADHERENCE_ROUNDS / _TARGET",
         "prompt-adherence escalation ladder settings"],
        ["PROMPTFORGE_AUTO_INSTALL / _FIRST_RUN_SETUP",
         "allow automatic model/pack downloads (on) / first-run scout"],
        ["PROMPTFORGE_LAN_SHARE / _RENDER / _PEER_HOSTS / _PEER_AUTO_UPDATE",
         "LAN model sharing (on), accept peer renders (on), static peer "
         "list, follow newer peers (on)"],
        ["PROMPTFORGE_AUTO_UPDATE", "launcher git auto-update (on; 0 pins "
         "the machine to its current version)"],
        ["PROMPTFORGE_JOB_MAX_RETRIES / _JOB_BACKOFF_S",
         "queue retry policy"],
        ["PROMPTFORGE_MAX_UPLOAD_MB / _MAX_VIDEO_SECONDS",
         "upload and clip limits"],
        ["PROMPTFORGE_CIVITAI_TOKEN", "unlocks token-gated Civitai "
         "downloads (never logged)"],
        ["PROMPTFORGE_WORKFLOW_REPAIRS", "LLM graph self-repair rounds"],
    ]
    story.append(T([[Paragraph(c, small) for c in r] for r in cfg],
                   [175, W - 175]))
    P("Source: backend/app/config.py", src)

    # -- 13 layout -----------------------------------------------------------
    P("13. On-disk layout and logs", h1)
    story.append(T([
        ["Path", "What lives there"],
        ["backend\\", "Flask app + core (its .venv is created on first "
         "launch)"],
        ["frontend\\dist\\", "built UI the backend serves (rebuilt when "
         "sources change)"],
        ["tools\\ComfyUI\\", "auto-installed render engine (an existing "
         "install elsewhere is found and reused, including venv-only "
         "ROCm layouts)"],
        ["data\\models\\", "typed model folders ComfyUI loads directly "
         "(checkpoints, diffusion_models, text_encoders, vae, loras, "
         "controlnet, upscale_models, segmentation, ...)"],
        ["data\\gallery, data\\promptforge.db", "images/videos and the "
         "database (assets, jobs, experience, model knowledge)"],
        ["data\\logs\\", "launch.log (full launcher transcript), "
         "comfyui[-err|-install|-repair].log, directml-install.log, "
         "sage/xformers/sam-install.log, doctor-report.txt - the same "
         "set a LAN peer can read remotely for diagnosis"],
        ["doctor.ps1", "read-only end-to-end health check -> "
         "data\\logs\\doctor-report.txt"],
        ["allow-lan.ps1", "one-time firewall opening for the LAN "
         "features (run as admin on both PCs)"],
    ], [150, W - 150]))
    S(10)
    P("This document is generated by docs\\generate_documentation.py — "
      "if it disagrees with the code, regenerate it; if it still "
      "disagrees, the generator is the bug.", src)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="PromptForge Documentation",
        author="PromptForge")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#8fa3c4"))
        canvas.drawCentredString(A4[0] / 2, 8 * mm,
                                 f"PromptForge documentation - page "
                                 f"{canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"written: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
