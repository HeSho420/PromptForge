# PromptForge launcher — starts everything the app needs, then the app itself.
#
#   .\launch.ps1              # start dependencies + app, open the browser
#   .\launch.ps1 -NoBrowser   # same, but don't open a browser tab
#
# What it does, in order (each step self-repairs when it can):
#   1. Creates the Python venv + installs backend deps (first run only)
#   2. Builds the UI (first run only, needs Node 20+)
#   3. Starts Ollama if installed and not already running (local LLM)
#      and pulls the configured model if it is missing (with retries)
#   4. Starts ComfyUI if it can find an install (real rendering); if its
#      Python environment is broken or missing, it repairs it automatically
#   5. Frees port 8000 from a stale PromptForge instance, then starts the
#      server at http://127.0.0.1:8000
# Dependencies it started itself are stopped again when you close it.

param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$started = @()   # child processes we own and must clean up
$logDir = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

# Fresh installs (Node, Ollama) may not be on this session's PATH yet.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path

function Test-Http($url) {
    try {
        $req = [System.Net.WebRequest]::Create($url)
        $req.Timeout = 2000
        $resp = $req.GetResponse()
        $resp.Close()
        return $true
    } catch { return $false }
}

function Wait-Http($url, $seconds, $proc) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http $url) { return $true }
        if ($proc -and $proc.HasExited) { return $false }
        Start-Sleep -Milliseconds 700
    }
    return (Test-Http $url)
}

function Get-GpuMode {
    # Mirrors the installer's decision (keep in sync with installer.ps1):
    # working NVIDIA driver -> cuda; ROCm-capable Radeon -> rocm; else cpu.
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) {
            $smi = @{ Source = "$env:SystemRoot\System32\nvidia-smi.exe" }
        }
        if ($smi) { return "cuda" }
    } catch {}
    try {
        $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
                  ForEach-Object { $_.Name } | Where-Object { $_ })
    } catch { $gpus = @() }
    foreach ($n in $gpus) {
        if ($n -match "Radeon.+(RX\s?90\d0|RX\s?7900|RX\s?7800|RX\s?7700|8[89]0M|860M|80[456]0S)") {
            return "rocm"
        }
    }
    return "cpu"
}

# AMD ROCm-on-Windows wheels - same list as installer.ps1, Python 3.12 only.
$rocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
$rocmSdkUrls = @("rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                 "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                 "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                 "rocm-7.2.1.tar.gz") | ForEach-Object { $rocmBase + $_ }
$rocmTorchUrls = @("torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                   "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                   "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl") | ForEach-Object { $rocmBase + $_ }

function Get-TorchPipLines([string]$pipPath, [string]$logPath) {
    # Returns the pip command string(s) that install the right torch build
    # for this machine, for use inside a background powershell -Command.
    $mode = Get-GpuMode
    if ($mode -eq "cuda") {
        return ("& '$pipPath' install --retries 10 --timeout 180 torch torchvision " +
                "--index-url https://download.pytorch.org/whl/cu126 *> '$logPath'; ")
    }
    if ($mode -eq "rocm") {
        $sdk = ($rocmSdkUrls | ForEach-Object { "'" + $_ + "'" }) -join " "
        $trio = ($rocmTorchUrls | ForEach-Object { "'" + $_ + "'" }) -join " "
        return ("& '$pipPath' install --retries 10 --timeout 300 $sdk *> '$logPath'; " +
                "& '$pipPath' install --retries 10 --timeout 300 $trio *>> '$logPath'; ")
    }
    return ("& '$pipPath' install --retries 10 --timeout 180 torch torchvision *> '$logPath'; ")
}

function Test-PyImport([string]$py, [string]$mods) {
    # PS 5.1 landmine: a native command writing to a REDIRECTED stderr while
    # $ErrorActionPreference is "Stop" becomes a terminating
    # NativeCommandError. A plain `& $py ... 2>$null` probe therefore kills
    # the launcher on the EXACT machines where the module is missing and
    # self-repair should run. Relax the preference around the probe.
    if (-not (Test-Path $py)) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $py -c "import $mods" 2>$null | Out-Null } catch {}
    $ErrorActionPreference = $prev
    return ($LASTEXITCODE -eq 0)
}

Write-Host ""
Write-Host "  == PromptForge ==" -ForegroundColor Cyan
Write-Host ""

# --- 1. Python backend environment -------------------------------------------
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "  First run: creating the Python environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv (Join-Path $root "backend\.venv")
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv (Join-Path $root "backend\.venv")
    } else {
        throw "Python 3.12+ not found. Install it from python.org and re-run."
    }
}
# Self-repair: make sure backend deps import (cheap check, fixes broken installs).
if (-not (Test-PyImport $python "flask, PIL, imageio_ffmpeg")) {
    Write-Host "  Installing backend dependencies..." -ForegroundColor DarkGray
    & (Join-Path $root "backend\.venv\Scripts\pip.exe") install -q --retries 8 --timeout 120 `
        -r (Join-Path $root "backend\requirements.txt")
    if (-not (Test-PyImport $python "flask, PIL, imageio_ffmpeg")) {
        throw ("Backend dependencies did not install. Check the internet " +
               "connection and run PromptForge again.")
    }
}
# Real segmentation (SAM) needs torch — install in the background on first
# run so launch isn't blocked; masks work once it finishes (~2.5 GB, GPU build).
if (-not (Test-PyImport $python "torch, segment_anything")) {
    Write-Host "  Installing SAM segmentation stack in the background (one time)..." -ForegroundColor DarkGray
    $samLog = Join-Path $logDir "sam-install.log"
    $pipPath = Join-Path $root "backend\.venv\Scripts\pip.exe"
    Start-Process -WindowStyle Hidden powershell -ArgumentList "-NoProfile", "-Command", (
        (Get-TorchPipLines $pipPath $samLog) +
        "& '$pipPath' install --retries 10 --timeout 180 numpy " +
        "'https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip' *>> '$samLog'"
    ) | Out-Null
}

# --- 2. UI build ---------------------------------------------------------------
if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "  First run: building the UI (one time)..."
        Push-Location (Join-Path $root "frontend")
        try {
            if (-not (Test-Path "node_modules")) { npm install --no-fund --no-audit | Out-Null }
            npm run build | Out-Null
        } finally { Pop-Location }
    } else {
        Write-Warning "Node.js not found - running API-only. Install Node 20+ from nodejs.org for the UI."
    }
}

# --- 3. Ollama (local LLM that plans workflows) ---------------------------------
function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
        return "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    }
    return $null
}

$ollamaExe = Find-Ollama
if (-not $ollamaExe -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    # First run on a fresh machine: Ollama is the first dependency — the LLM
    # it hosts decides what else this machine needs.
    Write-Host "  Installing Ollama (local LLM host)..." -ForegroundColor Yellow
    winget install Ollama.Ollama --source winget --accept-package-agreements `
        --accept-source-agreements --disable-interactivity | Out-Null
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
    $ollamaExe = Find-Ollama
}

# Hardware-adaptive model choice: more VRAM/RAM -> bigger planning model.
$vramGb = 0.0
try {
    $vramMb = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
               Select-Object -First 1)
    if ($vramMb) { $vramGb = [math]::Round([double]$vramMb / 1024, 1) }
} catch {}
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

$llmModel = $env:PROMPTFORGE_LLM_MODEL
if (-not $llmModel) {
    if ($vramGb -ge 16) { $llmModel = "qwen2.5:14b" }
    elseif ($vramGb -ge 6 -and $ramGb -ge 12) { $llmModel = "qwen2.5:7b" }
    else { $llmModel = "qwen2.5:3b" }
    $env:PROMPTFORGE_LLM_MODEL = $llmModel   # backend inherits the choice
}
Write-Host "  Hardware: $vramGb GB VRAM, $ramGb GB RAM -> planning model $llmModel" -ForegroundColor DarkGray

if ($ollamaExe) {
    if (-not (Test-Http "http://127.0.0.1:11434/api/version")) {
        Write-Host "  Starting Ollama (local LLM)..." -ForegroundColor DarkGray
        $p = Start-Process -PassThru -WindowStyle Hidden $ollamaExe -ArgumentList "serve"
        $started += $p
        if (-not (Wait-Http "http://127.0.0.1:11434/api/version" 20 $p)) {
            # Rare: a zombie instance holds the port. Restart Ollama cleanly.
            Write-Host "  Ollama did not answer - restarting it..." -ForegroundColor Yellow
            Get-Process ollama -ErrorAction SilentlyContinue |
                Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
            $p = Start-Process -PassThru -WindowStyle Hidden $ollamaExe -ArgumentList "serve"
            $started += $p
            Wait-Http "http://127.0.0.1:11434/api/version" 20 $p | Out-Null
        }
    }
    if (Test-Http "http://127.0.0.1:11434/api/version") {
        Write-Host "  [ok] Ollama running" -ForegroundColor Green
        # Pull the model in the background (with retries) if it is missing.
        $have = cmd /c "`"$ollamaExe`" list 2>nul" | Out-String
        if ($have -notmatch [regex]::Escape($llmModel)) {
            Write-Host "  Downloading local model $llmModel in the background..." -ForegroundColor DarkGray
            $pullCmd = "for /l %i in (1,1,5) do (`"$ollamaExe`" pull $llmModel && exit /b 0)"
            Start-Process -WindowStyle Hidden cmd -ArgumentList "/c", $pullCmd | Out-Null
        }
    } else {
        Write-Host "  [--] Ollama installed but not answering - Forge falls back to the API." -ForegroundColor Yellow
    }
} else {
    Write-Host "  [--] Ollama not installed - workflow planning falls back to the API (needs ANTHROPIC_API_KEY)." -ForegroundColor Yellow
}

# --- 4. ComfyUI (real rendering), if an install can be found --------------------
function Find-ComfyUI {
    $candidates = @(
        $env:PROMPTFORGE_COMFYUI_PATH,
        (Join-Path $root "tools\ComfyUI"),
        "$env:USERPROFILE\ComfyUI",
        "$env:USERPROFILE\Documents\ComfyUI",
        "$env:USERPROFILE\Desktop\ComfyUI_windows_portable",
        "$env:USERPROFILE\Downloads\ComfyUI_windows_portable"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }
    return $null
}

function Repair-ComfyVenv($dir) {
    # Repo layout only: create/fix the venv so ComfyUI can actually start.
    $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        Write-Host "  ComfyUI has no Python environment - creating one..." -ForegroundColor Yellow
        if (Get-Command py -ErrorAction SilentlyContinue) {
            py -3 -m venv (Join-Path $dir ".venv")
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -m venv (Join-Path $dir ".venv")
        } else { return $false }
    }
    if (-not (Test-PyImport $venvPy "torch, yaml, aiohttp, requests")) {
        $repairLog = Join-Path $logDir "comfyui-repair.log"
        Write-Host "  ComfyUI's Python packages are broken/missing." -ForegroundColor Yellow
        Write-Host "  Repairing (GPU torch + requirements, several GB - one time)..." -ForegroundColor Yellow
        Write-Host "  Progress: $repairLog" -ForegroundColor DarkGray
        $pip = Join-Path $dir ".venv\Scripts\pip.exe"
        # pip writes warnings to stderr; with *> redirection under "Stop"
        # that would terminate the launcher mid-repair (same PS 5.1 quirk
        # as Test-PyImport). Relax while the repair runs.
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $mode = Get-GpuMode
        if ($mode -eq "cuda") {
            & $pip install --retries 10 --timeout 180 torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/cu126 *> $repairLog
        } elseif ($mode -eq "rocm") {
            & $pip install --retries 10 --timeout 300 @($rocmSdkUrls) *> $repairLog
            & $pip install --retries 10 --timeout 300 @($rocmTorchUrls) *>> $repairLog
        } else {
            & $pip install --retries 10 --timeout 180 torch torchvision torchaudio *> $repairLog
        }
        & $pip install --retries 10 --timeout 180 -r (Join-Path $dir "requirements.txt") *>> $repairLog
        $ErrorActionPreference = $prevEap
        if (-not (Test-PyImport $venvPy "torch, yaml, aiohttp, requests")) {
            Write-Host "  [--] Repair failed - see $repairLog. Continuing without ComfyUI." -ForegroundColor Yellow
            return $false
        }
        Write-Host "  ComfyUI environment repaired." -ForegroundColor Green
    }
    return $true
}

function Start-ComfyUI($dir) {
    $out = Join-Path $logDir "comfyui.log"
    $err = Join-Path $logDir "comfyui-err.log"
    # Portable build: <dir>\ComfyUI\main.py + <dir>\python_embeded\python.exe
    $portablePy = Join-Path $dir "python_embeded\python.exe"
    $portableMain = Join-Path $dir "ComfyUI\main.py"
    if ((Test-Path $portablePy) -and (Test-Path $portableMain)) {
        return Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory (Join-Path $dir "ComfyUI") `
            -RedirectStandardOutput $out -RedirectStandardError $err `
            $portablePy -ArgumentList "-s", $portableMain, "--listen", "127.0.0.1"
    }
    # Repo layout: <dir>\main.py with a venv
    $repoMain = Join-Path $dir "main.py"
    if (Test-Path $repoMain) {
        if (-not (Repair-ComfyVenv $dir)) { return $null }
        $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
        if (-not (Test-Path $venvPy)) { $venvPy = Join-Path $dir "venv\Scripts\python.exe" }
        if (Test-Path $venvPy) {
            # <=20 GB RAM: don't let ComfyUI cache checkpoints in RAM between
            # renders — cached leftovers under the WAN video stack OOM-kill
            # the process on 16 GB machines.
            $comfyArgs = @($repoMain, "--listen", "127.0.0.1")
            if ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB -le 20) {
                $comfyArgs += "--disable-smart-memory"
            }
            return Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $dir `
                -RedirectStandardOutput $out -RedirectStandardError $err `
                $venvPy -ArgumentList $comfyArgs
        }
    }
    return $null
}

$comfyDir = Find-ComfyUI
$comfyUp = Test-Http "http://127.0.0.1:8188/system_stats"
if (-not $comfyUp -and $comfyDir) {
    Write-Host "  Starting ComfyUI from $comfyDir ..." -ForegroundColor DarkGray
    # Let ComfyUI load checkpoints straight from PromptForge's model folder.
    $comfyBase = $comfyDir
    if (Test-Path (Join-Path $comfyDir "ComfyUI")) { $comfyBase = Join-Path $comfyDir "ComfyUI" }
    # Always rewrite our mapping: ComfyUI loads checkpoints, video models,
    # text encoders and VAEs straight from PromptForge's flat models folder.
    $extra = Join-Path $comfyBase "extra_model_paths.yaml"
    @"
promptforge:
  base_path: $root\data\models
  checkpoints: checkpoints
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  clip_vision: clip_vision
  loras: loras
  photomaker: photomaker
  upscale_models: upscale_models
  controlnet: controlnet
  instantid: instantid
  insightface: insightface
  sams: segmentation
  ultralytics: ultralytics
  rmbg: rmbg
"@ | Out-File -Encoding utf8 $extra
    $p = Start-ComfyUI $comfyDir
    if ($p) {
        $started += $p
        # First boot can take a while (model scanning, CUDA init).
        $comfyUp = Wait-Http "http://127.0.0.1:8188/system_stats" 90 $p
        if (-not $comfyUp -and $p.HasExited) {
            Write-Host "  ComfyUI crashed on start - last errors:" -ForegroundColor Yellow
            Get-Content (Join-Path $logDir "comfyui-err.log") -Tail 5 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        }
    }
}
if ($comfyDir) {
    # Backend uses this to restart ComfyUI automatically if it crashes.
    $env:PROMPTFORGE_COMFYUI_DIR = $comfyDir
}
if ($comfyUp) {
    $env:PROMPTFORGE_INPAINT_BACKEND = "comfyui"
    Write-Host "  [ok] ComfyUI running - real rendering enabled" -ForegroundColor Green
} else {
    Write-Host "  [--] ComfyUI not available - edits use the clearly-labeled mock renderer." -ForegroundColor Yellow
    Write-Host "       (Logs: $logDir\comfyui*.log - or set PROMPTFORGE_COMFYUI_PATH.)" -ForegroundColor DarkGray
}

# --- 5. PromptForge itself -------------------------------------------------------
# Free port 8000 if a stale PromptForge backend is still holding it.
if (Test-Http "http://127.0.0.1:8000/api/health") {
    Write-Host "  A PromptForge instance is already running on port 8000 - reusing it." -ForegroundColor Yellow
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:8000" }
    exit 0
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -like (Join-Path $root "backend\*") -or
                   $_.ExecutablePath -like "*promptforge\backend*" } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -Confirm:$false } catch {} }

Write-Host ""
Write-Host "  Starting PromptForge at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "  (close this window or press Ctrl+C to stop everything)" -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        $tries = 0
        while ($tries -lt 60) {
            try {
                $r = [System.Net.WebRequest]::Create("http://127.0.0.1:8000/api/health")
                $r.Timeout = 1500
                $r.GetResponse().Close()
                Start-Process "http://127.0.0.1:8000"
                break
            } catch { Start-Sleep -Milliseconds 500; $tries++ }
        }
    } | Out-Null
}

try {
    # Flask logs to stderr; under "Stop" + redirection PowerShell 5.1 would
    # promote that to a terminating error. Relax while the server runs.
    $ErrorActionPreference = "Continue"
    & $python (Join-Path $root "backend\run.py")
} finally {
    foreach ($p in $started) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -Confirm:$false } catch {}
        }
    }
    Write-Host "  PromptForge stopped." -ForegroundColor DarkGray
}
