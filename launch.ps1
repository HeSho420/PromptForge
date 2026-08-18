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
# Everything the launcher prints is mirrored to data\logs\launch.log — the
# peer log endpoint serves it, so a broken install on one machine can be
# diagnosed from another without pasting console output around. Fresh file
# per launch; stopped before the long-running server so it stays small.
try { Start-Transcript -Path (Join-Path $logDir "launch.log") -Force | Out-Null } catch {}

# Last-resort net: whatever still escapes ends as ONE readable message, the
# transcript path, and a window that stays open - never a raw stack trace.
trap {
    Write-Host ""
    Write-Host "  [X] PromptForge hit a problem it could not fix by itself:" -ForegroundColor Red
    Write-Host ("      " + $_.Exception.Message) -ForegroundColor Yellow
    Write-Host "      Details: $logDir\launch.log" -ForegroundColor DarkGray
    Write-Host "      Run launch.bat again - most problems self-repair on the next try." -ForegroundColor Yellow
    try { Stop-Transcript | Out-Null } catch {}
    try { Read-Host "  Press Enter to close" | Out-Null } catch {}
    exit 1
}

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
    # One mode per machine, best hardware first:
    #   cuda      NVIDIA (nvidia-smi answers, or the adapter is present -
    #             the CUDA build self-heals the moment the driver works)
    #   rocm      Radeon on AMD's ROCm-wheel list (RDNA3/4)
    #   directml  any OTHER Radeon - torch-directml gives real GPU
    #             rendering on every DX12 AMD card (RDNA2 included)
    #   xpu       Intel Arc (discrete or Core Ultra) - native torch XPU
    #   cpu       nothing usable
    # Memoized: callers hit this ~15 times per launch and CIM is not free.
    if ($script:pfGpuMode) { return $script:pfGpuMode }
    $mode = $null
    # nvidia-smi must actually ANSWER: a stale exe left in System32 after a
    # GPU swap must not force the CUDA path on a non-NVIDIA machine.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) {
            $smi = @{ Source = "$env:SystemRoot\System32\nvidia-smi.exe" }
        }
        if ($smi) {
            $ans = (& $smi.Source --query-gpu=name --format=csv,noheader 2>$null |
                    Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and $ans) { $mode = "cuda" }
        }
    } catch {}
    $ErrorActionPreference = $prev
    if (-not $mode) {
        try {
            $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
                      ForEach-Object { $_.Name } | Where-Object { $_ })
        } catch { $gpus = @() }
        foreach ($n in $gpus) {
            # NVIDIA adapter with a broken driver: still choose cuda - the
            # CUDA torch renders on CPU until the driver is fixed, then the
            # GPU starts working with NO reinstall (Repair-CudaTorch also
            # force-repairs a CPU-only torch on these machines).
            if ($n -match "NVIDIA|GeForce|Quadro|RTX \d") { $mode = "cuda"; break }
        }
        if (-not $mode) {
            foreach ($n in $gpus) {
                if ($n -match "Radeon.+(RX\s?90\d0|RX\s?7900|RX\s?7800|RX\s?7700|8[89]0M|860M|80[456]0S)") {
                    $mode = "rocm"; break
                }
            }
        }
        if (-not $mode) {
            # Intel Arc (A/B-series discrete and the Arc iGPU in Core
            # Ultra) has native torch XPU wheels on Windows - current
            # torch, current ComfyUI. Checked BEFORE the generic AMD
            # catch-all: a Ryzen desktop iGPU enumerates as 'AMD
            # Radeon(TM) Graphics', and an Arc card next to it must win
            # over the frozen DirectML stack. Older Iris Xe/UHD iGPUs
            # are NOT covered by XPU - those stay on CPU.
            foreach ($n in $gpus) {
                if ($n -match "Intel.+Arc") { $mode = "xpu"; break }
            }
        }
        if (-not $mode) {
            foreach ($n in $gpus) {
                if ($n -match "Radeon|AMD") { $mode = "directml"; break }
            }
        }
        if (-not $mode) { $mode = "cpu" }
    }
    $script:pfGpuMode = $mode
    return $mode
}

function Get-GpuVramGb {
    # Real VRAM regardless of vendor. nvidia-smi first (authoritative);
    # otherwise the display-class registry key the driver itself writes
    # (HardwareInformation.qwMemorySize) - WMI's AdapterRAM is a 32-bit
    # relic that caps at 4 GB. This is what stops '0 GB VRAM' appearing on
    # machines with a perfectly good AMD or Intel card.
    if ($script:pfVramGb -ne $null) { return $script:pfVramGb }
    $gb = 0.0
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $mb = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
               Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $mb) { $gb = [math]::Round([double]("$mb".Trim()) / 1024, 1) }
    } catch {}
    $ErrorActionPreference = $prev
    if ($gb -le 0) {
        try {
            $cls = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            $best = [long]0
            Get-ChildItem $cls -ErrorAction SilentlyContinue |
                Where-Object { $_.PSChildName -match '^\d{4}$' } |
                ForEach-Object {
                    $v = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue)."HardwareInformation.qwMemorySize"
                    if ($v -and [long]$v -gt $best) { $best = [long]$v }
                }
            if ($best -gt 0) { $gb = [math]::Round($best / 1GB, 1) }
        } catch {}
    }
    $script:pfVramGb = $gb
    return $gb
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
    if ($mode -eq "directml") {
        # PINNED on purpose: torch-directml hard-requires torch==2.4.1
        # while an unpinned torchvision resolves to a build that demands a
        # newer torch - pip then fails the whole install (measured live on
        # the RX 6700 XT machine as "torch-directml missing").
        return ("& '$pipPath' install --retries 10 --timeout 300 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 *> '$logPath'; " +
                "& '$pipPath' install --retries 10 --timeout 300 torch-directml *>> '$logPath'; ")
    }
    if ($mode -eq "xpu") {
        # Intel Arc: native torch XPU wheels (torch.xpu device).
        return ("& '$pipPath' install --retries 10 --timeout 300 torch torchvision " +
                "--index-url https://download.pytorch.org/whl/xpu *> '$logPath'; ")
    }
    return ("& '$pipPath' install --retries 10 --timeout 180 torch torchvision *> '$logPath'; ")
}

function Get-PyVersion([string]$py) {
    # "3.12" / "3.13" of a venv's python, or "" - EAP relaxed around the
    # native call (the usual PS 5.1 stderr landmine).
    if (-not (Test-Path $py)) { return "" }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $v = ""
    try {
        $v = (& $py -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null | Out-String).Trim()
    } catch {}
    $ErrorActionPreference = $prev
    return $v
}

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
}

function Install-PythonDirect([string]$version = "3.12.10") {
    # winget is missing or broken on plenty of machines (LTSC, old Win10,
    # corporate images). python.org's own silent installer needs nothing:
    # per-user, no admin, adds the py launcher. Returns $true on success.
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
    $setup = Join-Path $env:TEMP "pf-python-setup.exe"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        Write-Host "  Downloading Python $version from python.org (one time)..." -ForegroundColor Yellow
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 300 -Uri $url -OutFile $setup
        if ((Get-Item $setup).Length -lt 10MB) { return $false }
        $p = Start-Process -PassThru -Wait $setup -ArgumentList `
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "Include_test=0"
        Update-SessionPath
        return ($p.ExitCode -eq 0)
    } catch {
        return $false
    } finally {
        Remove-Item $setup -Force -ErrorAction SilentlyContinue
        $ErrorActionPreference = $prev
    }
}

function Install-Tool([string]$wingetId, [string]$label) {
    # Quiet winget install + PATH refresh; $false when winget is absent or
    # the install failed - callers always have a fallback or degrade path.
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { return $false }
    Write-Host "  Installing $label (one time)..." -ForegroundColor Yellow
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    winget install --id $wingetId -e --source winget --accept-package-agreements `
        --accept-source-agreements --disable-interactivity 2>&1 | Out-Null
    $ErrorActionPreference = $prev
    Update-SessionPath
    return $true
}

function Get-Python312 {
    # AMD's ROCm-on-Windows torch wheels exist for Python 3.12 ONLY, so on
    # AMD machines every venv must be 3.12. Returns @("py","-3.12") when
    # available (installing it via winget when not), else $null.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $have = ""
    try { $have = (& py -3.12 -c "print(1)" 2>$null | Out-String).Trim() } catch {}
    $ErrorActionPreference = $prev
    if ($have -eq "1") { return @("py", "-3.12") }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  Installing Python 3.12 (required for AMD GPU rendering)..." -ForegroundColor Yellow
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        winget install Python.Python.3.12 --source winget --accept-package-agreements `
            --accept-source-agreements --disable-interactivity 2>&1 | Out-Null
        $ErrorActionPreference = $prev
        Update-SessionPath
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { $have = (& py -3.12 -c "print(1)" 2>$null | Out-String).Trim() } catch {}
        $ErrorActionPreference = $prev
        if ($have -eq "1") { return @("py", "-3.12") }
    }
    # winget missing or failed: python.org's own silent installer.
    if (Install-PythonDirect) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { $have = (& py -3.12 -c "print(1)" 2>$null | Out-String).Trim() } catch {}
        $ErrorActionPreference = $prev
        if ($have -eq "1") { return @("py", "-3.12") }
    }
    return $null
}

function Get-AmdGpuName {
    try {
        return (Get-CimInstance Win32_VideoController -ErrorAction Stop |
                Where-Object { $_.Name -match "Radeon|AMD" } |
                Select-Object -First 1 -ExpandProperty Name)
    } catch { return $null }
}

function Get-AmdGfxArch {
    # The gfx architecture id for AMD's NATIVE multi-arch torch channel
    # (rocm.nightlies.amd.com/whl-multi-arch: an arch-generic torch 2.9
    # plus a per-card amd-torch-device-<gfx> kernel package). A FULL torch
    # with every op — WAN video and Flux Kontext included — on cards the
    # official ROCm wheel list skips; proven on this household's RX 6700
    # XT, which rendered videos on exactly this stack. Only unambiguous
    # discrete names map; iGPUs stay on DirectML.
    $n = Get-AmdGpuName
    if (-not $n) { return $null }
    if ($n -match "RX\s?6[89]\d0") { return "gfx1030" }
    if ($n -match "RX\s?67\d0") { return "gfx1031" }
    if ($n -match "RX\s?66\d0") { return "gfx1032" }
    if ($n -match "RX\s?6[45]\d0") { return "gfx1034" }
    if ($n -match "RX\s?76\d0") { return "gfx1102" }
    if ($n -match "RX\s?5[67]\d0") { return "gfx1010" }
    if ($n -match "RX\s?55\d0") { return "gfx1012" }
    return $null
}

function Install-RocmNative([string]$dir, [string]$arch) {
    # Try AMD's native multi-arch stack in this venv. TRUE only when torch
    # afterwards actually SEES the GPU; anything less is rolled back by
    # the caller. Nightly channel — every failure path lands back on the
    # proven DirectML pins, so this can only ever upgrade a machine.
    $pipN = Join-Path $dir ".venv\Scripts\pip.exe"
    $pyN = Join-Path $dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $pipN)) { return $false }
    $log = Join-Path $logDir "rocm-native-install.log"
    Write-Host "  Trying AMD's native ROCm torch for this card ($arch) -" -ForegroundColor Yellow
    Write-Host "  a full GPU stack that also runs WAN video and Kontext (several GB, one time)..." -ForegroundColor Yellow
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pipN install --pre --retries 10 --timeout 600 `
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
        torch torchvision torchaudio "amd-torch-device-$arch" *> $log
    $sees = (& $pyN -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
    $ErrorActionPreference = $prev
    if ($sees -eq "1") {
        Write-Host "  [ok] Native ROCm torch is live - $((Get-AmdGpuName)) now renders EVERYTHING, video included" -ForegroundColor Green
        return $true
    }
    Write-Host "  Native ROCm did not take on this machine (see $log) - staying on DirectML (images render fine there)." -ForegroundColor Yellow
    return $false
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
    # 2>&1 (not 2>$null): under 5.1 the first stderr line otherwise still
    # surfaces as a NativeCommandError record on the console/transcript.
    try { & $py -c "import $mods" 2>&1 | Out-Null } catch {}
    $ErrorActionPreference = $prev
    return ($LASTEXITCODE -eq 0)
}

Write-Host ""
Write-Host "  == PromptForge ==" -ForegroundColor Cyan
Write-Host ""

# git powers the update channel and the most reliable ComfyUI download
# (schannel TLS). Best effort - everything has a git-less fallback.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Install-Tool "Git.Git" "git (updates + reliable downloads)" | Out-Null
}

# --- 0. Updates pushed through git -----------------------------------------------
# Push to the repository and every install picks it up here on next launch.
# Fast-forward only, refused when local edits exist, and data/ (models,
# photos, database) is untracked so an update can never touch it. Runs
# BEFORE the self-repair steps so new dependencies install in this same run.
# Set PROMPTFORGE_AUTO_UPDATE=0 to keep a machine on its current version.
$pfSelfUpdated = $false
if (($env:PROMPTFORGE_AUTO_UPDATE -ne "0") -and
    (Test-Path (Join-Path $root ".git")) -and
    (Get-Command git -ErrorAction SilentlyContinue)) {
    Push-Location $root
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # git talks on stderr; PS 5.1 landmine
    try {
        git fetch --quiet origin 2>$null
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
        $behind = (git rev-list --count "HEAD..origin/$branch" 2>$null | Out-String).Trim()
        $dirty  = (git status --porcelain --untracked-files=no 2>$null | Out-String).Trim()
        if ($behind -match '^\d+$' -and [int]$behind -gt 0) {
            if ($dirty) {
                Write-Host "  [--] $behind update(s) pushed, but local file edits block them (see git status)." -ForegroundColor Yellow
            } else {
                Write-Host "  Updating: $behind new commit(s) were pushed..." -ForegroundColor Cyan
                $wasAt = (git rev-parse HEAD 2>$null | Out-String).Trim()
                git pull --ff-only origin $branch 2>&1 | Out-Null
                $nowAt = (git rev-parse HEAD 2>$null | Out-String).Trim()
                $now = (git rev-parse --short HEAD 2>$null | Out-String).Trim()
                Write-Host "  [ok] Now on $now" -ForegroundColor Green
                if ($nowAt -and $wasAt -and $nowAt -ne $wasAt) { $pfSelfUpdated = $true }
            }
        }
    } catch {} finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
}
if ($pfSelfUpdated -and $env:PROMPTFORGE_RELAUNCHED -ne "1") {
    # The files on disk are new, but THIS process still runs the parse it
    # started with - launcher fixes were applying one launch LATE (bit us
    # live: a crash-repair shipped and verified, yet the very next launch
    # executed the previous version and hit the old crash). Hand over to
    # the updated script immediately; the env flag prevents any loop.
    Write-Host "  Restarting the launcher on the updated version..." -ForegroundColor Cyan
    $env:PROMPTFORGE_RELAUNCHED = "1"
    try { Stop-Transcript | Out-Null } catch {}
    $argList = @()
    if ($NoBrowser) { $argList += "-NoBrowser" }
    & powershell.exe -ExecutionPolicy Bypass -File $PSCommandPath @argList
    exit $LASTEXITCODE
}

# --- 1. Python backend environment -------------------------------------------
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
# AMD machines: the GPU torch wheels are Python-3.12-only, so a venv built
# with any other version can NEVER install them — measured as a machine
# that quietly ran the mock renderer. Rebuild wrong-version venvs.
$gpuModeEarly = Get-GpuMode
# rocm AND directml: both AMD GPU stacks are Python-3.12-only (ROCm wheels
# are cp312; torch-directml is frozen at torch 2.4.1 which stops at 3.12).
if ($gpuModeEarly -in @("rocm", "directml") -and (Test-Path $python)) {
    $backendVer = Get-PyVersion $python
    if ($backendVer -and $backendVer -ne "3.12") {
        Write-Host "  This AMD machine needs Python 3.12 for GPU work (environment is $backendVer) - rebuilding..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force (Join-Path $root "backend\.venv")
    }
}
if (-not (Test-Path $python)) {
    Write-Host "  First run: creating the Python environment..."
    $made = $false
    if ($gpuModeEarly -in @("rocm", "directml")) {
        $pl = Get-Python312
        if ($pl) {
            & $pl[0] $pl[1] -m venv (Join-Path $root "backend\.venv")
            $made = $true
        } else {
            Write-Host "  [--] Python 3.12 unavailable - AMD GPU stack cannot install; continuing on CPU." -ForegroundColor Yellow
        }
    }
    if (-not $made) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue) -and
            -not (Get-Command python -ErrorAction SilentlyContinue)) {
            # A fresh clone on a fresh machine: install Python silently
            # rather than sending the user to a website. winget first,
            # python.org's own silent installer when winget is missing or
            # fails - EVERY machine ends up with a working Python.
            Install-Tool "Python.Python.3.12" "Python 3.12" | Out-Null
            if (-not (Get-Command py -ErrorAction SilentlyContinue) -and
                -not (Get-Command python -ErrorAction SilentlyContinue)) {
                Install-PythonDirect | Out-Null
            }
        }
        if (Get-Command py -ErrorAction SilentlyContinue) {
            py -3 -m venv (Join-Path $root "backend\.venv")
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -m venv (Join-Path $root "backend\.venv")
        } else {
            throw ("Python could not be installed automatically (winget and " +
                   "the python.org download both failed - is this PC online?). " +
                   "Install Python 3.12 from python.org, then run launch.bat again.")
        }
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
# Build when the bundle is MISSING or STALE. Staleness matters as much as
# absence: the launcher's own git pull updates frontend\src on disk but the
# old check only looked for dist\index.html existing, so updated machines
# kept serving the previous UI forever (new panels simply never appeared).
$uiDist = Join-Path $root "frontend\dist\index.html"
$uiBuild = -not (Test-Path $uiDist)
if (-not $uiBuild) {
    try {
        $distTime = (Get-Item $uiDist).LastWriteTime
        $newestSrc = (Get-ChildItem (Join-Path $root "frontend\src") -Recurse -File |
                      Measure-Object -Property LastWriteTime -Maximum).Maximum
        if ($newestSrc -and $newestSrc -gt $distTime) { $uiBuild = $true }
    } catch { }
}
if ($uiBuild) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue) -and
        (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing Node.js LTS for the UI (one time)..." -ForegroundColor Yellow
        winget install OpenJS.NodeJS.LTS --source winget --accept-package-agreements `
            --accept-source-agreements --disable-interactivity | Out-Null
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                    [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
    }
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "  Building the UI (missing or older than the code)..."
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
if (-not $ollamaExe) {
    # First run on a fresh machine: winget first, then Ollama's own silent
    # installer for machines where winget is missing or broken.
    Install-Tool "Ollama.Ollama" "Ollama (local LLM host)" | Out-Null
    $ollamaExe = Find-Ollama
    if (-not $ollamaExe) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            Write-Host "  Downloading Ollama from ollama.com (one time)..." -ForegroundColor Yellow
            $setup = Join-Path $env:TEMP "pf-ollama-setup.exe"
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 `
                -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $setup
            if ((Get-Item $setup).Length -gt 50MB) {
                Start-Process -Wait $setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" | Out-Null
            }
            Remove-Item $setup -Force -ErrorAction SilentlyContinue
        } catch {}
        $ErrorActionPreference = $prev
        Update-SessionPath
        $ollamaExe = Find-Ollama
    }
}

# Hardware-adaptive model choice: more VRAM/RAM -> bigger planning model.
# Get-GpuVramGb reads the registry when nvidia-smi is absent, so AMD and
# Intel machines size by their REAL VRAM instead of a false zero.
$vramGb = Get-GpuVramGb
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

$llmModel = $env:PROMPTFORGE_LLM_MODEL
if (-not $llmModel) {
    # 14b needs the GPU (Ollama/CUDA) or serious RAM to run at usable speed.
    if ($vramGb -ge 16 -and ((Get-GpuMode) -eq "cuda" -or $ramGb -ge 32)) { $llmModel = "qwen2.5:14b" }
    elseif ($vramGb -ge 6 -and $ramGb -ge 12) { $llmModel = "qwen2.5:7b" }
    elseif ($ramGb -ge 32) { $llmModel = "qwen2.5:7b" }
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
# torch-directml is frozen at torch 2.4.1; newer ComfyUI releases pull a
# comfy-kitchen whose neighborhood-attention custom op torch 2.4.1 cannot
# even import (infer_schema crash, reproduced on a real 2.4.1 env).
# Measured boundary: v0.30.2 PASSES on 2.4.1, v0.31.x FAILS.
$comfyDmlTag = "v0.30.2"

function Find-ComfyUI {
    # Only a RUNNABLE layout counts: a folder that merely exists (models
    # dump, nested clone, leftovers) made the starter bail silently and
    # blocked the auto-download - measured live as an AMD machine whose
    # repair and DirectML code never even ran.
    $candidates = @(
        $env:PROMPTFORGE_COMFYUI_PATH,
        (Join-Path $root "tools\ComfyUI"),
        "$env:USERPROFILE\ComfyUI",
        "$env:USERPROFILE\Documents\ComfyUI",
        "$env:USERPROFILE\Desktop\ComfyUI_windows_portable",
        "$env:USERPROFILE\Downloads\ComfyUI_windows_portable"
    )
    foreach ($c in $candidates) {
        if (-not ($c -and (Test-Path $c))) { continue }
        $p = (Resolve-Path $c).Path
        $portable = (Test-Path (Join-Path $p "python_embeded\python.exe")) -and
                    (Test-Path (Join-Path $p "ComfyUI\main.py"))
        $repo = Test-Path (Join-Path $p "main.py")
        # Nested: a venv at the top and ComfyUI cloned into a subfolder —
        # the layout AMD's own ROCm guides produce, found live holding a
        # WORKING GPU torch that the old checks walked straight past.
        $nested = (Test-Path (Join-Path $p ".venv\Scripts\python.exe")) -and
                  (Test-Path (Join-Path $p "ComfyUI\main.py"))
        if ($portable -or $repo -or $nested) { return $p }
        Write-Host "  (skipping $p - a folder, but not a runnable ComfyUI)" -ForegroundColor DarkGray
    }
    return $null
}

function Test-NativeGpuVenv($venvPy) {
    # TRUE when this venv's torch sees a GPU natively (CUDA, or a ROCm-SDK
    # build reporting through the same API). Costs seconds on a cold torch,
    # so callers gate it to AMD machines.
    if (-not ($venvPy -and (Test-Path $venvPy))) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $sees = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
    $ErrorActionPreference = $prev
    return ($sees -eq "1")
}

function Get-NativeGpuComfyHome {
    # DirectML machines only: a RUNNABLE install whose venv torch already
    # sees the GPU natively outranks every other candidate. DirectML is
    # frozen on torch 2.4.1, which current ComfyUI can no longer import
    # (RegisterOperators schema crash, measured live on an RX 6700 XT),
    # while a native ROCm-SDK stack runs master. Never let a stale tools
    # install shadow a working native one.
    foreach ($c in @($env:PROMPTFORGE_COMFYUI_PATH,
                     "$env:USERPROFILE\ComfyUI",
                     "$env:USERPROFILE\Documents\ComfyUI",
                     (Join-Path $root "tools\ComfyUI"))) {
        if (-not ($c -and (Test-Path $c))) { continue }
        $p = (Resolve-Path $c).Path
        $runnable = (Test-Path (Join-Path $p "main.py")) -or
                    (Test-Path (Join-Path $p "ComfyUI\main.py"))
        if (-not $runnable) { continue }
        if (Test-NativeGpuVenv (Join-Path $p ".venv\Scripts\python.exe")) { return $p }
    }
    return $null
}

function Repair-ComfyVenv($dir, $srcDir = $null) {
    # Repo layout only: create/fix the venv so ComfyUI can actually start.
    # $srcDir = where ComfyUI's own files (requirements.txt) live — equal to
    # $dir except in the nested layout, where the code sits in <dir>\ComfyUI.
    if (-not $srcDir) { $srcDir = $dir }
    $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
    $mode = Get-GpuMode
    # AMD: BOTH GPU stacks (ROCm wheels and torch-directml) top out at
    # Python 3.12. A venv on any other version can never install them —
    # the exact quiet failure that left a 64 GB AMD machine on the mock
    # renderer. Verify the interpreter FIRST and rebuild when wrong.
    if ($mode -in @("rocm", "directml") -and (Test-Path $venvPy)) {
        $comfyVer = Get-PyVersion $venvPy
        if ($comfyVer -and $comfyVer -ne "3.12") {
            Write-Host "  ComfyUI's environment is Python $comfyVer, but AMD GPU wheels need 3.12 - rebuilding..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force (Join-Path $dir ".venv")
        }
    }
    if (-not (Test-Path $venvPy)) {
        Write-Host "  ComfyUI has no Python environment - creating one..." -ForegroundColor Yellow
        $made = $false
        if ($mode -in @("rocm", "directml")) {
            $pl = Get-Python312
            if ($pl) {
                & $pl[0] $pl[1] -m venv (Join-Path $dir ".venv")
                $made = $true
            } else {
                Write-Host "  [--] Python 3.12 unavailable - the AMD GPU stack cannot install; ComfyUI will use CPU." -ForegroundColor Yellow
            }
        }
        if (-not $made) {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                py -3 -m venv (Join-Path $dir ".venv")
            } elseif (Get-Command python -ErrorAction SilentlyContinue) {
                python -m venv (Join-Path $dir ".venv")
            } else { return $false }
        }
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
        } elseif ($mode -eq "directml") {
            # Never clobber a torch that already sees the GPU (ROCm-SDK
            # build); otherwise install the pinned DirectML stack in ONE
            # pass — latest-torch-then-swap was two multi-GB downloads.
            $natR = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
            if ($natR -eq "1") {
                "keeping existing native GPU torch (ROCm SDK build)" | Out-File $repairLog -Encoding utf8
            } else {
                # Fresh venv: try AMD's NATIVE multi-arch stack first — a
                # full torch that also runs video and Kontext (this very
                # card class rendered videos on it). DirectML only when
                # the native one does not take.
                $archF = Get-AmdGfxArch
                $gotNative = $false
                if ($archF) { $gotNative = Install-RocmNative $dir $archF }
                if (-not $gotNative) {
                    & $pip install --retries 10 --timeout 300 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 *> $repairLog
                    & $pip install --retries 10 --timeout 300 torch-directml *>> $repairLog
                }
            }
        } elseif ($mode -eq "xpu") {
            # Intel Arc: torch's native XPU wheels; ComfyUI detects the
            # torch.xpu device by itself, no flag needed.
            & $pip install --retries 10 --timeout 300 torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/xpu *> $repairLog
        } else {
            & $pip install --retries 10 --timeout 180 torch torchvision torchaudio *> $repairLog
        }
        & $pip install --retries 10 --timeout 180 -r (Join-Path $srcDir "requirements.txt") *>> $repairLog
        $ErrorActionPreference = $prevEap
        if (-not (Test-PyImport $venvPy "torch, yaml, aiohttp, requests")) {
            Write-Host "  [--] Repair failed - see $repairLog. Continuing without ComfyUI." -ForegroundColor Yellow
            return $false
        }
        Write-Host "  ComfyUI environment repaired." -ForegroundColor Green
    }
    # VERIFY the GPU stack rather than assume it: torch importing is not
    # torch seeing the GPU. Reported per brand, honestly.
    if ($mode -eq "rocm") {
        $prevV = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $hip = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
        $ErrorActionPreference = $prevV
        if ($hip -eq "1") {
            Write-Host "  [ok] AMD GPU visible to torch (ROCm) - GPU rendering enabled" -ForegroundColor Green
        } else {
            Write-Host "  [!] ROCm torch installed but the AMD GPU is not visible -" -ForegroundColor Yellow
            Write-Host "      renders will use the CPU. Update the AMD driver (Adrenalin) and relaunch." -ForegroundColor Yellow
        }
    } elseif ($mode -eq "directml") {
        # This Radeon is outside AMD's classic ROCm-wheel list, so DirectML
        # is the default GPU path here. BUT AMD's newer ROCm-SDK torch builds
        # reach some of these cards (seen live: an RX 6700 XT running
        # torch 2.9.0+rocmsdk with the GPU visible). A native stack that
        # already sees the GPU beats DirectML — keep it, never bulldoze it.
        $amd = Get-AmdGpuName
        $prevV = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $nativeOk = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
        $ErrorActionPreference = $prevV
        if ($nativeOk -eq "1") {
            Write-Host "  [ok] $amd is already visible to torch (ROCm SDK build) - native GPU rendering enabled" -ForegroundColor Green
            return $true
        }
        # One-shot UPGRADE for machines already running DirectML: this
        # card class rendered videos on AMD's native stack, so try it
        # once. Failure marks a flag (delete data\logs\rocm-native.tried
        # to retry) and rolls back: the uninstall below makes the
        # existing DirectML repair path reinstall the proven pins.
        $archU = Get-AmdGfxArch
        $triedFlag = Join-Path $logDir "rocm-native.tried"
        if ($archU -and -not (Test-Path $triedFlag)) {
            New-Item -ItemType File -Force $triedFlag | Out-Null
            if (Install-RocmNative $dir $archU) { return $true }
            $pipR = Join-Path $dir ".venv\Scripts\pip.exe"
            $prevR = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $pipR uninstall -y torch torchvision torchaudio `
                "amd-torch-device-$archU" *>> (Join-Path $logDir "rocm-native-install.log")
            $ErrorActionPreference = $prevR
        }
        if (-not (Test-PyImport $venvPy "torch_directml")) {
            Write-Host "  $amd renders through DirectML - installing that stack (one time)..." -ForegroundColor Yellow
            $dmlLog = Join-Path $logDir "directml-install.log"
            $pipD = Join-Path $dir ".venv\Scripts\pip.exe"
            $prevV = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $pipD uninstall -y torch torchvision torchaudio torch-directml *> $dmlLog
            # PINNED: torch-directml requires torch==2.4.1 exactly; letting
            # pip resolve torchvision freely ends in an impossible-
            # resolution failure (measured live on this very card).
            & $pipD install --retries 10 --timeout 300 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 *>> $dmlLog
            & $pipD install --retries 10 --timeout 300 torch-directml *>> $dmlLog
            $ErrorActionPreference = $prevV
        }
        if (Test-PyImport $venvPy "torch_directml") {
            Write-Host "  [ok] DirectML ready - $amd renders on the GPU" -ForegroundColor Green
        } else {
            Write-Host "  [!] DirectML did not install - the installer's own words:" -ForegroundColor Yellow
            Get-Content (Join-Path $logDir "directml-install.log") -Tail 8 -ErrorAction SilentlyContinue |
                Where-Object { $_ -match "ERROR|error:|Could not|No matching|SSLError|CERTIFICATE" } |
                Select-Object -First 4 | ForEach-Object { Write-Host "      | $_" -ForegroundColor DarkGray }
            Write-Host "      Full log: $logDir\directml-install.log - renders fall back to the CPU (slow, but real)." -ForegroundColor Yellow
        }
    } elseif ($mode -eq "xpu") {
        $prevV = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $xok = (& $venvPy -c "import torch; print(int(torch.xpu.is_available()))" 2>$null | Out-String).Trim()
        $ErrorActionPreference = $prevV
        if ($xok -eq "1") {
            Write-Host "  [ok] Intel GPU visible to torch (XPU) - GPU rendering enabled" -ForegroundColor Green
        } else {
            Write-Host "  [!] XPU torch installed but the Intel GPU is not visible -" -ForegroundColor Yellow
            Write-Host "      renders will use the CPU. Update the Intel Arc driver and relaunch." -ForegroundColor Yellow
        }
    }
    return $true
}

function Start-ComfyUI($dir, $extraArgs = @()) {
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
    # Repo layout: <dir>\main.py with a venv — or the nested variant with
    # the venv at <dir>\.venv and the code at <dir>\ComfyUI\main.py (the
    # layout AMD's ROCm-SDK setup guides produce).
    $repoMain = Join-Path $dir "main.py"
    $workDir = $dir
    $srcDir = $dir
    if (-not (Test-Path $repoMain)) {
        $nestedMain = Join-Path $dir "ComfyUI\main.py"
        if ((Test-Path $nestedMain) -and (Test-Path (Join-Path $dir ".venv\Scripts\python.exe"))) {
            $repoMain = $nestedMain
            $workDir = Join-Path $dir "ComfyUI"
            $srcDir = $workDir
        }
    }
    if (Test-Path $repoMain) {
        if (-not (Repair-ComfyVenv $dir $srcDir)) { return $null }
        Repair-CudaTorch $dir
        Optimize-ComfyPerf $dir
        $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
        if (-not (Test-Path $venvPy)) { $venvPy = Join-Path $dir "venv\Scripts\python.exe" }
        if (Test-Path $venvPy) {
            # <=20 GB RAM: don't let ComfyUI cache checkpoints in RAM between
            # renders — cached leftovers under the WAN video stack OOM-kill
            # the process on 16 GB machines.
            $comfyArgs = @($repoMain, "--listen", "127.0.0.1") + $extraArgs
            if ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB -le 20) {
                $comfyArgs += "--disable-smart-memory"
            }
            # The flag aborts startup when the package is missing, so it is
            # gated on the same probe the backend's own spawner uses.
            if (Test-PyImport $venvPy "sageattention") {
                $comfyArgs += "--use-sage-attention"
            }
            # Non-ROCm Radeons render through DirectML — unless this venv's
            # torch already sees the GPU natively (ROCm SDK build), in which
            # case the flag would force the slower path.
            if ((Get-GpuMode) -eq "directml" -and
                (Test-PyImport $venvPy "torch_directml")) {
                $prevN = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $nativeSees = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
                $ErrorActionPreference = $prevN
                if ($nativeSees -ne "1") { $comfyArgs += "--directml" }
            }
            return Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $workDir `
                -RedirectStandardOutput $out -RedirectStandardError $err `
                $venvPy -ArgumentList $comfyArgs
        }
    }
    return $null
}

function Get-VenvOnlyComfyHome {
    # A folder like Documents\ComfyUI that holds a WORKING GPU Python
    # environment but no ComfyUI code — a user who followed AMD's ROCm-SDK
    # guide as far as the venv (found live on a real machine). Installing
    # the code next to that venv reuses the native GPU stack, which beats
    # building a fresh DirectML environment from nothing. Memoized — the
    # torch probe costs seconds and two call sites need the answer.
    if ($script:pfVenvOnlyChecked) { return $script:pfVenvOnlyHome }
    $script:pfVenvOnlyChecked = $true
    $script:pfVenvOnlyHome = $null
    foreach ($c in @($env:PROMPTFORGE_COMFYUI_PATH,
                     "$env:USERPROFILE\ComfyUI",
                     "$env:USERPROFILE\Documents\ComfyUI")) {
        if (-not ($c -and (Test-Path $c))) { continue }
        $p = (Resolve-Path $c).Path
        if (Test-Path (Join-Path $p "main.py")) { continue }
        if (Test-Path (Join-Path $p "ComfyUI\main.py")) { continue }
        if (Test-NativeGpuVenv (Join-Path $p ".venv\Scripts\python.exe")) {
            $script:pfVenvOnlyHome = $p
            return $p
        }
    }
    return $null
}

function Repair-DirectmlComfyCode($dir) {
    # ComfyUI code too new for torch 2.4.1 (RegisterOperators schema crash):
    # replace the CODE with the pinned tag while keeping the environment and
    # user content untouched. Handles root and nested layouts.
    $src = $dir
    if (-not (Test-Path (Join-Path $dir "main.py"))) { $src = Join-Path $dir "ComfyUI" }
    if (-not (Test-Path (Join-Path $src "main.py"))) { return $false }
    Write-Host "  This ComfyUI is too new for torch 2.4.1 (DirectML) - swapping its code to $comfyDmlTag..." -ForegroundColor Yellow
    $stage = Join-Path $env:TEMP "pf-comfy-dml-code"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
        (& git -c http.sslBackend=schannel clone --depth 1 --branch $comfyDmlTag `
            https://github.com/comfyanonymous/ComfyUI.git $stage 2>&1 | Out-String) |
            Out-File (Join-Path $logDir "comfyui-install.log") -Append -Encoding utf8
        if (-not (Test-Path (Join-Path $stage "main.py"))) { return $false }
        $keep = @(".venv", "venv", "custom_nodes", "models", "user",
                  "input", "output", "extra_model_paths.yaml")
        Get-ChildItem $src -Force | Where-Object { $keep -notcontains $_.Name } |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
        Get-ChildItem $stage -Force | Where-Object { $keep -notcontains $_.Name } |
            ForEach-Object { Move-Item $_.FullName (Join-Path $src $_.Name) -Force }
        if (-not (Test-Path (Join-Path $src "main.py"))) { return $false }
        # The venv still holds the NEWER helper packages (comfy-kitchen is
        # the module that actually crashes torch 2.4.1) — align them with
        # the swapped code's own pins. torch stays: the requirement is
        # unpinned and 2.4.1 satisfies it.
        $venvPip = Join-Path $dir ".venv\Scripts\pip.exe"
        if (Test-Path $venvPip) {
            (& $venvPip install --retries 8 --timeout 300 -r (Join-Path $src "requirements.txt") 2>&1 |
                Out-String) | Out-File (Join-Path $logDir "comfyui-install.log") -Append -Encoding utf8
        }
        return $true
    } finally {
        Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
        $ErrorActionPreference = $prev
    }
}

function Install-ComfyUI {
    # A fresh clone has no ComfyUI. Prefer installing the code NEXT TO an
    # existing venv that already sees the GPU (nested layout); otherwise
    # fetch into tools\ComfyUI. git clone goes first — its schannel TLS uses
    # the Windows certificate store, so it survives the AV certificate
    # interception that breaks PowerShell downloads on machines here — then
    # the zip, with retries; every attempt is logged for remote diagnosis.
    $log = Join-Path $logDir "comfyui-install.log"
    "=== ComfyUI install $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8
    $venvHome = Get-VenvOnlyComfyHome
    $branchArgs = @()
    $zipRef = "refs/heads/master"
    if ($venvHome) {
        $dest = Join-Path $venvHome "ComfyUI"
        $ret = $venvHome
        Write-Host "  Found a working GPU Python environment at $venvHome -" -ForegroundColor Yellow
        Write-Host "  installing ComfyUI's code next to it (reuses the native GPU stack)." -ForegroundColor Yellow
    } else {
        $dest = Join-Path $root "tools\ComfyUI"
        $ret = $dest
        if (Test-Path (Join-Path $dest "main.py")) { return $dest }
        Write-Host "  Installing ComfyUI (one time, the render engine)..." -ForegroundColor Yellow
        if ((Get-GpuMode) -eq "directml") {
            # torch-directml is frozen at torch 2.4.1 and current master
            # cannot even import on that (RegisterOperators schema crash,
            # measured live) - pin the newest release verified against a
            # real torch 2.4.1 environment.
            $branchArgs = @("--branch", $comfyDmlTag)
            $zipRef = "refs/tags/$comfyDmlTag"
            Write-Host "  DirectML machine: installing ComfyUI $comfyDmlTag (newest release that runs on torch 2.4.1)." -ForegroundColor Yellow
        }
    }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    # git/IWR write progress to stderr; under "Stop" + redirection PS 5.1
    # would turn that into a terminating error mid-download.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            for ($i = 1; $i -le 2; $i++) {
                if (Test-Path $dest) { Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue }
                "--- git clone attempt $i ($zipRef)" | Out-File $log -Append -Encoding utf8
                (& git -c http.sslBackend=schannel clone --depth 1 @branchArgs `
                    https://github.com/comfyanonymous/ComfyUI.git $dest 2>&1 |
                    Out-String) | Out-File $log -Append -Encoding utf8
                if (Test-Path (Join-Path $dest "main.py")) {
                    Write-Host "  ComfyUI files in place (git)." -ForegroundColor Green
                    return $ret
                }
                Start-Sleep ([int][Math]::Pow(2, $i))
            }
        }
        $zip = Join-Path $env:TEMP "pf-comfyui.zip"
        $stage = Join-Path $env:TEMP "pf-comfyui-unzip"
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        for ($i = 1; $i -le 3; $i++) {
            try {
                "--- zip attempt $i" | Out-File $log -Append -Encoding utf8
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 300 `
                    -Headers @{ "User-Agent" = "PromptForge-installer" } `
                    -Uri "https://github.com/comfyanonymous/ComfyUI/archive/$zipRef.zip" `
                    -OutFile $zip
                if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
                Expand-Archive -Path $zip -DestinationPath $stage -Force
                $inner = Get-ChildItem $stage -Directory | Select-Object -First 1
                if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
                Move-Item $inner.FullName $dest
                Write-Host "  ComfyUI files in place (zip)." -ForegroundColor Green
                return $ret
            } catch {
                "zip attempt $i failed: $($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
                Start-Sleep ([int][Math]::Pow(2, $i))
            } finally {
                Remove-Item $zip -ErrorAction SilentlyContinue
                Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
            }
        }
        Write-Host "  [--] Could not download ComfyUI (5 attempts) - renders fall back to the mock." -ForegroundColor Yellow
        Write-Host "       The installer's own words ($log):" -ForegroundColor Yellow
        Get-Content $log -Tail 6 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "       | $_" -ForegroundColor DarkGray }
        return $null
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Repair-CudaTorch($dir) {
    # The commonest broken install: an NVIDIA machine whose ComfyUI venv
    # ended up with CPU-only torch (partial download, wrong index). It
    # imports fine, so the ordinary self-repair never fires - but every
    # render would crawl or fail. Detect it and put the GPU build back.
    if ((Get-GpuMode) -ne "cuda") { return }
    $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { return }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $cudaOk = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
    $ErrorActionPreference = $prev
    if ($cudaOk -eq "1") { return }
    Write-Host "  [!] NVIDIA GPU present but ComfyUI's torch cannot use it -" -ForegroundColor Yellow
    Write-Host "      reinstalling the GPU build (one time, ~2.5 GB)..." -ForegroundColor Yellow
    $pip = Join-Path $dir ".venv\Scripts\pip.exe"
    $log = Join-Path $logDir "torch-cuda-repair.log"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pip install --force-reinstall --retries 10 --timeout 300 torch torchvision torchaudio `
        --index-url https://download.pytorch.org/whl/cu126 *> $log
    $cudaOk = (& $venvPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim()
    $ErrorActionPreference = $prev
    if ($cudaOk -eq "1") {
        Write-Host "  [ok] GPU torch restored" -ForegroundColor Green
    } else {
        Write-Host "  [--] torch still cannot see the GPU - check the NVIDIA driver ($log)" -ForegroundColor Yellow
    }
}

function Test-ComfyRender($baseUrl) {
    # Answering HTTP is not working: prove ComfyUI can RENDER by pushing a
    # tiny model-free graph (EmptyImage -> SaveImage) through the real API
    # and waiting for an output. Returns @($ok, $whyNot).
    try {
        $graph = @{ prompt = @{
            "1" = @{ class_type = "EmptyImage"
                     inputs = @{ width = 64; height = 64; batch_size = 1; color = 8355711 } }
            "2" = @{ class_type = "SaveImage"
                     inputs = @{ filename_prefix = "pf_verify"; images = @("1", 0) } }
        } } | ConvertTo-Json -Depth 6
        $resp = Invoke-RestMethod -Uri "$baseUrl/prompt" -Method Post `
            -ContentType "application/json" -Body $graph -TimeoutSec 15
        $promptId = $resp.prompt_id
        if (-not $promptId) { return @($false, "the queue refused the test graph") }
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Milliseconds 700
            try { $hist = Invoke-RestMethod -Uri "$baseUrl/history/$promptId" -TimeoutSec 5 }
            catch { continue }
            $entry = $hist.$promptId
            if ($entry) {
                if ($entry.status -and $entry.status.status_str -eq "error") {
                    $detail = ($entry.status | ConvertTo-Json -Compress -Depth 5)
                    if ($detail.Length -gt 220) { $detail = $detail.Substring(0, 220) }
                    return @($false, "the test render errored: $detail")
                }
                if ($entry.outputs) { return @($true, "") }
            }
        }
        return @($false, "the test render never finished (28s)")
    } catch {
        return @($false, $_.Exception.Message)
    }
}

function Optimize-ComfyPerf($dir) {
    # Measured on an RTX 4060: SageAttention via Triton = 11% faster renders
    # with the image unchanged. NVIDIA-only (Triton kernels), never fatal.
    if ((Get-GpuMode) -ne "cuda") { return }
    $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { return }
    $pip = Join-Path $dir ".venv\Scripts\pip.exe"
    if (-not (Test-PyImport $venvPy "sageattention")) {
        Write-Host "  Installing SageAttention (faster renders on NVIDIA)..." -ForegroundColor DarkGray
        $log = Join-Path $logDir "sage-install.log"
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $pip install --retries 6 --timeout 120 triton-windows sageattention *> $log
        $ErrorActionPreference = $prev
        if (Test-PyImport $venvPy "sageattention") {
            Write-Host "  [ok] SageAttention ready (+~11% render speed)" -ForegroundColor Green
        }
    }
    # xformers: attention fallback ComfyUI uses automatically when present.
    # --no-deps is load-bearing - a bare 'pip install xformers' may pull a
    # DIFFERENT torch and break the working CUDA stack. If the wheel does
    # not match this torch, the import fails and it is removed again;
    # either way the venv is never left worse than before.
    if (-not (Test-PyImport $venvPy "xformers")) {
        $log2 = Join-Path $logDir "xformers-install.log"
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $pip install --no-deps --retries 6 --timeout 180 xformers *> $log2
        if (-not (Test-PyImport $venvPy "xformers, xformers.ops")) {
            & $pip uninstall -y xformers *>> $log2
        } else {
            Write-Host "  [ok] xformers ready (extra attention backend)" -ForegroundColor Green
        }
        $ErrorActionPreference = $prev
    }
}

$comfyDir = $null
if ((Get-GpuMode) -eq "directml") {
    # On these machines a native ROCm-SDK stack (when one exists) beats
    # DirectML outright: torch-directml is frozen at torch 2.4.1, which
    # limits ComfyUI to older releases, while native torch runs current
    # code on the GPU. Prefer a runnable native install; failing that,
    # install ComfyUI's code next to a native venv that lacks it.
    $comfyDir = Get-NativeGpuComfyHome
    if ($comfyDir) {
        Write-Host "  Using the native-GPU ComfyUI at $comfyDir (beats DirectML)." -ForegroundColor Green
    } elseif (Get-VenvOnlyComfyHome) {
        $comfyDir = Install-ComfyUI
    }
}
if (-not $comfyDir) { $comfyDir = Find-ComfyUI }
if (-not $comfyDir) { $comfyDir = Install-ComfyUI }
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
  ultralytics_bbox: ultralytics/bbox
  ultralytics_segm: ultralytics/segm
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
            # Code newer than the frozen DirectML torch can carry: swap the
            # code to the pinned release and try the GPU again before giving
            # up on it. (A native venv runs master - never downgrade those.)
            $errTail = (Get-Content (Join-Path $logDir "comfyui-err.log") -Tail 40 -ErrorAction SilentlyContinue | Out-String)
            if ((Get-GpuMode) -eq "directml" -and
                $errTail -match "RegisterOperators|torch\.library|infer_schema" -and
                -not (Test-NativeGpuVenv (Join-Path $comfyDir ".venv\Scripts\python.exe"))) {
                if (Repair-DirectmlComfyCode $comfyDir) {
                    Write-Host "  Retrying ComfyUI with the version-matched code..." -ForegroundColor Yellow
                    $p = Start-ComfyUI $comfyDir
                    if ($p) {
                        $started += $p
                        $comfyUp = Wait-Http "http://127.0.0.1:8188/system_stats" 90 $p
                    }
                }
            }
            if (-not $comfyUp) {
                # ANY GPU stack can crash on init (driver mismatch, broken
                # update, OOM at model scan). A CPU start still gives real
                # renders - always better than the mock, on every machine.
                Write-Host "  Retrying ComfyUI on the CPU (GPU init failed - slow but real renders)..." -ForegroundColor Yellow
                $p = Start-ComfyUI $comfyDir @("--cpu")
                if ($p) {
                    $started += $p
                    $comfyUp = Wait-Http "http://127.0.0.1:8188/system_stats" 90 $p
                }
            }
        }
    }
}
if ($comfyDir) {
    # Backend uses this to restart ComfyUI automatically if it crashes.
    $env:PROMPTFORGE_COMFYUI_DIR = $comfyDir
}
if ($comfyUp) {
    # Verify, not assume: a running ComfyUI that cannot render is exactly
    # as useless as a stopped one, and far quieter about it.
    $verify = Test-ComfyRender "http://127.0.0.1:8188"
    if ($verify[0]) {
        $devLine = ""
        try {
            $st = Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -TimeoutSec 5
            $dev = $st.devices[0]
            $devLine = " on $($dev.type) ($($dev.name))"
            if ($dev.type -eq "cpu" -and (Get-GpuMode) -eq "cuda") {
                Write-Host "  [!] ComfyUI is rendering on the CPU although an NVIDIA GPU exists -" -ForegroundColor Yellow
                Write-Host "      renders will be very slow. Close PromptForge and relaunch to auto-repair." -ForegroundColor Yellow
            }
        } catch {}
        $env:PROMPTFORGE_INPAINT_BACKEND = "comfyui"
        Write-Host "  [ok] ComfyUI VERIFIED - test image rendered$devLine" -ForegroundColor Green
    } else {
        Write-Host "  [!!] ComfyUI answers but CANNOT RENDER: $($verify[1])" -ForegroundColor Red
        # One more real-render attempt before any mock: when WE started this
        # ComfyUI, restart it on the CPU - a GPU stack broken at render time
        # (bad op, driver fault) usually still renders fine there.
        if ($p -and -not $p.HasExited) {
            Write-Host "  Restarting ComfyUI on the CPU (GPU render path is broken - slow but real renders)..." -ForegroundColor Yellow
            try { Stop-Process -Id $p.Id -Force -Confirm:$false } catch {}
            Start-Sleep -Seconds 2
            $p = Start-ComfyUI $comfyDir @("--cpu")
            if ($p) {
                $started += $p
                if (Wait-Http "http://127.0.0.1:8188/system_stats" 90 $p) {
                    $verify = Test-ComfyRender "http://127.0.0.1:8188"
                    if ($verify[0]) {
                        $env:PROMPTFORGE_INPAINT_BACKEND = "comfyui"
                        Write-Host "  [ok] ComfyUI VERIFIED on the CPU - real renders, no mock" -ForegroundColor Green
                    }
                }
            }
        }
        if ($env:PROMPTFORGE_INPAINT_BACKEND -ne "comfyui") {
            Write-Host "       Renders fall back to the clearly-labeled mock. Logs: $logDir\comfyui-err.log" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  [--] ComfyUI not available - edits use the clearly-labeled mock renderer." -ForegroundColor Yellow
    Write-Host "       Why, in order of likelihood:" -ForegroundColor DarkGray
    Write-Host "         1. Its download failed: $logDir\comfyui-install.log" -ForegroundColor DarkGray
    Write-Host "         2. Its Python packages failed to install: $logDir\comfyui-repair.log" -ForegroundColor DarkGray
    Write-Host "         3. It crashed on start: $logDir\comfyui-err.log" -ForegroundColor DarkGray
    Write-Host "         4. No usable GPU stack (AMD needs Python 3.12; Intel needs an Arc-class GPU + driver)" -ForegroundColor DarkGray
    Get-Content (Join-Path $logDir "comfyui-err.log") -Tail 5 -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "       | $_" -ForegroundColor DarkGray }
}

# --- Performance summary ---------------------------------------------------------
# Every tuning decision in one place, so "is it using my hardware?" has an
# answer without reading logs. The backend scales the rest automatically:
# model sizes, mesh octree detail and video limits all follow VRAM/RAM.
$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$gpuMode = Get-GpuMode
$sageOn = $false
if ($comfyDir) {
    $comfyPy = Join-Path $comfyDir ".venv\Scripts\python.exe"
    if (Test-Path $comfyPy) { $sageOn = Test-PyImport $comfyPy "sageattention" }
}
Write-Host ""
Write-Host "  Tuned for this machine:" -ForegroundColor Cyan
Write-Host ("    GPU: {0} ({1} GB VRAM) | RAM: {2} GB | CPU threads: {3}" -f `
    $gpuMode, $vramGb, $ramGb, $cores) -ForegroundColor DarkGray
Write-Host ("    torch build: {0} | planner LLM: {1} | SageAttention: {2}" -f `
    $gpuMode, $llmModel, $(if ($sageOn) { "on (+11% renders)" } else { "off" })) -ForegroundColor DarkGray
Write-Host ("    checkpoint RAM cache: {0}" -f `
    $(if ($ramGb -le 20) { "dropped between renders (protects $ramGb GB RAM)" }
      else { "kept between renders (RAM headroom)" })) -ForegroundColor DarkGray
Write-Host "    LAN: models copy from other PromptForge machines; idle peers accept renders" -ForegroundColor DarkGray

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
try { Stop-Transcript | Out-Null } catch {}

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
