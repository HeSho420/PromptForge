# PromptForge installer core (runs on the TARGET device).
#
# This script is packed inside PromptForge-Setup.bat by build-installer.ps1.
# The .bat unpacks the payload to a temp folder and starts this script with
# -Staging <temp>. Default is a small GUI (pick folder -> Install); pass
# -Silent -InstallDir <path> for a headless install.
#
# Design rules (why it looks the way it does):
#  * Windows PowerShell 5.1 syntax only - no '&&', no ternary, ASCII only.
#  * No admin rights needed: Python installs per-user, Ollama installs
#    per-user, ComfyUI is a plain folder inside the chosen directory.
#  * Every step is idempotent - a re-run skips what is already done, so a
#    failed install is resumed by simply running the installer again.
#  * Every download retries with backoff and verifies a minimum size.
#  * Direct downloads first (python.org / ollama.com / github.com), winget
#    only as a fallback - many machines have no working winget.
#  * GPU-aware: NVIDIA present -> CUDA torch wheels; otherwise CPU wheels.
#  * launch.ps1 (installed with the app) self-repairs at every start, so
#    anything this installer could not finish is retried at first launch.

param(
    [switch]$Silent,
    [string]$InstallDir = "",
    [string]$Staging = "",
    [switch]$NoShortcut,
    [switch]$NoLaunch,
    [switch]$AllowOneDrive,
    [switch]$UiSmokeTest
)

$ErrorActionPreference = "Stop"

# TLS 1.2 (and 1.3 where the OS supports it) for every download.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor 3072 -bor 12288
} catch {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor 3072
}

$script:PyVersion   = "3.12.10"
$script:PyUrl       = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$script:OllamaUrl   = "https://ollama.com/download/OllamaSetup.exe"
$script:ComfyZipUrl = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip"
$script:SamZipUrl   = "https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip"
$script:TorchGpuIndex = "https://download.pytorch.org/whl/cu126"
# AMD ROCm-on-Windows wheels (official, repo.radeon.com). Python 3.12 ONLY
# (cp312 builds) and Adrenalin driver 26.2.2+. Two pip steps: SDK, then torch.
# Keep in sync with the copy in launch.ps1 (self-repair uses the same list).
$script:RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
$script:RocmSdkWheels = @(
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "rocm-7.2.1.tar.gz")
$script:RocmTorchWheels = @(
    "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl")
$script:GpuMode = $null   # resolved once: cuda | rocm | directml | xpu | cpu
$script:TorchXpuIndex = "https://download.pytorch.org/whl/xpu"
# torch-directml is frozen at torch 2.4.1; ComfyUI releases newer than this
# tag cannot even import on it (comfy-kitchen custom-op schema crash,
# measured on a real 2.4.1 environment). Keep in sync with launch.ps1.
$script:ComfyDmlTag = "v0.30.2"
# Internal: PROMPTFORGE_SETUP_LIGHT=1 skips the multi-GB steps (torch, SAM,
# ComfyUI, Ollama) so the file/venv plumbing can be tested quickly.
$script:LightMode = ($env:PROMPTFORGE_SETUP_LIGHT -eq "1")
$script:AllowOneDrive = [bool]$AllowOneDrive

# A trailing backslash right before the closing quote ('-InstallDir "C:\X\"')
# survives cmd's %* expansion as an escaped quote and drags any following
# switches into the path. Recover the real path part and normalize.
if ($InstallDir) {
    $InstallDir = ($InstallDir -split '"')[0].Trim().TrimEnd("\")
}

if (-not $Staging) { $Staging = $PSScriptRoot }
$script:Staging  = $Staging
$script:LogFile  = $null
$script:LogBox   = $null      # GUI textbox, when the GUI is up
$script:Progress = $null      # GUI progress bar
$script:StatusLb = $null      # GUI status label
$script:Failed   = @()

# ---------------------------------------------------------------- logging --

function Update-Ui {
    if ($script:LogBox -ne $null) {
        [System.Windows.Forms.Application]::DoEvents()
    }
}

function Write-Log([string]$msg, [string]$level = "info") {
    $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    if ($script:LogFile) {
        try { Add-Content -Path $script:LogFile -Value $line -Encoding ASCII } catch {}
    }
    if ($script:LogBox -ne $null) {
        $script:LogBox.AppendText($line + [Environment]::NewLine)
        Update-Ui
    } else {
        if ($level -eq "error") { Write-Host $line -ForegroundColor Red }
        elseif ($level -eq "warn") { Write-Host $line -ForegroundColor Yellow }
        else { Write-Host $line }
    }
}

function Set-Status([string]$text) {
    if ($script:StatusLb -ne $null) { $script:StatusLb.Text = $text; Update-Ui }
}

function Invoke-Step([string]$name, [scriptblock]$body) {
    Write-Log ("== " + $name + " ==")
    Set-Status $name
    try {
        & $body
        Write-Log ("[ok] " + $name)
        return $true
    } catch {
        $script:Failed += $name
        Write-Log ("[FAILED] " + $name + " -- " + $_.Exception.Message) "error"
        return $false
    }
}

# ------------------------------------------------------------- primitives --

function Download-File([string[]]$urls, [string]$dest, [long]$minBytes = 4096) {
    # Tries every URL in order, 4 rounds - flaky networks (github zip
    # generation stalling mid-stream was observed live) often succeed on a
    # later attempt or an alternate host for the same content.
    $tries = @()
    for ($round = 1; $round -le 4; $round++) {
        foreach ($u in $urls) { $tries += @{ url = $u; round = $round } }
    }
    $attempt = 0
    foreach ($t in $tries) {
        $attempt++
        $url = $t.url
        try {
            if (Test-Path $dest) { Remove-Item $dest -Force }
            Write-Log ("downloading " + $url + " (attempt " + $attempt + ")")
            $req = [System.Net.HttpWebRequest]::Create($url)
            $req.UserAgent = "PromptForge-Installer/1.0"
            $req.Timeout = 60000
            $req.ReadWriteTimeout = 120000
            $req.AllowAutoRedirect = $true
            $resp = $req.GetResponse()
            $total = $resp.ContentLength
            $in = $resp.GetResponseStream()
            $out = [System.IO.File]::Create($dest)
            try {
                $buf = New-Object byte[] 262144
                $done = [long]0
                $lastTick = 0
                while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                    $out.Write($buf, 0, $n)
                    $done += $n
                    if ([Environment]::TickCount - $lastTick -gt 500) {
                        $lastTick = [Environment]::TickCount
                        if ($total -gt 0) {
                            Set-Status ("Downloading... {0:N0} / {1:N0} MB" -f
                                ($done / 1MB), ($total / 1MB))
                        } else {
                            Set-Status ("Downloading... {0:N0} MB" -f ($done / 1MB))
                        }
                        Update-Ui
                    }
                }
            } finally {
                $out.Close(); $in.Close(); $resp.Close()
            }
            $size = (Get-Item $dest).Length
            if ($size -lt $minBytes) {
                throw ("download is only " + $size + " bytes - truncated or blocked")
            }
            try { Unblock-File -Path $dest -ErrorAction SilentlyContinue } catch {}
            Write-Log ("downloaded {0:N1} MB" -f ($size / 1MB))
            return
        } catch {
            Write-Log ("download attempt " + $attempt + " failed: " +
                       $_.Exception.Message) "warn"
            if ($attempt -lt $tries.Count) {
                Start-Sleep -Seconds ([Math]::Min(30, 5 * $t.round))
            }
        }
    }
    throw ("Could not download " + $urls[0] + " after " + $tries.Count +
           " attempts. Check the internet connection (or a firewall/proxy) " +
           "and run the installer again.")
}

function Invoke-Process([string]$exe, [string]$argString, [int]$timeoutMin = 60) {
    # Run a child process with hidden window, capture its output into the
    # log, keep the GUI responsive, enforce a timeout. Returns the exit code.
    $so = [System.IO.Path]::GetTempFileName()
    $se = [System.IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $exe -ArgumentList $argString -PassThru `
            -WindowStyle Hidden -RedirectStandardOutput $so -RedirectStandardError $se
        # PS 5.1 quirk: without touching .Handle now, .ExitCode reads as
        # $null after the process exits - which looks like a failure.
        $null = $p.Handle
        $deadline = (Get-Date).AddMinutes($timeoutMin)
        while (-not $p.HasExited) {
            if ((Get-Date) -gt $deadline) {
                try { Stop-Process -Id $p.Id -Force -Confirm:$false } catch {}
                throw ("'" + $exe + "' still running after " + $timeoutMin +
                       " minutes - aborted")
            }
            Update-Ui
            Start-Sleep -Milliseconds 400
        }
        $p.WaitForExit()
        $code = $p.ExitCode
        if ($null -eq $code) {
            # Handle was lost anyway - trust the on-disk outcome checks the
            # callers all perform, and only log the uncertainty.
            Write-Log ("    (exit code unreadable for " +
                       [System.IO.Path]::GetFileName($exe) + " - assuming 0)") "warn"
            $code = 0
        }
        foreach ($f in @($so, $se)) {
            try {
                $tail = Get-Content $f -ErrorAction SilentlyContinue |
                        Select-Object -Last 12
                foreach ($ln in $tail) {
                    if ($ln.Trim()) { Write-Log ("    " + $ln.Trim()) }
                }
            } catch {}
        }
        return $code
    } finally {
        Remove-Item $so, $se -Force -ErrorAction SilentlyContinue
    }
}

function Expand-Zip([string]$zip, [string]$dest) {
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
    } catch {
        # Fall back to the cmdlet (handles a pre-existing destination).
        Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
    }
}

# ------------------------------------------------------------- discovery --

function Test-PythonExe([string]$exe) {
    # True when $exe is a 64-bit CPython 3.10 - 3.13.
    try {
        $out = & $exe -c "import sys,struct;print('%d.%d.%d' % (sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8))" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $parts = "$out".Trim().Split(".")
        if ($parts.Count -lt 3) { return $false }
        $major = [int]$parts[0]; $minor = [int]$parts[1]; $bits = [int]$parts[2]
        return ($major -eq 3 -and $minor -ge 10 -and $minor -le 13 -and $bits -eq 64)
    } catch { return $false }
}

function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.13", "3.10")) {
            try {
                $exe = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe) { $candidates += "$exe".Trim() }
            } catch {}
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += $cmd.Source }
    foreach ($v in @("Python312", "Python311", "Python313", "Python310")) {
        $candidates += (Join-Path $env:LOCALAPPDATA ("Programs\Python\" + $v + "\python.exe"))
        $candidates += ("C:\Program Files\" + $v + "\python.exe")
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c) -and (Test-PythonExe $c)) { return $c }
    }
    return $null
}

function Test-NvidiaGpu {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        $sys32 = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path $sys32) { $smi = @{ Source = $sys32 } }
    }
    if (-not $smi) { return $false }
    try {
        $out = & $smi.Source --query-gpu=name --format=csv,noheader 2>$null
        return ($LASTEXITCODE -eq 0 -and "$out".Trim().Length -gt 0)
    } catch { return $false }
}

function Get-PyMinor([string]$exe) {
    # "3.12" for a working python, "" when it cannot run.
    try {
        $out = & $exe -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return "$out".Trim() }
    } catch {}
    return ""
}

function Get-GpuMode {
    # Which torch build fits this machine: cuda | rocm | directml | xpu | cpu.
    # Keep in sync with launch.ps1 (its self-repair uses the same ladder).
    if (Test-NvidiaGpu) { return "cuda" }
    try {
        $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
                  ForEach-Object { $_.Name } | Where-Object { $_ })
    } catch { $gpus = @() }
    foreach ($n in $gpus) {
        if ($n -match "NVIDIA|GeForce|Quadro|RTX \d") {
            # Broken/absent driver: still install the CUDA build. It renders
            # on CPU today and uses the GPU the moment the driver is fixed -
            # no reinstall (launch.ps1 also self-repairs a CPU-only torch).
            Write-Log ("NVIDIA GPU present (" + $n + ") but its driver is " +
                       "not answering - installing the CUDA build anyway; " +
                       "install the GeForce driver to enable the GPU.") "warn"
            return "cuda"
        }
    }
    # ROCm on native Windows covers RX 9000, the high-end RX 7000 cards and
    # Ryzen AI APU graphics (860M/880M/890M, 8040S/8050S/8060S).
    $rocmPattern = "Radeon.+(RX\s?90\d0|RX\s?7900|RX\s?7800|RX\s?7700|8[89]0M|860M|80[456]0S)"
    foreach ($n in $gpus) {
        if ($n -match $rocmPattern) {
            Write-Log ("AMD GPU with Windows ROCm support detected: " + $n)
            return "rocm"
        }
    }
    foreach ($n in $gpus) {
        if ($n -match "Intel.+Arc") {
            # Intel Arc (discrete or Core Ultra iGPU): native torch XPU -
            # current torch, current ComfyUI. Checked BEFORE the generic
            # AMD catch-all: a Ryzen desktop iGPU enumerates as 'AMD
            # Radeon(TM) Graphics', and an Arc card next to it must win
            # over the frozen DirectML stack. Older Iris Xe/UHD iGPUs
            # are not covered and stay on CPU.
            Write-Log ("Intel Arc GPU detected (" + $n + ") - using the " +
                       "torch XPU stack")
            return "xpu"
        }
    }
    foreach ($n in $gpus) {
        if ($n -match "AMD|Radeon") {
            # Every other DX12 Radeon (RX 6000/5000, Ryzen iGPUs) renders
            # for real through torch-directml.
            Write-Log ("AMD GPU without Windows ROCm wheels (" + $n + ") - " +
                       "using the DirectML GPU stack")
            return "directml"
        }
    }
    return "cpu"
}

function Resolve-GpuMode {
    if (-not $script:GpuMode) {
        $script:GpuMode = Get-GpuMode
        Write-Log ("render acceleration: " + $script:GpuMode)
    }
    return $script:GpuMode
}

function Install-RocmTorch([string]$venvPy) {
    # AMD's two-step official install: ROCm SDK wheels first, then torch.
    $sdk = ($script:RocmSdkWheels | ForEach-Object { '"' + $script:RocmBase + $_ + '"' }) -join " "
    $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " + $sdk) 90
    if ($code -ne 0) { return $false }
    $trio = ($script:RocmTorchWheels | ForEach-Object { '"' + $script:RocmBase + $_ + '"' }) -join " "
    $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " + $trio) 120
    return ($code -eq 0)
}

function Install-DirectmlTorch([string]$venvPy) {
    # NATIVE first: AMD's multi-arch channel ships a FULL torch (video-
    # capable, all ops) for cards the official wheel list skips - proven
    # on an RX 6700 XT, which rendered videos on exactly this stack.
    $gfx = $null
    try {
        $n = (Get-CimInstance Win32_VideoController -ErrorAction Stop |
              Where-Object { $_.Name -match "Radeon|AMD" } |
              Select-Object -First 1 -ExpandProperty Name)
        if ($n -match "RX\s?6[89]\d0") { $gfx = "gfx1030" }
        elseif ($n -match "RX\s?67\d0") { $gfx = "gfx1031" }
        elseif ($n -match "RX\s?66\d0") { $gfx = "gfx1032" }
        elseif ($n -match "RX\s?6[45]\d0") { $gfx = "gfx1034" }
        elseif ($n -match "RX\s?76\d0") { $gfx = "gfx1102" }
        elseif ($n -match "RX\s?5[67]\d0") { $gfx = "gfx1010" }
        elseif ($n -match "RX\s?55\d0") { $gfx = "gfx1012" }
    } catch {}
    if ($gfx) {
        Write-Log ("trying AMD's native ROCm torch for " + $gfx +
                   " (full stack, video-capable; several GB)")
        $code = Invoke-Process $venvPy ("-m pip install --pre --retries 10 " +
            "--timeout 600 --index-url " +
            "https://rocm.nightlies.amd.com/whl-multi-arch/ " +
            "torch torchvision torchaudio amd-torch-device-" + $gfx) 120
        if ($code -eq 0) {
            $ok = Invoke-Process $venvPy `
                "-c ""import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)""" 5
            if ($ok -eq 0) {
                Write-Log "[ok] native ROCm torch is live - this GPU renders everything, video included"
                return $true
            }
        }
        Write-Log "native ROCm did not take - falling back to the DirectML pins" "warn"
        Invoke-Process $venvPy ("-m pip uninstall -y torch torchvision " +
            "torchaudio amd-torch-device-" + $gfx) 10 | Out-Null
    }
    # PINNED: torch-directml hard-requires torch==2.4.1; an unpinned
    # torchvision resolves to a newer torch and pip fails the whole install
    # (measured live on an RX 6700 XT).
    $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
        "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1") 90
    if ($code -ne 0) { return $false }
    $code = Invoke-Process $venvPy "-m pip install --retries 10 --timeout 300 torch-directml" 60
    return ($code -eq 0)
}

function Get-TorchModeFor([string]$venvPy) {
    # The per-venv decision: both AMD stacks are Python-3.12-only (ROCm
    # wheels are cp312; torch-directml stops at torch 2.4.1 = 3.12 max).
    $mode = Resolve-GpuMode
    if ($mode -in @("rocm", "directml")) {
        $minor = Get-PyMinor $venvPy
        if ($minor -ne "3.12") {
            Write-Log ("the AMD GPU stack needs Python 3.12 but this " +
                       "environment is Python " + $minor + " - falling back " +
                       "to CPU torch. Install Python 3.12 and re-run for " +
                       "AMD GPU rendering.") "warn"
            return "cpu"
        }
    }
    return $mode
}

function Test-TorchGpu([string]$venvPy) {
    # Warn-only, per-stack: a GPU build still renders on CPU if the GPU is
    # unusable, so a failure here never blocks the install.
    $mode = Resolve-GpuMode
    if ($mode -eq "directml") {
        $probe = "import sys, torch_directml; sys.exit(0 if torch_directml.device_count() > 0 else 1)"
    } elseif ($mode -eq "xpu") {
        $probe = "import sys, torch; sys.exit(0 if torch.xpu.is_available() else 1)"
    } else {
        $probe = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    }
    $code = Invoke-Process $venvPy ("-c """ + $probe + """") 5
    if ($code -ne 0) {
        Write-Log ("torch is installed but cannot use the GPU - renders run " +
                   "on the CPU for now. Update the GPU driver (AMD: Adrenalin " +
                   "26.2.2+ / NVIDIA: current GeForce / Intel: current Arc " +
                   "driver) and renders speed up automatically.") "warn"
    }
}

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $local = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $local) { return $local }
    return $null
}

function Test-OneDrivePath([string]$path) {
    # True when $path sits inside a OneDrive-synced folder. Installing 5-8 GB
    # of venvs there causes quota exhaustion, sync/file-lock races during pip
    # installs, and Files On-Demand dehydrating files the app needs.
    $roots = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)
    $p = $path.TrimEnd("\")
    foreach ($od in $roots) {
        if (-not $od) { continue }
        $r = $od.TrimEnd("\")
        if ($p.StartsWith($r, [StringComparison]::OrdinalIgnoreCase)) {
            $rest = $p.Substring($r.Length)
            if ($rest -eq "" -or $rest.StartsWith("\")) { return $true }
        }
    }
    return $false
}

function Test-Url([string]$url) {
    try {
        $req = [System.Net.HttpWebRequest]::Create($url)
        $req.Method = "HEAD"
        $req.Timeout = 8000
        $req.UserAgent = "PromptForge-Installer/1.0"
        $resp = $req.GetResponse()
        $resp.Close()
        return $true
    } catch { return $false }
}

# ---------------------------------------------------------------- steps ----

function Step-Preflight([string]$dest) {
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "PromptForge needs 64-bit Windows."
    }
    $ver = [Environment]::OSVersion.Version
    if ($ver.Major -lt 10) {
        throw ("Windows 10 or 11 is required (found " + $ver + ").")
    }
    # A bare drive root is never a sane install folder, and it would make the
    # running-instance check below match every process on the drive.
    if ($dest.TrimEnd("\") -match "^[A-Za-z]:$") {
        throw ("'" + $dest + "' is a drive root - pick a folder on it " +
               "instead, for example " + $dest.TrimEnd("\") + "\PromptForge.")
    }
    if (Test-OneDrivePath $dest) {
        if ($script:AllowOneDrive) {
            Write-Log ("install folder is inside OneDrive - continuing on " +
                       "explicit request. Consider pausing OneDrive sync " +
                       "during the install.") "warn"
        } else {
            throw ("'" + $dest + "' is synced by OneDrive. The install puts " +
                   "5-8 GB of Python environments there, which fills the " +
                   "OneDrive quota and causes sync conflicts and file-lock " +
                   "errors. Pick a folder outside OneDrive (for example " +
                   "C:\PromptForge), or pass -AllowOneDrive to override.")
        }
    }
    # Disk space on the destination drive.
    try {
        $qualifier = Split-Path -Qualifier $dest
        if ($qualifier) {
            $drive = New-Object System.IO.DriveInfo ($qualifier + "\")
            $freeGb = [Math]::Round($drive.AvailableFreeSpace / 1GB, 1)
            Write-Log ("free space on " + $qualifier + " -> " + $freeGb + " GB")
            if ($freeGb -lt 12) {
                throw ("Only " + $freeGb + " GB free on " + $qualifier +
                       " - at least 12 GB is needed for the software alone " +
                       "(AI models later need 20-40 GB more). Pick another drive.")
            }
            if ($freeGb -lt 40) {
                Write-Log ("NOTE: " + $freeGb + " GB free. The software fits, " +
                           "but AI models downloaded on first launch need " +
                           "20-40 GB more.") "warn"
            }
        }
    } catch {
        if ($_.Exception.Message -like "*Pick another drive*") { throw }
        Write-Log ("disk space check skipped: " + $_.Exception.Message) "warn"
    }
    # Internet reachability (needed for Python/torch/ComfyUI/Ollama/models).
    $reachable = 0
    foreach ($u in @("https://pypi.org", "https://github.com", "https://www.python.org")) {
        if (Test-Url $u) { $reachable++ } else { Write-Log ("unreachable: " + $u) "warn" }
    }
    if ($reachable -eq 0) {
        throw ("No internet connection (pypi.org, github.com and python.org " +
               "are all unreachable). The installer needs to download " +
               "components - connect and run it again.")
    }
    # A PromptForge already running FROM the destination would hold file locks.
    # Match on the normalized folder prefix ('C:\PromptForge\') so a sibling
    # like C:\PromptForge2 never triggers a false positive.
    try {
        $prefix = ($dest.TrimEnd("\") + "\")
        $running = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -and
                           $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) }
        if ($running) {
            throw ("PromptForge (or ComfyUI) is running from '" + $dest +
                   "'. Close it, then run the installer again.")
        }
    } catch {
        if ($_.Exception.Message -like "*Close it*") { throw }
    }
    # Write access.
    if (-not (Test-Path $dest)) {
        New-Item -ItemType Directory -Force $dest | Out-Null
    }
    $probe = Join-Path $dest (".write-test-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType File $probe -Force | Out-Null
    Remove-Item $probe -Force
    Write-Log "preflight passed"
}

function Step-CopyApp([string]$dest) {
    $src = Join-Path $script:Staging "app"
    if (-not (Test-Path (Join-Path $src "launch.ps1"))) {
        throw ("installer payload is incomplete (no app files at " + $src + ")")
    }
    # The payload never contains data\ or .venv\, so a re-install or upgrade
    # overwrites code while the user's gallery, models and settings survive.
    # Code-only folders are replaced wholesale first: a plain overwrite would
    # keep files that were DELETED from the app, and a stale .py module can
    # shadow current code. (backend\.venv sits next to backend\app - safe.)
    foreach ($d in @("backend\app", "frontend\dist", "docs")) {
        $newDir = Join-Path $src $d
        $oldDir = Join-Path $dest $d
        if ((Test-Path $newDir) -and (Test-Path $oldDir)) {
            Remove-Item $oldDir -Recurse -Force
        }
    }
    Copy-Item -Path (Join-Path $src "*") -Destination $dest -Recurse -Force
    $info = Join-Path $script:Staging "build-info.json"
    if (Test-Path $info) {
        Copy-Item $info (Join-Path $dest "build-info.json") -Force
        try {
            $bi = Get-Content $info -Raw | ConvertFrom-Json
            Write-Log ("app payload built " + $bi.built_at)
        } catch {}
    }
    Write-Log ("app files installed to " + $dest)
}

function Step-GitUpdates([string]$dest) {
    # Turn the installed folder into a real git clone (best effort): that is
    # what makes launch.ps1's auto-update work, so pushed fixes arrive on
    # this machine at every launch. Never fatal - without git or GitHub
    # access the machine simply stays on the payload version, and re-running
    # the installer updates it instead.
    if (Test-Path (Join-Path $dest ".git")) {
        Write-Log "already a git clone - the update channel is active"
        return
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Log ("git is not installed - automatic updates stay off " +
                   "(re-run this installer to update)") "warn"
        return
    }
    $repo = "https://github.com/HeSho420/PromptForge.git"
    $ok = $false
    $env:GIT_TERMINAL_PROMPT = "0"   # NEVER hang the install on a prompt
    try {
        Invoke-Process $git.Source ("-C """ + $dest + """ init") 2 | Out-Null
        Invoke-Process $git.Source ("-C """ + $dest + """ remote add origin " + $repo) 2 | Out-Null
        # schannel TLS survives AV certificate interception; interactive
        # credentials are disabled so a machine without stored GitHub
        # access fails fast instead of hanging a silent install.
        $code = Invoke-Process $git.Source ("-C """ + $dest + """ -c http.sslBackend=schannel " +
            "-c credential.interactive=false fetch --depth 1 origin main") 5
        if ($code -eq 0) {
            # data\, venvs, tools\ and frontend\dist are gitignored or
            # untracked - this aligns only the CODE with the repository,
            # and the pre-built UI the payload shipped stays in place.
            $code = Invoke-Process $git.Source ("-C """ + $dest + """ checkout -f -B main origin/main") 3
            if ($code -eq 0) { $ok = $true }
        }
    } finally {
        Remove-Item Env:GIT_TERMINAL_PROMPT -ErrorAction SilentlyContinue
    }
    if ($ok) {
        Write-Log "connected to the update channel - new versions arrive automatically at launch"
    } else {
        Remove-Item (Join-Path $dest ".git") -Recurse -Force -ErrorAction SilentlyContinue
        Write-Log ("could not reach the PromptForge repository from this " +
                   "machine - automatic updates stay off (re-run this " +
                   "installer to update)") "warn"
    }
}

function Step-Python {
    $py = Find-Python
    if ($py) {
        # AMD GPU (either stack) -> the GPU wheels are Python-3.12-only, so
        # a machine that only has e.g. 3.11 still gets a 3.12 install.
        if ((Resolve-GpuMode) -in @("rocm", "directml") -and (Get-PyMinor $py) -ne "3.12") {
            Write-Log ("Python " + (Get-PyMinor $py) + " found, but the AMD " +
                       "GPU stack needs Python 3.12 - installing 3.12 too")
        } else {
            Write-Log ("Python found: " + $py)
            $script:PythonExe = $py
            return
        }
    } else {
        Write-Log ("No suitable Python (need 3.10-3.13 64-bit) - installing " +
                   $script:PyVersion + " for the current user")
    }
    $setup = Join-Path $env:TEMP "promptforge-python-setup.exe"
    $installed = $false
    try {
        Download-File $script:PyUrl $setup 10000000
        $code = Invoke-Process $setup ("/quiet InstallAllUsers=0 PrependPath=1 " +
                                       "Include_launcher=1 Include_test=0") 20
        if ($code -ne 0) { throw ("python installer exit code " + $code) }
        $installed = $true
    } catch {
        Write-Log ("direct Python install failed: " + $_.Exception.Message) "warn"
    }
    if (-not $installed) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Log "trying winget as a fallback"
            Invoke-Process $winget.Source ("install -e --id Python.Python.3.12 " +
                "--silent --accept-package-agreements --accept-source-agreements " +
                "--disable-interactivity") 25 | Out-Null
        }
    }
    Remove-Item $setup -Force -ErrorAction SilentlyContinue
    $py = Find-Python
    if (-not $py) {
        throw ("Python did not install. Install Python 3.12 (64-bit) manually " +
               "from python.org and run this installer again.")
    }
    Write-Log ("Python ready: " + $py)
    $script:PythonExe = $py
}

function Ensure-Venv([string]$venvDir, [string]$label) {
    # Returns the venv's python.exe, creating OR REBUILDING the venv when
    # needed. Existence of python.exe is not enough: it is a tiny redirector
    # into the base Python, so if that base was uninstalled or upgraded the
    # file still exists but every run fails - and a resume would be stuck
    # forever behind misleading pip errors. Probe it for real.
    $venvPy = Join-Path $venvDir "Scripts\python.exe"
    if (Test-Path $venvPy) {
        # try/catch: a corrupt python.exe makes Start-Process itself throw -
        # that must count as "unhealthy", not abort the step.
        $healthy = $false
        try { $healthy = ((Invoke-Process $venvPy "-c ""import sys""" 5) -eq 0) } catch {}
        if ($healthy) {
            Write-Log ($label + " environment already exists")
            return $venvPy
        }
        Write-Log ($label + " environment exists but its Python no longer " +
                   "runs (base Python moved or was uninstalled) - rebuilding") "warn"
        Remove-Item $venvDir -Recurse -Force
    }
    Write-Log ("creating the " + $label + " Python environment")
    $code = Invoke-Process $script:PythonExe ("-m venv """ + $venvDir + """") 10
    if ($code -ne 0 -or -not (Test-Path $venvPy)) {
        throw ("could not create the " + $label + " venv at " + $venvDir)
    }
    return $venvPy
}

function Step-BackendVenv([string]$dest) {
    $venvPy = Ensure-Venv (Join-Path $dest "backend\.venv") "backend"
    $code = Invoke-Process $venvPy "-m pip install --upgrade pip --quiet" 10
    if ($code -ne 0) {
        Write-Log "pip self-update failed - continuing with the bundled pip" "warn"
    }
    $req = Join-Path $dest "backend\requirements.txt"
    $code = Invoke-Process $venvPy `
        ("-m pip install --retries 8 --timeout 120 -r """ + $req + """") 20
    if ($code -ne 0) { throw "pip could not install the backend requirements" }
    # Quick functional proof, not just exit codes:
    $code = Invoke-Process $venvPy "-c ""import flask, PIL, imageio_ffmpeg""" 5
    if ($code -ne 0) { throw "backend packages installed but do not import" }
    Write-Log "backend environment ready"
}

function Step-TorchSam([string]$dest) {
    if ($script:LightMode) { Write-Log "[light mode] torch + SAM skipped"; return }
    $venvPy = Join-Path $dest "backend\.venv\Scripts\python.exe"
    $code = Invoke-Process $venvPy "-c ""import torch, segment_anything""" 5
    if ($code -eq 0) { Write-Log "torch + SAM already installed"; return }
    $mode = Get-TorchModeFor $venvPy
    if ($mode -eq "cuda") {
        Write-Log "NVIDIA GPU detected - installing CUDA torch (about 2.5 GB, be patient)"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision --index-url " + $script:TorchGpuIndex) 120
        if ($code -ne 0) { throw "torch installation failed" }
    } elseif ($mode -eq "rocm") {
        Write-Log "AMD ROCm GPU - installing AMD's ROCm torch (several GB, be patient)"
        if (-not (Install-RocmTorch $venvPy)) {
            Write-Log "ROCm torch failed to install - falling back to CPU torch" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision") 90
            if ($code -ne 0) { throw "torch installation failed" }
        }
    } elseif ($mode -eq "directml") {
        Write-Log "AMD GPU - installing the DirectML torch stack (about 2 GB)"
        if (-not (Install-DirectmlTorch $venvPy)) {
            Write-Log "DirectML torch failed to install - falling back to CPU torch" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision") 90
            if ($code -ne 0) { throw "torch installation failed" }
        }
    } elseif ($mode -eq "xpu") {
        Write-Log "Intel Arc GPU - installing torch's XPU build (about 2 GB)"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision --index-url " + $script:TorchXpuIndex) 120
        if ($code -ne 0) {
            Write-Log "XPU torch failed to install - falling back to CPU torch" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision") 90
            if ($code -ne 0) { throw "torch installation failed" }
        }
    } else {
        Write-Log "no supported GPU - installing CPU torch (renders work, slowly)"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision") 90
        if ($code -ne 0) { throw "torch installation failed" }
    }
    if ($mode -ne "cpu") { Test-TorchGpu $venvPy }
    $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
        "numpy """ + $script:SamZipUrl + """") 30
    if ($code -ne 0) { throw "segment-anything installation failed" }
    $code = Invoke-Process $venvPy "-c ""import torch, segment_anything""" 5
    if ($code -ne 0) { throw "torch/SAM installed but do not import" }
    Write-Log "torch + SAM ready"
}

function Step-ComfyFiles([string]$dest) {
    if ($script:LightMode) { Write-Log "[light mode] ComfyUI download skipped"; return }
    $comfy = Join-Path $dest "tools\ComfyUI"
    if (Test-Path (Join-Path $comfy "main.py")) {
        Write-Log ("ComfyUI already present at " + $comfy)
        return
    }
    # master.zip is generated on-the-fly per request and stalls on flaky
    # networks (seen live: 4/4 read timeouts). Release-tag archives are
    # cached by github and far more reliable - resolve the latest tag first.
    $urls = @()
    if ((Resolve-GpuMode) -eq "directml") {
        # DirectML machines are pinned: torch-directml is frozen at torch
        # 2.4.1 and releases newer than this tag crash on import there.
        Write-Log ("DirectML machine - installing ComfyUI " + $script:ComfyDmlTag +
                   " (newest release that runs on torch 2.4.1)")
        $urls += ("https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/" + $script:ComfyDmlTag + ".zip")
        $urls += ("https://codeload.github.com/comfyanonymous/ComfyUI/zip/refs/tags/" + $script:ComfyDmlTag)
    } else {
        try {
            $req = [System.Net.HttpWebRequest]::Create(
                "https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest")
            $req.UserAgent = "PromptForge-Installer/1.0"
            $req.Timeout = 15000
            $resp = $req.GetResponse()
            $rd = New-Object System.IO.StreamReader ($resp.GetResponseStream())
            $json = $rd.ReadToEnd(); $rd.Close(); $resp.Close()
            if ($json -match '"tag_name"\s*:\s*"([^"]+)"') {
                $tag = $Matches[1]
                Write-Log ("latest ComfyUI release: " + $tag)
                $urls += ("https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/" + $tag + ".zip")
                $urls += ("https://codeload.github.com/comfyanonymous/ComfyUI/zip/refs/tags/" + $tag)
            }
        } catch {
            Write-Log ("could not resolve the latest ComfyUI release (" +
                       $_.Exception.Message + ") - using the master branch") "warn"
        }
        $urls += $script:ComfyZipUrl
        $urls += "https://codeload.github.com/comfyanonymous/ComfyUI/zip/refs/heads/master"
    }

    $zip = Join-Path $env:TEMP "promptforge-comfyui.zip"
    $tmp = Join-Path $env:TEMP ("promptforge-comfyui-" + [Guid]::NewGuid().ToString("N"))
    try {
        Download-File $urls $zip 1000000
        New-Item -ItemType Directory -Force $tmp | Out-Null
        Expand-Zip $zip $tmp
        $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
        if (-not $inner -or -not (Test-Path (Join-Path $inner.FullName "main.py"))) {
            throw "the ComfyUI archive did not contain main.py - github layout changed?"
        }
        New-Item -ItemType Directory -Force (Join-Path $dest "tools") | Out-Null
        if (Test-Path $comfy) {
            # Leftover from an interrupted attempt (no main.py, or the check
            # above would have returned). Moving INTO it would nest the new
            # folder one level too deep - replace it instead.
            Write-Log "removing an incomplete ComfyUI folder from an earlier attempt" "warn"
            Remove-Item $comfy -Recurse -Force
        }
        Move-Item $inner.FullName $comfy
    } finally {
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Log ("ComfyUI installed at " + $comfy +
               " (launch.ps1 finds tools\ComfyUI first)")
}

function Step-ComfyVenv([string]$dest) {
    if ($script:LightMode) { Write-Log "[light mode] ComfyUI environment skipped"; return }
    $comfy = Join-Path $dest "tools\ComfyUI"
    if (-not (Test-Path (Join-Path $comfy "main.py"))) {
        throw "ComfyUI files are missing - the previous step failed"
    }
    $venvPy = Ensure-Venv (Join-Path $comfy ".venv") "ComfyUI"
    $code = Invoke-Process $venvPy "-c ""import torch, yaml, aiohttp, requests""" 5
    if ($code -eq 0) { Write-Log "ComfyUI environment already complete"; return }
    $code = Invoke-Process $venvPy "-m pip install --upgrade pip --quiet" 10
    if ($code -ne 0) {
        Write-Log "pip self-update failed - continuing with the bundled pip" "warn"
    }
    $mode = Get-TorchModeFor $venvPy
    if ($mode -eq "cuda") {
        Write-Log "installing CUDA torch for ComfyUI (about 2.5 GB)"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision torchaudio --index-url " + $script:TorchGpuIndex) 120
    } elseif ($mode -eq "rocm") {
        Write-Log "installing AMD ROCm torch for ComfyUI (several GB)"
        if (Install-RocmTorch $venvPy) { $code = 0 } else {
            Write-Log "ROCm torch failed - falling back to CPU torch for ComfyUI" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision torchaudio") 90
        }
    } elseif ($mode -eq "directml") {
        Write-Log "installing the DirectML torch stack for ComfyUI (about 2 GB)"
        if (Install-DirectmlTorch $venvPy) { $code = 0 } else {
            Write-Log "DirectML torch failed - falling back to CPU torch for ComfyUI" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision torchaudio") 90
        }
    } elseif ($mode -eq "xpu") {
        Write-Log "installing torch's XPU build for ComfyUI (about 2 GB)"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision torchaudio --index-url " + $script:TorchXpuIndex) 120
        if ($code -ne 0) {
            Write-Log "XPU torch failed - falling back to CPU torch for ComfyUI" "warn"
            $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
                "torch torchvision torchaudio") 90
        }
    } else {
        Write-Log "installing CPU torch for ComfyUI"
        $code = Invoke-Process $venvPy ("-m pip install --retries 10 --timeout 300 " +
            "torch torchvision torchaudio") 90
    }
    if ($code -ne 0) { throw "torch installation for ComfyUI failed" }
    if ($mode -ne "cpu") { Test-TorchGpu $venvPy }
    $req = Join-Path $comfy "requirements.txt"
    $code = Invoke-Process $venvPy `
        ("-m pip install --retries 10 --timeout 300 -r """ + $req + """") 60
    if ($code -ne 0) { throw "pip could not install ComfyUI's requirements" }
    $code = Invoke-Process $venvPy "-c ""import torch, yaml, aiohttp, requests""" 5
    if ($code -ne 0) { throw "ComfyUI packages installed but do not import" }
    Write-Log "ComfyUI environment ready"
}

function Step-Ollama {
    if ($script:LightMode) { Write-Log "[light mode] Ollama skipped"; return }
    $ollama = Find-Ollama
    if ($ollama) { Write-Log ("Ollama found: " + $ollama); return }
    Write-Log "installing Ollama (local planning LLM host)"
    $setup = Join-Path $env:TEMP "promptforge-ollama-setup.exe"
    $installed = $false
    try {
        Download-File $script:OllamaUrl $setup 50000000
        $code = Invoke-Process $setup "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" 25
        if ($code -ne 0) { throw ("ollama installer exit code " + $code) }
        $installed = $true
    } catch {
        Write-Log ("direct Ollama install failed: " + $_.Exception.Message) "warn"
    }
    if (-not $installed) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Log "trying winget as a fallback"
            Invoke-Process $winget.Source ("install -e --id Ollama.Ollama --silent " +
                "--accept-package-agreements --accept-source-agreements " +
                "--disable-interactivity") 25 | Out-Null
        }
    }
    Remove-Item $setup -Force -ErrorAction SilentlyContinue
    if (-not (Find-Ollama)) {
        throw ("Ollama did not install. Install it manually from ollama.com " +
               "and run this installer again (or just launch PromptForge - " +
               "its launcher retries this too).")
    }
    Write-Log "Ollama ready (the launcher pulls a model sized to this machine)"
}

function Step-Shortcut([string]$dest) {
    if ($NoShortcut) { Write-Log "shortcut skipped (-NoShortcut)"; return }
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnk = Join-Path $desktop "PromptForge.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe")
    $sc.Arguments = ("-NoProfile -ExecutionPolicy Bypass -File """ +
                     (Join-Path $dest "launch.ps1") + """")
    $sc.WorkingDirectory = $dest
    $sc.Description = "PromptForge - local AI image studio"
    $sc.Save()
    Write-Log ("desktop shortcut created: " + $lnk)
}

function Step-Verify([string]$dest) {
    $problems = @()
    if (-not (Test-Path (Join-Path $dest "launch.ps1"))) { $problems += "launch.ps1 missing" }
    if (-not (Test-Path (Join-Path $dest "frontend\dist\index.html"))) {
        $problems += "web UI files missing"
    }
    $venvPy = Join-Path $dest "backend\.venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        if ((Invoke-Process $venvPy "-c ""import flask, PIL, imageio_ffmpeg""" 5) -ne 0) {
            $problems += "backend packages broken"
        }
    } else { $problems += "backend venv missing" }
    if (-not $script:LightMode) {
        $comfyPy = Join-Path $dest "tools\ComfyUI\.venv\Scripts\python.exe"
        if (-not (Test-Path $comfyPy)) { $problems += "ComfyUI venv missing" }
        if (-not (Find-Ollama)) { $problems += "Ollama missing" }
    }
    foreach ($p in $problems) { Write-Log ("verify: " + $p) "warn" }
    if ($problems.Count -gt 0) {
        throw ($problems -join "; ")
    }
    Write-Log "all components verified"
}

# ------------------------------------------------------------- main flow ---

function Run-Install([string]$dest) {
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
    $script:LogFile = Join-Path $dest "install.log"
    $script:Failed = @()
    Write-Log ("=== PromptForge setup -> " + $dest + " ===")
    if ($script:LightMode) { Write-Log "LIGHT TEST MODE - heavy downloads skipped" "warn" }

    $steps = @(
        @{ n = "Checking this PC (preflight)";        b = { Step-Preflight $dest } },
        @{ n = "Installing PromptForge files";        b = { Step-CopyApp $dest } },
        @{ n = "Connecting automatic updates";        b = { Step-GitUpdates $dest } },
        @{ n = "Python 3.12";                         b = { Step-Python } },
        @{ n = "Backend environment";                 b = { Step-BackendVenv $dest } },
        @{ n = "AI libraries (torch + SAM)";          b = { Step-TorchSam $dest } },
        @{ n = "ComfyUI (render engine)";             b = { Step-ComfyFiles $dest } },
        @{ n = "ComfyUI environment";                 b = { Step-ComfyVenv $dest } },
        @{ n = "Ollama (local LLM host)";             b = { Step-Ollama } },
        @{ n = "Desktop shortcut";                    b = { Step-Shortcut $dest } },
        @{ n = "Final verification";                  b = { Step-Verify $dest } }
    )
    if ($script:Progress -ne $null) {
        $script:Progress.Minimum = 0
        $script:Progress.Maximum = $steps.Count
        $script:Progress.Value = 0
    }
    $i = 0
    $stop = $false
    foreach ($s in $steps) {
        $i++
        if (-not $stop) {
            $ok = Invoke-Step $s.n $s.b
            # The first two steps are load-bearing for everything after them.
            if (-not $ok -and $i -le 2) {
                Write-Log "stopping - later steps depend on this one" "error"
                $stop = $true
            }
        } else {
            Write-Log ("skipped: " + $s.n)
        }
        if ($script:Progress -ne $null) { $script:Progress.Value = $i; Update-Ui }
    }

    if ($script:Failed.Count -eq 0) {
        Write-Log "=== DONE - PromptForge is installed ==="
        Write-Log "First launch downloads the AI models (several GB) automatically."
        if (-not $NoLaunch) {
            # Auto-launch: -ExecutionPolicy Bypass makes this work on
            # machines where running .ps1 scripts is disabled by default.
            Write-Log "starting PromptForge..."
            try {
                # The script path is quoted BY HAND: PS 5.1's Start-Process
                # joins -ArgumentList with spaces and no quoting, so an
                # install dir containing a space would otherwise split the
                # path and the app would never start (measured live).
                Start-Process -FilePath (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe") `
                    -WorkingDirectory $dest `
                    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                  ('"' + (Join-Path $dest "launch.ps1") + '"')
            } catch {
                Write-Log ("could not auto-start PromptForge (" +
                           $_.Exception.Message + ") - use the desktop shortcut") "warn"
            }
        }
        return $true
    }
    Write-Log ("=== FINISHED WITH PROBLEMS: " + ($script:Failed -join "; ") +
               " ===") "error"
    Write-Log ("Run the installer again to resume - completed steps are " +
               "skipped. PromptForge's own launcher also self-repairs " +
               "missing pieces at startup.") "warn"
    return $false
}

function Show-Gui {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "PromptForge Setup"
    $form.Size = New-Object System.Drawing.Size(680, 560)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = ("Choose the folder PromptForge will live in. Everything the " +
                 "app needs goes inside it (the Python and Ollama runtimes " +
                 "install into your user profile).")
    $lbl.Location = New-Object System.Drawing.Point(14, 12)
    $lbl.Size = New-Object System.Drawing.Size(640, 34)
    $form.Controls.Add($lbl)

    $dirBox = New-Object System.Windows.Forms.TextBox
    $dirBox.Location = New-Object System.Drawing.Point(14, 52)
    $dirBox.Size = New-Object System.Drawing.Size(520, 24)
    $dirBox.Text = "C:\PromptForge"
    $form.Controls.Add($dirBox)

    $browse = New-Object System.Windows.Forms.Button
    $browse.Text = "Browse..."
    $browse.Location = New-Object System.Drawing.Point(544, 50)
    $browse.Size = New-Object System.Drawing.Size(110, 26)
    $browse.Add_Click({
        $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $dlg.Description = "Pick the folder PromptForge will be installed in"
        if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $dirBox.Text = (Join-Path $dlg.SelectedPath "PromptForge")
        }
    })
    $form.Controls.Add($browse)

    $install = New-Object System.Windows.Forms.Button
    $install.Text = "Install"
    $install.Location = New-Object System.Drawing.Point(14, 86)
    $install.Size = New-Object System.Drawing.Size(140, 32)
    $form.Controls.Add($install)

    $status = New-Object System.Windows.Forms.Label
    $status.Text = "Ready."
    $status.Location = New-Object System.Drawing.Point(168, 94)
    $status.Size = New-Object System.Drawing.Size(486, 20)
    $form.Controls.Add($status)

    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.Location = New-Object System.Drawing.Point(14, 126)
    $bar.Size = New-Object System.Drawing.Size(640, 18)
    $form.Controls.Add($bar)

    $log = New-Object System.Windows.Forms.TextBox
    $log.Multiline = $true
    $log.ReadOnly = $true
    $log.ScrollBars = "Vertical"
    $log.Font = New-Object System.Drawing.Font("Consolas", 8.5)
    $log.Location = New-Object System.Drawing.Point(14, 154)
    $log.Size = New-Object System.Drawing.Size(640, 356)
    $form.Controls.Add($log)

    $script:LogBox = $log
    $script:Progress = $bar
    $script:StatusLb = $status

    $install.Add_Click({
        $dest = $dirBox.Text.Trim().TrimEnd("\")
        if (-not $dest) {
            [System.Windows.Forms.MessageBox]::Show("Pick an install folder first.")
            return
        }
        if ((Test-OneDrivePath $dest) -and -not $script:AllowOneDrive) {
            $ans = [System.Windows.Forms.MessageBox]::Show(
                ("The folder you picked is synced by OneDrive." +
                 [Environment]::NewLine + [Environment]::NewLine +
                 "PromptForge puts 5-8 GB of Python environments there, " +
                 "which fills your OneDrive quota and causes sync errors " +
                 "during the install." + [Environment]::NewLine +
                 [Environment]::NewLine +
                 "Recommended: choose No and pick a folder outside OneDrive " +
                 "(for example C:\PromptForge)." + [Environment]::NewLine +
                 [Environment]::NewLine + "Install into OneDrive anyway?"),
                "PromptForge Setup",
                [System.Windows.Forms.MessageBoxButtons]::YesNo,
                [System.Windows.Forms.MessageBoxIcon]::Warning)
            if ($ans -ne [System.Windows.Forms.DialogResult]::Yes) { return }
            $script:AllowOneDrive = $true   # user explicitly accepted
        }
        $install.Enabled = $false
        $browse.Enabled = $false
        $dirBox.Enabled = $false
        $form.ControlBox = $false
        $ok = $false
        try {
            $ok = Run-Install $dest
        } catch {
            Write-Log ("unexpected error: " + $_.Exception.Message) "error"
        }
        $form.ControlBox = $true
        Set-Status "Finished."
        if ($ok) {
            [System.Windows.Forms.MessageBox]::Show(
                ("PromptForge is installed and STARTING NOW - a browser " +
                 "tab opens when it is ready." + [Environment]::NewLine +
                 [Environment]::NewLine +
                 "The first launch downloads the AI models (several GB) " +
                 "and can take a while. Next time, use the PromptForge " +
                 "shortcut on the desktop."),
                "PromptForge Setup",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
        } else {
            [System.Windows.Forms.MessageBox]::Show(
                ("Some steps failed - see the log (also saved as install.log " +
                 "in the install folder). Run the installer again to resume; " +
                 "completed steps are skipped."),
                "PromptForge Setup",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning) | Out-Null
        }
        $install.Enabled = $true
        $browse.Enabled = $true
        $dirBox.Enabled = $true
    })

    if ($UiSmokeTest) {
        $form.Show()
        Update-Ui
        Write-Log "UI smoke test passed"
        $form.Close()
        $form.Dispose()
        return
    }
    [void]$form.ShowDialog()
}

# Any crash that escapes to here would close the console with no message
# (the .bat cannot pause a killed/failed PowerShell child usefully in GUI
# mode) - persist the error and show it before exiting.
try {
    if ($UiSmokeTest) {
        Show-Gui
    } elseif ($Silent) {
        if (-not $InstallDir) { throw "-Silent needs -InstallDir <folder>" }
        $ok = Run-Install $InstallDir
        if (-not $ok) { exit 1 }
        exit 0
    } else {
        Show-Gui
    }
} catch {
    $err = ("PromptForge setup crashed: " + $_.Exception.Message +
            [Environment]::NewLine + $_.ScriptStackTrace)
    try {
        [IO.File]::WriteAllText((Join-Path $env:TEMP "promptforge-setup-error.log"),
                                $err, [Text.Encoding]::ASCII)
    } catch {}
    Write-Host $err -ForegroundColor Red
    if (-not $Silent) {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.MessageBox]::Show($err, "PromptForge Setup",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
        } catch {}
    }
    exit 1
}
