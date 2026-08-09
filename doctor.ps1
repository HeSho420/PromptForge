# PromptForge doctor - verifies EVERYTHING on this machine and says what
# to do about anything broken. Read-only: it changes nothing.
#
#   .\doctor.ps1
#
# Writes the same report to data\logs\doctor-report.txt so it can be
# shared. launch.ps1 is the repair tool; this is the diagnosis.

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$reportPath = Join-Path $root "data\logs\doctor-report.txt"
New-Item -ItemType Directory -Force (Split-Path $reportPath) | Out-Null
$script:lines = @()

function Say($status, $text) {
    $color = switch ($status) {
        "OK" { "Green" }; "FIX" { "Yellow" }; "BAD" { "Red" }; default { "Gray" }
    }
    $line = "[{0,-3}] {1}" -f $status, $text
    Write-Host "  $line" -ForegroundColor $color
    $script:lines += $line
}

function Get-Ver($py) {
    if (-not (Test-Path $py)) { return "" }
    try { return (& $py -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null | Out-String).Trim() } catch { return "" }
}

Write-Host ""
Write-Host "  == PromptForge doctor ==" -ForegroundColor Cyan
Write-Host ""

# --- git / updates ---------------------------------------------------------------
if ((Test-Path (Join-Path $root ".git")) -and (Get-Command git -ErrorAction SilentlyContinue)) {
    Push-Location $root
    $commit = (git rev-parse --short HEAD 2>$null | Out-String).Trim()
    git fetch --quiet origin 2>$null
    $branch = (git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
    $behind = (git rev-list --count "HEAD..origin/$branch" 2>$null | Out-String).Trim()
    Pop-Location
    if ($behind -match '^\d+$' -and [int]$behind -gt 0) {
        Say "FIX" "version $commit is $behind update(s) behind - run launch.ps1 (it pulls automatically)"
    } else {
        Say "OK" "version $commit is current"
    }
} else {
    Say "FIX" "not a git clone (or git missing) - updates cannot arrive; clone the repository"
}

# --- GPU -------------------------------------------------------------------------
$gpuMode = "cpu"
$gpuName = ""
try {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = $true }
    if ($smi) { $gpuMode = "cuda" }
} catch {}
try {
    $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name })
    $gpuName = ($gpus -join ", ")
    if ($gpuMode -ne "cuda") {
        foreach ($n in $gpus) {
            if ($n -match "Radeon.+(RX\s?90\d0|RX\s?7900|RX\s?7800|RX\s?7700|8[89]0M|860M|80[456]0S)") { $gpuMode = "rocm" }
        }
        if ($gpuMode -ne "rocm") {
            foreach ($n in $gpus) { if ($n -match "Radeon|AMD") { $gpuMode = "directml" } }
        }
    }
} catch {}
Say "OK" "GPU: $gpuName -> mode '$gpuMode'"
if ($gpuMode -eq "directml") {
    Say "OK" "this Radeon is outside AMD's classic ROCm wheel list - it renders through DirectML (or a ROCm-SDK torch when one is installed)"
}
$needPy = if ($gpuMode -in @("rocm", "directml")) { "3.12" } else { "" }

# --- backend environment ---------------------------------------------------------
$bpy = Join-Path $root "backend\.venv\Scripts\python.exe"
$bver = Get-Ver $bpy
if (-not $bver) {
    Say "FIX" "backend environment missing - run launch.ps1"
} elseif ($needPy -and $bver -ne $needPy) {
    Say "FIX" "backend is Python $bver but AMD GPU wheels need $needPy - run launch.ps1 (it rebuilds)"
} else {
    Say "OK" "backend environment: Python $bver"
    $probe = (& $bpy -c "import flask, PIL; print('ok')" 2>$null | Out-String).Trim()
    if ($probe -eq "ok") { Say "OK" "backend dependencies import" }
    else { Say "FIX" "backend dependencies broken - run launch.ps1 (self-repairs)" }
}

# --- ComfyUI ----------------------------------------------------------------------
$comfyDir = $null
foreach ($c in @($env:PROMPTFORGE_COMFYUI_PATH, (Join-Path $root "tools\ComfyUI"),
                 "$env:USERPROFILE\ComfyUI", "$env:USERPROFILE\Documents\ComfyUI")) {
    if ($c -and (Test-Path $c)) { $comfyDir = $c; break }
}
if (-not $comfyDir) {
    Say "FIX" "no ComfyUI install found - run launch.ps1 (it downloads one)"
} else {
    Say "OK" "ComfyUI at $comfyDir"
    $cpy = Join-Path $comfyDir ".venv\Scripts\python.exe"
    $cver = Get-Ver $cpy
    if (-not $cver) {
        Say "FIX" "ComfyUI has no Python environment - run launch.ps1"
    } elseif ($needPy -and $cver -ne $needPy) {
        Say "FIX" "ComfyUI env is Python $cver but AMD GPU wheels need $needPy - run launch.ps1 (it rebuilds)"
    } else {
        $torchLine = (& $cpy -c "import torch;print(torch.__version__);print(int(torch.cuda.is_available()))" 2>$null | Out-String).Trim() -split "`r?`n"
        if (-not $torchLine[0]) {
            Say "FIX" "ComfyUI env (Python $cver) has NO torch - run launch.ps1 (installs the right build)"
        } else {
            Say "OK" "ComfyUI env: Python $cver, torch $($torchLine[0])"
            if ($gpuMode -eq "directml") {
                if ($torchLine.Count -gt 1 -and $torchLine[1] -eq "1") {
                    Say "OK" "torch sees the Radeon natively (ROCm SDK build) - DirectML not needed"
                } else {
                    $dml = (& $cpy -c "import torch_directml;d=torch_directml.device();print('dev-ok')" 2>$null | Out-String).Trim()
                    if ($dml -match "dev-ok") { Say "OK" "DirectML device opens - Radeon renders on the GPU" }
                    else { Say "FIX" "torch-directml missing or its device will not open - run launch.ps1 (it swaps the stack in); if it keeps failing, share data\logs\directml-install.log" }
                }
            } elseif ($gpuMode -ne "cpu") {
                if ($torchLine.Count -gt 1 -and $torchLine[1] -eq "1") {
                    Say "OK" "torch sees the GPU"
                } else {
                    Say "FIX" "torch CANNOT see the GPU - run launch.ps1 (auto-repairs on NVIDIA; on AMD check the Adrenalin driver)"
                }
            }
        }
    }
    # live render check when it is up
    $up = $false
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -TimeoutSec 3
        $up = $true
    } catch {}
    if ($up) {
        try {
            $graph = @{ prompt = @{
                "1" = @{ class_type = "EmptyImage"; inputs = @{ width = 64; height = 64; batch_size = 1; color = 8355711 } }
                "2" = @{ class_type = "SaveImage"; inputs = @{ filename_prefix = "pf_doctor"; images = @("1", 0) } }
            } } | ConvertTo-Json -Depth 6
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8188/prompt" -Method Post -ContentType "application/json" -Body $graph -TimeoutSec 10
            $ok = $false
            for ($i = 0; $i -lt 25; $i++) {
                Start-Sleep -Milliseconds 600
                try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:8188/history/$($resp.prompt_id)" -TimeoutSec 5 } catch { continue }
                if ($h.($resp.prompt_id).outputs) { $ok = $true; break }
            }
            if ($ok) { Say "OK" "ComfyUI RENDERS (live test image produced)" }
            else { Say "BAD" "ComfyUI runs but the test render never finished - see data\logs\comfyui-err.log" }
        } catch { Say "BAD" "ComfyUI runs but refused the test graph: $($_.Exception.Message)" }
    } else {
        Say "FIX" "ComfyUI is not running right now - launch.ps1 starts and verifies it"
    }
}

# --- backend + network -----------------------------------------------------------
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
    Say "OK" "PromptForge backend is running"
} catch { Say "FIX" "backend not running - start with launch.ps1" }
$fw = @(Get-NetFirewallRule -DisplayName "PromptForge LAN*" -ErrorAction SilentlyContinue)
if ($fw.Count -ge 2) { Say "OK" "firewall rules for LAN features present" }
else { Say "FIX" "LAN firewall rules missing - run allow-lan.ps1 as administrator (peer features stay blocked until then)" }
$profileCat = (Get-NetConnectionProfile | Select-Object -First 1).NetworkCategory
if ("$profileCat" -eq "Public") { Say "FIX" "network profile is PUBLIC - set it to Private or LAN features stay blocked" }
else { Say "OK" "network profile: $profileCat" }

# --- disk -------------------------------------------------------------------------
$drive = (Get-Item $root).PSDrive
$freeGb = [math]::Round($drive.Free / 1GB, 1)
if ($freeGb -lt 25) { Say "FIX" "only $freeGb GB free on $($drive.Name): - models need room (100+ GB for the full library)" }
else { Say "OK" "$freeGb GB free on drive $($drive.Name):" }

Write-Host ""
$script:lines | Out-File -Encoding utf8 $reportPath
Write-Host "  Report written to $reportPath" -ForegroundColor Cyan
Write-Host "  Anything marked FIX: run .\launch.ps1 (or the named script) and re-run the doctor." -ForegroundColor Cyan
if ($MyInvocation.InvocationName -ne ".") { Read-Host "Press Enter to close" | Out-Null }
