# PromptForge — Windows setup & run (PowerShell)
# Usage, from the project root:
#   .\setup.ps1            # first-time setup + start the app
#   .\setup.ps1 -Dev       # setup + start backend AND the Vite dev server
# If scripts are blocked, run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param([switch]$Dev)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "== PromptForge setup ==" -ForegroundColor Cyan

# --- backend ---
# Prefer the py launcher (present even when python.exe is not on PATH, which
# is the Windows default); fall back to python.exe.
$sysPython = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $sysPython = { py -3 @args }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $sysPython = { python @args }
} else {
    throw "Python not found. Install Python 3.12+ from python.org and re-run."
}

$venv = Join-Path $root "backend\.venv"
if (-not (Test-Path $venv)) {
    Write-Host "Creating Python venv..."
    & $sysPython -m venv $venv
}
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"
& $pip install -q -r (Join-Path $root "backend\requirements.txt")
Write-Host "Backend dependencies ready." -ForegroundColor Green

# --- frontend ---
$npm = Get-Command npm -ErrorAction SilentlyContinue
if ($npm) {
    Push-Location (Join-Path $root "frontend")
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing frontend dependencies (one-time)..."
        npm install
    }
    if (-not $Dev) {
        Write-Host "Building frontend..."
        npm run build   # backend serves frontend/dist at http://127.0.0.1:8000
    }
    Pop-Location
} else {
    Write-Warning "Node.js/npm not found - skipping the frontend. Install Node 20+ from nodejs.org for the UI."
}

# --- run ---
if ($Dev -and $npm) {
    Write-Host "Starting backend (8000) + Vite dev server (5173)..." -ForegroundColor Cyan
    $backend = Start-Process -PassThru -NoNewWindow $python -ArgumentList (Join-Path $root "backend\run.py")
    try {
        Push-Location (Join-Path $root "frontend")
        npm run dev
    } finally {
        Pop-Location
        if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id }
    }
} else {
    Write-Host "Starting PromptForge at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    & $python (Join-Path $root "backend\run.py")
}
