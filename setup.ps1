# PromptForge - legacy entry point, kept so old instructions stay valid.
# Everything now lives in launch.ps1 (GPU detection, self-repair, updates).
# Prefer double-clicking launch.bat - it works even when running PowerShell
# scripts is disabled on this PC.
param([switch]$Dev)
if ($Dev) {
    Write-Host "The -Dev flow moved: run 'npm run dev' in frontend\ next to a running backend." -ForegroundColor Yellow
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "launch.ps1")
exit $LASTEXITCODE
