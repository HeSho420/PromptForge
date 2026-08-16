# PromptForge zip bootstrap (started by install.bat from an unpacked
# "Download ZIP" copy of the repository).
#
# A zip download has no .git folder, so the app could never receive pushed
# updates. This converts the unpacked folder into a REAL clone in place
# (best effort - offline or credential-less machines simply skip it and
# still get a working app), then hands over to launch.ps1 which installs
# everything else.
#
# Windows PowerShell 5.1 syntax only; never fatal before the launch line.

$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent   # ...\PromptForge (repo root)

Write-Host "  == PromptForge bootstrap ==" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $root ".git"))) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        Write-Host "  Connecting this folder to the update channel..." -ForegroundColor DarkGray
        $env:GIT_TERMINAL_PROMPT = "0"
        try {
            & $git.Source -C $root init 2>&1 | Out-Null
            & $git.Source -C $root remote add origin https://github.com/HeSho420/PromptForge.git 2>&1 | Out-Null
            & $git.Source -C $root -c http.sslBackend=schannel -c credential.interactive=false `
                fetch --depth 1 origin main 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                & $git.Source -C $root checkout -f -B main origin/main 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  [ok] Updates will arrive automatically at every launch." -ForegroundColor Green
                } else {
                    Remove-Item (Join-Path $root ".git") -Recurse -Force -ErrorAction SilentlyContinue
                }
            } else {
                Remove-Item (Join-Path $root ".git") -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  (repository not reachable from this machine - the app still works; updates stay off)" -ForegroundColor DarkGray
            }
        } finally {
            Remove-Item Env:GIT_TERMINAL_PROMPT -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "  (git not installed - the app still works; updates stay off)" -ForegroundColor DarkGray
    }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "launch.ps1")
exit $LASTEXITCODE
