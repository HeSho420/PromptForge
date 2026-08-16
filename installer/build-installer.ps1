# Builds the single-file PromptForge installer.
#
#   .\installer\build-installer.ps1
#       -> writes PromptForge-Setup.bat to the Desktop (default)
#
# Run it again after ANY change to the app - the output always embeds the
# current code. The produced .bat is fully self-contained: copy it to any
# 64-bit Windows 10/11 PC, double-click, pick a folder, press Install.
#
# How the single file works: this script zips the runtime files (backend
# source, pre-built web UI, launcher, installer core), base64-encodes the
# zip and appends it to a small batch header after a ::PAYLOAD:: marker.
# On the target the header extracts the payload with PowerShell (built into
# Windows - no other tools needed) and starts installer.ps1.
#
# The web UI ships PRE-BUILT (frontend\dist), so the target never needs
# Node.js. This script rebuilds dist when the sources changed (needs npm on
# THIS machine only); use -SkipFrontendBuild to pack the existing dist as-is.

param(
    [string]$OutFile = (Join-Path ([Environment]::GetFolderPath("Desktop")) "PromptForge-Setup.bat"),
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent   # ...\promptforge

Write-Host "== Building the PromptForge single-file installer =="
Write-Host ("repo: " + $root)

# --- 1. Make sure the web UI is built (target machines have no Node) --------
$dist = Join-Path $root "frontend\dist\index.html"
$needBuild = $false
if (-not (Test-Path $dist)) {
    $needBuild = $true
} else {
    $distTime = (Get-Item $dist).LastWriteTime
    $newest = Get-ChildItem (Join-Path $root "frontend\src") -Recurse -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest -and $newest.LastWriteTime -gt $distTime) { $needBuild = $true }
}
if ($SkipFrontendBuild) { $needBuild = $false }
if ($needBuild) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        if (Test-Path $dist) {
            Write-Warning ("frontend sources are newer than dist but npm is " +
                           "missing - packing the OLD dist. Install Node 20+ " +
                           "and rebuild for an up-to-date UI.")
        } else {
            throw ("frontend\dist is missing and npm is not installed - the " +
                   "installer would ship without a UI. Install Node 20+ and " +
                   "run 'npm install' + 'npm run build' in frontend\ first.")
        }
    } else {
        Write-Host "frontend changed - rebuilding the web UI..."
        Push-Location (Join-Path $root "frontend")
        try {
            if (-not (Test-Path "node_modules")) {
                & npm install --no-fund --no-audit | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
            }
            & npm run build | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
        } finally { Pop-Location }
        Write-Host "web UI rebuilt."
    }
}
if (-not (Test-Path $dist)) { throw "frontend\dist\index.html is missing" }

# --- 2. Stage the runtime files (never data\, venvs, git, caches) -----------
$staging = Join-Path $env:TEMP ("pf-installer-staging-" + [Guid]::NewGuid().ToString("N"))
$app = Join-Path $staging "app"
New-Item -ItemType Directory -Force $app | Out-Null
try {
    foreach ($f in @("launch.ps1", "launch.bat", "install.bat", "setup.ps1",
                     "doctor.ps1", "allow-lan.ps1")) {
        $p = Join-Path $root $f
        if (Test-Path $p) { Copy-Item $p (Join-Path $app $f) }
    }
    New-Item -ItemType Directory -Force (Join-Path $app "backend") | Out-Null
    Copy-Item (Join-Path $root "backend\run.py") (Join-Path $app "backend\run.py")
    Copy-Item (Join-Path $root "backend\requirements.txt") (Join-Path $app "backend\requirements.txt")
    Copy-Item (Join-Path $root "backend\app") (Join-Path $app "backend\app") -Recurse
    New-Item -ItemType Directory -Force (Join-Path $app "frontend") | Out-Null
    Copy-Item (Join-Path $root "frontend\dist") (Join-Path $app "frontend\dist") -Recurse
    if (Test-Path (Join-Path $root "docs")) {
        Copy-Item (Join-Path $root "docs") (Join-Path $app "docs") -Recurse
    }
    # Strip caches that Copy-Item dragged along.
    Get-ChildItem $app -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Get-ChildItem $app -Recurse -File -Include "*.pyc" -ErrorAction SilentlyContinue |
        Remove-Item -Force

    # install.bat (packed above) hands unpacked-zip starts to this script.
    New-Item -ItemType Directory -Force (Join-Path $app "installer") | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "bootstrap.ps1") (Join-Path $app "installer\bootstrap.ps1")

    # The installer core sits at the zip root - the .bat starts it directly.
    Copy-Item (Join-Path $PSScriptRoot "installer.ps1") (Join-Path $staging "installer.ps1")

    $fileCount = (Get-ChildItem $app -Recurse -File).Count
    $buildInfo = @{
        app      = "PromptForge"
        built_at = (Get-Date -Format "yyyy-MM-dd HH:mm")
        machine  = $env:COMPUTERNAME
        files    = $fileCount
    } | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $staging "build-info.json"), $buildInfo,
                            [Text.Encoding]::ASCII)

    # --- 3. Zip + base64 -----------------------------------------------------
    $zip = Join-Path $env:TEMP ("pf-installer-" + [Guid]::NewGuid().ToString("N") + ".zip")
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($staging, $zip,
        [IO.Compression.CompressionLevel]::Optimal, $false)
    $zipBytes = [IO.File]::ReadAllBytes($zip)
    $b64 = [Convert]::ToBase64String($zipBytes)
    Remove-Item $zip -Force

    # --- 4. Compose the self-extracting batch file ---------------------------
    # NOTE: inside the embedded PowerShell command the payload marker is built
    # from pieces (':'+':PAYLOAD:'+':') so IndexOf never matches the command
    # line itself - only the real marker line further down.
    $hdr = @(
        "@echo off",
        "setlocal EnableExtensions",
        "title PromptForge Setup",
        "echo.",
        "echo   == PromptForge Setup ==",
        "echo   Unpacking the installer (this takes a few seconds)...",
        "set ""PF_SELF=%~f0""",
        "set ""PF_TMP=%TEMP%\promptforge-setup-%RANDOM%%RANDOM%""",
        "set ""PF_LOG=%TEMP%\promptforge-setup-last-run.log""",
        "echo %date% %time% setup started from ""%PF_SELF%"" > ""%PF_LOG%""",
        "echo If no lines follow this one, the setup process was terminated from outside - >> ""%PF_LOG%""",
        "echo usually an antivirus. Allow/restore PromptForge-Setup.bat there and retry. >> ""%PF_LOG%""",
        "mkdir ""%PF_TMP%"" >nul 2>&1",
        ("powershell -NoProfile -ExecutionPolicy Bypass -Command " +
         """`$ErrorActionPreference='Stop'; " +
         "`$raw=[IO.File]::ReadAllText(`$env:PF_SELF); " +
         "`$m=':'+':PAYLOAD:'+':'; `$i=`$raw.IndexOf(`$m); " +
         "if(`$i -lt 0){throw 'payload marker missing'}; " +
         "`$b64=(`$raw.Substring(`$i+`$m.Length) -replace '\s',''); " +
         "[IO.File]::WriteAllBytes((Join-Path `$env:PF_TMP 'payload.zip')," +
         "[Convert]::FromBase64String(`$b64))"""),
        "if errorlevel 1 goto :unpackfail",
        ("powershell -NoProfile -ExecutionPolicy Bypass -Command " +
         """`$ErrorActionPreference='Stop'; " +
         "Add-Type -AssemblyName System.IO.Compression.FileSystem; " +
         "[IO.Compression.ZipFile]::ExtractToDirectory(" +
         "(Join-Path `$env:PF_TMP 'payload.zip'),`$env:PF_TMP)"""),
        "if errorlevel 1 goto :unpackfail",
        "echo %date% %time% payload unpacked ok >> ""%PF_LOG%""",
        # Everything from the installer launch onward lives on ONE physical
        # line: cmd re-reads a batch file at a byte offset after every
        # command, so if this .bat is rebuilt or re-synced while an
        # hour-long install runs, any FOLLOWING line would be read from the
        # NEW file at a garbage offset (seen live). A single line is parsed
        # once, before the long call starts.
        ("powershell -NoProfile -ExecutionPolicy Bypass -File ""%PF_TMP%\installer.ps1"" -Staging ""%PF_TMP%"" %* " +
         "&& (echo installer finished ok >> ""%PF_LOG%"" & rd /s /q ""%PF_TMP%"" >nul 2>&1 & exit /b 0) " +
         "|| (echo installer FAILED >> ""%PF_LOG%"" & rd /s /q ""%PF_TMP%"" >nul 2>&1 & " +
         "if ""%~1""=="""" (echo. & echo   Setup ended with an error. Details: & " +
         "echo     ""%TEMP%\promptforge-setup-error.log"" & echo     ""%PF_LOG%"" & " +
         "echo   If this window normally closes instantly with no message, an & " +
         "echo   antivirus is likely blocking the installer - allow this file & " +
         "echo   there and run it again. & pause & exit /b 1) else (exit /b 1))"),
        ":unpackfail",
        "echo %date% %time% UNPACK FAILED >> ""%PF_LOG%""",
        "echo.",
        "echo   Could not unpack the installer. The file may be incomplete",
        "echo   (copy it again) or PowerShell is blocked on this PC.",
        "rd /s /q ""%PF_TMP%"" >nul 2>&1",
        "pause",
        "exit /b 1",
        "::PAYLOAD::"
    )

    $sb = New-Object System.Text.StringBuilder
    foreach ($line in $hdr) { [void]$sb.Append($line).Append("`r`n") }
    for ($i = 0; $i -lt $b64.Length; $i += 512) {
        $len = [Math]::Min(512, $b64.Length - $i)
        [void]$sb.Append($b64.Substring($i, $len)).Append("`r`n")
    }
    [IO.File]::WriteAllText($OutFile, $sb.ToString(), [Text.Encoding]::ASCII)

    $sizeMb = [Math]::Round((Get-Item $OutFile).Length / 1MB, 1)
    Write-Host ""
    Write-Host ("== Done: " + $OutFile + " (" + $sizeMb + " MB, " +
                $fileCount + " app files) ==") -ForegroundColor Green
    Write-Host "Copy this ONE file to any Windows 10/11 PC and double-click it."
} finally {
    Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
}
