# Prints the full path of a working git.exe as the ONLY stdout line
# (everything else is silenced), installing git if needed:
#   1. already on PATH -> done
#   2. winget install (most Win10/11)
#   3. Portable Git direct download from git-for-windows releases
#      (machines without winget), extracted under LOCALAPPDATA
# Exit 0 with a path, exit 1 without one. Never throws to the console.
$ErrorActionPreference = 'Continue'
$c = Get-Command git -ErrorAction SilentlyContinue
if ($c) { $c.Source; exit 0 }
try {
    winget install --id Git.Git -e --silent --accept-package-agreements `
        --accept-source-agreements --disable-interactivity 2>&1 | Out-Null
} catch {}
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
$c = Get-Command git -ErrorAction SilentlyContinue
if ($c) { $c.Source; exit 0 }
$d = Join-Path $env:LOCALAPPDATA 'PromptForge\PortableGit'
$g = Join-Path $d 'cmd\git.exe'
if (-not (Test-Path $g)) {
    try {
        $t = Join-Path $env:TEMP 'pf-portablegit.7z.exe'
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -Uri `
            'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/PortableGit-2.47.1-64-bit.7z.exe' `
            -OutFile $t 2>&1 | Out-Null
        if ((Get-Item $t).Length -gt 40MB) {
            New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null
            $p = Start-Process -Wait -PassThru $t -ArgumentList ('-o"' + $d + '"'), '-y'
            $null = $p
        }
        Remove-Item $t -Force -ErrorAction SilentlyContinue
    } catch {}
}
if (Test-Path $g) { $g; exit 0 }
exit 1
