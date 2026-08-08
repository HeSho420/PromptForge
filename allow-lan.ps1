# Allow PromptForge's LAN features through the Windows firewall.
#
# Run ONCE per machine, as administrator:
#   Right-click allow-lan.ps1 -> Run with PowerShell (accept the UAC prompt)
#   or from an elevated PowerShell:  .\allow-lan.ps1
#
# What it opens, and why (Private/Domain networks only, never Public):
#   TCP 8765       model transfers between your own PromptForge machines,
#                  and the render-delegation proxy
#   UDP 8766-8769  the discovery beacon that lets the machines find each
#                  other automatically
#
# The main app stays on 127.0.0.1 and is NOT exposed by these rules.
# Remove everything again with:
#   Get-NetFirewallRule -DisplayName "PromptForge LAN*" | Remove-NetFirewallRule

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "This needs administrator rights (it changes firewall rules)." -ForegroundColor Yellow
    Write-Host "Re-launching elevated - accept the prompt..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList `
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`""
    exit
}

foreach ($rule in @(
    @{ Name = "PromptForge LAN transfers (TCP 8765)"; Protocol = "TCP"; Port = "8765" },
    @{ Name = "PromptForge LAN discovery (UDP 8766-8769)"; Protocol = "UDP"; Port = "8766-8769" }
)) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  [ok] rule already present: $($rule.Name)" -ForegroundColor Green
        continue
    }
    New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound `
        -Action Allow -Protocol $rule.Protocol -LocalPort $rule.Port `
        -Profile Private, Domain | Out-Null
    Write-Host "  [ok] added: $($rule.Name)" -ForegroundColor Green
}

$profileCategory = (Get-NetConnectionProfile | Select-Object -First 1).NetworkCategory
if ($profileCategory -eq "Public") {
    Write-Host ""
    Write-Host "  [!] Your network is marked PUBLIC - these rules only apply on" -ForegroundColor Yellow
    Write-Host "      Private networks. Settings > Network & internet > properties" -ForegroundColor Yellow
    Write-Host "      -> set the network profile to Private on BOTH machines." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Done. Do the same on the other machine, then check Settings -> Network" -ForegroundColor Cyan
Write-Host "  in PromptForge - the peers should list each other within seconds." -ForegroundColor Cyan
if ($MyInvocation.InvocationName -ne ".") { Read-Host "Press Enter to close" | Out-Null }
