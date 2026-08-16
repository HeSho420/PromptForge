@echo off
rem PromptForge - double-click to start. Works even when PowerShell scripts
rem are disabled on this PC (the Windows default): -ExecutionPolicy Bypass
rem applies to this one process only and changes no system settings.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
if errorlevel 1 pause
