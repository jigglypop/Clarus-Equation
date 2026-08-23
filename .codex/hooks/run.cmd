@echo off
rem Fast path for hooks on Windows: run the prebuilt binary directly and
rem skip PowerShell startup. Hook events never trigger a build (that would
rem stall the prompt); manual commands fall back to run.ps1 which rebuilds.
set "BIN=%LOCALAPPDATA%\ce-research-core\release\ce-research-core.exe"
if "%~1"=="hook" (
    if exist "%BIN%" (
        rem Hooks must never stall prompting. A policy-blocked/stale binary
        rem degrades to the documented silent no-op until the prerequisite is repaired.
        "%BIN%" %* 2>nul
        exit /b 0
    )
    exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %errorlevel%
