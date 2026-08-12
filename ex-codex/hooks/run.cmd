@echo off
rem Fast path for hooks on Windows: run the prebuilt binary directly and
rem skip PowerShell startup. Hook events never trigger a build (that would
rem stall the prompt); manual commands fall back to run.ps1 which rebuilds.
set "BIN=%LOCALAPPDATA%\ce-research-core\release\ce-research-core.exe"
if exist "%BIN%" (
    "%BIN%" %*
    exit /b %errorlevel%
)
if "%~1"=="hook" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %errorlevel%
