@echo off
rem Fast path for hooks on Windows: run the prebuilt binary directly and
rem skip PowerShell startup. Falls back to run.ps1 (which rebuilds) only
rem when the binary is missing.
set "BIN=%LOCALAPPDATA%\ce-research-core\release\ce-research-core.exe"
if exist "%BIN%" (
    "%BIN%" %*
    exit /b %errorlevel%
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %errorlevel%
