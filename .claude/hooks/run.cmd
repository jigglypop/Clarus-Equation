@echo off
rem Thin wrapper for ce-research-core (Claude edition): prebuilt binary only,
rem never builds. Hook events no-op silently when the binary is missing.
set "BIN=%LOCALAPPDATA%\ce-research-core\release\ce-research-core.exe"
if exist "%BIN%" (
    "%BIN%" %*
    exit /b %errorlevel%
)
if "%~1"=="hook" exit /b 0
echo ce-research-core: no prebuilt binary. Build once from ex-codex\skills\ce-research\core. 1>&2
exit /b 2
