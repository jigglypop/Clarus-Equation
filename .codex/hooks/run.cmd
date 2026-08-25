@echo off
rem Fast path for hooks on Windows: run the prebuilt binary directly and
rem skip shell startup. Hook events never trigger a build (that would stall
rem the prompt); manual commands use Cargo directly so Windows script execution
rem policy is neither weakened nor bypassed.
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
where cargo >nul 2>nul
if not errorlevel 1 (
    cargo run --quiet --locked --release --target-dir "%LOCALAPPDATA%\ce-research-core" --manifest-path "%~dp0..\skills\ce-research\core\Cargo.toml" -- %*
    exit /b %errorlevel%
)
if exist "%BIN%" (
    "%BIN%" %*
    exit /b %errorlevel%
)
echo ce-research-core: cargo and prebuilt binary are unavailable 1>&2
exit /b 2
