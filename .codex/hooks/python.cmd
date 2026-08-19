@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Bootstrap only an already policy-allowed system Python.  No PowerShell
rem execution-policy override, venv creation, uv resolution, or install occurs.
set "CE_BOOTSTRAP="

if defined CE_PYTHON if exist "%CE_PYTHON%" set "CE_BOOTSTRAP=%CE_PYTHON%"
if not defined CE_BOOTSTRAP if defined LOCALAPPDATA (
    if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "CE_BOOTSTRAP=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "CE_BOOTSTRAP=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "CE_BOOTSTRAP=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "CE_BOOTSTRAP=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "CE_BOOTSTRAP=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)

if not defined CE_BOOTSTRAP (
    >&2 echo CE Python harness: no policy-allowed system Python ^>=3.10 was found.
    >&2 echo Set CE_PYTHON to an approved system interpreter; do not bypass Application Control.
    exit /b 2
)

"%CE_BOOTSTRAP%" -B "%~dp0python_harness.py" %*
exit /b %errorlevel%
