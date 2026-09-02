@echo off
rem Claude Code hook wrapper. stdin JSON passes through to lib\session_start.py via the policy-allowed Python.
call "%~dp0python.cmd" python "%~dp0lib\session_start.py"
exit /b %errorlevel%
