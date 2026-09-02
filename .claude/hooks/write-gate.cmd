@echo off
rem Claude Code hook wrapper. stdin JSON passes through to lib\write_gate.py via the policy-allowed Python.
call "%~dp0python.cmd" python "%~dp0lib\write_gate.py"
exit /b %errorlevel%
