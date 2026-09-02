@echo off
rem Claude Code hook wrapper. stdin JSON passes through to lib\verify_on_save.py via the policy-allowed Python.
call "%~dp0python.cmd" python "%~dp0lib\verify_on_save.py"
exit /b %errorlevel%
