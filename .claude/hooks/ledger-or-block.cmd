@echo off
rem Claude Code hook wrapper. stdin JSON passes through to lib\ledger_or_block.py via the policy-allowed Python.
call "%~dp0python.cmd" python "%~dp0lib\ledger_or_block.py"
exit /b %errorlevel%
