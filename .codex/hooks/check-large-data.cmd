@echo off
rem Native Windows entry point for the staged/outgoing Git-blob gate.
call "%~dp0python.cmd" python "%~dp0check_large_data.py" %*
exit /b %errorlevel%
