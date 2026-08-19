@echo off
rem Claude Code delegates to the canonical native Windows Git-blob gate.
call "%~dp0..\..\.codex\hooks\check-large-data.cmd" %*
exit /b %errorlevel%
