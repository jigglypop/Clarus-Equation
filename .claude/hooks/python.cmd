@echo off
rem Claude Code delegates to the canonical Codex Windows Python launcher.
call "%~dp0..\..\.codex\hooks\python.cmd" %*
exit /b %errorlevel%
