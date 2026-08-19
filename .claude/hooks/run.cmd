@echo off
rem Claude Code delegates to the canonical Codex Windows launcher.
call "%~dp0..\..\.codex\hooks\run.cmd" %*
exit /b %errorlevel%
