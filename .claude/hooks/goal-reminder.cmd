@echo off
rem UserPromptSubmit hook: print the current research target (progress ledger section 2).
rem Delegates to the canonical Codex Python wrapper; never builds, never prompts.
call "%~dp0python.cmd" python "%~dp0..\..\.codex\hooks\goal_reminder.py"
exit /b 0
