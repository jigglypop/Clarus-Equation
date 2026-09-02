@echo off
rem Headless research-loop driver. Delegates to the policy-allowed Python.
call "%~dp0..\.claude\hooks\python.cmd" python "%~dp0research_loop.py" %*
exit /b %errorlevel%
