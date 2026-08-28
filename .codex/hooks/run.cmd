@echo off
rem CE_RUN workspace orchestration is retired. Canonical results belong in paper/.
if "%~1"=="hook" exit /b 0
>&2 echo CE run workspace commands are retired; update canonical paper/ files directly.
exit /b 2
