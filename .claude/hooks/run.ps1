# Claude Code wrapper for the canonical CE research-core launcher.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot '..\..\.codex\hooks\run.ps1'
& $runner @args
exit $LASTEXITCODE
