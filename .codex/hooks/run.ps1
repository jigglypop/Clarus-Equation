param([ValidateSet('route', 'stop')][string]$Event)
$ErrorActionPreference = 'Stop'
$config = Split-Path $PSScriptRoot -Parent
$core = Join-Path $config 'skills/ce-research/core'
$target = Join-Path $config 'target/ce-research-core'
$bin = Join-Path $target 'release/ce-research-core.exe'
$built = if (Test-Path $bin) { (Get-Item $bin).LastWriteTimeUtc } else { [datetime]::MinValue }
$stale = (Get-Item "$core/src/main.rs").LastWriteTimeUtc -gt $built -or (Get-Item "$core/Cargo.toml").LastWriteTimeUtc -gt $built
if ($stale) { cargo build --quiet --locked --release --target-dir $target --manifest-path "$core/Cargo.toml"; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
& $bin hook $Event
exit $LASTEXITCODE

