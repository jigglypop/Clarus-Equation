# Wrapper for ce-research-core: builds on demand into %LOCALAPPDATA%
# (outside OneDrive), then forwards all arguments. Hook events degrade
# to a silent no-op when no toolchain is available.
$ErrorActionPreference = 'Stop'
$config = Split-Path $PSScriptRoot -Parent
$core = Join-Path $config 'skills/ce-research/core'
$target = Join-Path $env:LOCALAPPDATA 'ce-research-core'
$bin = Join-Path $target 'release/ce-research-core.exe'

$built = if (Test-Path $bin) { (Get-Item $bin).LastWriteTimeUtc } else { [datetime]::MinValue }
$stale = @('src/main.rs', 'Cargo.toml', 'Cargo.lock') | Where-Object {
    $src = Join-Path $core $_
    (Test-Path $src) -and (Get-Item $src).LastWriteTimeUtc -gt $built
}
if ($stale -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    cargo build --quiet --locked --release --target-dir $target --manifest-path "$core/Cargo.toml"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
if (Test-Path $bin) {
    & $bin @args
    exit $LASTEXITCODE
}
if ($args.Count -gt 0 -and $args[0] -eq 'hook') { exit 0 }
Write-Error 'ce-research-core: no prebuilt binary and cargo not found'
exit 2
