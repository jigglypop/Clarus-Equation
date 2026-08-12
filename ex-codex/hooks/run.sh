#!/bin/sh
# Wrapper for ce-research-core: builds on demand into the local cache
# (outside any cloud-synced folder), then forwards all arguments.
# Hook events NEVER build (a cargo build would stall the prompt); they use
# the existing binary or degrade to a silent no-op.
set -eu
config=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
core="$config/skills/ce-research/core"
target="${XDG_CACHE_HOME:-$HOME/.cache}/ce-research-core"
bin="$target/release/ce-research-core"

if [ "${1:-}" = hook ]; then
    [ -x "$bin" ] && exec "$bin" "$@"
    exit 0
fi

stale=0
[ -x "$bin" ] || stale=1
for src in "$core/src/main.rs" "$core/Cargo.toml" "$core/Cargo.lock"; do
    [ "$src" -nt "$bin" ] && stale=1
done
if [ "$stale" = 1 ] && command -v cargo >/dev/null 2>&1; then
    cargo build --quiet --locked --release --target-dir "$target" --manifest-path "$core/Cargo.toml"
fi
if [ -x "$bin" ]; then
    exec "$bin" "$@"
fi
echo "ce-research-core: no prebuilt binary and cargo not found" >&2
exit 2
