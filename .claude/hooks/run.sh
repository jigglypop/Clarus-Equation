#!/bin/sh
# Thin wrapper for ce-research-core (Claude edition): forwards to the shared
# prebuilt binary and NEVER builds — the Rust source of truth lives in
# ex-codex/skills/ce-research/core. Hook events degrade to a silent no-op
# when no binary is available; manual commands print how to build one.
set -eu
if [ -n "${LOCALAPPDATA:-}" ]; then
    win_bin=$(printf '%s' "$LOCALAPPDATA/ce-research-core/release/ce-research-core.exe" | tr '\\' '/')
    [ -x "$win_bin" ] && exec "$win_bin" "$@"
fi
posix_bin="${XDG_CACHE_HOME:-$HOME/.cache}/ce-research-core/release/ce-research-core"
[ -x "$posix_bin" ] && exec "$posix_bin" "$@"
if [ "${1:-}" = hook ]; then
    exit 0
fi
echo "ce-research-core: no prebuilt binary. Build once from the codex core:" >&2
echo "  cargo build --locked --release --target-dir <cache> --manifest-path ex-codex/skills/ce-research/core/Cargo.toml" >&2
exit 2
