#!/bin/sh
set -eu
config=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
core="$config/skills/ce-research/core"
target="$config/target/ce-research-core"
bin="$target/release/ce-research-core"
[ -x "$bin" ] && [ ! "$core/src/main.rs" -nt "$bin" ] && [ ! "$core/Cargo.toml" -nt "$bin" ] || cargo build --quiet --locked --release --target-dir "$target" --manifest-path "$core/Cargo.toml"
exec "$bin" hook "$1"

