#!/bin/sh
# Claude Code delegates to the canonical Codex launcher so both providers use
# the same source path, stale check, cache location, and hook no-op behavior.
set -eu
project=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
exec "$project/.codex/hooks/run.sh" "$@"
