#!/bin/sh
# CE_RUN workspace orchestration is retired. Canonical results belong in docs/.
[ "${1:-}" = hook ] && exit 0
echo "CE run workspace commands are retired; update canonical docs/ files directly." >&2
exit 2
