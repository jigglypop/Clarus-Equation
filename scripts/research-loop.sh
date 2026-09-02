#!/bin/sh
# Headless research-loop driver (POSIX). Uses CE_PYTHON if set, else python on PATH.
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
exec "${CE_PYTHON:-python}" -B "$here/research_loop.py" "$@"
