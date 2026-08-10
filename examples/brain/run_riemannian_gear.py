#!/usr/bin/env python
"""Run the complete synthetic Riemannian gear experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "reality_stone" / "python" / "reality_stone" / "clarus" / "riemannian_gear.py"
SPEC = importlib.util.spec_from_file_location("riemannian_gear_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

main = MODULE.main


if __name__ == "__main__":
    raise SystemExit(main())
