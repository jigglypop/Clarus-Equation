#!/usr/bin/env python
"""Run the NumPy-only nonlinear object-permanence gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "reality_stone" / "python" / "reality_stone" / "clarus" / "nonlinear_object_world.py"
)
SPEC = importlib.util.spec_from_file_location("nonlinear_object_world_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    raise SystemExit(MODULE.main())
