#!/usr/bin/env python
"""Run the NumPy-only compositional causal OOD gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reality_stone" / "python"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from reality_stone.clarus.compositional_causal_world import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
