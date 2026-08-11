#!/usr/bin/env python
"""Run the V3 latent-context causal bridge gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reality_stone" / "python"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from reality_stone.clarus.latent_causal_bridge import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
