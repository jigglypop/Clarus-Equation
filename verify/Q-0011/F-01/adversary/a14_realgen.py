"""Fair number: the card's REAL power_profile_parent generator, small (m,p) grid at n=2e5."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from a_core import stats_fast  # noqa: E402
from check_families import power_profile_parent  # noqa: E402

rows = []
n = 200000
for m in (100, 300, 1000, 3000, 10000):
    for p in (2.0, 8.0, 16.0, 64.0, 128.0, 1024.0):
        c = stats_fast(power_profile_parent(n, m, p))["c"]
        rows.append({"m": m, "p": p, "c": c})
rows.sort(key=lambda r: r["c"])
out = {"n": n, "c_min": rows[0]["c"], "argmin": rows[0], "all": rows}
print(json.dumps({"n": n, "c_min": rows[0]["c"], "argmin": rows[0], "lowest5": rows[:5]}, indent=2))
(HERE / "a14_realgen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
