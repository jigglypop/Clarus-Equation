"""The card's OWN battery family (power_profile_parent, p=128) at n beyond the disclosed 2e5."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from check_families import power_profile_parent, tree_stats  # noqa: E402

rows = []
for n, m, p in ((10 ** 6, 2000, 128.0), (10 ** 6, 5000, 128.0), (5 * 10 ** 6, 5000, 128.0),
                (5 * 10 ** 6, 12000, 128.0)):
    t0 = time.time()
    c = tree_stats(power_profile_parent(n, m, p))["c"]
    rows.append({"n": n, "m": m, "p": p, "c_from_card_tree_stats": c,
                 "outside_window": bool(not (0.25 <= c <= 2.0)), "sec": round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
(HERE / "a15_cardfam_bign.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
