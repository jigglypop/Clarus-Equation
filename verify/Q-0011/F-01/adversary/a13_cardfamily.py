"""Why the card's own battery missed the floor: its power_profile parametrisation cannot host the
optimal weight profile.  s_j = n(1-(j/m)^p)  =>  weight density rho(w) ~ w^{1/p - 1}, i.e. tail
G(t) = m(w_max^{1/p} - t^{1/p}) is BOUNDED; the optimum needs G(t) ~ 1/t (density ~ w^-2),
which is p = -1 -- outside the family for every p > 0."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from a4_cat import cat_stats  # noqa: E402
from a_core import stats_fast  # noqa: E402
from check_families import power_profile_parent  # noqa: E402


def pp_sizes(n, m, p):
    j = np.arange(m, dtype=np.float64)
    s = np.maximum(1, np.floor(n * (1.0 - (j / m) ** p)).astype(np.int64))
    s = np.unique(s)[::-1]
    return s[(s >= 1) & (s <= n - 1)]


out = {}
# equivalence of my caterpillar evaluator with the card's generator
eq = []
for n, m, p in ((5000, 50, 8.0), (20000, 300, 16.0), (20000, 300, 128.0)):
    a = cat_stats(n, pp_sizes(n, m, p))["c"]
    b = stats_fast(power_profile_parent(n, m, p))["c"]
    eq.append({"n": n, "m": m, "p": p, "cat_stats_c": a, "card_generator_c": b, "diff": a - b})
out["equivalence_check"] = eq

best = {}
for n in (10 ** 5, 10 ** 6, 10 ** 8, 10 ** 12, 10 ** 16):
    lo = (10.0, None)
    for m in np.unique(np.round(np.geomspace(10, min(3e5, n / 10), 24)).astype(np.int64)):
        for p in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0, 4096.0):
            s = pp_sizes(n, int(m), p)
            if len(s) < 5:
                continue
            c = cat_stats(n, s)["c"]
            if c < lo[0]:
                lo = (c, {"m": int(m), "p": p, "spine": int(len(s))})
    best[str(n)] = {"c_min_card_family": lo[0], **(lo[1] or {})}
    print(f"card power_profile family, n={n:.0e}: c_min = {lo[0]:.4f}  {lo[1]}", flush=True)
out["card_power_profile_floor"] = best
(HERE / "a13_cardfamily.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out["equivalence_check"], indent=2))
