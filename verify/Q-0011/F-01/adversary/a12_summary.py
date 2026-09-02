"""Violator anatomy, exact 2-split upper limit, and the c_min(n) ~ 4/ln n trend."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a4_cat import cat_stats  # noqa: E402
from a_core import stats_fast  # noqa: E402
import a_fam as F  # noqa: E402

out = {}
# ---- 1. anatomy of the K6 violator
sizes = np.load(HERE / "a6_violator_sizes.npy")
n = 4051399
st = cat_stats(n, sizes)
s = np.sort(sizes)[::-1].astype(float)
w = 1 - s / n
out["violator"] = {
    "n": n, "spine_vertices": int(len(s)), "leaves": int(n - 1 - len(s)),
    "max_depth": int(len(s)) + 1,
    "largest_spine_subtree": int(s[0]), "smallest_spine_subtree": int(s[-1]),
    "weights_w_min": float(w[0]), "weights_w_max": float(w[-1]),
    "dynamic_range_R": float(w[-1] / w[0]),
    "leaves_at_root": int(n - 1 - s[0]),
    "c": st["c"], "A_over_At": st["A"] / st["At"], "B_over_D": st["B"] / st["D"],
    "shape": "caterpillar: a spine of 912 vertices whose subtree sizes follow a power law, "
             "every size drop hung as leaves at that spine vertex",
}

# ---- 2. upper end: 2-split of two stars, exact, any split fraction
def split2_c(n, a):
    b = n - 1 - a
    w1, w2, wL = 1 - a / n, 1 - b / n, 1 - 1 / n
    diag = a ** 2 * w1 ** 2 + b ** 2 * w2 ** 2 + (n - 3) * wL ** 2
    A = diag + 2 * ((a - 1) * w1 ** 2 + (b - 1) * w2 ** 2)
    At = diag + 2 * wL * ((a - 1) * w1 + (b - 1) * w2)
    B = (2.0 / n ** 2) * ((n - 3) * (n - 4) / 2 + a ** 2 * b ** 2 + (a - 1) * b ** 2 + (b - 1) * a ** 2)
    return (A + B) / At

chk = []
for nn in (2001, 20001):
    for frac in (0.1, 0.5):
        a = int(frac * (nn - 1))
        p = F.split_stars(nn, [a, nn - 1 - a])
        chk.append(abs(split2_c(nn, a) - stats_fast(p)["c"]))
out["split2_formula_vs_On_max_abs"] = max(chk)
tab = {}
for nn in (10 ** 3, 10 ** 5, 10 ** 7, 10 ** 9, 10 ** 12):
    row = {}
    for frac in (0.05, 0.2, 0.5, 0.8):
        a = max(1, int(frac * (nn - 1)))
        row[str(frac)] = split2_c(nn, a)
    row["max_over_fine_grid"] = max(split2_c(nn, max(1, int(f * (nn - 1))))
                                    for f in np.linspace(0.02, 0.98, 97))
    tab[str(nn)] = row
out["split2_table"] = tab
out["split2_sup"] = max(r["max_over_fine_grid"] for r in tab.values())

# ---- 3. c_min(n) trend from a6/a5
trend = json.loads((HERE / "a6_refine.json").read_text(encoding="utf-8"))["per_n"]
out["c_min_trend"] = {k: {"c_min": v["c_min"], "c_min_times_ln_n": v["c_min"] * math.log(float(k))}
                      for k, v in trend.items() if v.get("c_min", 10) < 5}
print(json.dumps(out, indent=2))
(HERE / "a12_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
