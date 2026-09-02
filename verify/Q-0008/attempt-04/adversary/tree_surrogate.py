"""adversary attempt-04 audit: tree-only surrogate.

Checks (a) that the Pruefer generator actually used by check_modes samples uniform rooted labelled
trees (MC E[D] vs the exact combinatorics that produced the pre-registered numbers), (b) how much of
the observed her_slope scatter is pure 256-tree sampling noise, (c) driver_fast == driver_matrix.
Seed 777001 (declared before running).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import driver_numbers as dn  # noqa: E402

SIZES = (8, 16, 32, 64, 128)
M = 20000
R = 20000
rng = np.random.default_rng(777001)
out = {"M": M, "R": R, "seed": 777001}

worst = 0.0
for n in (5, 9, 17, 33):
    for _ in range(15):
        p = dn.uniform_rooted_tree(n, rng)
        a, b = dn.driver_fast(p)[0], dn.driver_matrix(p)
        worst = max(worst, abs(a - b) / (1 + b))
out["driver_fast_vs_matrix_max_rel_err"] = worst

pool = {}
for n in SIZES:
    d = np.empty(M)
    tr = np.empty(M)
    for t in range(M):
        p = dn.uniform_rooted_tree(n, rng)
        dd, trh, _ = dn.driver_fast(p)
        d[t] = dd
        tr[t] = trh
    pool[n] = d
    ex = dn.cayley_exact(n)
    out.setdefault("per_n", {})[str(n)] = {
        "MC_E_D": float(d.mean()), "exact_E_D": ex["E_D"],
        "MC_minus_exact_in_mcse": float((d.mean() - ex["E_D"]) / (d.std(ddof=1) / math.sqrt(M))),
        "rel_dev": float(d.mean() / ex["E_D"] - 1.0),
        "cv_D": float(d.std(ddof=1) / d.mean()),
        "MC_E_trHk": float(tr.mean()), "exact_E_trHk": ex["E_trHk"],
        "trHk_dev_in_mcse": float((tr.mean() - ex["E_trHk"]) / (tr.std(ddof=1) / math.sqrt(M))),
        "rel_se_of_mean_over_256": float(d.std(ddof=1) / d.mean() / math.sqrt(256)),
    }

xs = np.log(np.array(SIZES, float))
slopes = np.empty(R)
ratios = np.empty(R)
for r in range(R):
    ys = []
    for n in SIZES:
        idx = rng.integers(0, M, 256)
        ys.append(math.sqrt(pool[n][idx].mean()) / n)
    slopes[r] = np.polyfit(xs, np.log(ys), 1)[0]
    ratios[r] = ys[-1] * 128 / math.sqrt(127)
out["tree_only_slope"] = {
    "mean": float(slopes.mean()), "sd": float(slopes.std(ddof=1)),
    "exact_grid_slope": dn.slope(SIZES, [math.sqrt(dn.cayley_exact(n)["E_D"]) / n for n in SIZES]),
    "P_ge_observed_0.5576107": float(np.mean(slopes >= 0.5576106551570001)),
    "P_outside_window": float(np.mean((slopes < 0.43) | (slopes > 0.63))),
    "q": [float(np.percentile(slopes, q)) for q in (2.5, 50, 97.5)],
}
out["tree_only_ratio128"] = {"mean": float(ratios.mean()), "sd": float(ratios.std(ddof=1)),
                             "P_ge_observed_34.242": float(np.mean(ratios >= 34.24199061956648))}
json.dump(out, open(HERE / "tree_surrogate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
