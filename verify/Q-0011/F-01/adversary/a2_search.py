"""Audit 1 / K6: hunt for a rooted tree with c = D/(n^2 mu2_eff) outside [1/4, 2].

Part A: structured extremes NOT in the card's battery (hub-at-depth-1, spindles, combs,
        double brooms, k-splits of every branch shape, complete k-ary, GW(0.5)/GW(2),
        BA/preferential attachment, Kesten, nested splits, profile trees).
Part B: simulated annealing over parent arrays (all shapes reachable), maximizing and
        minimizing c at several n.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_core import stats_fast  # noqa: E402
import a_fam as F  # noqa: E402

rng = np.random.default_rng(20260902)
rows = []


def rec(name, parent):
    st = stats_fast(parent)
    rows.append({"family": name, "n": int(st["n"]), "c": st["c"], "B_over_D": st["B"] / st["D"] if st["D"] else 0.0})
    return st["c"]


# ---------------------------------------------------------------- Part A
NS = (10, 12, 16, 24, 32, 48, 64, 100, 128, 256, 512, 1000, 2000)
for n in NS:
    rec(f"hub_depth1", F.hub_at_depth(n, 1))
    for d in (2, 3, 5, max(1, n // 4), max(1, n // 2), n - 2):
        if 1 <= d <= n - 2:
            rec(f"hub_depth_{'q' if d == n // 4 else d}", F.hub_at_depth(n, d))
    for pre in (1, 2, 3, max(1, n // 8), max(1, n // 3)):
        for frac in (0.25, 0.5, 0.75, 0.9):
            nl = int(frac * (n - pre - 1))
            if nl >= 1:
                rec("spindle", F.spindle(n, pre, nl))
    for t in (1, 2, 3, 5, 10):
        rec("comb", F.comb(n, t))
    for e in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95):
        rec("broom", F.broom(n, e))
    for m in (2, 3, 5, max(2, n // 4), max(2, n // 2)):
        if m <= n - 2:
            rec("double_broom", F.double_broom(n, m))
    for k in (2, 3, 4, 5, 8, 16):
        if k <= n // 2:
            rec("split_stars_equal", F.split_stars(n, [1.0] * k))
            rec("split_chains", F.split_chains(n, k))
            rec("split_of_brooms", F.split_of(n, k, lambda m: F.broom(max(m, 3), 0.5)))
            rec("split_of_combs", F.split_of(n, k, lambda m: F.comb(max(m, 3), 2)))
            rec("split_of_splits", F.split_of(n, k, lambda m: F.split_stars(max(m, 5), [1.0, 1.0])))
    for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rec("split2_unequal", F.split_stars(n, [a, 1 - a]))
        rec("split2_unequal_chain_star", F.split_of(n, 2, lambda m: F.chain(max(m, 2))))
    for a in (0.05, 0.1, 0.2, 0.33):
        rec("split3_unequal", F.split_stars(n, [a, a, 1 - 2 * a]))
    for m, p in ((5, 2.0), (10, 4.0), (20, 8.0), (30, 16.0), (50, 32.0), (min(n - 2, 100), 64.0), (min(n - 2, 300), 128.0)):
        if 2 <= m <= n - 2:
            rec("power_profile", F.power_profile(n, m, p))
for k in (2, 3, 4, 5):
    for d in (2, 3, 4, 5, 6, 8, 10):
        p = F.kary(k, d)
        if len(p) <= 200000:
            rec(f"kary{k}", p)
for n in (32, 128, 512, 2000):
    for trial in range(30):
        rec("gw_0.5", F.gw_tree(n, 0.5, rng))
        rec("gw_2.0", F.gw_tree(n, 2.0, rng))
        rec("gw_1.0", F.gw_tree(n, 1.0, rng))
        rec("kesten", F.kesten_trunc(n, rng))
        rec("ba_tree", F.ba_tree(n, rng))
        rec("rrt", F.rrt(n, rng))
# geometric / two-block profile trees (the analytic minimiser candidate)
for n in (200, 1000, 5000, 20000):
    for a in (0.02, 0.05, 0.1, 0.2, 0.3):
        for b in (0.3, 0.5, 0.7, 0.9):
            lo = [int(round((1 - x) * n)) for x in np.linspace(a / 4, a, 40)]
            hi = [int(round((1 - x) * n)) for x in np.linspace(b, min(0.99, b + 0.1), 40)]
            rec("two_block_profile", F.profile_tree(n, lo + hi))
    for g in (1.1, 1.5, 2.0, 3.0):
        ws = [min(0.99, 0.001 * g ** i) for i in range(0, 40)]
        rec("geom_profile", F.profile_tree(n, [int(round((1 - w) * n)) for w in ws]))

partA = {"rows": len(rows),
         "c_min": min(r["c"] for r in rows), "c_max": max(r["c"] for r in rows)}
srt = sorted(rows, key=lambda r: r["c"])
partA["lowest"] = srt[:12]
partA["highest"] = srt[-12:]
print(json.dumps(partA, ensure_ascii=False, indent=2))
(HERE / "a2_partA.json").write_text(json.dumps({"summary": partA, "all": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
