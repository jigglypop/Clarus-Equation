"""K6 attack, lower end: minimise c over 'spine + leaves' trees.

Continuum motivation (all weights small, so (1-w)^2 ~ 1):
    A/n^4 ~ int int min(w,w')^2 dnu dnu,   Atilde/n^4 ~ int int w w' dnu dnu = (int w dnu)^2
    => c ~ J[G] = 2 int t G(t)^2 dt / (int G dt)^2,  G(t) = nu({w > t}),  -G' <= 1  (integrality)
A truncated power law G ~ t^{-alpha} gives J -> 1 - alpha, so the sandwich floor 1/4 should be
crossed once the tree is large enough to host the required dynamic range.  Tested exactly below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a4_cat import cat_stats  # noqa: E402


def powerlaw_sizes(n, alpha, m_spine, w_max):
    """Spine weights from the maximal-density truncated power law with `m_spine` vertices."""
    delta = alpha * m_spine / n            # density cap rho(delta) = 1
    c = delta ** (alpha + 1) / alpha
    kmax = n * c * (delta ** -alpha - w_max ** -alpha)
    k = np.arange(1, int(min(m_spine, max(kmax, 1))) + 1, dtype=np.float64)
    inv = delta ** -alpha - k / (n * c)
    inv = inv[inv > 0]
    w = inv ** (-1.0 / alpha)
    w = w[w <= w_max]
    sizes = np.unique(np.round(n * (1.0 - w)).astype(np.int64))
    sizes = sizes[(sizes >= 1) & (sizes <= n - 1)]
    return sizes


rows = []
best = (10.0, None)
for n_exp in (4, 5, 6, 7, 8, 9, 10, 12, 15, 20):
    n = 10 ** n_exp
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
        for m_spine in (10 ** 3, 10 ** 4, 10 ** 5, 3 * 10 ** 5):
            if m_spine > n / 10:
                continue
            for w_max in (0.1, 0.3, 0.6, 0.9, 0.99):
                try:
                    sizes = powerlaw_sizes(n, alpha, m_spine, w_max)
                    if len(sizes) < 10:
                        continue
                    st = cat_stats(n, sizes)
                except Exception as exc:  # noqa: BLE001
                    continue
                row = {"n": n, "alpha": alpha, "m_spine": m_spine, "w_max": w_max,
                       "m_used": st["m"], "c": st["c"], "B_over_D": st["B"] / st["D"]}
                rows.append(row)
                if st["c"] < best[0]:
                    best = (st["c"], row)

rows.sort(key=lambda r: r["c"])
summary = {"n_rows": len(rows), "c_min": rows[0]["c"], "best": rows[0], "lowest10": rows[:10]}
print(json.dumps(summary, indent=2))
best_by_n = {}
for r in rows:
    key = r["n"]
    if key not in best_by_n or r["c"] < best_by_n[key]["c"]:
        best_by_n[key] = r
print("best per n:")
for k in sorted(best_by_n):
    r = best_by_n[k]
    print(f"  n=1e{len(str(k))-1:<3d} c={r['c']:.5f}  alpha={r['alpha']} m={r['m_used']} w_max={r['w_max']}")
(HERE / "a5_min.json").write_text(json.dumps({"summary": summary, "best_by_n": {str(k): v for k, v in best_by_n.items()}, "rows": rows[:400]}, indent=2), encoding="utf-8")
