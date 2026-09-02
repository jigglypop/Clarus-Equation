"""Refine the K6 violator: smallest n with c < 1/4, plus the c_min(n) trend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a4_cat import cat_stats  # noqa: E402

ALPHAS = [0.85, 0.92, 0.96, 0.98, 0.99, 0.997]
WMAX = [0.3, 0.9]


def powerlaw_sizes(n, alpha, m_spine, w_max):
    delta = alpha * m_spine / n
    if not (0 < delta < w_max):
        return None
    c = delta ** (alpha + 1) / alpha
    k = np.arange(1, int(m_spine) + 1, dtype=np.float64)
    inv = delta ** -alpha - k / (n * c)
    inv = inv[inv > 0]
    if len(inv) < 10:
        return None
    w = inv ** (-1.0 / alpha)
    w = w[w <= w_max]
    sizes = np.unique(np.round(n * (1.0 - w)).astype(np.int64))
    sizes = sizes[(sizes >= 1) & (sizes <= n - 1)]
    return sizes if len(sizes) >= 10 else None


def best_at(n, m_cap=100000, nm=12):
    ms = np.unique(np.round(np.geomspace(30, max(60, min(m_cap, n / 20)), nm)).astype(np.int64))
    best = (10.0, None, None)
    for alpha in ALPHAS:
        for m in ms:
            for wm in WMAX:
                sizes = powerlaw_sizes(n, alpha, int(m), wm)
                if sizes is None:
                    continue
                st = cat_stats(n, sizes)
                if st["c"] < best[0]:
                    best = (st["c"], {"alpha": alpha, "m": int(m), "w_max": wm, "m_used": st["m"],
                                      "B_over_D": st["B"] / st["D"]}, sizes)
    return best


out = {"per_n": {}}
for n in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 10, 10 ** 14, 10 ** 20, 10 ** 40):
    c, params, sizes = best_at(n)
    out["per_n"][str(n)] = {"c_min": c, **(params or {})}
    print(f"n=1e{len(str(n))-1:<3} c_min={c:.5f}  {params}", flush=True)

lo, hi = 10 ** 6, 10 ** 8
for _ in range(16):
    mid = int(round((lo * hi) ** 0.5))
    c, params, sizes = best_at(mid)
    print(f"  bisect n={mid:<12} c={c:.5f}", flush=True)
    if c < 0.25:
        hi = mid
    else:
        lo = mid
c, params, sizes = best_at(hi, nm=20)
out["smallest_n_below_quarter"] = {"n": hi, "largest_n_still_above": lo, "c_at_hi": c, "params": params}
print("smallest n with c<1/4:", hi, "c =", c, params, flush=True)
np.save(HERE / "a6_violator_sizes.npy", sizes)
out["violator"] = {"n": hi, "c": c, "params": params, "n_spine_sizes": int(len(sizes)),
                   "sizes_head": [int(x) for x in sizes[:6]], "sizes_tail": [int(x) for x in sizes[-6:]]}
(HERE / "a6_refine.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
