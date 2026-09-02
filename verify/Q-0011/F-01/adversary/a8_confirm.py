"""K6 counterexample: exact rational c, then the CARD'S OWN tree_stats on the explicit tree."""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from a4_cat import cat_parent, cat_stats  # noqa: E402
from a_core import stats_fast  # noqa: E402


def cat_stats_exact(n: int, sizes) -> dict:
    s = sorted({int(x) for x in sizes if 1 <= int(x) <= n - 1}, reverse=True)
    m = len(s)
    N = Fraction(n)
    w = [Fraction(n - si, n) for si in s]
    wL = Fraction(n - 1, n)
    L0 = n - 1 - s[0]
    Li = [s[i] - (s[i + 1] if i + 1 < m else 0) - 1 for i in range(m)]
    P, Q = [Fraction(0)] * m, [Fraction(0)] * m
    for i in range(1, m):
        P[i] = P[i - 1] + w[i - 1]
        Q[i] = Q[i - 1] + w[i - 1] ** 2
    Ltot = L0 + sum(Li)
    diag = sum(Fraction(s[i] ** 2) * w[i] ** 2 for i in range(m)) + Ltot * wL ** 2
    A = diag + 2 * (sum(Fraction(s[i] ** 2) * Q[i] for i in range(m))
                    + sum(Li[i] * (Q[i] + w[i] ** 2) for i in range(m)))
    At = diag + 2 * (sum(Fraction(s[i] ** 2) * w[i] * P[i] for i in range(m))
                     + wL * sum(Li[i] * (P[i] + w[i]) for i in range(m)))
    tail = [0] * m
    acc = 0
    for i in range(m - 1, -1, -1):
        tail[i] = acc
        acc += s[i] ** 2
    B = Fraction(2, n ** 2) * (Fraction(Ltot * (Ltot - 1), 2) + L0 * sum(si ** 2 for si in s)
                               + sum(Li[i] * tail[i] for i in range(m)))
    D = A + B
    return {"D": D, "A": A, "B": B, "At": At, "c": D / At}


sizes = np.load(HERE / "a6_violator_sizes.npy")
n = 4051399
out = {"n": n, "spine_len": int(len(sizes)),
       "construction": "root -> spine of m vertices with the listed subtree sizes; every size drop is "
                       "hung as leaves at that spine vertex (a caterpillar with a power-law size profile)"}
st = cat_stats(n, sizes)
out["float_spine_formula"] = {k: st[k] for k in ("D", "A", "B", "At", "c")}
ex = cat_stats_exact(n, sizes)
out["exact_rational_c"] = float(ex["c"])
out["exact_rational_c_str"] = f"{ex['c'].numerator}/{ex['c'].denominator}"
out["exact_minus_float"] = float(ex["c"]) - st["c"]
out["violates_lower_bound_0.25"] = bool(ex["c"] < Fraction(1, 4))
out["margin_below_quarter"] = float(Fraction(1, 4) - ex["c"])
print(json.dumps({k: out[k] for k in ("n", "spine_len", "exact_rational_c", "exact_minus_float",
                                      "violates_lower_bound_0.25", "margin_below_quarter")}, indent=2), flush=True)

# ---- literal K6: the card's own tree_stats on the explicit parent array
t0 = time.time()
try:
    parent = cat_parent(n, sizes)
    out["parent_len"] = len(parent)
    mine = stats_fast(parent)
    out["adversary_O(n)_on_parent_array"] = {"c": mine["c"], "D": mine["D"], "At": mine["n2_mu2_eff"]}
    print("adversary O(n) on explicit parent array: c =", mine["c"], flush=True)
    from check_families import tree_stats as card_stats  # noqa: PLC0415
    cd = card_stats(parent)
    out["card_tree_stats"] = {"c": cd["c"], "D": cd["D"], "n2_mu2_eff": cd["n2_mu2_eff"],
                              "max_depth": cd["max_depth"]}
    out["card_c_outside_window"] = bool(not (0.25 <= cd["c"] <= 2.0))
    print("CARD tree_stats c =", cd["c"], " outside [0.25,2.0]:", out["card_c_outside_window"], flush=True)
except MemoryError as exc:
    out["card_tree_stats_error"] = f"MemoryError: {exc}"
    print("MemoryError on the explicit array", flush=True)
out["seconds"] = time.time() - t0
(HERE / "a8_confirm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("float_spine_formula",)}, indent=2))
