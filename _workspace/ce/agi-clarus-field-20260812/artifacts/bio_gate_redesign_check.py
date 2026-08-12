"""Pre-exploration for bio gate redesign: headroom-normalized delta contrast.

Reads existing confirmatory JSONs (recording-level medians only; no refit).
Statistic per recording: Delta' = median_delta / (1 - median_r2_current).
Caveat: ratio of medians, not median of ratios (targets not stored in JSON).
Exact Mann-Whitney U (one-sided, AML32 > AML18) via full enumeration.
"""
import json, itertools, math
from pathlib import Path

ROOT = Path(r"c:/Users/dongh/OneDrive/Desktop/Clarus-Equation/artifacts/agi")

def load(strain, h):
    d = json.load(open(ROOT / f"local_memory_{strain}_h{h}_confirmatory.json"))
    out = []
    for rec in d["result"]["recordings"]:
        dm = rec["median_delta_memory"]
        rc = rec["median_r2_current_nonlinear"]
        out.append((rec["recording_id"], dm, rc, dm / (1.0 - rc)))
    return out

def mw_exact_p_greater(x, y):
    """P(U >= u_obs) under permutation of ranks, exact enumeration."""
    n, m = len(x), len(y)
    u_obs = sum(1 for a in x for b in y if a > b) + 0.5 * sum(1 for a in x for b in y if a == b)
    allv = x + y
    idx = range(n + m)
    count = 0; total = 0
    for comb in itertools.combinations(idx, n):
        xs = [allv[i] for i in comb]
        ys = [allv[i] for i in idx if i not in comb]
        u = sum(1 for a in xs for b in ys if a > b) + 0.5 * sum(1 for a in xs for b in ys if a == b)
        total += 1
        if u >= u_obs:
            count += 1
    return u_obs, count / total

for h in (1, 6):
    g = load("aml32", h); f = load("aml18", h)
    print(f"\n=== h={h} ===")
    print(f"{'recording':34s} {'strain':6s} {'delta':>7s} {'r2cur':>7s} {'deltaN':>7s}")
    for rid, dm, rc, dn in g:
        print(f"{rid:34s} AML32  {dm:7.4f} {rc:7.4f} {dn:7.4f}")
    for rid, dm, rc, dn in f:
        print(f"{rid:34s} AML18  {dm:7.4f} {rc:7.4f} {dn:7.4f}")
    for label, col in (("raw delta", 1), ("normalized delta'", 3)):
        xg = [r[col] for r in g]; xf = [r[col] for r in f]
        med = lambda v: sorted(v)[len(v)//2] if len(v)%2 else 0.5*(sorted(v)[len(v)//2-1]+sorted(v)[len(v)//2])
        u, p = mw_exact_p_greater(xg, xf)
        print(f"[{label}] AML32 med={med(xg):.4f} range=({min(xg):.4f},{max(xg):.4f}) | "
              f"AML18 med={med(xf):.4f} range=({min(xf):.4f},{max(xf):.4f}) | "
              f"U={u:.1f}/{len(xg)*len(xf)} exact one-sided p(AML32>AML18)={p:.4f}")
