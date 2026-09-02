"""Adversary b1 (re-audit, card revision 2): how much margin does the NEW K4 window have?

The pre-registered K4 grid is n in {8,16,32}, delta = 0.2, 64 trials, seed 20260902.  That grid is
NOT run here (it belongs to the prover / judge).  Instead this measures the SAME statistic on
sizes OUTSIDE the grid (6, 12, 24, 48) with INDEPENDENT seeds, to answer:

  (a) does closure_alt2 / closure_null7 stay above the pre-registered floor 1e-3 with margin, or is
      the smoke value 1.78e-3 (n in {4,6}, 2 trials) close to a false kill?
  (b) does the statistic grow with n?  (the K4 statistic is max_n RMS over {8,16,32}, so a growing
      trend means the pre-registered max is taken at n = 32 and the margin is LARGER than at n = 8)
  (c) does closure_align / closure_alt1 stay at machine zero (window <= 1e-12) at delta = 0.2?
  (d) MIN_DET rejection rate at delta = 0.2 -- a high rate would mean the statistic is conditioned
      on small deformations and could be biased downwards (towards a false kill of the contrast).
  (e) delta scaling: is the contrast residual O(delta^2)?
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, rms, MIN_DET  # noqa: E402
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)


def diag_proj(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


PROJ = {
    "align": diag_proj(GROUPS["scale"] + GROUPS["sd"]),
    "alt1": diag_proj(GROUPS["scale"] + GROUPS["asd"]),
    "alt2": diag_proj(GROUPS["sd"] + GROUPS["asd"][:1]),
    "null7": diag_proj(GROUPS["scale"] + GROUPS["sd"] + GROUPS["asd"]),
}
TRIALS = 64            # card value
DELTA = 0.2            # card value
OUT_SIZES = (6, 12, 24, 48)          # deliberately DISJOINT from the pre-registered {8,16,32}
SEEDS = (777001, 313317)             # NOT the card seed 20260902


def labels(c):
    return (c @ FLAT).reshape(-1, 4, 4)


def rule(parent, c, P):
    order, *_ = tree_arrays(parent)
    tr = c @ P.T
    acc = np.zeros_like(c)
    for v in order:
        p = parent[v]
        acc[v] = tr[v] + (acc[p] if p >= 0 else 0.0)
    return acc + (c - tr)


def closure_samples(n, P, seed, delta=DELTA, trials=TRIALS):
    """fold = 0, exactly as check_align.sample_closure; also counts MIN_DET rejections."""
    rng = np.random.default_rng(seed)
    vals, rejected = [], 0
    while len(vals) < trials:
        par = uniform_rooted_tree(n, rng)
        c = rng.normal(size=(len(par), 16)) @ P.T
        v = block_residual(labels(rule(par, c, P)), delta)
        if math.isfinite(v):
            vals.append(v)
        else:
            rejected += 1
    return np.asarray(vals), rejected


print("MIN_DET =", MIN_DET, " delta =", DELTA, " trials =", TRIALS,
      " sizes(out-of-grid) =", OUT_SIZES, " seeds =", SEEDS)
print("\n[a,b,c,d] closure statistic per size (RMS) and rejection count, OUT-OF-GRID sizes only")
print("  %-6s %-8s %-12s %-12s %-12s %-12s" % ("seed", "n", "align", "alt1", "alt2", "null7"))
table = {}
for seed in SEEDS:
    for n in OUT_SIZES:
        row, rej = {}, {}
        for name, P in PROJ.items():
            v, r = closure_samples(n, P, seed)
            row[name], rej[name] = rms(v), r
        table[(seed, n)] = row
        print("  %-6d %-8d %-12.4e %-12.4e %-12.4e %-12.4e   rejects=%s"
              % (seed, n, row["align"], row["alt1"], row["alt2"], row["null7"],
                 {k: rej[k] for k in ("align", "alt1", "alt2", "null7")}))

print("\n  n-trend (seed-averaged RMS):")
for name in ("align", "alt1", "alt2", "null7"):
    vals = [np.mean([table[(s, n)][name] for s in SEEDS]) for n in OUT_SIZES]
    print("    %-6s %s" % (name, "  ".join("n=%2d:%.3e" % (n, v) for n, v in zip(OUT_SIZES, vals))))
    if name in ("alt2", "null7"):
        lo = min(vals); print("           min over out-of-grid sizes = %.3e -> margin vs 1e-3 floor = %.2fx"
                             % (lo, lo / 1e-3))

print("\n  worst single (seed,size) contrast values and the implied false-kill margin:")
for name in ("alt2", "null7"):
    worst = min(table[k][name] for k in table)
    print("    %-6s worst over all out-of-grid (seed,size) = %.4e  -> %.2fx the 1e-3 floor"
          % (name, worst, worst / 1e-3))
for name in ("align", "alt1"):
    worst = max(table[k][name] for k in table)
    print("    %-6s worst (largest) over all out-of-grid (seed,size) = %.4e  -> %.2e of the 1e-12 ceiling"
          % (name, worst, worst / 1e-12))

print("\n[e] delta scaling of the contrast (n = 12, seed 777001):")
for delta in (0.05, 0.1, 0.2, 0.4):
    line = []
    for name in ("align", "alt2", "null7"):
        v, r = closure_samples(12, PROJ[name], 777001, delta=delta, trials=32)
        line.append("%s=%.3e(rej %d)" % (name, rms(v), r))
    print("    delta=%.2f  %s" % (delta, "  ".join(line)))
