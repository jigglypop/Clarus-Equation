"""Adversary a7 (kill_executable): do the pre-registered windows fire on a TRUE card?

 (1) rho_spread = max_n rho(n) / min_n rho(n) is a max/min of 5 noisy ratios whose true value is 1.
     Its Monte-Carlo floor is > 1 by construction.  Bootstrap the sampling law of a 256-trial RMS
     from real residual samples (sizes 10, 20 -- NOT the pre-registered grid) and compute the
     false-kill rate P[spread > 1.25] and P[ratio outside (0.85,1.18)] for a card that is exactly right.
 (2) exact-orbit closure of the transmitted channel: with fold = 0, is the block residual zero at
     LARGE delta for the card's plane and for the alternative blind planes?
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, rms  # noqa: E402
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


P_ALIGN = diag_proj(GROUPS["scale"] + GROUPS["sd"])
TRIALS = 256
DELTA = 0.005


def samples(n, P, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < TRIALS:
        par = uniform_rooted_tree(n, rng)
        c = rng.normal(size=(len(par), 16))
        v = block_residual(labels(rule(par, c, P) if P is not None else c), DELTA)
        if math.isfinite(v):
            out.append(v)
    return np.asarray(out)


print("[1] sampling law of the K1 statistics for a card that is EXACTLY true")
pools = {}
for n in (10, 20):
    pools[("align", n)] = samples(n, P_ALIGN, 606060 + n)
    pools[("iid", n)] = samples(n, None, 707070 + n)
    a, i = pools[("align", n)], pools[("iid", n)]
    print("    n=%3d  RMS_align=%.4e RMS_iid=%.4e  ratio=%.4f  kurtosis(align)=%.2f"
          % (n, rms(a), rms(i), rms(a) / rms(i), float(np.mean(a ** 4) / np.mean(a ** 2) ** 2)))

rng = np.random.default_rng(24680)
B = 20000
rel = []
for n in (10, 20):
    a, i = pools[("align", n)], pools[("iid", n)]
    ra = np.sqrt(np.mean(a[rng.integers(0, TRIALS, size=(B, TRIALS))] ** 2, axis=1))
    ri = np.sqrt(np.mean(i[rng.integers(0, TRIALS, size=(B, TRIALS))] ** 2, axis=1))
    r = (ra / rms(a)) / (ri / rms(i))
    rel.append(r)
    print("    n=%3d bootstrap ratio: sd=%.4f  1%%/99%% = %.3f/%.3f" % (n, r.std(), *np.percentile(r, [1, 99])))
rel = np.concatenate(rel)
draw = rel[rng.integers(0, len(rel), size=(200000, 5))]
spread = draw.max(axis=1) / draw.min(axis=1)
print("    simulated rho_spread (5 sizes, true rho=1): median=%.3f  90%%=%.3f  95%%=%.3f"
      % (np.median(spread), *np.percentile(spread, [90, 95])))
print("    FALSE-KILL RATE  P[spread > 1.25]                 = %.3f" % float(np.mean(spread > 1.25)))
print("    FALSE-KILL RATE  P[any of 5 ratios outside 0.85-1.18] = %.3f"
      % float(np.mean((draw.min(axis=1) < 0.85) | (draw.max(axis=1) > 1.18))))
print("    combined K1 false-kill rate (spread or ratio_32/128 outside) = %.3f"
      % float(np.mean((spread > 1.25) | (draw[:, 0] < 0.85) | (draw[:, 0] > 1.18)
                      | (draw[:, 1] < 0.85) | (draw[:, 1] > 1.18))))

print("\n[2] transmitted channel with fold = 0: is the block exactly simple at LARGE delta?")
cases = {
    "P_align scale+sd (CARD)": diag_proj(GROUPS["scale"] + GROUPS["sd"]),
    "P_alt1  scale+asd":       diag_proj(GROUPS["scale"] + GROUPS["asd"]),
    "P_alt2  sd+asd_1":        diag_proj(GROUPS["sd"] + GROUPS["asd"][:1]),
    "P_null7 scale+sd+asd":    diag_proj(GROUPS["scale"] + GROUPS["sd"] + GROUPS["asd"]),
    "P_asd   asd only":        diag_proj(GROUPS["asd"]),
}
for name, P in cases.items():
    line = []
    for delta in (0.01, 0.05, 0.2):
        rr = np.random.default_rng(4242)
        vals = []
        for _ in range(16):
            par = uniform_rooted_tree(16, rr)
            cc = rr.normal(size=(16, 16)) @ P.T
            v = block_residual(labels(rule(par, cc, P)), delta)
            if math.isfinite(v):
                vals.append(v)
        line.append("d=%.2f:%.2e" % (delta, rms(vals)))
    print("    %-26s %s" % (name, "  ".join(line)))
