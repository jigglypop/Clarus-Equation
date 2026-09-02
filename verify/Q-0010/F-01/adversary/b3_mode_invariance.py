"""Adversary b3 (re-audit): (i) did adding the closure mode change the align / rand4 / qspine paths?
Independent re-implementation of run_grid and run_qspine at the SMOKE sizes with the card seeds; the
card script smoke printed align [3.799871814032248e-05, 3.1870882694453024e-05],
rand4 [2.8829904829812426e-05, 4.154274203771799e-05], qspine_ratio 0.407770581211982.
A bit-level match means the frozen paths compute exactly what the card documents.
(ii) extrapolate the K4 contrast to the pre-registered sizes from the OUT-OF-GRID b1 measurements
(the pre-registered grid itself is not run here).
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, fit_slope, rms  # noqa: E402
from driver_numbers import qspine_block, tree_arrays, uniform_rooted_tree  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
SEED, SEED_IID, DELTA = 20260902, 20260903, 0.005


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


def sample_align(n, rng, P, delta):
    while True:
        par = uniform_rooted_tree(n, rng)
        c = rng.normal(size=(len(par), 16))
        v = block_residual(labels(rule(par, c, P)), delta)
        if math.isfinite(v):
            return v


def sample_iid(n, rng, delta):
    while True:
        v = block_residual(labels(rng.normal(size=(n, 16))), delta)
        if math.isfinite(v):
            return v


def grid(P, sizes, trials, delta, seed, seed_iid):
    ra, ri = np.random.default_rng(seed), np.random.default_rng(seed_iid)
    a = [rms([sample_align(n, ra, P, delta) for _ in range(trials)]) for n in sizes]
    i = [rms([sample_iid(n, ri, delta) for _ in range(trials)]) for n in sizes]
    return a, i


P_ALIGN = diag_proj(GROUPS["scale"] + GROUPS["sd"])
W, _ = np.linalg.qr(np.random.default_rng(SEED).normal(size=(16, 4)))
P_RAND4 = W @ W.T

print("[i] independent re-implementation at the smoke sizes (4, 6), 2 trials, card seeds")
a, i = grid(P_ALIGN, (4, 6), 2, DELTA, SEED, SEED_IID)
card_align = [3.799871814032248e-05, 3.1870882694453024e-05]
print("    align rms      mine = %r" % a)
print("    align rms      card = %r" % card_align)
print("    max abs diff = %.3e" % max(abs(x - y) for x, y in zip(a, card_align)))
b, _ = grid(P_RAND4, (4, 6), 2, DELTA, SEED, SEED_IID)
card_rand4 = [2.8829904829812426e-05, 4.154274203771799e-05]
print("    rand4 rms      mine = %r" % b)
print("    rand4 rms      card = %r" % card_rand4)
print("    max abs diff = %.3e" % max(abs(x - y) for x, y in zip(b, card_rand4)))

ra, ri = np.random.default_rng(SEED), np.random.default_rng(SEED_IID)
vals = []
while len(vals) < 2:
    par = qspine_block(3, ra)
    c = ra.normal(size=(len(par), 16))
    v = block_residual(labels(rule(par, c, P_ALIGN)), DELTA)
    if math.isfinite(v):
        vals.append(v)
r_iid = rms([sample_iid(6, ri, DELTA) for _ in range(2)])
print("    qspine ratio   mine = %.15f   card = 0.407770581211982   diff = %.3e"
      % (rms(vals) / r_iid, abs(rms(vals) / r_iid - 0.407770581211982)))


print("")
print("[ii] K4 contrast extrapolated to the pre-registered sizes from OUT-OF-GRID b1 data")
print("     (b1: seed-averaged RMS at n = 6, 12, 24, 48, delta = 0.2, 64 trials, seeds 777001/313317)")
out_n = np.array([6.0, 12.0, 24.0, 48.0])
data = {"alt2": np.array([2.638e-3, 2.908e-3, 5.190e-3, 9.131e-3]),
        "null7": np.array([6.317e-3, 1.113e-2, 1.586e-2, 2.889e-2])}
for name, y in data.items():
    slope, intercept = np.polyfit(np.log(out_n), np.log(y), 1)
    pred = {int(n): float(math.exp(intercept + slope * math.log(n))) for n in (8, 16, 32)}
    stat = max(pred.values())
    print("     %-6s log-log slope = %+.3f   extrapolated n=8/16/32: %.2e / %.2e / %.2e"
          % (name, slope, pred[8], pred[16], pred[32]))
    print("            -> pre-registered statistic max_n RMS approx %.2e = %.1fx the 1e-3 floor" % (stat, stat / 1e-3))
print("     align/alt1 measured 1.0e-16..1.9e-16 at every out-of-grid size, n-independent")
print("            -> ceiling 1e-12 is 5e3 x above the observed level; the align side cannot fire")
