"""Adversary a4 (content): run the FROZEN F-02 block pipeline with alternative transmitted
subspaces.  If non-orbit-tangent projectors reproduce the card's K1/K3 statistics, then the
card's discriminant ('the orbit tangent is what makes the transition') is over-claimed.

Deliberately OUTSIDE the pre-registered configuration so this cannot pre-empt K1/K2/K3:
  sizes (6, 12, 24)   [card: 8,16,32,64,128]     trials 96      [card: 256]
  seeds 424242/424243 [card: 20260902/20260903]  delta 0.005    [same regime, declared]
Common random numbers across projectors (same seed per projector run) so the comparison is sharp.
"""
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, fit_slope, rms  # noqa: E402
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)

SIZES = (6, 12, 24)
TRIALS = 96
SEED, SEED_IID = 424242, 424243
DELTA = 0.005


def diag_proj(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


def rand_proj(seed, dim=4):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(16, dim)))
    return q @ q.T


def labels(coeff):
    return (coeff @ FLAT).reshape(-1, 4, 4)


def rule(parent, coeff, P):
    order, *_ = tree_arrays(parent)
    trans = coeff @ P.T
    fold = coeff - trans
    acc = np.zeros_like(coeff)
    for v in order:
        p = parent[v]
        acc[v] = trans[v] + (acc[p] if p >= 0 else 0.0)
    return acc + fold


def sample(n, rng, P, delta=DELTA):
    while True:
        parent = uniform_rooted_tree(n, rng)
        c = rng.normal(size=(len(parent), 16))
        v = block_residual(labels(rule(parent, c, P)), delta)
        if math.isfinite(v):
            return v


def sample_iid(n, rng, delta=DELTA):
    while True:
        v = block_residual(labels(rng.normal(size=(n, 16))), delta)
        if math.isfinite(v):
            return v


rng_i = np.random.default_rng(SEED_IID)
iid = [rms([sample_iid(n, rng_i) for _ in range(TRIALS)]) for n in SIZES]
print("iid RMS at sizes", SIZES, "=", np.array(iid))
print("exact sqrt(n-1)/n slope on this grid =", fit_slope(SIZES, [math.sqrt(n - 1) / n for n in SIZES]))
print("measured iid slope                   =", fit_slope(SIZES, iid))

cases = {
    "P_align  scale+sd (CARD)   ": diag_proj(GROUPS["scale"] + GROUPS["sd"]),
    "P_alt1   scale+asd  (NOT T)": diag_proj(GROUPS["scale"] + GROUPS["asd"]),
    "P_alt2   sd+asd_1   (NOT T)": diag_proj(GROUPS["sd"] + GROUPS["asd"][:1]),
    "P_scale  scale only (1-dim)": diag_proj(GROUPS["scale"]),
    "P_asd    asd only   (3-dim)": diag_proj(GROUPS["asd"]),
    "P_null7  scale+sd+asd      ": diag_proj(GROUPS["scale"] + GROUPS["sd"] + GROUPS["asd"]),
    "P_rand4  seed 424242       ": rand_proj(424242),
    "P_I      full (F-02 her)   ": np.eye(16),
}
t0 = time.time()
print("\n%-28s %-34s %-10s %s" % ("projector", "RMS_align", "slope", "rho = RMS/RMS_iid"))
for name, P in cases.items():
    rng = np.random.default_rng(SEED)
    vals = [rms([sample(n, rng, P) for _ in range(TRIALS)]) for n in SIZES]
    print("%-28s %-34s %-10.4f %s" % (name, np.array2string(np.array(vals), precision=3),
                                      fit_slope(SIZES, vals),
                                      np.array2string(np.array(vals) / np.array(iid), precision=3)))
print("elapsed %.1f s" % (time.time() - t0))
