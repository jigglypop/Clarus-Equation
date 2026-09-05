"""Adversary a5 (hidden assumptions): the transmitted part is only invisible to FIRST order at a
FIXED base point B0.  Two questions the card's scope asserts but does not measure:

 (A) contamination size.  Per trial, compare
        E_full  = residual(label_v = sum_{u<=v} P xi_u + (1-P) xi_v)      [card rule]
        E_fold  = residual(label_v = (1-P) xi_v)                          [transmitted part deleted]
     with COMMON random numbers.  If the transmitted channel were exactly invisible, the ratio is
     exactly 1.  Its departure is the O(delta) / O(n) correction.  Fit  dev ~ C delta^a n^b.
 (B) base-point dependence.  The card transmits a LINEAR sum of tangent vectors at the fixed B0.
     The geometric reading of the same axiom is a GROUP action: e_v = A_v (I + delta fold_v) with
     A_v = exp(delta sum_{u<=v} P xi_u) in R_+ x SO(3)_sd.  Both are 'transmit the orbit tangent';
     they differ at O(delta^2 depth).  Measured side by side.

Sizes 10/40/160, trials 128, seeds 515151 -- all outside the pre-registered K1/K2/K3 configuration.
"""
import math
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, MIN_DET, REFERENCE  # noqa: E402
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual  # noqa: E402
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
P_IDX = GROUPS["scale"] + GROUPS["sd"]
P = np.zeros((16, 16))
for i in P_IDX:
    P[i, i] = 1.0


def labels(c):
    return (c @ FLAT).reshape(-1, 4, 4)


def accum(parent, trans):
    order, *_ = tree_arrays(parent)
    acc = np.zeros_like(trans)
    for v in order:
        p = parent[v]
        acc[v] = trans[v] + (acc[p] if p >= 0 else 0.0)
    return acc


def residual_group(parent, coeff, delta):
    """Geometric reading: transmitted part acts as a group element on the tetrad."""
    trans = accum(parent, coeff @ P.T)          # accumulated tangent coefficients
    fold = labels(coeff - coeff @ P.T)
    blocked = np.zeros_like(REFERENCE)
    for v in range(len(parent)):
        A = expm(delta * (trans[v] @ FLAT).reshape(4, 4))
        tet = A @ (np.eye(4) + delta * fold[v])
        if float(np.linalg.det(tet)) <= MIN_DET:
            return math.nan
        blocked += optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(tet)).aligned_candidate
    return float(simplicity_residual(blocked))


def run(n, delta, trials, seed):
    rng = np.random.default_rng(seed)
    full, fold, grp = [], [], []
    while len(full) < trials:
        parent = uniform_rooted_tree(n, rng)
        c = rng.normal(size=(len(parent), 16))
        trans = accum(parent, c @ P.T)
        a = block_residual(labels(trans + (c - c @ P.T)), delta)      # card rule
        b = block_residual(labels(c - c @ P.T), delta)                # transmitted deleted
        g = residual_group(parent, c, delta)                          # group action reading
        if all(math.isfinite(x) for x in (a, b, g)):
            full.append(a); fold.append(b); grp.append(g)
    full, fold, grp = map(np.asarray, (full, fold, grp))
    return {"n": n, "delta": delta,
            "dev_linear": float(np.mean(np.abs(full / fold - 1.0))),
            "dev_group": float(np.mean(np.abs(grp / fold - 1.0))),
            "rms_ratio_linear": float(np.sqrt(np.mean(full ** 2) / np.mean(fold ** 2))),
            "rms_ratio_group": float(np.sqrt(np.mean(grp ** 2) / np.mean(fold ** 2)))}


t0 = time.time()
rows = []
print("%6s %8s | %12s %12s | %12s %12s" % ("n", "delta", "dev_lin", "dev_grp", "rmsratio_lin", "rmsratio_grp"))
for delta in (0.005, 0.02, 0.08):
    for n in (10, 40, 160):
        r = run(n, delta, 128, 515151 + n)
        rows.append(r)
        print("%6d %8.3f | %12.3e %12.3e | %12.6f %12.6f"
              % (n, delta, r["dev_linear"], r["dev_group"], r["rms_ratio_linear"], r["rms_ratio_group"]))

A = np.array([[1.0, math.log(r["delta"]), math.log(r["n"])] for r in rows])
for key in ("dev_linear", "dev_group"):
    y = np.array([math.log(r[key]) for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    print("fit %s ~ C delta^a n^b :  a = %.3f  b = %.3f  C = %.3e"
          % (key, coef[1], coef[2], math.exp(coef[0])))
    a, b, C = coef[1], coef[2], math.exp(coef[0])
    n_star = math.exp((math.log(0.18) - math.log(C) - a * math.log(0.005)) / b)
    print("    -> at delta = 0.005 the deviation reaches the K1 window half-width 0.18 at n ~ %.3g" % n_star)
print("elapsed %.1f s" % (time.time() - t0))
