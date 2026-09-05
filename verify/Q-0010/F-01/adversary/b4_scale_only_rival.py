"""Adversary b4 (re-audit, content/dof): a STRICTLY WEAKER rival axiom.

R1 = transmit only the scale direction (1-dim), fold sd + asd + sym (15 dims).
R1 is inside ker M~, so a2 already gives budget (0,0,1) and rho(128) = 1.0000 for it: it passes K1
and K3 exactly.  Question: does K4 (orbit closure, fold = 0) exclude it?  And is there ANY statistic
in the frozen pipeline that separates the card 4-plane from R1 and from alt1?
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
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple  # noqa: E402
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
I4 = np.eye(4)
B0 = geometric_self_dual_triple(I4)


def diag_proj(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


PROJ = {
    "align  scale+sd (card, 4d)": diag_proj(GROUPS["scale"] + GROUPS["sd"]),
    "alt1   scale+asd (4d)":      diag_proj(GROUPS["scale"] + GROUPS["asd"]),
    "R1     scale only (1d)":     diag_proj(GROUPS["scale"]),
    "R2     sd only (3d)":        diag_proj(GROUPS["sd"]),
}


def rule(parent, c, P):
    order, *_ = tree_arrays(parent)
    tr = c @ P.T
    acc = np.zeros_like(c)
    for v in order:
        p = parent[v]
        acc[v] = tr[v] + (acc[p] if p >= 0 else 0.0)
    return acc + (c - tr)


def closure_and_spread(P, n, seed, delta=0.2, trials=32):
    rr = np.random.default_rng(seed)
    res, spread = [], []
    while len(res) < trials:
        par = uniform_rooted_tree(n, rr)
        c = rr.normal(size=(len(par), 16)) @ P.T
        lab = (rule(par, c, P) @ FLAT).reshape(-1, 4, 4)
        v = block_residual(lab, delta)
        if not math.isfinite(v):
            continue
        angles = []
        for l in lab:
            R = optimal_internal_alignment(B0, geometric_self_dual_triple(I4 + delta * l)).rotation
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))
        res.append(v); spread.append(float(np.std(angles)))
    return rms(res), float(np.mean(spread))


print("fold = 0, delta = 0.2, 32 trials, OUT-OF-GRID sizes, seeds not the card seed")
print("  %-28s %-6s %-14s %-14s" % ("projector", "n", "closure residual", "rotation spread (deg)"))
for name, P in PROJ.items():
    for n in (12, 24):
        r, s = closure_and_spread(P, n, 606061 + n)
        print("  %-28s %-6d %-14.3e %-14.3f" % (name, n, r, s))
print("")
print("K4 as written cannot exclude R1 (scale only, 1-dim) or alt1: both close the orbit exactly.")
print("The within-block polar rotation spread separates the card 4-plane from BOTH of them.")
