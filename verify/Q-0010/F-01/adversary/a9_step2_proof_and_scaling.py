"""Adversary a9: (i) ladder step 2 is a ONE-LINE proof, not a numerical fact; (ii) per-cell
scaling of the true residual-threshold counting exponent.

(i)  Every geometric triple is simple:  tl G(Phi(e), Phi(e)) == 0 for all nondegenerate e.
     Differentiate at e = I:            2 tl G(B0, dPhi xi) == 0 for ALL xi.
     Since dPhi(scale) = B0, the 'scale column of M is zero' (card step 2, 3.9e-14) is a corollary,
     and the stronger statement (B0 is tl-G-orthogonal to the WHOLE image of dPhi) also follows.
(ii) exponent of P[eps <= x] ~ x^k for n = 2, 3, 4 folded-only blocks (per-cell increment).
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402

I4 = np.eye(4)


def tl(m):
    return m - np.trace(m) / 3.0 * np.eye(3)


def Gb(A, B):
    return (plebanski_gram(A + B) - plebanski_gram(A - B)) / 4.0


rng = np.random.default_rng(2468)
worst_simple = 0.0
for _ in range(200):
    e = I4 + 0.9 * rng.normal(size=(4, 4))
    if abs(float(np.linalg.det(e))) < 0.05:
        continue
    S = geometric_self_dual_triple(e)
    worst_simple = max(worst_simple, float(np.linalg.norm(tl(plebanski_gram(S))) / np.linalg.norm(plebanski_gram(S))))
print("(i) max normalized ||tl G(Phi(e),Phi(e))|| over 200 random LARGE tetrads =", worst_simple)

B0 = geometric_self_dual_triple(I4)
worst_grad = 0.0
for _ in range(200):
    xi = rng.normal(size=(4, 4))
    d = (geometric_self_dual_triple(I4 + 0.37 * xi) - geometric_self_dual_triple(I4 - 0.37 * xi)) / 0.74
    worst_grad = max(worst_grad, float(np.linalg.norm(tl(Gb(B0, d)))))
print("    max ||tl G(B0, dPhi xi)|| over 200 random xi   =", worst_grad,
      "  (=> B0 tl-G-orthogonal to the whole image of dPhi; card only states the scale column)")

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
FOLD = np.eye(16)
for i in GROUPS["scale"] + GROUPS["sd"]:
    FOLD[i, i] = 0.0

print("(ii) lower-tail exponent k of P[eps_block <= x] ~ x^k, folded-only ensemble, delta = 0.005")
prev = None
for n, trials in ((2, 30000), (3, 20000), (4, 15000)):
    r = np.random.default_rng(13579)
    vals = []
    while len(vals) < trials:
        c = r.normal(size=(n, 16)) @ FOLD.T
        v = block_residual((c @ FLAT).reshape(-1, 4, 4), 0.005)
        if math.isfinite(v):
            vals.append(v)
    x = np.sort(np.asarray(vals)); p = np.arange(1, len(x) + 1) / len(x)
    lo, hi = int(0.002 * len(x)), int(0.05 * len(x))
    k = float(np.polyfit(np.log(x[lo:hi]), np.log(p[lo:hi]), 1)[0])
    print("     n=%d  k=%.3f   %s" % (n, k, "" if prev is None else "per-cell increment = %.3f" % (k - prev)))
    prev = k
print("     card's per-cell exponent (predicts[6]) = 9.000")
