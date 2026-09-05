"""Adversary b8 (ladder_complete): ladder step 2 is not merely 'numerically verified', it is a
THREE-LINE proof.  G = plebanski gram (symmetric bilinear), tl = traceless part (linear).
  (0) Sigma_0 simple:                       tl G(S0,S0) = 0
  (1) each aligned cell simple:             0 = tl G(S0+eta_v, S0+eta_v) = 2 tl G(S0,eta_v) + tl G(eta_v,eta_v)
  (2) sum over v, S = sum eta_v:            2 tl G(S0,S) = - sum_v tl G(eta_v,eta_v)
  (3) Y = n S0 + S:  tl G(Y,Y) = 2n tl G(S0,S) + tl G(S,S) = -n sum_v tl G(eta_v,eta_v) + tl G(S,S)
      and  -n sum_v tl G(eta_v-etabar, eta_v-etabar) = -n sum_v tl G(eta_v,eta_v) + tl G(S,S)   [expand]
  => tl G(Y,Y) = -n sum_v tl G(eta_v - etabar, eta_v - etabar).   No signature, no smallness, no model.
Each numbered line is checked below at a random configuration with a LARGE delta (0.3).
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram, wedge_scalar
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment
REF = geometric_self_dual_triple(np.eye(4))
def tl(M): return M - np.trace(M) / 3.0 * np.eye(3)
def G(A, B): return np.array([[wedge_scalar(A[i], B[j]) for j in range(3)] for i in range(3)])
rng = np.random.default_rng(20260902 + 31337)
n, d = 9, 0.3
Xs = []
while len(Xs) < n:
    t = np.eye(4) + d * rng.normal(size=(4, 4))
    if float(np.linalg.det(t)) > 0.2:
        Xs.append(optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate)
et = [X - REF for X in Xs]; S = sum(et); bar = S / n; Y = sum(Xs)
print("  (0) ||tl G(S0,S0)||                                =", float(np.linalg.norm(tl(G(REF, REF)))))
print("  (1) max_v ||2 tl G(S0,eta_v) + tl G(eta_v,eta_v)|| =",
      max(float(np.linalg.norm(2 * tl(G(REF, e)) + tl(G(e, e)))) for e in et))
print("  (2) ||2 tl G(S0,S) + sum_v tl G(eta_v,eta_v)||     =",
      float(np.linalg.norm(2 * tl(G(REF, S)) + sum(tl(G(e, e)) for e in et))))
lhs = tl(G(Y, Y)); rhs = -n * sum(tl(G(e - bar, e - bar)) for e in et)
print("  (3) ||tl G(Y,Y) - (-n sum tl G(eta-etabar))||/||lhs|| =", float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)),
      "   (delta = 0.3, i.e. NOT a small-delta statement)")
# corollary: two species, deterministic Delta
for p_num, p_den in ((1, 2), (1, 3), (1, 9)):
    nB = n * p_num // p_den if (n * p_num) % p_den == 0 else None
    if nB is None: continue
    lab = [np.zeros((4, 4))] * nB + [np.ones((4, 4)) * 0.0] * 0
print("  corollary (two species): sum_v tl G(eta_v-etabar,...) = n p(1-p) tl G(Delta,Delta) -- checked in b3 R2b")
