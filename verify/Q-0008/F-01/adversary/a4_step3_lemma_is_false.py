"""Adversary 4: ladder step 3 ('residual = |A sum_v delta e_v| / n + O(delta^2)') is FALSE.

Every single cell triple Sigma(e) is EXACTLY Plebanski-simple, and the SO(3) polar
alignment preserves simplicity.  Write Y = sum_v X_v = n*Sigma_0 + S, S = sum_v eta_v,
eta_v = X_v - Sigma_0.  Bilinearity of the Plebanski gram plus per-cell simplicity
  0 = tl[gram(Sigma_0 + eta_v)] = 2 tl[cross(Sigma_0, eta_v)] + tl[gram(eta_v)]
gives the EXACT identity
  tl[gram(Y)] = -(n-1) * sum_v tl[gram(eta_v)]  +  sum_{v != w} tl[G(v,w)] ,
so the O(delta) term cancels identically.  The residual is O(delta^2) and its leading
n-dependence comes from the fluctuation of sum_v tl[gram(eta_v)] -- a sum of the
SECOND-order per-cell invariants -- not from the label sum S.
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (
    geometric_self_dual_triple, simplicity_residual, plebanski_gram, wedge_scalar)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

REF = geometric_self_dual_triple(np.eye(4))
def tl(M): return M - np.trace(M)/3.0*np.eye(3)
def G(A,B): return np.array([[wedge_scalar(A[i],B[j]) for j in range(3)] for i in range(3)])

print("== each aligned cell is EXACTLY simple (so no first-order residual per cell) ==")
rng=np.random.default_rng(3)
worst=0.0
for _ in range(50):
    t=np.eye(4)+0.2*rng.normal(size=(4,4))
    if abs(np.linalg.det(t))<0.2: continue
    X=optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate
    worst=max(worst, simplicity_residual(X))
print("   max simplicity_residual over 50 aligned single cells =", worst)

print("\n== exact cancellation identity, n=6, delta=0.1 ==")
rng=np.random.default_rng(5)
n, d = 6, 0.1
Xs=[optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4)+d*rng.normal(size=(4,4)))).aligned_candidate for _ in range(n)]
Y=sum(Xs); etas=[X-REF for X in Xs]
lhs=tl(plebanski_gram(Y))
rhs=-(n-1)*sum(tl(G(e,e)) for e in etas) + sum(tl(G(etas[v],etas[w])) for v in range(n) for w in range(n) if v!=w)
print("   ||tl gram(Y) - identity RHS|| =", float(np.linalg.norm(lhs-rhs)), " (||lhs|| =", float(np.linalg.norm(lhs)),")")
lin = 2*n*tl(G(REF, sum(etas)))
print("   ||first-order term 2n*tl cross(Sigma0,S)|| =", float(np.linalg.norm(lin)),
      " vs || -n*sum_v tl gram(eta_v)|| =", float(np.linalg.norm(-n*sum(tl(G(e,e)) for e in etas))),
      " -> these two cancel exactly, killing the O(delta) term")

print("\n== is the per-cell second-order mean traceless part zero (step 3's isotropy claim)? ==")
rng=np.random.default_rng(9)
acc=np.zeros((3,3)); M=4000
for _ in range(M):
    t=np.eye(4)+0.01*rng.normal(size=(4,4))
    X=optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate
    acc+=tl(G(X-REF,X-REF))
acc/=M
print("   ||E tl gram(eta)|| =", float(np.linalg.norm(acc)),
      " ; per-sample RMS ||tl gram(eta)|| scale ~", 1e-4)
print("   (a nonzero mean would give an n-INDEPENDENT residual floor for every mode)")
