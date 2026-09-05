"""Adversary 6: does the per-cell second-order mean have a nonzero traceless part?
Ladder step 3 asserts E[tl gram(eta)] = 0 by SO(4) isotropy ('no residual floor').
A nonzero mean makes the residual tend to an n-INDEPENDENT floor, driving every
exponent to 0 at large n and killing K1 from above."""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram, wedge_scalar, simplicity_residual)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment
REF=geometric_self_dual_triple(np.eye(4))
def tl(M): return M-np.trace(M)/3.0*np.eye(3)
def G(A,B): return np.array([[wedge_scalar(A[i],B[j]) for j in range(3)] for i in range(3)])
c=float(np.linalg.norm(plebanski_gram(REF)))
print("||gram(Sigma_0)|| =", c)
for d in (0.005,0.01,0.02,0.05):
    rng=np.random.default_rng(1234)
    M=20000; acc=np.zeros((3,3)); sq=0.0
    for _ in range(M):
        t=np.eye(4)+d*rng.normal(size=(4,4))
        if abs(float(np.linalg.det(t)))<0.05: continue
        e=optimal_internal_alignment(REF,geometric_self_dual_triple(t)).aligned_candidate-REF
        g=tl(G(e,e)); acc+=g; sq+=float(np.linalg.norm(g))**2
    mean=acc/M; sd=math.sqrt(sq/M)
    err=sd/math.sqrt(M)
    print(f"  delta={d:<6} ||E tl gram(eta)||={np.linalg.norm(mean):.4e}  MCerr~{err:.2e}  "
          f"z={np.linalg.norm(mean)/err:6.1f}   implied floor ||E||/||gram(Sigma0)||/1 = {np.linalg.norm(mean)/c:.3e}")
print("\n(floor formula: residual -> (n-1)n||E tl gram eta|| / (n^2 ||gram Sigma_0||) -> ||E||/||gram Sigma_0||)")
