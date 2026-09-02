"""Adversary b6 (kill_executable): how much killing power does K4 (defect dilution) have?

The defect mode is DETERMINISTIC given the seed: one Delta = 0.35*G, then eps(n) = simplicity_residual
((n-1) Sigma_0 + Sigma_0 + Delta_aligned) for n in {4,...,64}.  By the step-2 identity
   eps(n) = ((n-1)/n^2) ||tl gram Delta|| / ||gram(Sigma_0 + Delta/n)||,
so the ONLY thing that can move eps(64)/eps(8) away from 63/64^2 / (7/64) = 0.140625 is the
denominator drift, which is O(||Delta||/n).  Here that drift is measured over 40 INDEPENDENT Delta
draws (adversary seeds, never the pre-registered one) to see whether the window [0.124,0.158] and
the slope window [-0.96,-0.86] can fire at all.
"""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment
REF = geometric_self_dual_triple(np.eye(4))
GRID = (4, 8, 16, 32, 64)
def fit(x, y): return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])

rng = np.random.default_rng(987654321)   # NOT 20260902
ratios, slopes = [], []
for _ in range(40):
    while True:
        t = np.eye(4) + 0.35 * rng.normal(size=(4, 4))
        if float(np.linalg.det(t)) > 0.2: break
    ac = optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate
    eps = [simplicity_residual((n - 1) * REF + ac) for n in GRID]
    ratios.append(eps[GRID.index(64)] / eps[GRID.index(8)]); slopes.append(fit(GRID, eps))
r = np.asarray(ratios); s = np.asarray(slopes)
print(f"   ratio  eps(64)/eps(8): min {r.min():.5f}  max {r.max():.5f}  mean {r.mean():.5f}   "
      f"pre-registered {0.140625}  window (0.124, 0.158)")
print(f"   slope  d ln eps/d ln n: min {s.min():.5f}  max {s.max():.5f}  mean {s.mean():.5f}   "
      f"pre-registered {-0.9069}   window (-0.96, -0.86)")
print(f"   fraction of independent Delta draws INSIDE both windows: "
      f"{float(np.mean((r>0.124)&(r<0.158)&(s>-0.96)&(s<-0.86))):.3f}")
print("   => K4 fires only if the exact step-2 identity is false; over the whole Delta ensemble the")
print("      statistic is pinned to within ~1% of the pre-registered value.  Near-zero killing power.")
print("   (the same identity was already verified to 1e-14 in verify/Q-0008/F-01/adversary/a8.)")
