"""Adversary b7: K4's error budget is MIS-CENTRED, and the disclosed eps(4),eps(8) peek is material.

(i) over 200 independent Delta draws (adversary seeds), the distribution of the two pre-registered
    K4 statistics is compared with the pre-registered centres 0.140625 / -0.9069 and the windows.
(ii) the disclosed quantity r48 = eps(8)/eps(4) (prover ran it at the pre-registered seed and got
     0.587 vs the identity's 0.5833) is correlated with the two pre-registered statistics: if the
     correlation is high, knowing r48 largely determines whether K4 fires.
"""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment
REF = geometric_self_dual_triple(np.eye(4))
GRID = (4, 8, 16, 32, 64)
def fit(x, y): return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])
rng = np.random.default_rng(13571113)
rows = []
for _ in range(200):
    while True:
        t = np.eye(4) + 0.35 * rng.normal(size=(4, 4))
        if float(np.linalg.det(t)) > 0.2: break
    ac = optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate
    e = {n: simplicity_residual((n - 1) * REF + ac) for n in GRID}
    rows.append((e[8] / e[4], e[64] / e[8], fit(GRID, [e[n] for n in GRID]),
                 fit((8, 16, 32, 64), [e[n] for n in (8, 16, 32, 64)])))
a = np.asarray(rows)
r48, r648, sl5, sl4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
print(f"   identity (delta_c -> 0):  r48 = {7/64/(3/16):.4f}   r64/8 = 0.140625   slope5 = -0.9069")
print(f"   r48    : mean {r48.mean():.4f} sd {r48.std():.4f}   [prover disclosed 0.587 at the prereg seed]")
print(f"   r64/8  : mean {r648.mean():.4f} sd {r648.std():.4f}   window (0.124,0.158)  "
      f"P(fire) = {float(np.mean((r648<0.124)|(r648>0.158))):.3f}")
print(f"   slope5 : mean {sl5.mean():.4f} sd {sl5.std():.4f}   window (-0.96,-0.86)  "
      f"P(fire) = {float(np.mean((sl5<-0.96)|(sl5>-0.86))):.3f}")
print(f"   P(K4 fires although the exact identity is TRUE) = {float(np.mean((r648<0.124)|(r648>0.158)|(sl5<-0.96)|(sl5>-0.86))):.3f}")
print(f"   (slope restricted to n=8..64 would be: mean {sl4.mean():.4f} sd {sl4.std():.4f} -- the n=4 point carries the drift)")
print(f"\n   corr(r48, r64/8)  = {np.corrcoef(r48, r648)[0,1]:+.3f}")
print(f"   corr(r48, slope5) = {np.corrcoef(r48, sl5)[0,1]:+.3f}")
sel = np.abs(r48 - 0.587) < 0.004
print(f"   conditional on r48 in 0.587+-0.004 ({int(sel.sum())} draws): r64/8 mean {r648[sel].mean():.4f} sd {r648[sel].std():.4f},"
      f" slope5 mean {sl5[sel].mean():.4f} sd {sl5[sel].std():.4f}")
print(f"   P(K4 fires | r48 = 0.587) = {float(np.mean((r648[sel]<0.124)|(r648[sel]>0.158)|(sl5[sel]<-0.96)|(sl5[sel]>-0.86))):.3f}")
