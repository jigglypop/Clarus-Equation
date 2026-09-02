"""Adversary 3: two hidden assumptions of the card's ladder step 3 / kill scripts.
(A) Is the residual actually LINEAR in delta at the pre-registered delta?  The lemma
    'residual = |A sum delta e|/n + O(delta^2)' and the claim 'delta cancels in the
    exponent' both require the linear regime.  For the heritable mode the per-cell
    perturbation grows like sqrt(depth) ~ n^{1/4}, so delta_eff GROWS with n.
(B) The MIN_DET=0.05 accept/reject filter (undeclared in the card) rejects the whole
    configuration if ANY cell has det(I+de) <= 0.05.  Rejection probability grows with n
    for the heritable mode -> n-dependent conditioning that truncates the very tail that
    produces the +1/4 exponent.
This script does NOT run the pre-registered kill statistic (sizes x 64 trials).
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"verify"/"Q-0008"/"F-01"))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment
import check_modes as CM

def her_labels(n, rng):
    parent = CM.uniform_rooted_tree(n, rng)
    xi = rng.normal(size=(n,4,4))
    lab = np.zeros((n,4,4)); ch=[[] for _ in range(n)]
    for v,p in enumerate(parent):
        if p>=0: ch[p].append(v)
    root=parent.index(-1); order=[root]; i=0
    while i<len(order):
        x=order[i]; i+=1; order.extend(ch[x])
    for v in order:
        p=parent[v]; lab[v]=xi[v]+(lab[p] if p>=0 else 0.0)
    return lab

def resid(pert):
    blocked=np.zeros((3,6))
    for de in pert:
        t=np.eye(4)+de
        if float(np.linalg.det(t))<=CM.MIN_DET: return math.nan
        blocked+=optimal_internal_alignment(CM.REFERENCE, geometric_self_dual_triple(t)).aligned_candidate
    return simplicity_residual(blocked)

print("== (A) delta-linearity of the heritable residual (n=64, 24 trials, SAME labels reused) ==")
rng=np.random.default_rng(7)
labs=[her_labels(64,rng) for _ in range(24)]
deltas=[0.0005,0.001,0.002,0.005,0.01,0.02,0.04]
rmss=[]
for d in deltas:
    vals=[resid(d*L) for L in labs]
    vals=[v for v in vals if math.isfinite(v)]
    rmss.append(float(np.sqrt(np.mean(np.square(vals)))))
    print(f"   delta={d:<7} RMS={rmss[-1]:.6g}  n_finite={len(vals)}/24")
for i in range(len(deltas)-1):
    print(f"   local d ln RMS / d ln delta on [{deltas[i]},{deltas[i+1]}] = "
          f"{math.log(rmss[i+1]/rmss[i])/math.log(deltas[i+1]/deltas[i]):.4f}   (linear lemma requires 1.000)")

print("\n== (A') same for the i.i.d. mode at n=64 (K1 uses delta=0.05, K2/K3 use 0.02) ==")
rng=np.random.default_rng(11)
xis=[rng.normal(size=(64,4,4)) for _ in range(24)]
rms2=[]
for d in [0.005,0.01,0.02,0.05,0.1]:
    vals=[resid(d*X) for X in xis]; vals=[v for v in vals if math.isfinite(v)]
    rms2.append(float(np.sqrt(np.mean(np.square(vals)))))
    print(f"   delta={d:<6} RMS={rms2[-1]:.6g}")
dd=[0.005,0.01,0.02,0.05,0.1]
for i in range(len(dd)-1):
    print(f"   local slope in delta [{dd[i]},{dd[i+1]}] = {math.log(rms2[i+1]/rms2[i])/math.log(dd[i+1]/dd[i]):.4f}")

print("\n== (B) MIN_DET=0.05 rejection probability vs n, heritable mode, delta=0.02 ==")
for n in (8,16,32,64,128):
    rng=np.random.default_rng(1000+n); rej=0; tot=200
    for _ in range(tot):
        L=her_labels(n,rng)
        bad=any(float(np.linalg.det(np.eye(4)+0.02*L[v])) <= CM.MIN_DET for v in range(n))
        rej+=bad
    print(f"   n={n:<4} reject-whole-configuration rate = {rej/tot:.3f}")
print("   (iid mode at delta=0.02, for contrast)")
for n in (8,128):
    rng=np.random.default_rng(2000+n); rej=0
    for _ in range(200):
        X=rng.normal(size=(n,4,4))
        rej += any(float(np.linalg.det(np.eye(4)+0.02*X[v]))<=CM.MIN_DET for v in range(n))
    print(f"   n={n:<4} reject rate = {rej/200:.3f}")
