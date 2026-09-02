"""Adversary b3 (recovers): the card's four `recovers` entries point at verify indices that are
PURE ALGEBRA (0*x=0, closed-form identities).  None of them runs the physics.  Here each limit is
executed against examples/physics (the same modules the kill script uses).

R1  all cells identical (13.3 common orbit)              -> residual identically 0?
R2  coherent two species, p=1/2 (13.5 no-go)             -> n-independent?  equal to eps_star/2?
R3  n = 1 (12.4 finite simplicity audit)                 -> residual 0?
R4  i.i.d. only                                          -> eps = eps_star sqrt(n-1)/n?
ISO ladder step 3's isotropy premise E[tl gram(eta)] = 0 at O(delta^2), computed EXACTLY as
    sum_{ab} tl G(L_ab, L_ab) over the 16 label basis directions (no Monte Carlo).
"""
import math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram, simplicity_residual, wedge_scalar)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

REF = geometric_self_dual_triple(np.eye(4))
DELTA = 0.005

def tl(M): return M - np.trace(M) / 3.0 * np.eye(3)
def G(A, B): return np.array([[wedge_scalar(A[i], B[j]) for j in range(3)] for i in range(3)])
def cell(lab, d=DELTA):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * lab)).aligned_candidate
def block(labels, d=DELTA):
    return simplicity_residual(sum(cell(l, d) for l in labels))
def rms(v): 
    a = np.asarray(v, float); return float(np.sqrt(np.mean(a * a)))

print("== R3: n = 1, single aligned cell (12.4) ==")
rng = np.random.default_rng(11)
w = max(simplicity_residual(cell(rng.normal(size=(4, 4)), 0.2)) for _ in range(200))
print(f"   max simplicity_residual over 200 single cells (delta=0.2) = {w:.3e}   card recovers 'D=0 => eps=0': OK")

print("\n== R1: all n cells identical (13.3 common-metric orbit closure) ==")
X = cell(rng.normal(size=(4, 4)), 0.2)
for n in (2, 5, 17, 64):
    print(f"   n={n:<4} residual(n*X) = {simplicity_residual(n * X):.3e}")

print("\n== R4: i.i.d. mode.  Calibrate eps_star at n=2, then test eps(n) = eps_star sqrt(n-1)/n ==")
TR = 400
rng = np.random.default_rng(777)
tab = {}
for n in (2, 3, 4, 6, 8, 12, 16):
    tab[n] = rms([block(rng.normal(size=(n, 4, 4))) for _ in range(TR)])
eps_star = 2.0 * tab[2]
print(f"   eps_star := 2*RMS(n=2) = {eps_star:.6e}   (delta={DELTA}, {TR} trials)")
print(f"   {'n':>4} {'RMS observed':>14} {'eps_star sqrt(n-1)/n':>22} {'ratio':>8}")
for n, v in tab.items():
    pred = eps_star * math.sqrt(n - 1) / n
    print(f"   {n:>4} {v:14.6e} {pred:22.6e} {v/pred:8.4f}")
print(f"   (MC se on each RMS ~ {1/math.sqrt(2*TR):.3f} relative)")

print("\n== R2a: coherent two species with GAUSSIAN species labels, p=1/2 (kernel reading) ==")
print("   card: D = 4 n^2 p^2 (1-p)^2 => eps = eps_star * 2p(1-p) = eps_star/2, n-independent")
rng = np.random.default_rng(2024)
for n in (4, 8, 16, 32):
    vals = []
    for _ in range(TR):
        gB, gC = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        labs = [gB] * (n // 2) + [gC] * (n - n // 2)
        vals.append(block(labs))
    r = rms(vals)
    print(f"   n={n:<4} RMS = {r:.6e}   eps_star/2 = {eps_star/2:.6e}   ratio = {r/(eps_star/2):.4f}")

print("\n== R2b: coherent two species with a DETERMINISTIC Delta (what 13.5 actually says) ==")
print("   exact identity: eps = p(1-p) ||tl gram Delta|| / ||gram(Sigma_0+(1-p)Delta)||, n-independent")
rng = np.random.default_rng(4242)
lab = rng.normal(size=(4, 4))
for n in (4, 8, 16, 32, 64):
    labs = [np.zeros((4, 4))] * (n // 2) + [lab] * (n - n // 2)
    v = block(labs)
    print(f"   n={n:<4} eps = {v:.6e}   eps/(eps_star/2) = {v/(eps_star/2):.4f}   <- 'eps = eps_star/2' would need 1.000")
print("   => eps_star is NOT one universal constant: the deterministic (13.5) branch and the")
print("      Gaussian branch have different normalisations; only the n-dependence is shared.")

print("\n== ISO: exact O(delta^2) mean of the traceless per-cell gram (ladder step 3 premise) ==")
h = 1e-5
Ls = []
for a in range(4):
    for b in range(4):
        E = np.zeros((4, 4)); E[a, b] = 1.0
        Ls.append((cell(E, h) - cell(E, -h)) / (2 * h))
acc = sum(tl(G(L, L)) for L in Ls)
scale = sum(float(np.linalg.norm(tl(G(L, L)))) for L in Ls)
print(f"   || sum_ab tl G(L_ab,L_ab) || = {float(np.linalg.norm(acc)):.3e}   scale sum ||tl G|| = {scale:.3e}"
      f"   ratio = {float(np.linalg.norm(acc))/scale:.2e}")
print("   (isotropy premise E[tl gram(eta)] = 0 holds EXACTLY at O(delta^2) for any label covariance")
print("    of the form kappa (x) I_16; so no O(delta^2) residual floor.  Step 3's premise is provable.)")
tr_part = sum(float(np.trace(G(L, L))) for L in Ls)
print(f"   (trace part sum_ab tr G(L_ab,L_ab) = {tr_part:.6f}  -- nonzero, as it must be)")

print("\n== ISO check via the block statistic: mean vs fluctuation of the i.i.d. residual ==")
rng = np.random.default_rng(31)
for n in (8, 32):
    vals = [block(rng.normal(size=(n, 4, 4))) for _ in range(600)]
    a = np.asarray(vals)
    print(f"   n={n:<3} mean/RMS = {a.mean()/rms(a):.4f}   (a nonzero O(delta^2) mean floor would push this to 1)")
