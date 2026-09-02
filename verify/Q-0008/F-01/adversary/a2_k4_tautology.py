"""Adversary 2: is K4 (gamma_coh = 0) a physical test or an algebraic identity?
simplicity_residual is normalized by ||gram||, and gram is quadratic in the triple,
so residual(r*X) == residual(X) for every r>0 and every X. K4 therefore cannot fire."""
import sys, math
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment, repeated_coherent_mismatch_residual

rng = np.random.default_rng(20260902)
REF = geometric_self_dual_triple(np.eye(4))
while True:
    t = np.eye(4) + 0.35*rng.normal(size=(4,4))
    if float(np.linalg.det(t)) > 0.2: break
cand = geometric_self_dual_triple(t)
rs = [repeated_coherent_mismatch_residual(REF, cand, repeats=r) for r in (1,4,16,64,256)]
print("K4 residuals r=1,4,16,64,256:", [f"{x:.17g}" for x in rs])
print("spread max-min =", max(rs)-min(rs))
print("K4 fitted slope =", float(np.polyfit(np.log([1,4,16,64,256]),np.log(rs),1)[0]))

# scale invariance for 200 ARBITRARY random triples (not just the coherent one)
worst = 0.0
for _ in range(200):
    X = rng.normal(size=(3,6))
    for r in (2.0, 7.5, 1e3, 1e-3):
        worst = max(worst, abs(simplicity_residual(r*X) - simplicity_residual(X)))
print("worst |residual(rX)-residual(X)| over 200 random triples x 4 scales =", worst)
print("=> gamma_coh = 0 is degree-0 homogeneity of the normalized residual, not an n-flow result.")

# Does the coherent branch of the FORMULA say anything beyond that?
# eps_n = eps_1 * w_coh  for all n : this is forced by residual(n*Y)=residual(Y).
Y = REF + optimal_internal_alignment(REF, cand).aligned_candidate
print("residual(1*Y), residual(1000*Y):", simplicity_residual(Y), simplicity_residual(1000*Y))
