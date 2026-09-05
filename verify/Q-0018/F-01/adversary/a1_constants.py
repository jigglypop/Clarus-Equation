"""Adversary a1: independent re-derivation of the Q-0018 F-01 constants.

Independence: the M_ab tensor is rebuilt by NUMERICAL differentiation of the actual
physics code (geometric_self_dual_triple + optimal_internal_alignment + plebanski_gram),
not from the sympy exact_M in c_delta.py.  Stiefel moments are obtained three ways:
(i) invariant-theory linear solve, (ii) exact operator identity, (iii) Haar MC with a
different sampler (SO(4) rotation of a fixed pair) and a different seed.
"""
from __future__ import annotations
import json, math, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

SEED = 424243
out = {"script": "a1_constants", "seed": SEED}
I4 = np.eye(4)
REF = geometric_self_dual_triple(I4)
G0 = plebanski_gram(REF)
out["G0"] = {"matrix": G0.tolist(), "norm_sq": float(np.sum(G0 * G0)),
             "is_2I": bool(np.allclose(G0, 2 * np.eye(3), atol=1e-12))}


def aligned(t):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate


def L_of(xi, h=1.0e-3):
    def f(s):
        return aligned(I4 + s * xi)
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


def gram_bilinear(A, B):
    return np.array([[float(A[i][0]*B[j][3] + A[i][3]*B[j][0] + A[i][1]*B[j][4]
                            + A[i][4]*B[j][1] + A[i][2]*B[j][5] + A[i][5]*B[j][2])
                      for j in range(3)] for i in range(3)])


def tl(X):
    return X - np.trace(X) / 3.0 * np.eye(3)


basis = []
for a in range(16):
    e = np.zeros((4, 4)); e[a // 4, a % 4] = 1.0
    basis.append(e)
Lb = [L_of(b) for b in basis]
M = np.zeros((16, 16, 3, 3))
for a in range(16):
    for b in range(16):
        M[a, b] = tl((gram_bilinear(Lb[a], Lb[b]) + gram_bilinear(Lb[b], Lb[a])) / 2.0)

T2 = float(np.einsum("abij,abij->", M, M))
T4 = float(np.einsum("aaij,aaij->", M, M))
Msum = np.einsum("aaij->ij", M)
out["A_M_tensor_from_physics_code"] = {
    "T2": T2, "T2_dev_from_60": T2 - 60.0,
    "T4": T4, "T4_dev_from_2": T4 - 2.0,
    "sum_a_M_aa_maxabs": float(np.max(np.abs(Msum))),
    "M_symmetric_in_ab": float(np.max(np.abs(M - np.swapaxes(M, 0, 1)))),
}


def M_of(X, Y):
    x = np.asarray(X, float).reshape(16); y = np.asarray(Y, float).reshape(16)
    return np.einsum("a,b,abij->ij", x, y, M)


rng = np.random.default_rng(SEED)
anti_basis = []
for i in range(4):
    for j in range(i + 1, 4):
        A_ = np.zeros((4, 4)); A_[i, j] = 1.0; A_[j, i] = -1.0
        anti_basis.append(A_ / math.sqrt(2))
worst_anti_L = max(float(np.max(np.abs(L_of(A_)))) for A_ in anti_basis)
worst_anti_M = 0.0
for A_ in anti_basis:
    for b in basis:
        worst_anti_M = max(worst_anti_M, float(np.max(np.abs(M_of(A_, b)))))
worst_trace_M = max(float(np.max(np.abs(M_of(I4 / 2.0, b)))) for b in basis)
out["B_support"] = {
    "max_abs_L_on_antisym_basis": worst_anti_L,
    "max_abs_M_antisym_any": worst_anti_M,
    "max_abs_M_identity_any": worst_trace_M,
    "max_abs_M_overall_scale": float(np.max(np.abs(M))),
    "claim_L_kills_antisymmetric": bool(worst_anti_L < 1e-8),
    "claim_M_kills_trace": bool(worst_trace_M < 1e-8),
}

A_lin = np.array([[4.0, 2.0], [4.0, 20.0]]); rhs = np.array([0.25, 0.0])
a_sol, b_sol = np.linalg.solve(A_lin, rhs)
a_ex, b_ex = Fraction(5, 72), Fraction(-1, 72)
NS = 500000
g = rng.normal(size=(NS, 4, 4))
q, r = np.linalg.qr(g)
q = q * np.sign(np.einsum("tii->ti", r))[:, None, :]
det = np.linalg.det(q)
q[det < 0, :, 0] *= -1.0
n_ = q[:, :, 0]; m_ = q[:, :, 1]
xi = np.einsum("ti,tk->tik", n_, m_).reshape(NS, 16)
C1_mc = xi.T @ xi / NS
C1_ex = np.zeros((16, 16))
for i in range(4):
    for k in range(4):
        for j in range(4):
            for l in range(4):
                C1_ex[4*i+k, 4*j+l] = (float(a_ex)*(i == j)*(k == l)
                                       + float(b_ex)*((i == k)*(j == l) + (i == l)*(j == k)))
eig = np.sort(np.linalg.eigvalsh(C1_ex))
Xr = rng.normal(size=(4, 4))
op_lhs = (C1_ex @ Xr.reshape(16)).reshape(4, 4)
op_rhs = float(a_ex)*Xr + float(b_ex)*Xr.T + float(b_ex)*np.trace(Xr)*np.eye(4)
out["C_stiefel_C1"] = {
    "invariant_theory_solve": [float(a_sol), float(b_sol)],
    "exact_rationals": [str(a_ex), str(b_ex)],
    "solve_matches_exact": bool(abs(a_sol - float(a_ex)) < 1e-14 and abs(b_sol - float(b_ex)) < 1e-14),
    "haar_mc_max_abs_dev": float(np.max(np.abs(C1_mc - C1_ex))),
    "haar_mc_samples": NS,
    "trace": float(np.trace(C1_ex)),
    "eig_mult_0": int(np.sum(np.abs(eig) < 1e-12)),
    "eig_mult_1_18": int(np.sum(np.abs(eig - 1/18) < 1e-12)),
    "eig_mult_1_12": int(np.sum(np.abs(eig - 1/12) < 1e-12)),
    "operator_identity_max_dev": float(np.max(np.abs(op_lhs - op_rhs))),
}

T2C1 = float(np.einsum("ac,bd,abij,cdij->", C1_ex, C1_ex, M, M))
floor = np.einsum("ab,abij->ij", C1_ex, M)
E01 = np.zeros((4, 4)); E01[0, 1] = 1.0
Q4_1 = float(np.sum(M_of(E01, E01) ** 2))
E00 = np.zeros((4, 4)); E00[0, 0] = 1.0
Q4_stretch = float(np.sum(M_of(E00, E00) ** 2))
S01 = (E01 + E01.T) / 2.0
Q4_sym = float(np.sum(M_of(S01, S01) ** 2))
orbit = []
for _ in range(30):
    gg = rng.normal(size=(4, 4)); qq, rr = np.linalg.qr(gg)
    qq = qq @ np.diag(np.sign(np.diag(rr)))
    if np.linalg.det(qq) < 0:
        qq[:, 0] *= -1
    Xc = qq @ E01 @ qq.T
    orbit.append(float(np.sum(M_of(Xc, Xc) ** 2)))
s4 = 256.0
c2 = 2 * s4 * T2C1 / 12
c4 = s4 * (Q4_1 - 2 * T2C1) / 12
out["D_contractions"] = {
    "T2_C1": T2C1, "T2_C1_minus_5_over_27": T2C1 - 5/27,
    "identity_T2C1_minus_(1_18)^2_T2": T2C1 - (1/18)**2 * 60,
    "mean_floor_maxabs": float(np.max(np.abs(floor))),
    "Q4_1": Q4_1, "Q4_1_minus_1_over_6": Q4_1 - 1/6,
    "Q4_uniaxial_E00": Q4_stretch,
    "Q4_symmetric_part_only": Q4_sym,
    "Q4_1_minus_Q4_sym": Q4_1 - Q4_sym,
    "orbit_min": min(orbit), "orbit_max": max(orbit), "orbit_spread": max(orbit) - min(orbit),
    "c2": c2, "c2_minus_640_over_81": c2 - 640/81,
    "c4": c4, "c4_plus_352_over_81": c4 + 352/81,
    "c_delta": c2 + c4, "c_delta_minus_32_over_9": c2 + c4 - 32/9,
    "c_delta_minus_256Q4_over_12": (c2 + c4) - 256 * Q4_1 / 12,
}
np.save(HERE / "M_tensor_adversary.npy", M)
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a1_constants.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
