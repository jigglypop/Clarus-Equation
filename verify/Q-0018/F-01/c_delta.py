"""Q-0018 F-01 -- formula-side numbers for the deterministic split-shear branch (competing hypothesis C).

Model (card): cell label increment is a fixed-magnitude rank-1 shear
    Xi_u = 4 n_u m_u^T,   (n_u, m_u) a Haar-random orthonormal pair in R^4  (SO(4)-conjugation orbit of E_01),
so det(I + delta Xi_u) = 1 exactly, E||Xi||^2 = 16 (same per-cell second moment as the F-02 Gaussian
kappa (x) I_16 with kappa = I), and the cell-cell kernel kappa_vw := E[Xi_v . Xi_w]/16 is the F-02 tree
kernel unchanged.  What changes is the within-cell 16x16 second-moment tensor C (no longer I_16) and the
fourth-moment contraction Q4 (no longer Isserlis).  Fourth-moment law (card step 2):

    E||Phi||^2 = 2 T2(C) D_kappa + (Q4 - 2 T2(C)) S_gen,      Phi = sum_{u,w} (A^T H A)_{uw} M(zeta_u, zeta_w),
    T2(C) = sum_{abcd} C_ac C_bd <M_ab, M_cd>,   Q4 = E||M(zeta,zeta)||^2,   S_gen = sum_u (A^T H A)_uu^2,

and eps_bar^2 = delta^4 E||Phi||^2 / (12 n^2)  =  (delta^4 / n^2) [ c2 D + c4 S_gen ].

This script computes, EXACTLY (sympy / Fraction) where the object is algebraic and by Monte Carlo only for
the design check:
  A  exact M_ab (a1 method: L = Sigma_prime + Omega Sigma_0, Omega = skew(C1)/(2c)); T2 = 60, T4 = 2, ||G0||^2 = 12
  B  exact C1 (unit shear second moment) = (5/72) d_ij d_kl - (1/72)(d_ik d_jl + d_il d_jk); MC cross-check
  C  Q4_1 = ||M(E_01,E_01)||^2 exactly, and its constancy on the SO(4) orbit (20 Haar draws, numeric)
  D  exact T2(C1), exact mean floor sum_ab C1_ab M_ab = 0, c2, c4, c_Delta = c2 + c4 = 256 Q4_1 / 12
  E  pre-registered numbers: c(n) for iid / chain / two-species n in {2,4,8,16}; structure ratio R_str;
     det-branch gamma_iid on {8..128}; det-branch Cayley her/iid ratio at n = 128 and gamma_her (Meir-Moon)
  F  Q-0012 naive per-component-kurtosis reading of the same model (shows it cannot reach c_Delta)
  G  tetrad-free surrogate MC of the fourth-moment law (design check, 1e5 trials, tol 2%) and the CV that
     fixes the physical-MC windows by the declared rule  w = max(0.05, 3 CV / sqrt(2000))
  H  physical single-configuration recovery (no MC): two cells {I, I + 4 delta E_01}, eps/delta^2 -> sqrt(2)/3

All constants below are declared before running; nothing is edited after seeing results.
Seed 20260902.  Writes verify/Q-0018/F-01/result.json.  The kill script (physical MC, mode det) is NOT run here.
"""
from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))

from driver_numbers import cayley_exact, n_k_log  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    _PAIR_INDEX,
    geometric_self_dual_triple,
    plebanski_gram,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

# ------------------------------------------------------------------ declared constants (frozen)
SEED = 20260902
DELTA = 0.005
SHEAR_SCALE = 4.0                 # Xi = 4 n m^T  ->  E||Xi||^2 = 16, kappa = I (F-02 normalisation)
GRID_N = (2, 4, 8, 16)
KERNELS = ("iid", "chain", "coh")
SUR_TRIALS = 100_000
TOL_SUR = 0.02                    # G: |MC/law - 1|
ORBIT_DRAWS = 20
TOL_ORBIT = 1.0e-9                # C: |Q4(Lambda E01 Lambda^T) - Q4_1|
TOL_EXACT = 1.0e-12               # D: exact-vs-numeric agreement of T2(C1)
C1_MC_SAMPLES = 400_000
TOL_C1_MC = 3.0e-3                # B: max |MC C1 - exact C1| (MC se ~ 1.6e-3 on O(0.07) entries)
PHYS_TRIALS_PLANNED = 2000        # kill script design (not run here)
WINDOW_FLOOR = 0.05               # window rule: w = max(0.05, 3 CV / sqrt(PHYS_TRIALS_PLANNED))
WINDOW_SE_MULT = 3.0
SLOPE_GRID = (8, 16, 32, 64, 128)
RECOVERY_DELTAS = (0.02, 0.01, 0.005)
TOL_RECOVERY = 0.02               # H: |eps/delta^2 / (sqrt2/3) - 1| at delta = 0.005

PAIRS = [(0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2)]
assert list(_PAIR_INDEX) == [tuple(p) for p in PAIRS]
EPS3 = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
REFERENCE = geometric_self_dual_triple(np.eye(4))


# ------------------------------------------------------------------ Part A: exact M_ab (a1 method)
def _two_form(u, v):
    m = sp.Matrix(4, 4, lambda i, j: u[i] * v[j] - u[j] * v[i])
    return sp.Matrix([m[i, j] for (i, j) in PAIRS])


def _wedge(a, b):
    return a[0] * b[3] + a[3] * b[0] + a[1] * b[4] + a[4] * b[1] + a[2] * b[5] + a[5] * b[2]


def _triple(e):
    out = []
    for i in range(3):
        form = _two_form(e.row(0).T, e.row(i + 1).T)
        for j in range(3):
            for k in range(3):
                s = EPS3.get((i, j, k), 0)
                if s:
                    form = form + sp.Rational(s, 2) * _two_form(e.row(j + 1).T, e.row(k + 1).T)
        out.append(sp.expand(form))
    return out


def _gram(A, B):
    return sp.Matrix(3, 3, lambda i, j: _wedge(A[i], B[j]))


def _tl(M):
    return M - (M.trace() / 3) * sp.eye(3)


def _frac(x) -> Fraction:
    r = sp.Rational(x)
    return Fraction(int(r.p), int(r.q))


def exact_M() -> tuple[list, dict]:
    d = sp.symbols("d")
    I4 = sp.eye(4)
    S0 = _triple(I4)
    G0 = _gram(S0, S0)
    c = sp.simplify(G0[0, 0])
    assert sp.simplify(G0 - c * sp.eye(3)) == sp.zeros(3, 3)
    L = []
    for a in range(16):
        xi = sp.zeros(4, 4)
        xi[a // 4, a % 4] = 1
        Sd = _triple(I4 + d * xi)
        Sp = [sp.Matrix([sp.diff(comp, d).subs(d, 0) for comp in vec]) for vec in Sd]
        C1 = _gram(S0, Sp)
        Om = (C1 - C1.T) / (2 * c)
        La = []
        for i in range(3):
            La.append(sp.expand(Sp[i] + sum((Om[i, j] * S0[j] for j in range(3)), sp.zeros(6, 1))))
        L.append(La)
    M = [[_tl((_gram(L[a], L[b]) + _gram(L[b], L[a])) / 2) for b in range(16)] for a in range(16)]
    MF = [[[[_frac(M[a][b][i, j]) for j in range(3)] for i in range(3)] for b in range(16)] for a in range(16)]
    T2 = sum(x * x for a in range(16) for b in range(16) for r in MF[a][b] for x in r)
    T4 = sum(x * x for a in range(16) for r in MF[a][a] for x in r)
    Msum = [[sum(MF[a][a][i][j] for a in range(16)) for j in range(3)] for i in range(3)]
    normG0 = sum(_frac(x) ** 2 for x in G0)
    info = {"T2": str(T2), "T4": str(T4), "normG0_sq": str(normG0),
            "sum_a_M_aa_is_zero": all(v == 0 for r in Msum for v in r), "c": str(c)}
    return MF, info


def M_numeric(MF) -> np.ndarray:
    return np.array([[[[float(MF[a][b][i][j]) for j in range(3)] for i in range(3)] for b in range(16)] for a in range(16)])


# ------------------------------------------------------------------ Part B: exact C1 of a Haar orthonormal pair
def exact_C1() -> list[list[Fraction]]:
    """C1[(i,k),(j,l)] = E[n_i m_k n_j m_l] = (5/72) d_ij d_kl - (1/72)(d_ik d_jl + d_il d_jk), a = 4 i + k."""
    C = [[Fraction(0)] * 16 for _ in range(16)]
    for i in range(4):
        for k in range(4):
            for j in range(4):
                for l in range(4):
                    v = Fraction(5, 72) * int(i == j) * int(k == l) - Fraction(1, 72) * (int(i == k) * int(j == l) + int(i == l) * int(j == k))
                    C[4 * i + k][4 * j + l] = v
    return C


def haar_pairs(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    g = rng.normal(size=(count, 4, 2))
    n = g[:, :, 0] / np.linalg.norm(g[:, :, 0], axis=1, keepdims=True)
    m = g[:, :, 1] - np.sum(g[:, :, 1] * n, axis=1, keepdims=True) * n
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return n, m


def so4(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


# ------------------------------------------------------------------ Part D: exact contractions
def exact_T2_of_C(MF, C) -> Fraction:
    """T2(C) = sum_{abcd} C_ac C_bd <M_ab, M_cd> = sum_cd <N_cd, M_cd>, N_cd = sum_ab C_ac C_bd M_ab."""
    nz = [[(c, C[a][c]) for c in range(16) if C[a][c] != 0] for a in range(16)]
    total = Fraction(0)
    N = [[[[Fraction(0)] * 3 for _ in range(3)] for _ in range(16)] for _ in range(16)]
    for a in range(16):
        for b in range(16):
            Mab = MF[a][b]
            if all(x == 0 for r in Mab for x in r):
                continue
            for c, cac in nz[a]:
                for dd, cbd in nz[b]:
                    w = cac * cbd
                    Ncd = N[c][dd]
                    for i in range(3):
                        for j in range(3):
                            Ncd[i][j] += w * Mab[i][j]
    for c in range(16):
        for dd in range(16):
            total += sum(N[c][dd][i][j] * MF[c][dd][i][j] for i in range(3) for j in range(3))
    return total


def exact_mean_floor(MF, C) -> list[list[Fraction]]:
    S = [[Fraction(0)] * 3 for _ in range(3)]
    for a in range(16):
        for b in range(16):
            if C[a][b] != 0:
                for i in range(3):
                    for j in range(3):
                        S[i][j] += C[a][b] * MF[a][b][i][j]
    return S


# ------------------------------------------------------------------ lattices
def centering(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def generator(name: str, n: int) -> np.ndarray:
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))
    if name == "coh":
        nB = n // 2
        A = np.zeros((n, 2))
        A[:nB, 0] = 1.0
        A[nB:, 1] = 1.0
        return A
    raise ValueError(name)


def D_and_Sgen(A: np.ndarray) -> tuple[float, float, float]:
    n = A.shape[0]
    H = centering(n)
    B = A.T @ H @ A
    K = H @ A @ A.T @ H
    return float(np.sum(B * B)), float(np.sum(np.diag(B) ** 2)), float(np.sum(np.diag(K) ** 2))


def chain_S_exact(n: int) -> Fraction:
    return sum(Fraction(k * k) * (1 - Fraction(k, n)) ** 2 for k in range(1, n + 1))


def chain_D_exact(n: int) -> Fraction:
    return Fraction((n * n - 1) * (2 * n * n + 7), 180)


def cayley_E_Sgen(n: int) -> float:
    """E sum_u s_u^2 (1 - s_u/n)^2 over uniform rooted Cayley trees (Meir-Moon N_k)."""
    return sum(math.exp(n_k_log(n, k)) * k * k * (1 - k / n) ** 2 for k in range(1, n + 1))


def fit_slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ------------------------------------------------------------------ Part H: physical single configuration
def aligned(tetrad: np.ndarray) -> np.ndarray:
    return optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(tetrad)).aligned_candidate


def block_eps(tetrads: list[np.ndarray]) -> float:
    Y = sum(aligned(t) for t in tetrads)
    g = plebanski_gram(Y)
    t = g - np.trace(g) / 3 * np.eye(3)
    return float(np.linalg.norm(t) / np.linalg.norm(g))


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out: dict = {"card": "F-01", "question": "Q-0018", "seed": SEED, "delta": DELTA, "shear_scale": SHEAR_SCALE,
                 "declared": {"tol_sur": TOL_SUR, "tol_orbit": TOL_ORBIT, "tol_exact": TOL_EXACT, "tol_c1_mc": TOL_C1_MC,
                              "tol_recovery": TOL_RECOVERY,
                              "window_rule": f"max({WINDOW_FLOOR}, {WINDOW_SE_MULT}*CV/sqrt({PHYS_TRIALS_PLANNED}))",
                              "sur_trials": SUR_TRIALS, "orbit_draws": ORBIT_DRAWS, "c1_mc_samples": C1_MC_SAMPLES}}

    # ---- A
    MF, info = exact_M()
    M = M_numeric(MF)
    out["A_exact_M"] = info
    ok_A = info["T2"] == "60" and info["T4"] == "2" and info["normG0_sq"] == "12" and info["sum_a_M_aa_is_zero"]

    # ---- B
    C1 = exact_C1()
    C1n = np.array([[float(x) for x in row] for row in C1])
    n_, m_ = haar_pairs(rng, C1_MC_SAMPLES)
    xi = np.einsum("ti,tk->tik", n_, m_).reshape(C1_MC_SAMPLES, 16)
    C1_mc = xi.T @ xi / C1_MC_SAMPLES
    dev_C1 = float(np.max(np.abs(C1_mc - C1n)))
    trC1 = sum(C1[a][a] for a in range(16))
    eig = np.sort(np.linalg.eigvalsh(C1n))
    out["B_C1"] = {"trace": str(trC1), "mc_max_abs_dev": dev_C1, "eigenvalues_sorted": eig.tolist(),
                   "expected_eigs": "0 (x1, trace), 1/18 (x9, sym traceless), 1/12 (x6, antisymmetric)",
                   "entries": {"diag_ii": "3/72=1/24", "offdiag_ik": "5/72", "cross_ik_ki": "-1/72"}}
    ok_B = (trC1 == 1 and dev_C1 <= TOL_C1_MC and abs(eig[0]) < 1e-12
            and abs(eig[1] - 1 / 18) < 1e-12 and abs(eig[9] - 1 / 18) < 1e-12 and abs(eig[10] - 1 / 12) < 1e-12 and abs(eig[-1] - 1 / 12) < 1e-12)

    # ---- C
    a01 = 4 * 0 + 1
    Q4_1 = sum(x * x for r in MF[a01][a01] for x in r)
    orbit_dev = 0.0
    e0 = np.eye(4)[:, [0]]
    e1 = np.eye(4)[:, [1]]
    for _ in range(ORBIT_DRAWS):
        Lam = so4(rng)
        x = (Lam @ e0 @ e1.T @ Lam.T).reshape(16)
        phi = np.einsum("a,b,abij->ij", x, x, M)
        orbit_dev = max(orbit_dev, abs(float(np.sum(phi * phi)) - float(Q4_1)))
    Q4_stretch = sum(x * x for r in MF[0][0] for x in r)
    out["C_Q4"] = {"Q4_1_exact": str(Q4_1), "Q4_1_float": float(Q4_1), "orbit_max_abs_dev": orbit_dev,
                   "Q4_uniaxial_E00_exact": str(Q4_stretch),
                   "note": "Q4 = ||M(xi,xi)||^2 is SO(4)-conjugation invariant and SO(4) is transitive on orthonormal pairs => constant on the shear orbit; no Haar integral needed"}
    ok_C = orbit_dev <= TOL_ORBIT and Q4_1 == Fraction(1, 6) and Q4_stretch == 0

    # ---- D
    T2C1 = exact_T2_of_C(MF, C1)
    floor = exact_mean_floor(MF, C1)
    T2C1_num = float(np.einsum("ac,bd,abij,cdij->", C1n, C1n, M, M))
    s4 = Fraction(int(SHEAR_SCALE ** 4))
    c2 = 2 * s4 * T2C1 / 12
    c4 = s4 * (Q4_1 - 2 * T2C1) / 12
    c_delta = c2 + c4
    ratio = c_delta / 10
    T2_gauss = Fraction(60, 256)
    out["D_constants"] = {
        "T2_C1_exact": str(T2C1), "T2_C1_float": float(T2C1), "T2_C1_numeric_dev": abs(T2C1_num - float(T2C1)),
        "T2_gauss_I16_over_256": str(T2_gauss), "T2_C1_over_T2_gauss": str(T2C1 / T2_gauss),
        "mean_floor_exact": [[str(v) for v in r] for r in floor], "mean_floor_is_zero": all(v == 0 for r in floor for v in r),
        "c2_exact": str(c2), "c2_float": float(c2), "c4_exact": str(c4), "c4_float": float(c4),
        "c_delta_exact": str(c_delta), "c_delta_float": float(c_delta),
        "ratio_to_gauss_exact": str(ratio), "ratio_to_gauss_float": float(ratio),
        "c2_over_10": str(c2 / 10),
    }
    ok_D = abs(T2C1_num - float(T2C1)) <= TOL_EXACT and all(v == 0 for r in floor for v in r) and c_delta == Fraction(32, 9)

    # ---- E: pre-registered numbers
    pre: dict = {}
    c2f, c4f = float(c2), float(c4)
    for n in GRID_N:
        for name in KERNELS:
            A = generator(name, n)
            D, S, Sker = D_and_Sgen(A)
            if name == "iid":
                exact_S, exact_D = Fraction((n - 1) ** 2, n), Fraction(n - 1)
            elif name == "chain":
                exact_S, exact_D = chain_S_exact(n), chain_D_exact(n)
            else:
                p = Fraction(n // 2, n)
                exact_S, exact_D = 2 * n * n * p * p * (1 - p) ** 2, 4 * n * n * p * p * (1 - p) ** 2
            assert abs(D - float(exact_D)) < 1e-9 and abs(S - float(exact_S)) < 1e-9
            cn = c2 + c4 * exact_S / exact_D
            cn_ker = c2f + c4f * Sker / float(exact_D)
            pre[f"{name}_n{n}"] = {"D": str(exact_D), "S_gen": str(exact_S), "S_gen_over_D": float(exact_S / exact_D),
                                   "c_pred_exact": str(cn), "c_pred": float(cn),
                                   "c_alt_kernel_diag": cn_ker, "S_ker": Sker,
                                   "c_gauss_F02": 10.0}
    S16, D16 = chain_S_exact(16), chain_D_exact(16)
    R_str = (Fraction(15, 16) - S16 / D16) / (Fraction(15, 16) - Fraction(1, 2))
    Sker16 = pre["chain_n16"]["S_ker"]
    R_str_ker = (15 / 16 - Sker16 / float(D16)) / (15 / 16 - 1 / 2)
    pre["R_str"] = {"exact": str(R_str), "float": float(R_str), "alt_kernel_diag": R_str_ker,
                    "definition": "(c_iid(16)-c_chain(16))/(c_iid(16)-c_iid(2)) = (15/16 - S_ch/D_ch)/(15/16 - 1/2)"}
    rms_iid = [math.sqrt((c2f + c4f * (n - 1) / n) * (n - 1)) / n for n in SLOPE_GRID]
    rms_iid_gauss = [math.sqrt(10 * (n - 1)) / n for n in SLOPE_GRID]
    pre["gamma_iid_det"] = fit_slope(SLOPE_GRID, rms_iid)
    pre["gamma_iid_gauss_F02"] = fit_slope(SLOPE_GRID, rms_iid_gauss)
    cay = {}
    for n in SLOPE_GRID:
        cay[n] = (cayley_exact(n)["E_D"], cayley_E_Sgen(n))
    rms_her = [math.sqrt(c2f * cay[n][0] + c4f * cay[n][1]) / n for n in SLOPE_GRID]
    rms_her_gauss = [math.sqrt(10 * cay[n][0]) / n for n in SLOPE_GRID]
    pre["cayley"] = {str(n): {"E_D": cay[n][0], "E_S_gen": cay[n][1], "S_over_D": cay[n][1] / cay[n][0]} for n in SLOPE_GRID}
    pre["gamma_her_det"] = fit_slope(SLOPE_GRID, rms_her)
    pre["gamma_her_gauss_F02"] = fit_slope(SLOPE_GRID, rms_her_gauss)
    pre["her_over_iid_128_det"] = rms_her[-1] / rms_iid[-1]
    pre["her_over_iid_128_gauss_F02"] = rms_her_gauss[-1] / rms_iid_gauss[-1]
    pre["her_over_iid_128_shift_pct"] = 100 * (pre["her_over_iid_128_det"] / pre["her_over_iid_128_gauss_F02"] - 1)
    pre["two_species_fixed_Delta_coeff"] = {"eps2_over_delta4_p2q2_exact": str(s4 * Q4_1 / 12), "equals_c_delta": s4 * Q4_1 / 12 == c_delta,
                                           "p_half_n_any_eps_over_delta2": math.sqrt(float(c_delta) / 16)}
    out["E_preregistered"] = pre

    # ---- F: Q-0012 naive per-component kurtosis reading (cannot reach c_delta)
    n_, m_ = haar_pairs(rng, C1_MC_SAMPLES)
    off = SHEAR_SCALE * n_[:, 0] * m_[:, 1]
    dia = SHEAR_SCALE * n_[:, 0] * m_[:, 0]
    k4_off = float(np.mean(off ** 4) / np.mean(off ** 2) ** 2 - 3)
    k4_dia = float(np.mean(dia ** 4) / np.mean(dia ** 2) ** 2 - 3)
    k4_needed = 60 * (float(ratio) - 1)
    out["F_q0012_naive"] = {"kurt_offdiag_component_mc": k4_off, "kurt_diag_component_mc": k4_dia,
                            "q0012_c_iid_inf_if_k4_offdiag": 10 * (1 + k4_off / 60),
                            "q0012_lower_bound_pearson": 10 * (1 - 2 / 60),
                            "k4_needed_to_reach_c_delta": k4_needed, "pearson_bound": -2.0,
                            "conclusion": "shear branch lies outside the 16-iid-component class of Q-0012: the required kurtosis is below the Pearson bound"}

    # ---- G: tetrad-free surrogate MC of the fourth-moment law + CV for windows
    sur = {"trials": SUR_TRIALS, "tol": TOL_SUR, "cases": {}}
    ok_G = True
    for n in GRID_N:
        H = centering(n)
        for name in KERNELS:
            A = generator(name, n)
            D, S, _ = D_and_Sgen(A)
            law = float(s4) * (2 * float(T2C1) * D + (float(Q4_1) - 2 * float(T2C1)) * S)
            HA = H @ A
            m_inc = A.shape[1]
            vals = np.empty(SUR_TRIALS)
            batch = 5000
            for s0 in range(0, SUR_TRIALS, batch):
                nn, mm = haar_pairs(rng, batch * m_inc)
                z = (SHEAR_SCALE * np.einsum("ti,tk->tik", nn, mm)).reshape(batch, m_inc, 16)
                xt = np.einsum("vu,tua->tva", HA, z)
                phi = np.einsum("tva,tvb,abij->tij", xt, xt, M)
                vals[s0:s0 + batch] = np.einsum("tij,tij->t", phi, phi)
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / math.sqrt(SUR_TRIALS))
            cv = float(vals.std(ddof=1) / mean)
            rel = mean / law - 1
            w = max(WINDOW_FLOOR, WINDOW_SE_MULT * cv / math.sqrt(PHYS_TRIALS_PLANNED))
            c_pred = float(pre[f"{name}_n{n}"]["c_pred"])
            sur["cases"][f"{name}_n{n}"] = {"n": n, "kernel": name, "D": D, "S_gen": S, "law": law, "mc_mean": mean, "mc_se": se,
                                          "rel_err": rel, "z": rel * law / se, "cv": cv,
                                          "c_mc": mean / (12 * D), "c_pred": c_pred,
                                          "window_halfwidth_rel": w, "window": [c_pred * (1 - w), c_pred * (1 + w)]}
            ok_G = ok_G and abs(rel) <= TOL_SUR
    sur["max_abs_rel_err"] = max(abs(c["rel_err"]) for c in sur["cases"].values())
    sur["pass"] = bool(ok_G)
    out["G_surrogate"] = sur

    # ---- H: physical single configuration (no MC)
    E01 = np.zeros((4, 4))
    E01[0, 1] = 1.0
    rows = []
    for dl in RECOVERY_DELTAS:
        eps = block_eps([np.eye(4), np.eye(4) + SHEAR_SCALE * dl * E01])
        Lam = so4(rng)
        eps_rot = block_eps([np.eye(4), np.eye(4) + SHEAR_SCALE * dl * (Lam @ E01 @ Lam.T)])
        rows.append({"delta": dl, "eps": eps, "eps_over_delta2": eps / dl ** 2, "eps_rot_over_delta2": eps_rot / dl ** 2,
                     "det_tetrad": float(np.linalg.det(np.eye(4) + SHEAR_SCALE * dl * E01))})
    target = math.sqrt(2) / 3
    out["H_physical_recovery"] = {"target_sqrt2_over_3": target, "rows": rows,
                                  "rel_dev_at_delta0005": rows[-1]["eps_over_delta2"] / target - 1,
                                  "rel_dev_rot_at_delta0005": rows[-1]["eps_rot_over_delta2"] / target - 1,
                                  "note": "two cells, p=1/2: eps^2 -> (32/9) delta^4 /16 = (2/9) delta^4; deviation is the O(delta) truncation of the exact identity"}
    ok_H = (abs(rows[-1]["eps_over_delta2"] / target - 1) <= TOL_RECOVERY
            and abs(rows[-1]["eps_rot_over_delta2"] / target - 1) <= TOL_RECOVERY)

    out["pass"] = {"A": bool(ok_A), "B": bool(ok_B), "C": bool(ok_C), "D": bool(ok_D), "G": bool(ok_G), "H": bool(ok_H)}
    out["all_pass"] = all(out["pass"].values())
    out["wall_seconds"] = time.time() - t0
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("A_exact_M", "B_C1", "C_Q4", "D_constants", "pass", "all_pass", "wall_seconds")}, ensure_ascii=False, indent=1))
    short = {}
    for k, v in pre.items():
        if k == "cayley":
            continue
        short[k] = v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("c_pred", "c_pred_exact", "S_gen_over_D", "c_alt_kernel_diag", "float", "exact", "alt_kernel_diag", "eps2_over_delta4_p2q2_exact", "p_half_n_any_eps_over_delta2")}
    print("preregistered:", json.dumps(short, ensure_ascii=False, indent=1))
    print("cayley:", json.dumps(pre["cayley"], indent=1))
    print("surrogate max|rel|:", sur["max_abs_rel_err"])
    for k, v in sur["cases"].items():
        print(f"  {k:10s} c_mc={v['c_mc']:.4f} c_pred={v['c_pred']:.4f} rel={v['rel_err']:+.4f} z={v['z']:+.2f} cv={v['cv']:.3f} w={v['window_halfwidth_rel']:.4f}")
    print("physical recovery:", json.dumps(out["H_physical_recovery"], indent=1))
    print("q0012 naive:", json.dumps(out["F_q0012_naive"], indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
