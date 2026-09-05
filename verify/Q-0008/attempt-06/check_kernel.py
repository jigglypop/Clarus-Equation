"""Q-0008 attempt-06 -- ladder step 3 (kernel law) machine checks.

    eps_bar^2 = eps_star^2 ||H kappa H||_F^2 / n^2,   eps_star^2 = 2 T2 delta^4 / ||G0||^2 = 10 delta^4.

Every tolerance, grid, trial count and seed below is DECLARED BEFORE RUNNING and is not edited
afterwards.  Seed 20260902 (repository convention).  The design pilot that fixed the trial counts
ran at seed 1 and is not part of the evidence.

Parts
  A  structure constants T2 = 60, T4 = 2, sum_a M_aa = 0, ||G0||^2 = 12 and the multiplicity table
     (numeric Richardson map of Q-0012 check_cumulant; exact rationals cited from
     verify/Q-0012/F-01/adversary/a1_exact_constants.json)
  B  isotropy lemma, all orders: tl gram Y(L xi L^T) = R0 tl gram Y(xi) R0^T for L in SO(4),
     R0 = G(rho^{-1} Sigma_0, Sigma_0) G0^{-1}  (delta = 0.2, n = 3 cells, 20 draws)
  C  Wick law E||tl gram Y||^2 = n^2 delta^4 * 2 T2 * ||H kappa H||^2:
     C-a tetrad-free quadratic-form surrogate, 100000 trials, tol 2%
     C-b physical polar-aligned block, 2000 trials, tol 5% (task specification), z-score reported
     C-c common-random-number ratio physical/surrogate, |mean - 1| tol 1%
  D  closed forms (iid, two species, chain, star; n <= 11), cross term tr(H kappa) = sum s_u (1 - s_u/n),
     general identity ||H kappa H||^2 = tr kappa^2 - (2/n) 1^T kappa^2 1 + (1^T kappa 1)^2 / n^2
  E  gamma_her scope table (tree only, report): D/(n^2 depth^2) for chain / caterpillar / Cayley /
     star-of-chains / balanced binary / star
  F  delta three-point test (report only, not a verdict): delta in {0.02, 0.01, 0.005}, n in {8, 32},
     512 common-random-number trials, heritable Cayley and iid labels; fit RMS/delta^2 = r0 (1 + c delta^2).

Usage: python verify/Q-0008/attempt-06/check_kernel.py [--quick]
Writes verify/Q-0008/attempt-06/result.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))

from check_cumulant import geometry_constants, gram_form, linear_map, quadratic_tensor, tl  # noqa: E402
from driver_numbers import cayley_exact, tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    _PAIR_INDEX,
    geometric_self_dual_triple,
    plebanski_gram,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

# ------------------------------------------------------------------ declared constants (frozen)
SEED = 20260902
DELTA = 0.005                     # card convention (all stochastic modes)
MIN_DET = 0.05                    # card convention: resample configuration if any det(I + delta label) <= MIN_DET

TOL_CONST = 1.0e-6                # A: |T2/60 - 1|, |T4/2 - 1|, |normG0^2/12 - 1|
TOL_ISO = 1.0e-9                  # A: max |sum_a M_aa|
TOL_MULT = 1.0e-8                 # A: rounding for the multiplicity table
EXPECT_MULT = {"0": 52, "1/8": 96, "1/6": 24, "1/2": 72, "2/3": 12}

EQUIV_DELTA = 0.2                 # B
EQUIV_N = 3
EQUIV_TRIALS = 20
EQUIV_MIN_DET = 0.5               # B: keeps the positive Urbantke branch at delta = 0.2 (declared)
TOL_EQUIV = 1.0e-10               # B: ||gram Y' - R0 gram Y R0^T|| / ||gram Y||

GRID_N = (2, 4, 8, 16)            # C
KERNELS = ("iid", "chain", "coh")
SUR_TRIALS = 100_000              # C-a
TOL_SUR = 0.02
PHYS_TRIALS = 2000                # C-b (task specification)
TOL_PHYS = 0.05
TOL_CRN = 0.01                    # C-c

CLOSED_N_MAX = 11                 # D
TOL_CLOSED = 1.0e-9               # D: |direct - closed| / (1 + |closed|)

D3_DELTAS = (0.02, 0.01, 0.005)   # F (report only)
D3_SIZES = (8, 32)
D3_TRIALS = 512
D3_MODES = ("her_cayley", "iid")
D3_LABEL_NOISE = 0.005            # F: |tau| < 0.5%  -> "잡음" (not truncation); |tau| >= 1% and tau < 0 -> "O(δ⁴)"; else "미결"
D3_LABEL_TRUNC = 0.01

REFERENCE = geometric_self_dual_triple(np.eye(4))
G0 = plebanski_gram(REFERENCE)
T2_EXACT, T4_EXACT, NORM_G0_SQ_EXACT = 60.0, 2.0, 12.0


# ------------------------------------------------------------------ helpers
def centering(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def D_of(kappa: np.ndarray) -> float:
    n = kappa.shape[0]
    K = centering(n) @ kappa @ centering(n)
    return float(np.sum(K * K))


def generator(name: str, n: int) -> np.ndarray:
    """xi = A zeta with zeta iid N(0,1)^{16} per column; kappa = A A^T."""
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))                       # kappa_vw = min(v, w)
    if name == "coh":
        nB = n // 2
        A = np.zeros((n, 2))
        A[:nB, 0] = 1.0
        A[nB:, 1] = 1.0
        return A
    raise ValueError(name)


def anc_matrix(parent: list[int]) -> np.ndarray:
    n = len(parent)
    A = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            A[v, u] = 1.0
            u = parent[u]
    return A


def chain_parent(n: int) -> list[int]:
    return [-1] + list(range(n - 1))


def star_parent(n: int) -> list[int]:
    return [-1] + [0] * (n - 1)


def binary_parent(n: int) -> list[int]:
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


def caterpillar(k: int) -> list[int]:
    """spine k, each spine vertex carries k-1 leaves; n = k^2, depth = k-1 (adversary b2 definition)."""
    parent = [-1]
    spine = [0]
    for _ in range(1, k):
        parent.append(spine[-1])
        spine.append(len(parent) - 1)
    for v in spine:
        parent.extend([v] * (k - 1))
    return parent


def star_of_chains(k: int) -> list[int]:
    """root + k chains of length k; n = k^2 + 1, depth = k (adversary b2 definition)."""
    parent = [-1]
    for _ in range(k):
        prev = 0
        for _ in range(k):
            parent.append(prev)
            prev = len(parent) - 1
    return parent


def depth_of(parent: list[int]) -> int:
    _, depth, _, _ = tree_arrays(parent)
    return int(depth.max())


def heritable_labels(parent: list[int], xi: np.ndarray) -> np.ndarray:
    order, _, _, _ = tree_arrays(parent)
    labels = np.zeros_like(xi)
    for v in order:
        p = parent[v]
        labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
    return labels


def block_sum(labels: np.ndarray, delta: float) -> np.ndarray | None:
    """Y = sum_v polar-aligned Sigma(I + delta label_v); None if the MIN_DET rule rejects the draw."""
    Y = np.zeros_like(REFERENCE)
    for lab in labels:
        tetrad = np.eye(4) + delta * lab
        if float(np.linalg.det(tetrad)) <= MIN_DET:
            return None
        Y += optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(tetrad)).aligned_candidate
    return Y


def tl_gram_and_gram_norms(Y: np.ndarray) -> tuple[float, float, np.ndarray]:
    g = plebanski_gram(Y)
    t = tl(g)
    return float(np.sum(t * t)), float(np.sum(g * g)), t


def surrogate_phi(xt: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Phi = sum_v sum_ab xt_v^a xt_v^b M_ab for one configuration xt (n x 16)."""
    return np.einsum("va,vb,abij->ij", xt, xt, M)


def so4(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def mat6(v: np.ndarray) -> np.ndarray:
    m = np.zeros((4, 4))
    for comp, (a, b) in zip(v, _PAIR_INDEX):
        m[a, b] = comp
        m[b, a] = -comp
    return m


def vec6(m: np.ndarray) -> np.ndarray:
    return np.array([m[a, b] for (a, b) in _PAIR_INDEX])


def rho6(L: np.ndarray, triple: np.ndarray) -> np.ndarray:
    """component action of L in SO(4) on each 2-form row of the triple."""
    return np.array([vec6(L @ mat6(row) @ L.T) for row in triple])


def fit_slope(xs, ys) -> float:
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


# ------------------------------------------------------------------ Part A
def part_a() -> tuple[dict, np.ndarray]:
    M = quadratic_tensor(linear_map())
    gc = geometry_constants(M)
    K = np.einsum("abij,abij->ab", M, M)
    cnt = Counter(round(float(x) / TOL_MULT) * TOL_MULT for x in K.ravel())
    names = {0.0: "0", 0.125: "1/8", 1 / 6: "1/6", 0.5: "1/2", 2 / 3: "2/3"}
    mult = {}
    for val, c in cnt.items():
        key = next((nm for tv, nm in names.items() if abs(val - tv) < 10 * TOL_MULT), f"{val:.6f}")
        mult[key] = mult.get(key, 0) + c
    exact = json.loads((ROOT / "verify/Q-0012/F-01/adversary/a1_exact_constants.json").read_text(encoding="utf-8"))
    out = {
        "T2": gc["T2"], "T4": gc["T4"], "sum_a_Maa_max_abs": gc["isotropy_max_abs_sum_a_Maa"],
        "normG0_sq": float(np.sum(G0 * G0)), "G0": G0.tolist(),
        "multiplicities": mult, "expected_multiplicities": EXPECT_MULT,
        "exact_cited": {"T2": exact["T2"], "T4": exact["T4"], "normG0_sq": exact["normG0_sq"],
                        "sum_a_M_aa": exact["sum_a_M_aa"], "source": "verify/Q-0012/F-01/adversary/a1_exact_constants.json"},
        "eps_star_sq_over_delta4": 2 * gc["T2"] / float(np.sum(G0 * G0)),
    }
    out["pass"] = bool(
        abs(gc["T2"] / T2_EXACT - 1) <= TOL_CONST
        and abs(gc["T4"] / T4_EXACT - 1) <= TOL_CONST
        and abs(out["normG0_sq"] / NORM_G0_SQ_EXACT - 1) <= TOL_CONST
        and gc["isotropy_max_abs_sum_a_Maa"] <= TOL_ISO
        and mult == EXPECT_MULT
    )
    return out, M


# ------------------------------------------------------------------ Part B
def part_b(rng: np.random.Generator) -> dict:
    rows = []
    worst = 0.0
    for t in range(EQUIV_TRIALS):
        L = so4(rng)
        while True:
            xi = rng.normal(size=(EQUIV_N, 4, 4))
            if all(np.linalg.det(np.eye(4) + EQUIV_DELTA * x) > EQUIV_MIN_DET for x in xi):
                break
        Y = block_sum(xi, EQUIV_DELTA)
        Yc = block_sum(np.array([L @ x @ L.T for x in xi]), EQUIV_DELTA)
        S0inv = rho6(L.T, REFERENCE)                    # rho^{-1} Sigma_0
        R0 = gram_form(S0inv, REFERENCE) @ np.linalg.inv(G0)
        gY, gYc = plebanski_gram(Y), plebanski_gram(Yc)
        err_gram = float(np.linalg.norm(gYc - R0 @ gY @ R0.T) / np.linalg.norm(gY))
        err_tl = float(np.linalg.norm(tl(gYc) - R0 @ tl(gY) @ R0.T) / np.linalg.norm(gY))
        rows.append({
            "R0_orthogonality": float(np.linalg.norm(R0 @ R0.T - np.eye(3))),
            "R0_det": float(np.linalg.det(R0)),
            "rho_inv_Sigma0_eq_R0_Sigma0": float(np.linalg.norm(S0inv - R0 @ REFERENCE)),
            "err_gram": err_gram, "err_tl_gram": err_tl,
            "tl_over_gram": float(np.linalg.norm(tl(gY)) / np.linalg.norm(gY)),
        })
        worst = max(worst, err_gram, err_tl, rows[-1]["rho_inv_Sigma0_eq_R0_Sigma0"], rows[-1]["R0_orthogonality"],
                    abs(rows[-1]["R0_det"] - 1.0))
    return {"delta": EQUIV_DELTA, "n_cells": EQUIV_N, "trials": EQUIV_TRIALS, "worst": worst,
            "rows": rows, "pass": bool(worst <= TOL_EQUIV)}


# ------------------------------------------------------------------ Part C
def part_c_surrogate(rng: np.random.Generator, M: np.ndarray) -> dict:
    out = {"trials": SUR_TRIALS, "tol": TOL_SUR, "cases": {}}
    ok = True
    for n in GRID_N:
        H = centering(n)
        for name in KERNELS:
            A = generator(name, n)
            D = D_of(A @ A.T)
            wick = 2 * T2_EXACT * D
            HA = H @ A
            vals = np.empty(SUR_TRIALS)
            batch = 5000
            for s in range(0, SUR_TRIALS, batch):
                z = rng.normal(size=(batch, A.shape[1], 16))
                xt = np.einsum("vu,tua->tva", HA, z)
                phi = np.einsum("tva,tvb,abij->tij", xt, xt, M)
                vals[s:s + batch] = np.einsum("tij,tij->t", phi, phi)
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / math.sqrt(SUR_TRIALS))
            rel = mean / wick - 1.0
            case = {"n": n, "kernel": name, "D": D, "wick_2T2D": wick, "mc_mean": mean, "mc_se": se,
                    "rel_err": rel, "z": rel * wick / se if se > 0 else 0.0, "cv": float(vals.std(ddof=1) / mean)}
            out["cases"][f"{name}_n{n}"] = case
            ok = ok and abs(rel) <= TOL_SUR
    out["max_abs_rel_err"] = max(abs(c["rel_err"]) for c in out["cases"].values())
    out["pass"] = bool(ok)
    return out


def part_c_physical(rng: np.random.Generator, M: np.ndarray, trials: int) -> dict:
    out = {"trials": trials, "delta": DELTA, "tol_phys": TOL_PHYS, "tol_crn": TOL_CRN, "cases": {}}
    ok_phys = ok_crn = True
    rejections = 0
    for n in GRID_N:
        H = centering(n)
        for name in KERNELS:
            A = generator(name, n)
            D = D_of(A @ A.T)
            wick_num = n * n * DELTA**4 * 2 * T2_EXACT * D
            law_eps2 = 10.0 * DELTA**4 * D / (n * n)
            N_phys, N_sur, E2, TL = [], [], [], []
            while len(N_phys) < trials:
                z = rng.normal(size=(A.shape[1], 4, 4))
                labels = np.einsum("vu,uab->vab", A, z)
                Y = block_sum(labels, DELTA)
                if Y is None:
                    rejections += 1
                    continue
                num, den, t = tl_gram_and_gram_norms(Y)
                xt = H @ labels.reshape(n, 16)
                phi = surrogate_phi(xt, M)
                N_phys.append(num)
                N_sur.append(n * n * DELTA**4 * float(np.sum(phi * phi)))
                E2.append(num / den)
                TL.append(t)
            N_phys, N_sur, E2 = np.array(N_phys), np.array(N_sur), np.array(E2)
            ratio = N_phys / N_sur
            mean = float(N_phys.mean())
            se = float(N_phys.std(ddof=1) / math.sqrt(trials))
            rel = mean / wick_num - 1.0
            crn = float(ratio.mean())
            crn_se = float(ratio.std(ddof=1) / math.sqrt(trials))
            mean_tl = np.mean(np.array(TL), axis=0)
            case = {
                "n": n, "kernel": name, "D": D,
                "wick_E_tlgram_sq": wick_num, "mc_E_tlgram_sq": mean, "mc_se": se, "rel_err": rel,
                "z": rel * wick_num / se if se > 0 else 0.0,
                "law_E_eps2": law_eps2, "mc_E_eps2": float(E2.mean()), "rel_err_eps2": float(E2.mean() / law_eps2 - 1.0),
                "crn_mean_phys_over_sur": crn, "crn_se": crn_se, "crn_sd_per_trial": float(ratio.std(ddof=1)),
                "ratio_of_means_phys_over_sur": float(mean / N_sur.mean()),
                "mean_tlgram_over_rms": float(np.linalg.norm(mean_tl) / math.sqrt(mean)),
            }
            out["cases"][f"{name}_n{n}"] = case
            ok_phys = ok_phys and abs(rel) <= TOL_PHYS
            ok_crn = ok_crn and abs(crn - 1.0) <= TOL_CRN
    out["rejections"] = rejections
    out["max_abs_rel_err"] = max(abs(c["rel_err"]) for c in out["cases"].values())
    out["max_abs_z"] = max(abs(c["z"]) for c in out["cases"].values())
    out["max_abs_crn_minus_1"] = max(abs(c["crn_mean_phys_over_sur"] - 1.0) for c in out["cases"].values())
    out["pass_phys_5pct"] = bool(ok_phys)
    out["pass_crn_1pct"] = bool(ok_crn)
    out["pass"] = bool(ok_phys and ok_crn)
    return out


# ------------------------------------------------------------------ Part D
def part_d(rng: np.random.Generator) -> dict:
    worst = 0.0
    rows = {"iid": [], "coh": [], "chain": [], "star": []}
    for n in range(1, CLOSED_N_MAX + 1):
        d_iid = D_of(np.eye(n))
        rows["iid"].append((n, d_iid, n - 1))
        worst = max(worst, abs(d_iid - (n - 1)) / (1 + abs(n - 1)))
        for nB in range(0, n + 1):
            k = np.zeros((n, n))
            sB = np.arange(n) < nB
            k[np.ix_(sB, sB)] = 1.0
            k[np.ix_(~sB, ~sB)] = 1.0
            p = nB / n
            closed = 4 * n**2 * p**2 * (1 - p) ** 2
            d = D_of(k)
            rows["coh"].append((n, nB, d, closed))
            worst = max(worst, abs(d - closed) / (1 + abs(closed)))
        kc = anc_matrix(chain_parent(n)) @ anc_matrix(chain_parent(n)).T
        closed = (n**2 - 1) * (2 * n**2 + 7) / 180
        d = D_of(kc)
        rows["chain"].append((n, d, closed))
        worst = max(worst, abs(d - closed) / (1 + abs(closed)))
        ks = anc_matrix(star_parent(n)) @ anc_matrix(star_parent(n)).T
        closed = n - 2 + 1 / n**2
        d = D_of(ks)
        rows["star"].append((n, d, closed))
        worst = max(worst, abs(d - closed) / (1 + abs(closed)))
    # cross term tr(H kappa) = sum_u s_u (1 - s_u/n): chain, star, random Cayley
    cross = []
    for n in range(2, CLOSED_N_MAX + 1):
        trees = [chain_parent(n), star_parent(n)] + [uniform_rooted_tree(n, rng) for _ in range(3)]
        for parent in trees:
            A = anc_matrix(parent)
            _, _, sub, _ = tree_arrays(parent)
            s = sub.astype(float)
            direct = float(np.trace(centering(n) @ A @ A.T))
            closed = float(np.sum(s * (1 - s / n)))
            cross.append((n, direct, closed))
            worst = max(worst, abs(direct - closed) / (1 + abs(closed)))
    # general identity for random symmetric kappa
    gen = []
    for _ in range(10):
        n = 5
        B = rng.normal(size=(n, n))
        kappa = B + B.T
        one = np.ones(n)
        direct = D_of(kappa)
        closed = float(np.trace(kappa @ kappa) - 2 / n * one @ kappa @ kappa @ one + (one @ kappa @ one) ** 2 / n**2)
        gen.append((direct, closed))
        worst = max(worst, abs(direct - closed) / (1 + abs(closed)))
    return {"n_max": CLOSED_N_MAX, "worst_rel_err": worst, "pass": bool(worst <= TOL_CLOSED),
            "chain": rows["chain"], "star": rows["star"], "iid": rows["iid"],
            "coh_count": len(rows["coh"]), "cross_term_count": len(cross), "general_identity_count": len(gen)}


# ------------------------------------------------------------------ Part E
def part_e(rng: np.random.Generator) -> dict:
    def d_fast(parent):
        n = len(parent)
        _, depth, sub, prefix = tree_arrays(parent)
        s = sub.astype(float)
        w2 = float(np.sum(s * s))
        w2p = float(np.sum((2.0 * depth + 1.0) * s * s))
        s_row = float(np.sum(prefix.astype(float) ** 2))
        return w2p - 2.0 * s_row / n + w2 * w2 / (n * n), int(depth.max())

    out = {}
    out["chain"] = [{"n": n, "D_over_n2depth2": d_fast(chain_parent(n))[0] / (n * n * (n - 1) ** 2)} for n in (16, 64, 256, 1024)]
    out["chain_limit_1_over_90"] = 1 / 90
    out["caterpillar"] = []
    for k in (8, 16, 32, 64, 128):
        p = caterpillar(k)
        d, dep = d_fast(p)
        out["caterpillar"].append({"k": k, "n": len(p), "depth": dep, "D_over_n2depth2": d / (len(p) ** 2 * dep**2)})
    out["cayley_exact"] = []
    for n in (8, 16, 32, 64, 128, 256, 512, 1024):
        ED = cayley_exact(n)["E_D"]
        out["cayley_exact"].append({"n": n, "E_D": ED, "E_D_over_n3": ED / n**3,
                                    "note": "depth ~ n^{1/2} (Aldous), so D/(n^2 depth^2) ~ E_D/n^3"})
    out["star_of_chains"] = []
    for k in (8, 16, 32, 64):
        p = star_of_chains(k)
        d, dep = d_fast(p)
        out["star_of_chains"].append({"k": k, "n": len(p), "depth": dep,
                                      "D_over_n2depth2": d / (len(p) ** 2 * dep**2),
                                      "D_over_n2depth": d / (len(p) ** 2 * dep), "b2_asymptote_1_over_6": 1 / 6})
    out["balanced_binary"] = []
    for k in (4, 6, 8, 10):
        n = 2**k - 1
        p = binary_parent(n)
        d, dep = d_fast(p)
        out["balanced_binary"].append({"n": n, "depth": dep, "D_over_n2depth2": d / (n * n * dep**2), "D_over_n2": d / (n * n)})
    out["star"] = [{"n": n, "D_over_n2": (n - 2 + 1 / n**2) / (n * n)} for n in (8, 64, 512)]
    gammas = {}
    grid = (8, 16, 32, 64, 128)
    for name, fn in (("chain", chain_parent), ("star", star_parent), ("balanced_binary", binary_parent)):
        gammas[name] = fit_slope(np.log(grid), [math.log(math.sqrt(d_fast(fn(n))[0]) / n) for n in grid])
    gammas["cayley_exact_grid"] = fit_slope(np.log(grid), [math.log(math.sqrt(cayley_exact(n)["E_D"]) / n) for n in grid])
    ks = (8, 16, 32, 64)
    gammas["star_of_chains"] = fit_slope([math.log(k * k + 1) for k in ks],
                                         [math.log(math.sqrt(d_fast(star_of_chains(k))[0]) / (k * k + 1)) for k in ks])
    gammas["caterpillar"] = fit_slope([math.log(k * k) for k in ks],
                                      [math.log(math.sqrt(d_fast(caterpillar(k))[0]) / (k * k)) for k in ks])
    out["gamma_grid"] = gammas
    return out


# ------------------------------------------------------------------ Part F
def part_f(rng: np.random.Generator, trials: int) -> dict:
    out = {"deltas": list(D3_DELTAS), "sizes": list(D3_SIZES), "trials": trials, "modes": list(D3_MODES),
           "label_rule": {"noise_if_abs_tau_below": D3_LABEL_NOISE, "trunc_if_tau_below_minus": D3_LABEL_TRUNC}, "cases": {}}
    taus = []
    for mode in D3_MODES:
        for n in D3_SIZES:
            eps = {d: [] for d in D3_DELTAS}
            while len(eps[D3_DELTAS[0]]) < trials:
                if mode == "her_cayley":
                    parent = uniform_rooted_tree(n, rng)
                    labels = heritable_labels(parent, rng.normal(size=(n, 4, 4)))
                else:
                    labels = rng.normal(size=(n, 4, 4))
                vals = {}
                ok = True
                for d in D3_DELTAS:
                    Y = block_sum(labels, d)
                    if Y is None:
                        ok = False
                        break
                    num, den, _ = tl_gram_and_gram_norms(Y)
                    vals[d] = math.sqrt(num / den)                    # eps = ||tl gram Y|| / ||gram Y||
                if not ok:
                    continue
                for d in D3_DELTAS:
                    eps[d].append(vals[d])
            rms = {d: math.sqrt(float(np.mean(np.square(eps[d])))) for d in D3_DELTAS}
            r = {d: rms[d] / d**2 for d in D3_DELTAS}
            r_ref = r[0.005]
            x = np.array([d * d for d in D3_DELTAS])
            y = np.array([math.log(r[d]) for d in D3_DELTAS])
            c_sq = fit_slope(x, y)                                   # ln r = ln r0 + c delta^2
            resid_sq = float(np.max(np.abs(y - np.polyval(np.polyfit(x, y, 1), x))))
            xl = np.array(D3_DELTAS)
            c_lin = fit_slope(xl, y)                                 # alternative ln r = a + b delta
            resid_lin = float(np.max(np.abs(y - np.polyval(np.polyfit(xl, y, 1), xl))))
            tau = c_sq * 0.005**2
            # per-trial ratio sd: the O(delta) odd term (mean zero) -- shows why the mean is the right statistic
            per = np.array(eps[0.01]) / np.array(eps[0.005])
            # bootstrap SE (report only; B = 2000, separate rng so the main stream is untouched)
            brng = np.random.default_rng(SEED)
            E = np.array([eps[d] for d in D3_DELTAS])          # 3 x trials
            boot_ratio = {d: [] for d in D3_DELTAS}
            boot_tau = []
            for _ in range(2000):
                idx = brng.integers(0, trials, size=trials)
                rb = {d: math.sqrt(float(np.mean(E[i, idx] ** 2))) / d**2 for i, d in enumerate(D3_DELTAS)}
                for d in D3_DELTAS:
                    boot_ratio[d].append(rb[d] / rb[0.005] - 1.0)
                yb = np.array([math.log(rb[d]) for d in D3_DELTAS])
                boot_tau.append(fit_slope(x, yb) * 0.005**2)
            case = {"mode": mode, "n": n, "rms": {str(d): rms[d] for d in D3_DELTAS},
                    "rms_over_delta2": {str(d): r[d] for d in D3_DELTAS},
                    "ratio_to_0.005": {str(d): r[d] / r_ref - 1.0 for d in D3_DELTAS},
                    "ratio_to_0.005_boot_se": {str(d): float(np.std(boot_ratio[d], ddof=1)) for d in D3_DELTAS},
                    "fit_c_delta2": c_sq, "fit_resid_delta2": resid_sq,
                    "fit_c_delta1_alt": c_lin, "fit_resid_delta1_alt": resid_lin,
                    "implied_rms_truncation_at_0.005": tau,
                    "implied_rms_truncation_at_0.005_boot_se": float(np.std(boot_tau, ddof=1)),
                    "per_trial_ratio_0.01_over_0.005_mean": float(per.mean()), "per_trial_ratio_sd": float(per.std(ddof=1))}
            out["cases"][f"{mode}_n{n}"] = case
            taus.append(tau)
    tmax = max(taus, key=abs)
    if abs(tmax) < D3_LABEL_NOISE:
        label = "잡음"
    elif tmax <= -D3_LABEL_TRUNC:
        label = "O(δ⁴)"
    else:
        label = "미결"
    out["max_abs_implied_truncation_at_0.005"] = tmax
    out["conclusion"] = label
    out["conclusion_meaning"] = ("'잡음' = the implied O(delta^4) truncation at delta=0.005 is below 0.5% of the RMS, so the ~2% deficit "
                                 "seen in the step-7 amplitude ratio cannot be the truncation term; '미결' otherwise unless a "
                                 "negative >=1% truncation ('O(δ⁴)') is found.  Report only, not a verdict.")
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    quick = "--quick" in sys.argv
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    result: dict = {"question": "Q-0008", "attempt": 6, "ladder_step": 3, "seed": SEED, "delta": DELTA, "quick": quick}

    a, M = part_a()
    result["A_constants"] = a
    print("A constants:", {k: a[k] for k in ("T2", "T4", "sum_a_Maa_max_abs", "normG0_sq", "eps_star_sq_over_delta4", "pass")}, flush=True)

    result["B_equivariance"] = part_b(rng)
    print("B equivariance worst:", result["B_equivariance"]["worst"], "pass", result["B_equivariance"]["pass"], flush=True)

    result["C_surrogate"] = part_c_surrogate(rng, M)
    print("C-a surrogate max |rel|:", result["C_surrogate"]["max_abs_rel_err"], "pass", result["C_surrogate"]["pass"],
          f"({time.time() - t0:.0f}s)", flush=True)

    result["C_physical"] = part_c_physical(rng, M, 50 if quick else PHYS_TRIALS)
    cp = result["C_physical"]
    print("C-b physical max |rel|:", cp["max_abs_rel_err"], "max|z|", cp["max_abs_z"], "pass5%", cp["pass_phys_5pct"],
          "| C-c crn max|mean-1|:", cp["max_abs_crn_minus_1"], "pass1%", cp["pass_crn_1pct"], f"({time.time() - t0:.0f}s)", flush=True)

    result["D_closed_forms"] = part_d(rng)
    print("D closed forms worst:", result["D_closed_forms"]["worst_rel_err"], "pass", result["D_closed_forms"]["pass"], flush=True)

    result["E_scope_table"] = part_e(rng)
    print("E gamma grid:", {k: round(v, 4) for k, v in result["E_scope_table"]["gamma_grid"].items()}, flush=True)

    result["F_delta3"] = part_f(rng, 16 if quick else D3_TRIALS)
    print("F delta3 conclusion:", result["F_delta3"]["conclusion"], "max implied tau:", result["F_delta3"]["max_abs_implied_truncation_at_0.005"],
          f"({time.time() - t0:.0f}s)", flush=True)

    verdict_parts = {"A": a["pass"], "B": result["B_equivariance"]["pass"], "C_surrogate": result["C_surrogate"]["pass"],
                     "C_physical_5pct": cp["pass_phys_5pct"], "C_crn_1pct": cp["pass_crn_1pct"], "D": result["D_closed_forms"]["pass"]}
    result["verdict_parts"] = verdict_parts
    result["numeric"] = "pass" if all(verdict_parts.values()) else "fail"
    result["wall_seconds"] = time.time() - t0
    target = HERE / ("result_quick.json" if quick else "result.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("numeric:", result["numeric"], "->", target.name, f"({result['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
