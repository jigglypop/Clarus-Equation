"""Q-0020 F-01: Euclidean Regge 1->5 블록의 접힘 부피 ln Ω_fold — 예산식 Γ_eff = S_c − ħ ln Ω_fold 의 숫자.

unglued 다섯 cell 의 길이 자유도 x∈R^50 을 자기 작용 Hessian N = ⊕_a H_a (등분할 κ, 길이 차트; Q-0019 F-02 규약 그대로)
로 Gaussian 적분할 때, 접착 제약(공유 변 길이 일치 35개) + 게이지 고정(내부 정점 이동 4개, 분자에만) 이 지우는 정규화 부피:

  ln Ω_fold(ℓ) = m ln(ℓ/ℓ_P) − ½ ln det(Kᵀ N̂⁻¹ K) − (m/2) ln(16π²),   m = 39 = 35 + 4
  K = 접힘 방향(제약 행공간 ∪ 게이지)의 정규직교 기저(길이 차트 50×39), N̂ = N (Regge 작용 2차 동차 ⇒ Hessian 0차 동차),
  κ = ℓ²/(8π ℓ_P²) 가 S/ħ 의 무차원 계수 (S = S_geo/(8πG), ℓ_P² = ħG).
  DE 원장 §7.3 꼴: −½ ln det(KᵀN̂⁻¹K) = ½ Σ_i ln(1−σ_i²) + ½ ln det C,  A=J_pᵀN̂J_p (11), C=KᵀN̂K (39), W=A^{-1/2}(J_pᵀN̂K)C^{-1/2}.

부호 규약 셋을 모두 계산한다(F-02 가 규약 의존으로 죽었으므로 규약 의존을 예측으로 바꾼다):
  R: raw 부정부호 N, |det| 와 KᵀN⁻¹K 의 음의 고윳값 개수 n_neg.   W: GHP conformal Wick = cell 별 |H_a| (정부호).
  P: §7.3 전제의 양의 부분공간 = N_E=−N 의 양의 40차원 섹터(cell 별 8) 로 제한.
모드: predict(regular ℓ²=2, 스케일 2→8, 상수·교차 스케일 ℓ★) · two_level(K1·K4: 1→5→25 가법성) · irregular(K2·K3).
씨앗 20260902, numpy 만. Hessian 은 h=2e−3 과 h/2 의 Richardson 외삽. 출력 predictions.json 또는 result_<mode>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import regge_one_to_five_boundary_hessian as RG  # noqa: E402
from regge_one_to_five_refinement import BOUNDARY_TRIANGLES, BOUNDARY_VERTICES  # noqa: E402

SEED = 20260902
FD_STEP = 2.0e-3
CLUSTER_TOL = 1.0e-5
LN16PI2 = math.log(16.0 * math.pi**2)
M_LEVEL1 = 39                      # 35 접착 + 4 게이지
M_LEVEL2 = 234                     # 210 접착 + 24 게이지 = 6 × 39
KILL_K1_TOL = 1.0e-2               # |r_R| : R 규약 가법성 잔차 (FD 한계, 234차원)
KILL_K2_SPEC_TOL = 0.5             # |Δ ½Σln(1−σ²)|_W  비정규 경계
KILL_K3_WINDOW = (0.8, 1.25)       # ℓ★²(비정규)/ℓ★²(regular)
KILL_K4_MIN = 1.0e-2               # |r_W| : Wick 규약은 가법적이지 않다
# predict 모드 관측값(카드 사전등록) — irregular 모드의 기준. predict 실행 뒤 채운다.
REGULAR_REFERENCE = {"n_neg_R": 31, "half_sum_log_1msig2_W": -3.6986104, "lstar2_over_lP2": 62.0688}

assert list(BOUNDARY_TRIANGLES) == list(combinations(range(5), 3))
assert list(RG.BOUNDARY_EDGES) == list(combinations(range(5), 2))


# ---------------------------------------------------------------- 배경 기하 (F-02 predict_action_metric.py 그대로)
def points_from_squared(squared: np.ndarray) -> dict[int, np.ndarray]:
    d = np.zeros((5, 5))
    for k, (i, j) in enumerate(RG.BOUNDARY_EDGES):
        d[i, j] = d[j, i] = squared[k]
    gram = np.array([[0.5 * (d[0, i] + d[0, j] - d[i, j]) for j in range(1, 5)] for i in range(1, 5)])
    lower = np.linalg.cholesky(gram)
    verts = np.vstack((np.zeros(4), lower))
    verts = verts - verts.mean(axis=0)
    return {i: verts[i] for i in range(5)}


def refine(cells: list[tuple[int, ...]], points: dict[int, np.ndarray]) -> list[tuple[int, ...]]:
    out = []
    for cell in cells:
        label = max(points) + 1
        points[label] = np.mean([points[v] for v in cell], axis=0)
        for omitted in cell:
            out.append((label,) + tuple(v for v in cell if v != omitted))
    return out


def cell_lengths(cell: tuple[int, ...], points: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([np.linalg.norm(points[i] - points[j]) for i, j in combinations(cell, 2)])


def simplex_action(lengths: np.ndarray, kappas: np.ndarray) -> float:
    return float(
        sum(
            RG._triangle_area(t, lengths, None) * (kappas[n] - RG._dihedral_angle(BOUNDARY_VERTICES, t, lengths, None))
            for n, t in enumerate(BOUNDARY_TRIANGLES)
        )
    )


def richardson_hessian(fun, point: np.ndarray, step: float = FD_STEP) -> np.ndarray:
    _, h1 = RG._gradient_and_hessian(fun, point, step)
    _, h2 = RG._gradient_and_hessian(fun, point, step / 2.0)
    return (4.0 * h2 - h1) / 3.0


def simplex_hessian(lengths: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    return richardson_hessian(lambda v: simplex_action(v, kappas), lengths)


def clusters(values: np.ndarray, tol: float = CLUSTER_TOL) -> list[dict]:
    out: list[list[float]] = []
    for v in np.sort(np.asarray(values).real):
        if out and abs(out[-1][0] - v) < tol * max(1.0, abs(v)):
            out[-1].append(float(v))
        else:
            out.append([float(v)])
    return [{"value": float(np.mean(c)), "multiplicity": len(c)} for c in out]


# ---------------------------------------------------------------- 블록 구성: 등분할 κ, 접착 제약, 게이지 방향
def equal_split_kappas(cells: list[tuple[int, ...]], block_vertices: tuple[int, ...], kappa_coarse: np.ndarray) -> list[np.ndarray]:
    """블록 경계 삼각형(block_vertices 의 combinations 순)은 kappa_coarse[t], 내부 삼각형은 2π 를 품는 cell 수로 등분."""
    count: dict[tuple[int, ...], int] = {}
    for cell in cells:
        for t in combinations(cell, 3):
            key = tuple(sorted(t))
            count[key] = count.get(key, 0) + 1
    bidx = {tuple(sorted(t)): n for n, t in enumerate(combinations(block_vertices, 3))}
    out = []
    for cell in cells:
        k = []
        for t in combinations(cell, 3):
            key = tuple(sorted(t))
            total = float(kappa_coarse[bidx[key]]) if key in bidx else 2.0 * math.pi
            k.append(total / count[key])
        out.append(np.asarray(k))
    return out


def gluing_rows(cells: list[tuple[int, ...]]) -> np.ndarray:
    """공유 변 길이 일치 제약: 같은 변을 품는 cell 들의 길이 차 = 0 (첫 소유자 기준)."""
    owners: dict[tuple[int, int], list[int]] = {}
    for a, cell in enumerate(cells):
        for r, (i, j) in enumerate(combinations(cell, 2)):
            owners.setdefault(tuple(sorted((i, j))), []).append(10 * a + r)
    dof = 10 * len(cells)
    rows = []
    for idx in owners.values():
        for k in idx[1:]:
            row = np.zeros(dof)
            row[idx[0]] = 1.0
            row[k] = -1.0
            rows.append(row)
    return np.asarray(rows)


def gauge_directions(cells: list[tuple[int, ...]], points: dict[int, np.ndarray], internal: list[int]) -> np.ndarray:
    """내부 정점 v 를 e_k 방향으로 옮길 때 각 cell 길이의 1차 변화 (glued, 평탄 배위의 정확 대칭)."""
    dof = 10 * len(cells)
    cols = []
    for v in internal:
        for k in range(4):
            e = np.eye(4)[k]
            col = np.zeros(dof)
            for a, cell in enumerate(cells):
                if v not in cell:
                    continue
                for r, (i, j) in enumerate(combinations(cell, 2)):
                    if v == i:
                        u = j
                    elif v == j:
                        u = i
                    else:
                        continue
                    d = points[v] - points[u]
                    col[10 * a + r] = float(d @ e) / float(np.linalg.norm(d))
            cols.append(col)
    return np.asarray(cols).T


# ---------------------------------------------------------------- 접힘 부피
def fold(cells: list[tuple[int, ...]], points: dict[int, np.ndarray], kappas: list[np.ndarray], internal: list[int]) -> dict:
    lengths = [cell_lengths(c, points) for c in cells]
    hess = [simplex_hessian(l, k) for l, k in zip(lengths, kappas)]
    dof = 10 * len(cells)
    N = np.zeros((dof, dof))
    NW = np.zeros((dof, dof))
    vcols = []
    cell_sig = []
    for a, h in enumerate(hess):
        w, v = np.linalg.eigh(h)
        sl = slice(10 * a, 10 * a + 10)
        N[sl, sl] = h
        NW[sl, sl] = v @ np.diag(np.abs(w)) @ v.T
        neg = v[:, w < 0]
        vc = np.zeros((dof, neg.shape[1]))
        vc[sl] = neg
        vcols.append(vc)
        cell_sig.append((int(np.sum(w > 0)), int(np.sum(w < 0))))
    V = np.hstack(vcols)                      # N_E = −N 의 양의 부분공간 (정규직교)
    Gam = gluing_rows(cells)
    g = gauge_directions(cells, points, internal)
    rows = np.vstack([Gam, g.T])
    _, s, vt = np.linalg.svd(rows)
    rank = int(np.sum(s > 1.0e-9 * s[0]))
    K = vt[:rank].T                           # 접힘 방향 (dof × m)
    Jp = vt[rank:].T                          # 물리 glued 방향 (dof × (dof−m))
    m = rank
    # R
    Ninv = np.linalg.inv(N)
    wR = np.linalg.eigvalsh(K.T @ Ninv @ K)
    d_R = -0.5 * float(np.sum(np.log(np.abs(wR))))
    n_neg = int(np.sum(wR < 0))
    # W
    wW = np.linalg.eigvalsh(K.T @ np.linalg.inv(NW) @ K)
    d_W = -0.5 * float(np.sum(np.log(wW)))
    # P
    Np = -(V.T @ N @ V)
    G = K.T @ V
    wP = np.linalg.eigvalsh(G @ np.linalg.inv(Np) @ G.T)
    rank_P = int(np.sum(wP > 1.0e-9 * np.max(np.abs(wP))))       # 관측: 35 < 39 — 양의 부분공간은 게이지 고정과 양립하지 않음
    d_P = -0.5 * float(np.sum(np.log(np.abs(wP)))) if rank_P == m else float("nan")
    # P 의 살아남는 정의: 접착 제약(35)만, V_+ 위 (ln ℓ 계수 m_P = 35)
    _, sg, vtg = np.linalg.svd(Gam)
    Kg = vtg[: int(np.sum(sg > 1.0e-9 * sg[0]))].T
    Gg = Kg.T @ V
    wPg = np.linalg.eigvalsh(Gg @ np.linalg.inv(Np) @ Gg.T)
    m_P = int(Kg.shape[1])
    rank_P_glue = int(np.sum(wPg > 1.0e-9 * np.max(np.abs(wPg))))
    d_P35 = -0.5 * float(np.sum(np.log(np.abs(wPg)))) if rank_P_glue == m_P else float("nan")
    # §7.3 스펙트럼 (W: 정부호)
    A = Jp.T @ NW @ Jp
    C = K.T @ NW @ K
    B = Jp.T @ NW @ K
    wa, va = np.linalg.eigh(A)
    aih = va @ np.diag(wa**-0.5) @ va.T
    sig2 = np.linalg.eigvalsh(aih @ B @ np.linalg.solve(C, B.T) @ aih)
    half_sum = 0.5 * float(np.sum(np.log1p(-sig2)))
    ident_W = half_sum + 0.5 * float(np.linalg.slogdet(C)[1]) - d_W
    # R 스펙트럼 (부정부호: A⁻¹BC⁻¹Bᵀ 의 고윳값, 실수성 기록)
    AR = Jp.T @ N @ Jp
    CR = K.T @ N @ K
    BR = Jp.T @ N @ K
    rho = np.linalg.eigvals(np.linalg.solve(AR, BR @ np.linalg.solve(CR, BR.T)))
    ident_R = 0.5 * float(np.sum(np.log(np.abs(1.0 - rho)))) + 0.5 * float(np.linalg.slogdet(CR)[1]) - d_R
    # F-02 다리: A_R (J_pᵀN⁻¹J_p) 의 고윳값 = 1/(1−ρ)
    lam = np.linalg.eigvals(AR @ (Jp.T @ Ninv @ Jp))
    bridge = float(np.max(np.abs(np.sort(lam.real) - np.sort((1.0 / (1.0 - rho)).real)) / np.abs(np.sort(lam.real))))
    gauge_res = float(np.linalg.norm(g.T @ N @ g) / np.linalg.norm(N @ g))
    return {
        "cells": len(cells), "dof": dof, "m": m, "m_glue": int(Gam.shape[0]), "m_gauge": int(g.shape[1]),
        "d_R": d_R, "d_W": d_W, "d_P": d_P, "n_neg_R": n_neg, "rank_P": rank_P,
        "m_P": m_P, "rank_P_glue": rank_P_glue, "d_P35": d_P35, "c_P35": d_P35 - 0.5 * m_P * LN16PI2,
        "c_R": d_R - 0.5 * m * LN16PI2, "c_W": d_W - 0.5 * m * LN16PI2, "c_P": d_P - 0.5 * m * LN16PI2,
        "sigma2_W": sig2.tolist(), "sigma2_W_clusters": clusters(sig2), "sigma2_W_max": float(np.max(sig2)),
        "half_sum_log_1msig2_W": half_sum, "identity_7_3_W": ident_W, "identity_7_3_R": ident_R,
        "rho_R": [float(z.real) for z in rho], "rho_R_clusters": clusters(rho),
        "rho_R_max_imag_ratio": float(np.max(np.abs(rho.imag) / np.abs(rho))),
        "f02_bridge_residual": bridge, "gauge_gNg_over_Ng": gauge_res,
        "cell_signature_N": cell_sig[0], "signature_N": [sum(s[0] for s in cell_sig), sum(s[1] for s in cell_sig)],
    }


def regular_level1(squared_edge: float = 2.0) -> dict:
    points = points_from_squared(np.full(10, squared_edge))
    cells = refine([tuple(BOUNDARY_VERTICES)], points)
    kap = equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
    return fold(cells, points, kap, [5])


def coarse_action_unit() -> dict:
    """단위 제곱 변 길이당 coarse Regge 작용 Ŝ_c (모듈 규약, Euler 검사) 와 닫힌 꼴."""
    b0 = np.sqrt(np.full(10, 2.0))
    s2 = float(RG.coarse_euclidean_regge_boundary_action(b0))
    hc = richardson_hessian(RG.coarse_euclidean_regge_boundary_action, b0)
    closed = 10.0 * math.sqrt(3.0) / 4.0 * (math.pi - math.acos(0.25))
    return {"S_c_at_sq2": s2, "S_hat_c": s2 / 2.0, "S_hat_c_closed_form": closed,
            "euler_bHb_minus_2S": float(b0 @ hc @ b0 - 2.0 * s2)}


def crossover(s_hat: float, m: int, consts: dict[str, float]) -> dict:
    """Γ_eff(ℓ)/ħ = (ℓ²/8πℓ_P²) Ŝ_c − [m ln(ℓ/ℓ_P) + c]. 정류점 ℓ★² = 4π m/Ŝ_c (규약 무관), 최소값은 규약별."""
    l2 = 4.0 * math.pi * m / s_hat
    out = {"m": m, "lstar2_over_lP2": l2, "lstar_over_lP": math.sqrt(l2), "gamma_min_over_hbar": {},
           "quotient_dominates_somewhere": {}, "l_omega_over_lP": {}}
    for k, c in consts.items():
        gmin = m / 2.0 - 0.5 * m * math.log(l2) - c
        out["gamma_min_over_hbar"][k] = gmin
        out["quotient_dominates_somewhere"][k] = bool(gmin < 0)
        out["l_omega_over_lP"][k] = math.exp(-c / m)          # Ω_fold = 1 인 스케일
    return out


# ---------------------------------------------------------------- 모드
def run_predict() -> dict:
    reg = regular_level1(2.0)
    reg8 = regular_level1(8.0)
    act = coarse_action_unit()
    consts = {"R": reg["c_R"], "W": reg["c_W"]}
    cross = crossover(act["S_hat_c"], reg["m"], consts)
    cross_p = crossover(act["S_hat_c"], reg["m_P"], {"P35": reg["c_P35"]})
    return {
        "card": "Q-0020 F-01", "seed": SEED, "fd_step": FD_STEP,
        "convention": "ln Ω_fold = m ln(ℓ/ℓ_P) + d − (m/2) ln 16π², d = −½ ln det(KᵀN⁻¹K); R raw|det|, W cell별 |H_a|, P N_E 양의 40차원",
        "regular": reg,
        "scale_2_to_8": {"d_R": reg8["d_R"] - reg["d_R"], "d_W": reg8["d_W"] - reg["d_W"], "d_P35": reg8["d_P35"] - reg["d_P35"],
                          "sigma2_W_max_dev": float(np.max(np.abs(np.sort(reg8["sigma2_W"]) - np.sort(reg["sigma2_W"])))),
                          "ln_omega_shift_lP_fixed": reg["m"] * math.log(2.0)},
        "coarse_action": act,
        "crossover": cross, "crossover_P_glue_only": cross_p,
        "ln16pi2_half_m": 0.5 * reg["m"] * LN16PI2,
        "convention_differences": {"W_minus_R": reg["d_W"] - reg["d_R"], "P35_minus_R": reg["d_P35"] - reg["d_R"], "P35_minus_W": reg["d_P35"] - reg["d_W"]},
    }


def run_two_level() -> dict:
    """K1·K4: 직접 1→5→25 (250차원) vs 합성 (regular 1→5 + 다섯 cell 각각의 1→5, 경계 κ = level-1 등분할 κ_a)."""
    points = points_from_squared(np.full(10, 2.0))
    cells1 = refine([tuple(BOUNDARY_VERTICES)], points)
    kap1 = equal_split_kappas(cells1, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
    level1 = fold(cells1, points, kap1, [5])
    subs = []
    for cell, kc in zip(cells1, kap1):
        sq = cell_lengths(cell, points) ** 2
        pts = points_from_squared(sq)
        sc = refine([tuple(range(5))], pts)
        subs.append(fold(sc, pts, equal_split_kappas(sc, tuple(range(5)), kc), [5]))
    points2 = points_from_squared(np.full(10, 2.0))
    cells2 = refine(refine([tuple(BOUNDARY_VERTICES)], points2), points2)
    kap2 = equal_split_kappas(cells2, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
    direct = fold(cells2, points2, kap2, list(range(5, 11)))
    res = {}
    for k in ("d_R", "d_W", "d_P35"):
        res[k] = direct[k] - level1[k] - sum(s[k] for s in subs)
    m_add = level1["m"] + sum(s["m"] for s in subs)
    killed_k1 = abs(res["d_R"]) > KILL_K1_TOL or direct["m"] != M_LEVEL2 or m_add != M_LEVEL2
    killed_k4 = abs(res["d_W"]) < KILL_K4_MIN
    keys = ("cells", "dof", "m", "m_glue", "m_gauge", "d_R", "d_W", "d_P35", "n_neg_R", "rank_P", "identity_7_3_W", "gauge_gNg_over_Ng")
    return {
        "mode": "two_level", "direct": {k: direct[k] for k in keys},
        "level1_regular": {k: level1[k] for k in ("m", "d_R", "d_W", "d_P35")},
        "subcells": [{k: s[k] for k in ("m", "d_R", "d_W", "d_P35", "n_neg_R")} for s in subs],
        "m_composed": m_add, "additivity_residual": res,
        "kill_K1": {"tol": KILL_K1_TOL, "m_expected": M_LEVEL2, "killed": bool(killed_k1)},
        "kill_K4": {"min_abs_r_W": KILL_K4_MIN, "killed": bool(killed_k4)},
        "killed": bool(killed_k1 or killed_k4),
    }


def run_irregular(amplitude: float = 0.1) -> dict:
    """K2·K3: 제곱 변 길이 2(1±0.1), 부호 = default_rng(20260902).choice([-1,1],10) (F-02 K2 와 같은 섭동)."""
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1.0, 1.0], size=10)
    squared = 2.0 * (1.0 + amplitude * signs)
    points = points_from_squared(squared)
    cells = refine([tuple(BOUNDARY_VERTICES)], points)
    kap = equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
    irr = fold(cells, points, kap, [5])
    b0 = np.sqrt(squared)
    s_irr = float(RG.coarse_euclidean_regge_boundary_action(b0)) / float(np.mean(squared))
    lstar2 = 4.0 * math.pi * irr["m"] / s_irr if s_irr > 0 else float("nan")
    ratio = lstar2 / REGULAR_REFERENCE["lstar2_over_lP2"]
    spec_shift = irr["half_sum_log_1msig2_W"] - REGULAR_REFERENCE["half_sum_log_1msig2_W"]
    lo, hi = KILL_K3_WINDOW
    killed_k2 = irr["m"] != M_LEVEL1 or irr["n_neg_R"] != REGULAR_REFERENCE["n_neg_R"] or abs(spec_shift) > KILL_K2_SPEC_TOL
    killed_k3 = not (s_irr > 0) or not (lo <= ratio <= hi)
    keys = ("m", "m_glue", "m_gauge", "d_R", "d_W", "d_P35", "n_neg_R", "rank_P", "rank_P_glue", "sigma2_W", "sigma2_W_max",
            "half_sum_log_1msig2_W", "identity_7_3_W", "rho_R_max_imag_ratio", "gauge_gNg_over_Ng", "signature_N")
    return {
        "mode": "irregular", "squared_lengths": squared.tolist(), "fold": {k: irr[k] for k in keys},
        "S_hat_c_irregular": s_irr, "lstar2_over_lP2": lstar2, "lstar2_ratio_to_regular": ratio, "spec_shift_W": spec_shift,
        "reference": REGULAR_REFERENCE,
        "kill_K2": {"spec_tol": KILL_K2_SPEC_TOL, "killed": bool(killed_k2)},
        "kill_K3": {"window": list(KILL_K3_WINDOW), "killed": bool(killed_k3)},
        "killed": bool(killed_k2 or killed_k3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("predict", "two_level", "irregular"), default="predict")
    args = parser.parse_args()
    runner = {"predict": run_predict, "two_level": run_two_level, "irregular": run_irregular}[args.mode]
    result = runner()
    result["provenance"] = {
        "python": sys.executable, "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "nonfinite_policy": "undefined numerical entries are null, not zero",
    }
    result = json_safe(result)
    out = HERE / ("predictions.json" if args.mode == "predict" else f"result_{args.mode}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=1, allow_nan=False))


def json_safe(value):
    """정의되지 않은 수치를 명시적 null로 보존해 표준 JSON을 출력한다."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
