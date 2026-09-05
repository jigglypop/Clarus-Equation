"""Q-0019 F-02: Euclidean Regge 1->5 블록의 mismatch 증폭률을 작용 Hessian(Schur 보완) 계량으로 잰 펜슬 스펙트럼.

카드(derivations/Q-0019/F-02.formula.md)가 선언한 규약을 그대로 계산한다. 좌표는 고르지 않는다.

  * 배경: 제곱 변 길이 2의 regular 4-simplex(무게중심 원점), barycentric 1->5 분할(26장 flat section), 다섯 fine cell.
  * fine 형식 N: cell a의 등분할(equal-split) 단일 simplex 작용 S_a = Σ_{t⊂a} A_t(κ_t − θ_t^a), κ_t = (2π 또는 π)/n_t
    (n_t = 그 hinge를 품는 cell 수)의 길이 차트 Hessian H_a(10×10)를 cell별 길이 편차에 건 블록대각 2차 형식.
    장부 항등식: Σ_a J_aᵀ H_a J_a = H_f (15×15 glued fine Hessian, 모듈 euclidean_regge_one_to_five_action).
  * coarse 형식 M: 경계 변 길이 편차 Δ_c = 그 변을 품는 세 cell 길이 편차의 산술평균, M = Δ_cᵀ H_c Δ_c,
    H_c = coarse 경계 Hessian(모듈) = H_eff = A − B C⁺ Bᵀ (26장 Schur 정리, 잔차를 함께 기록).
  * λ_S² := (M, N) 펜슬의 일반화 고윳값 (det(M − λ² N) = 0). 좌표 변환 P 아래 congruence 불변. 자유 파라미터 0.
  * 게이지 4방향 G(내부 정점 이동): L G = 0 (M의 정확 영방향), GᵀNG = 0, 그러나 N G ≠ 0 (unglued 형식의 대칭이 아님).
  * 부호: N·H_c·H_f의 signature와 각 고유벡터의 N-부호 섹터를 기록한다.

모드: predict(카드 숫자·자기감사, 실행됨) · two_level(K1) · irregular(K2) · coords(K3).
K1·K2·K3 모드는 카드 작성 시점에 실행하지 않았다(사전등록). 씨앗 20260902(K3는 20260903·4·5), numpy만.
Hessian은 유한차분 h=2e-3과 h/2의 Richardson 외삽(정밀도 ~1e-7). 출력: predictions.json 또는 result_<mode>.json.
"""

from __future__ import annotations

import argparse
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
ZERO_TOL = 1.0e-6          # 펜슬 영고윳값 판정 (|λ²| < ZERO_TOL)
CLUSTER_TOL = 1.0e-5       # 스펙트럼 군집 판정
KILL_K1_WINDOW = (0.75, 1.25)          # λ₂²/(λ₁²)² 창 (세 군집 모두)
KILL_K2_RATIO_WINDOW = (0.8, 1.25)     # λ²_max·λ²_min 비 창
KILL_K2_COUNT_GT1 = (5, 6)             # 비정규에서 λ²>1 개수 허용값
KILL_K2_COUNT_LT1 = (4, 5)             # 비정규에서 λ²<1 개수 허용값
KILL_REALITY_TOL = 1.0e-6              # |Im λ²| / |λ²|
KILL_K3_TOL = 1.0e-8                   # 좌표 변환 3종 스펙트럼 불변
K3_SEEDS = (20260903, 20260904, 20260905)
LEVEL1_LAMBDA2 = {"triv": 1.0, "std": 0.4296818, "[3,2]": 1.4723737}  # predict 모드에서 관측(카드 predicts)

# ---------------------------------------------------------------- Sym(4) 정규직교 기저
def sym_basis() -> list[np.ndarray]:
    basis = []
    for i in range(4):
        m = np.zeros((4, 4))
        m[i, i] = 1.0
        basis.append(m)
    for i in range(4):
        for j in range(i + 1, 4):
            m = np.zeros((4, 4))
            m[i, j] = m[j, i] = 1.0 / math.sqrt(2.0)
            basis.append(m)
    return basis


BASIS = sym_basis()


def sym_to_vec(m: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.sum(b * m)) for b in BASIS])


# ---------------------------------------------------------------- 배경 기하
def points_from_squared(squared: np.ndarray) -> dict[int, np.ndarray]:
    """10개 경계 제곱 변 길이(BOUNDARY_EDGES 순서)에서 무게중심 원점의 정점 0..4 좌표."""
    d = np.zeros((5, 5))
    for k, (i, j) in enumerate(RG.BOUNDARY_EDGES):
        d[i, j] = d[j, i] = squared[k]
    gram = np.array([[0.5 * (d[0, i] + d[0, j] - d[i, j]) for j in range(1, 5)] for i in range(1, 5)])
    lower = np.linalg.cholesky(gram)
    verts = np.vstack((np.zeros(4), lower))
    verts = verts - verts.mean(axis=0)
    return {i: verts[i] for i in range(5)}


def refine(cells: list[tuple[int, ...]], points: dict[int, np.ndarray]) -> list[tuple[int, ...]]:
    """각 cell을 무게중심으로 1->5 분할 (26장 flat section). 새 정점 라벨은 이어 붙인다."""
    out = []
    for cell in cells:
        label = max(points) + 1
        points[label] = np.mean([points[v] for v in cell], axis=0)
        for omitted in cell:
            out.append((label,) + tuple(v for v in cell if v != omitted))
    return out


def cell_lengths(cell: tuple[int, ...], points: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([np.linalg.norm(points[i] - points[j]) for i, j in combinations(cell, 2)])


def kappas_for(cells: list[tuple[int, ...]]) -> list[np.ndarray]:
    """등분할 규약: hinge t의 상수 (외부 경계 π, 내부 2π)를 그 hinge를 품는 cell 수로 나눈다."""
    count: dict[tuple[int, ...], int] = {}
    for cell in cells:
        for t in combinations(cell, 3):
            key = tuple(sorted(t))
            count[key] = count.get(key, 0) + 1
    out = []
    for cell in cells:
        k = []
        for t in combinations(cell, 3):
            key = tuple(sorted(t))
            total = math.pi if all(v in BOUNDARY_VERTICES for v in key) else 2.0 * math.pi
            k.append(total / count[key])
        out.append(np.asarray(k))
    return out


# ---------------------------------------------------------------- 단일 simplex 작용과 Hessian (길이 차트, Richardson)
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


# ---------------------------------------------------------------- 형식 N·M (metric 좌표) 와 길이 좌표
class Complex:
    """cells 위의 mismatch 공간 R^{10·ncell}(cell별 Sym(4) 편차)과 두 2차 형식."""

    def __init__(self, cells: list[tuple[int, ...]], points: dict[int, np.ndarray], coarse_hessian: np.ndarray):
        self.cells = cells
        self.points = points
        self.n = len(cells)
        self.kappas = kappas_for(cells)
        self.lengths = [cell_lengths(c, points) for c in cells]
        self.hess = [simplex_hessian(l, k) for l, k in zip(self.lengths, self.kappas)]
        # T_a: metric 좌표 -> cell a 길이 편차 (δℓ_e = u_eᵀ M u_e / ℓ_e, g = I + 2δM)
        self.T = []
        for cell, lens in zip(cells, self.lengths):
            t = np.zeros((10, 10))
            for r, (i, j) in enumerate(combinations(cell, 2)):
                u = points[i] - points[j]
                t[r] = [float(u @ b @ u) / lens[r] for b in BASIS]
            self.T.append(t)
        dim = 10 * self.n
        self.N = np.zeros((dim, dim))
        for a, (t, h) in enumerate(zip(self.T, self.hess)):
            self.N[10 * a : 10 * a + 10, 10 * a : 10 * a + 10] = t.T @ h @ t
        self.N_len = np.zeros((dim, dim))
        for a, h in enumerate(self.hess):
            self.N_len[10 * a : 10 * a + 10, 10 * a : 10 * a + 10] = h
        # L: cell 길이 편차 -> coarse 경계 변 (품는 cell 산술평균)
        self.L_len = np.zeros((10, dim))
        for k, (i, j) in enumerate(RG.BOUNDARY_EDGES):
            owners = [(a, list(combinations(c, 2)).index((i, j) if c.index(i) < c.index(j) else (j, i)))
                      for a, c in enumerate(cells) if i in c and j in c]
            for a, r in owners:
                self.L_len[k, 10 * a + r] = 1.0 / len(owners)
        self.P = np.zeros((dim, dim))  # metric -> length 좌표 (블록대각)
        for a, t in enumerate(self.T):
            self.P[10 * a : 10 * a + 10, 10 * a : 10 * a + 10] = t
        self.L = self.L_len @ self.P
        self.Hc = coarse_hessian
        self.M = self.L.T @ coarse_hessian @ self.L
        self.M_len = self.L_len.T @ coarse_hessian @ self.L_len


def pencil_eigs(N: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(M, N) 펜슬의 0 아닌 일반화 고윳값(복소 가능), 실부 오름차순."""
    ev = np.linalg.eigvals(np.linalg.solve(N, M))
    ev = ev[np.abs(ev) > ZERO_TOL]
    return ev[np.argsort(ev.real)]


def clusters(values: np.ndarray, tol: float = CLUSTER_TOL) -> list[dict]:
    out: list[list[float]] = []
    for v in np.sort(values.real):
        if out and abs(out[-1][0] - v) < tol * max(1.0, abs(v)):
            out[-1].append(float(v))
        else:
            out.append([float(v)])
    return [{"lambda2": float(np.mean(c)), "multiplicity": len(c)} for c in out]


def label_clusters(cl: list[dict]) -> dict[str, dict]:
    names = {1: "triv", 4: "std", 5: "[3,2]"}
    return {names.get(c["multiplicity"], f"mult{c['multiplicity']}"): c for c in cl}


def signature(H: np.ndarray, rel: float = 1.0e-6) -> dict:
    w = np.linalg.eigvalsh(H)
    tol = rel * float(np.max(np.abs(w)))
    return {"pos": int(np.sum(w > tol)), "neg": int(np.sum(w < -tol)), "zero": int(np.sum(np.abs(w) <= tol))}


def max_imag_ratio(ev: np.ndarray) -> float:
    return float(np.max(np.abs(ev.imag) / np.abs(ev)))


def random_coordinate_change(rng: np.random.Generator, dim: int) -> np.ndarray:
    q1, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    q2, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q1 @ np.diag(np.exp(rng.uniform(-1.0, 1.0, size=dim))) @ q2


def spectrum_deviation(ref: np.ndarray, other: np.ndarray) -> float:
    return float(np.max(np.abs(np.sort(other.real) - np.sort(ref.real)) / np.abs(np.sort(ref.real))))


def level1_complex(squared: np.ndarray) -> tuple[Complex, np.ndarray, np.ndarray]:
    points = points_from_squared(squared)
    cells = refine([tuple(BOUNDARY_VERTICES)], points)
    b0 = np.sqrt(squared)
    hc = richardson_hessian(RG.coarse_euclidean_regge_boundary_action, b0)
    return Complex(cells, points, hc), b0, points[5]


# ---------------------------------------------------------------- 모드
def run_predict() -> dict:
    squared = np.full(10, 2.0)
    cx, b0, _ = level1_complex(squared)
    y0 = RG.barycentric_internal_lengths(b0)
    p0 = np.concatenate((b0, y0))
    hf = richardson_hessian(lambda v: RG.euclidean_regge_one_to_five_action(v[:10], v[10:]), p0)
    # 장부 항등식 1: Σ_a J_aᵀ H_a J_a = H_f (J_a: 15 글로벌 길이 -> cell a 10 길이)
    def gidx(i: int, j: int) -> int:
        i, j = sorted((i, j))
        return 10 + i if j == 5 else list(RG.BOUNDARY_EDGES).index((i, j))
    hsum = np.zeros((15, 15))
    for cell, h in zip(cx.cells, cx.hess):
        idx = [gidx(i, j) for i, j in combinations(cell, 2)]
        hsum[np.ix_(idx, idx)] += h
    glue_residual = float(np.linalg.norm(hsum - hf) / np.linalg.norm(hf))
    # 장부 항등식 2: Schur H_eff = A − B C⁺ Bᵀ = H_c, pullback, stationarity (26장)
    A, B, C = hf[:10, :10], hf[:10, 10:], hf[10:, 10:]
    u = np.ones(5) / math.sqrt(5.0)
    cpinv = np.outer(u, u) / (40.0 * math.sqrt(5.0))
    J = RG.barycentric_section_jacobian(b0)
    schur_residual = float(np.linalg.norm(A - B @ cpinv @ B.T - cx.Hc) / np.linalg.norm(cx.Hc))
    pullback_residual = float(np.linalg.norm(A + B @ J - cx.Hc) / np.linalg.norm(cx.Hc))
    stationarity_residual = float(np.linalg.norm(B.T + C @ J) / np.linalg.norm(C))
    # 펜슬 (metric 좌표)
    ev = pencil_eigs(cx.N, cx.M)
    cl = label_clusters(clusters(ev))
    # 좌표 3종: 길이 좌표(독립 구성), 임의 P(씨앗 20260902)
    ev_len = pencil_eigs(cx.N_len, cx.M_len)
    rng = np.random.default_rng(SEED)
    P = random_coordinate_change(rng, 50)
    ev_rand = pencil_eigs(P.T @ cx.N @ P, P.T @ cx.M @ P)
    coord_dev = {"length": spectrum_deviation(ev, ev_len), "random_P_seed_20260902": spectrum_deviation(ev, ev_rand),
                 "random_P_condition_number": float(np.linalg.cond(P))}
    # 고유벡터의 N-부호 섹터
    W = np.linalg.solve(cx.N, cx.M)
    wv, vv = np.linalg.eig(W)
    sectors = []
    for k in np.argsort(wv.real):
        if abs(wv[k]) > ZERO_TOL:
            v = vv[:, k].real
            sectors.append({"lambda2": float(wv[k].real), "N_sign": float(np.sign(v @ cx.N @ v)),
                            "M_sign": float(np.sign(v @ cx.M @ v))})
    # 게이지 4방향: e_a = I − q w_aᵀ
    G = np.zeros((50, 4))
    for k in range(4):
        q = np.eye(4)[k]
        for a, cell in enumerate(cx.cells):
            rows = np.array([cx.points[v] - cx.points[cell[0]] for v in cell[1:]])
            w = np.linalg.solve(rows, np.ones(4))
            eta = -np.outer(q, w)
            G[10 * a : 10 * a + 10, k] = sym_to_vec(0.5 * (eta + eta.T))
    gauge = {"LG": float(np.linalg.norm(cx.L @ G)), "GtNG_over_NG": float(np.linalg.norm(G.T @ cx.N @ G) / np.linalg.norm(cx.N @ G)),
             "NG": float(np.linalg.norm(cx.N @ G)), "N_norm": float(np.linalg.norm(cx.N))}
    # 게이지 quotient 변형 (G^{⊥N}/G) 의 펜슬 — 같은 스펙트럼이어야 규약 무관
    _, _, vt = np.linalg.svd(G.T @ cx.N)
    Z = vt[4:].T
    wz, U = np.linalg.eigh(Z.T @ cx.N @ Z)
    U = U[:, np.abs(wz) > 1.0e-5]
    ev_quot = pencil_eigs(U.T @ Z.T @ cx.N @ Z @ U, U.T @ Z.T @ cx.M @ Z @ U)
    # 코히런트 항등: Rayleigh = 1 (10 기저 방향), 공통 α, cell별 so(4) = 0
    coh = np.array([np.tile(sym_to_vec(b), 5) for b in BASIS]).T
    coh_rayleigh = [float((c @ cx.M @ c) / (c @ cx.N @ c)) for c in coh.T]
    alpha = np.tile(sym_to_vec(np.eye(4)), 5)
    euler = float(b0 @ cx.Hc @ b0 - 2.0 * RG.coarse_euclidean_regge_boundary_action(b0))
    anti = rng.normal(size=(4, 4))
    anti = anti - anti.T
    so4_vec = np.tile(sym_to_vec(0.5 * (anti + anti.T)), 5)
    # N-중심화(코히런트의 N-직교 여공간) 펜슬 = λ² − 1 항등식
    _, _, vt = np.linalg.svd(coh.T @ cx.N)
    Zc = vt[10:].T
    ev_c = pencil_eigs(Zc.T @ cx.N @ Zc, Zc.T @ cx.M @ Zc)
    nonunit = ev[np.abs(ev.real - 1.0) > CLUSTER_TOL]
    mu_residual = spectrum_deviation(nonunit - 1.0, ev_c)
    # δ 3점 선형화: 유한 mismatch 의 2차 차분 vs N·M
    v = rng.normal(size=50)
    v /= np.linalg.norm(v)
    def fine_total(delta: float) -> float:
        return sum(simplex_action(l + delta * (t @ v[10 * a : 10 * a + 10]), k)
                   for a, (l, t, k) in enumerate(zip(cx.lengths, cx.T, cx.kappas)))
    dc = cx.L @ v
    lin = {}
    for delta in (0.02, 0.01, 0.005):
        fine_dd = (fine_total(delta) + fine_total(-delta) - 2.0 * fine_total(0.0)) / delta**2
        coarse_dd = (RG.coarse_euclidean_regge_boundary_action(b0 + delta * dc) + RG.coarse_euclidean_regge_boundary_action(b0 - delta * dc)
                     - 2.0 * RG.coarse_euclidean_regge_boundary_action(b0)) / delta**2
        lin[str(delta)] = {"fine_rel": float(abs(fine_dd - v @ cx.N @ v) / abs(v @ cx.N @ v)),
                           "coarse_rel": float(abs(coarse_dd - v @ cx.M @ v) / abs(v @ cx.M @ v))}
    # l² 차트 감사 (비선형 재매개화는 불변이 아님: 관측만 기록)
    hess_l2 = []
    for l, k in zip(cx.lengths, cx.kappas):
        g, _ = RG._gradient_and_hessian(lambda x, k=k: simplex_action(x, k), l, FD_STEP)
        hess_l2.append(simplex_hessian(l, k) - np.diag(g / l))
    gc, _ = RG._gradient_and_hessian(RG.coarse_euclidean_regge_boundary_action, b0, FD_STEP)
    hc_l2 = cx.Hc - np.diag(gc / b0)
    N_l2 = np.zeros_like(cx.N)
    for a, (t, h) in enumerate(zip(cx.T, hess_l2)):
        N_l2[10 * a : 10 * a + 10, 10 * a : 10 * a + 10] = t.T @ h @ t
    ev_l2 = pencil_eigs(N_l2, cx.L.T @ hc_l2 @ cx.L)
    trace10 = float(np.sum(ev.real) / 10.0)
    return {
        "card": "Q-0019 F-02", "seed": SEED, "fd_step": FD_STEP,
        "convention": "λ_S² = (M,N) 펜슬 일반화 고윳값; N = ⊕_a 등분할 단일 simplex Hessian(길이 차트), M = LᵀH_cL, L = 3-cell 산술평균",
        "lambda2_clusters": cl,
        "lambda2_spectrum": [float(z.real) for z in ev],
        "max_imag_ratio": max_imag_ratio(ev),
        "lambda2_iso_trace_over_10": trace10,
        "lambda2_iso_negative_sector": float((np.sum(ev.real) - 1.0) / 9.0),
        "mu": {k: c["lambda2"] - 1.0 for k, c in cl.items()},
        "trace_budget_residual": float(sum(c["lambda2"] * c["multiplicity"] for c in cl.values()) / 10.0 - trace10),
        "signature": {"N": signature(cx.N), "H_c": signature(cx.Hc), "H_f": signature(hf), "H_cell": signature(cx.hess[0])},
        "eigenvector_sectors": sectors,
        "coordinate_invariance": coord_dev,
        "gauge": gauge,
        "gauge_quotient_spectrum_deviation": spectrum_deviation(ev, ev_quot),
        "identities": {"glue_sum_vs_Hf_rel": glue_residual, "schur_rel": schur_residual, "pullback_rel": pullback_residual,
                       "stationarity_rel": stationarity_residual},
        "coherent_rayleigh_max_dev": float(max(abs(r - 1.0) for r in coh_rayleigh)),
        "common_alpha_rayleigh": float((alpha @ cx.M @ alpha) / (alpha @ cx.N @ alpha)),
        "euler_bHb_minus_2S": euler,
        "so4_mismatch_norm": float(np.linalg.norm(so4_vec)),
        "n_centered_mu_identity_residual": mu_residual,
        "n_centered_spectrum": [float(z.real) for z in ev_c],
        "delta_linearization": lin,
        "chart_l2_audit": {"spectrum": [float(z.real) for z in ev_l2], "H_c_l2_signature": signature(hc_l2),
                           "note": "s=l² 차트 Hessian은 Euler 항등으로 스케일 모드를 죽인다 — 비선형 재매개화는 불변이 아님(scope)"},
        "rank_M": int(np.linalg.matrix_rank(cx.M, 1.0e-9)),
    }


def run_two_level() -> dict:
    """K1: 1->5->25. N₂ = 25 sub-sub-cell 등분할 Hessian, L₂ = 9-subcell 산술평균 (= L₁∘blockdiag L^{(a)})."""
    squared = np.full(10, 2.0)
    points = points_from_squared(squared)
    cells = refine(refine([tuple(BOUNDARY_VERTICES)], points), points)
    b0 = np.sqrt(squared)
    hc = richardson_hessian(RG.coarse_euclidean_regge_boundary_action, b0)
    cx = Complex(cells, points, hc)
    ev = pencil_eigs(cx.N, cx.M)
    cl = label_clusters(clusters(ev))
    ratios = {k: cl[k]["lambda2"] / LEVEL1_LAMBDA2[k] ** 2 for k in LEVEL1_LAMBDA2 if k in cl}
    lo, hi = KILL_K1_WINDOW
    return {
        "mode": "two_level", "cells": len(cells), "lambda2_clusters": cl, "lambda2_level1": LEVEL1_LAMBDA2,
        "lambda2_over_lambda1_squared": ratios, "max_imag_ratio": max_imag_ratio(ev),
        "signature_N": signature(cx.N), "rank_M": int(np.linalg.matrix_rank(cx.M, 1.0e-9)),
        "kill_window": list(KILL_K1_WINDOW), "reality_tol": KILL_REALITY_TOL,
        "killed": (len(ratios) != 3 or any(not (lo <= r <= hi) for r in ratios.values())
                   or max_imag_ratio(ev) > KILL_REALITY_TOL),
    }


def run_irregular(amplitude: float = 0.1) -> dict:
    """K2: 제곱 변 길이 2(1±0.1), 부호 = default_rng(20260902).choice([-1,1],10)."""
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1.0, 1.0], size=10)
    squared = 2.0 * (1.0 + amplitude * signs)
    cx, _, _ = level1_complex(squared)
    ev = pencil_eigs(cx.N, cx.M)
    lam2 = np.sort(ev.real)
    ratio_max = float(lam2[-1] / LEVEL1_LAMBDA2["[3,2]"])
    ratio_min = float(lam2[0] / LEVEL1_LAMBDA2["std"])
    n_gt1 = int(np.sum(lam2 > 1.0 + CLUSTER_TOL))
    n_lt1 = int(np.sum(lam2 < 1.0 - CLUSTER_TOL))
    lo, hi = KILL_K2_RATIO_WINDOW
    return {
        "mode": "irregular", "squared_lengths": squared.tolist(), "lambda2_spectrum": lam2.tolist(),
        "max_imag_ratio": max_imag_ratio(ev), "ratio_max": ratio_max, "ratio_min": ratio_min,
        "count_gt1": n_gt1, "count_lt1": n_lt1, "signature_N": signature(cx.N), "signature_Hc": signature(cx.Hc),
        "kill_ratio_window": list(KILL_K2_RATIO_WINDOW), "kill_count_gt1": list(KILL_K2_COUNT_GT1),
        "kill_count_lt1": list(KILL_K2_COUNT_LT1), "reality_tol": KILL_REALITY_TOL,
        "killed": (not (lo <= ratio_max <= hi) or not (lo <= ratio_min <= hi) or n_gt1 not in KILL_K2_COUNT_GT1
                   or n_lt1 not in KILL_K2_COUNT_LT1 or max_imag_ratio(ev) > KILL_REALITY_TOL),
    }


def run_coords() -> dict:
    """K3: 새 씨앗 3종의 임의 가역 선형 좌표 변환에서 펜슬 스펙트럼 불변."""
    cx, _, _ = level1_complex(np.full(10, 2.0))
    ev = pencil_eigs(cx.N, cx.M)
    devs = {}
    for seed in K3_SEEDS:
        P = random_coordinate_change(np.random.default_rng(seed), 50)
        devs[str(seed)] = {"deviation": spectrum_deviation(ev, pencil_eigs(P.T @ cx.N @ P, P.T @ cx.M @ P)),
                           "condition_number": float(np.linalg.cond(P))}
    worst = max(d["deviation"] for d in devs.values())
    return {"mode": "coords", "seeds": list(K3_SEEDS), "deviations": devs, "max_deviation": worst,
            "kill_tol": KILL_K3_TOL, "killed": worst > KILL_K3_TOL}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("predict", "two_level", "irregular", "coords"), default="predict")
    args = parser.parse_args()
    runner = {"predict": run_predict, "two_level": run_two_level, "irregular": run_irregular, "coords": run_coords}[args.mode]
    result = runner()
    out = HERE / ("predictions.json" if args.mode == "predict" else f"result_{args.mode}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
