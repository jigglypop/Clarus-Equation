"""Q-0019 F-01: Euclidean Regge 1->5 블록 위 공통 계량 고정점의 선형화 RG 사상 L과 증폭률 λ.

카드(derivations/Q-0019/F-01.formula.md)가 선언한 규약을 그대로 계산한다.

  * 배경: 제곱 변 길이 2의 regular 4-simplex(무게중심 원점), barycentric 1->5 분할, 다섯 fine cell a.
  * 섭동: cell a에 선형 사상 e_a = I + δ X_a (ambient 정규직교 좌표), 계량 g_a = e_aᵀ e_a.
    Regge 길이가 보는 것은 g_a뿐이므로(so(4) 회전은 정확히 불가시) mismatch 좌표는
    M_a := sym 편차의 중심화 값 (Sym(4), 다섯 cell 합 0; 40차원).
  * coarse 사상: 경계 변 (i,j)의 제곱 길이 = 그 변을 품는 fine cell 셋(a∉{i,j})의 u_ijᵀ g_a u_ij 평균;
    내부 길이는 flat section(26장 식 5–7, Schur 소거와 동치)으로 결정되므로 경계 사상에 나타나지 않는다.
    10개 경계 제곱 길이 -> coarse Gram G (10×10 선형 solve) -> coarse 편차 G−I.
    이 파이프라인은 g에 대해 정확히 선형이므로 L: R^50 -> R^10을 기저마다 계산한다.
  * λ(M) := ‖L M‖_F / ‖M‖_rms,  ‖M‖_rms = sqrt(mean_a ‖M_a‖_F²).
    λ_iso² = 5·‖L H‖_F²/40 (40차원 단위구 균등 평균의 정확식), λ_max = √5·σ_max(L H),
    λ_coh = 1 (코히런트 입력 M_a ≡ S 는 항등), λ_scale = std⊗scale 블록, 게이지 4방향 = 0, 공통 궤도 = 0.
  * 12.4 규약의 Plebanski 잔차(자기쌍대 삼중항 Σ(e), 극 정렬, 무흔적 gram)는 recovers·K3에 쓴다:
    쌍 잔차 ε12(ref, cand) = ε_simp(ref + R·cand).

모드: predict(카드 숫자, 실행됨) · two_level(K1) · irregular(K2) · delta(K3).
K1·K2·K3 모드는 카드 작성 시점에 실행하지 않았다(사전등록). 씨앗 20260902, numpy만.
출력: verify/Q-0019/F-01/predictions.json (predict) 또는 result_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20260902
MC_SAMPLES = 20000
EDGES = tuple(combinations(range(5), 2))
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# ---------------------------------------------------------------- Sym(4) 정규직교 기저 (Frobenius)
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


def vec_to_sym(v: np.ndarray) -> np.ndarray:
    return sum(float(c) * b for c, b in zip(v, BASIS))


# ---------------------------------------------------------------- 기하
def regular_simplex_vertices() -> np.ndarray:
    """제곱 변 길이 2, 무게중심 원점인 regular 4-simplex의 정점 (5×4)."""
    e = np.eye(5) - 1.0 / 5.0
    q, _ = np.linalg.qr(e[:, :4])
    return e @ q


def vertices_from_squared_lengths(squared: np.ndarray) -> np.ndarray:
    """10개 제곱 변 길이(EDGES 순서)에서 무게중심 원점의 정점 좌표 (5×4)."""
    d = np.zeros((5, 5))
    for k, (i, j) in enumerate(EDGES):
        d[i, j] = d[j, i] = squared[k]
    gram = np.array([[0.5 * (d[0, i] + d[0, j] - d[i, j]) for j in range(1, 5)] for i in range(1, 5)])
    lower = np.linalg.cholesky(gram)
    verts = np.vstack((np.zeros(4), lower))
    return verts - verts.mean(axis=0)


def sub_cells(parent: np.ndarray) -> list[np.ndarray]:
    """barycentric 분할의 다섯 sub-cell: [중심점, 부모 정점 j≠a]."""
    c = parent.mean(axis=0)
    return [np.vstack((c, parent[[j for j in range(5) if j != a]])) for a in range(5)]


def solve_sym_from_edges(parent: np.ndarray, ell2: np.ndarray) -> np.ndarray:
    """u_eᵀ G u_e = ell2_e (10개 경계 변)를 만족하는 G ∈ Sym(4)."""
    a = np.array([[float((parent[i] - parent[j]) @ b @ (parent[i] - parent[j])) for b in BASIS] for i, j in EDGES])
    return vec_to_sym(np.linalg.solve(a, ell2))


def coarse_metric(parent: np.ndarray, metrics: list[np.ndarray]) -> np.ndarray:
    """다섯 sub-cell 계량(ambient Sym(4)) -> 3-cell 평균 경계 길이 -> coarse 계량."""
    ell2 = np.empty(10)
    for k, (i, j) in enumerate(EDGES):
        u = parent[i] - parent[j]
        ell2[k] = np.mean([float(u @ metrics[a] @ u) for a in range(5) if a not in (i, j)])
    return solve_sym_from_edges(parent, ell2)


def linear_map(parent: np.ndarray) -> np.ndarray:
    """L: R^50(다섯 cell의 Sym(4) 편차) -> R^10(coarse Sym(4) 편차). 파이프라인이 정확 선형이므로 기저로 계산."""
    cols = []
    for a in range(5):
        for b in BASIS:
            metrics = [np.zeros((4, 4)) for _ in range(5)]
            metrics[a] = b
            cols.append(sym_to_vec(coarse_metric(parent, metrics)))
    return np.asarray(cols).T


def centering(n: int) -> np.ndarray:
    return np.kron(np.eye(n) - np.ones((n, n)) / n, np.eye(10))


def lambda_stats(l: np.ndarray, n: int) -> dict:
    lh = l @ centering(n)
    sv = np.linalg.svd(lh, compute_uv=False)
    iso = math.sqrt(n * float(np.sum(sv**2)) / (10.0 * (n - 1)))
    return {
        "lambda_iso": iso,
        "lambda_max": math.sqrt(n) * float(sv[0]),
        "singular_values_times_sqrt_n": [math.sqrt(n) * float(s) for s in sv],
        "gamma_geo": math.log(iso) / math.log(n),
    }


# ---------------------------------------------------------------- 12.4 Plebanski 잔차 (numpy 자급)
def two_form(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray([a[i] * b[j] - a[j] * b[i] for i, j in PAIRS])


def wedge_scalar(f: np.ndarray, g: np.ndarray) -> float:
    # e0123 계수: F01G23 − F02G13 + F03G12 + F12G03 − F13G02 + F23G01
    return float(f[0] * g[5] - f[1] * g[4] + f[2] * g[3] + f[3] * g[2] - f[4] * g[1] + f[5] * g[0])


def self_dual_triple(tetrad: np.ndarray) -> np.ndarray:
    """Σ^i(e) = e^0∧e^i + ½ ε_ijk e^j∧e^k, 행이 1-형식."""
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    out = []
    for i in range(3):
        form = two_form(tetrad[0], tetrad[i + 1])
        for j in range(3):
            for k in range(3):
                if eps[i, j, k]:
                    form = form + 0.5 * eps[i, j, k] * two_form(tetrad[j + 1], tetrad[k + 1])
        out.append(form)
    return np.asarray(out)


def plebanski_gram(triple: np.ndarray) -> np.ndarray:
    return np.array([[wedge_scalar(triple[i], triple[j]) for j in range(3)] for i in range(3)])


def simplicity_residual(triple: np.ndarray) -> float:
    gram = plebanski_gram(triple)
    traceless = gram - np.trace(gram) / 3.0 * np.eye(3)
    return float(np.linalg.norm(traceless) / np.linalg.norm(gram))


def polar_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    cross = np.array([[wedge_scalar(reference[i], candidate[j]) for j in range(3)] for i in range(3)])
    left, _, right_t = np.linalg.svd(cross)
    rot = left @ right_t
    if np.linalg.det(rot) < 0.0:
        left[:, -1] *= -1.0
        rot = left @ right_t
    return rot @ candidate


def pair_residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    return simplicity_residual(reference + polar_align(reference, candidate))


def tetrad_from_metric(g: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(g).T  # eᵀe = g


def cayley_rotation(a: np.ndarray) -> np.ndarray:
    """반대칭 a에서 정확 직교 행렬 (I−a/2)^{-1}(I+a/2)."""
    return np.linalg.solve(np.eye(4) - 0.5 * a, np.eye(4) + 0.5 * a)


# ---------------------------------------------------------------- 비선형 파이프라인 (K3·recovers)
def nonlinear_block(parent: np.ndarray, tetrads: list[np.ndarray]) -> dict:
    metrics = [e.T @ e for e in tetrads]
    g_c = coarse_metric(parent, metrics)
    g_bar = sum(metrics) / 5.0
    ref = self_dual_triple(tetrad_from_metric(g_bar))
    fine = [pair_residual(ref, self_dual_triple(e)) for e in tetrads]
    coarse = pair_residual(ref, self_dual_triple(tetrad_from_metric(g_c)))
    dev = [m - g_bar for m in metrics]
    return {
        "coarse_metric": g_c,
        "eps12_fine_rms": math.sqrt(float(np.mean(np.square(fine)))),
        "eps12_coarse": coarse,
        "metric_fine_rms": math.sqrt(float(np.mean([np.sum(d * d) for d in dev]))),
        "metric_coarse": float(np.linalg.norm(g_c - g_bar)),
    }


def gauge_directions(parent: np.ndarray) -> list[list[np.ndarray]]:
    """내부점 이동 q(4방향)에 대응하는 정확 cell 사상 e_a = I − q w_aᵀ, w_a = E_a^{-1} 1."""
    cells = sub_cells(parent)
    dirs = []
    for k in range(4):
        q = np.zeros(4)
        q[k] = 0.1
        tetrads = []
        for cell in cells:
            rows = cell[1:] - cell[0]
            w = np.linalg.solve(rows, np.ones(4))
            tetrads.append(np.eye(4) - np.outer(q, w))
        dirs.append(tetrads)
    return dirs


# ---------------------------------------------------------------- 모드
def run_predict() -> dict:
    parent = regular_simplex_vertices()
    l = linear_map(parent)
    h = centering(5)
    stats = lambda_stats(l, 5)
    # 코히런트 항등: L(1⊗S) = S
    coh = max(float(np.linalg.norm(l @ np.tile(sym_to_vec(b), 5) - sym_to_vec(b))) for b in BASIS)
    # 스케일 mismatch 블록: x = c⊗I_4, Σc=0
    c = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    x = np.kron(c, sym_to_vec(np.eye(4)))
    lam_scale = math.sqrt(5.0) * float(np.linalg.norm(l @ x) / np.linalg.norm(x))
    # 스펙트럼 군집(중복도)
    sv = np.asarray(stats["singular_values_times_sqrt_n"])
    clusters: list[list[float]] = []
    for s in sv:
        if clusters and abs(clusters[-1][0] - s) < 1.0e-9:
            clusters[-1].append(float(s))
        else:
            clusters.append([float(s)])
    # 게이지 4방향: 선형 L 위 0, 비선형 경계 정확 불변
    gauge_lin = 0.0
    gauge_nl = 0.0
    for tetrads in gauge_directions(parent):
        x = np.concatenate([sym_to_vec(0.5 * (e + e.T) - np.eye(4)) for e in tetrads])
        gauge_lin = max(gauge_lin, float(np.linalg.norm(l @ h @ x)))
        blk = nonlinear_block(parent, tetrads)
        gauge_nl = max(gauge_nl, float(np.linalg.norm(blk["coarse_metric"] - np.eye(4))), blk["eps12_coarse"])
    # 공통 궤도(공통 α, cell별 so(4) 회전): 분자·분모 모두 0 (13.3)
    rng = np.random.default_rng(SEED)
    alpha = 1.3
    tetrads = []
    for _ in range(5):
        a = rng.normal(size=(4, 4))
        a = 0.2 * (a - a.T)
        tetrads.append(alpha * cayley_rotation(a))
    orbit = nonlinear_block(parent, tetrads)
    # MC 등방 평균 (trace 정확식 확인)
    xs = rng.normal(size=(MC_SAMPLES, 50)) @ h.T
    num = np.linalg.norm(xs @ l.T, axis=1)
    den = np.linalg.norm(xs, axis=1) / math.sqrt(5.0)
    lam = num / den
    return {
        "card": "Q-0019 F-01",
        "seed": SEED,
        "convention": "λ = ‖L M‖_F / ‖M‖_rms, M = 중심화 Sym(4) 편차, 경계 제곱길이 = 3-cell 평균, 내부 길이 = flat section",
        "lambda_iso": stats["lambda_iso"],
        "lambda_iso_squared": stats["lambda_iso"] ** 2,
        "lambda_max": stats["lambda_max"],
        "lambda_max_squared": stats["lambda_max"] ** 2,
        "lambda_coh": 1.0,
        "coherent_identity_residual": coh,
        "lambda_scale": lam_scale,
        "lambda_scale_squared": lam_scale**2,
        "gamma_geo": stats["gamma_geo"],
        "spectrum_sqrt5_sigma": stats["singular_values_times_sqrt_n"],
        "spectrum_clusters": [{"value": cl[0], "value_squared": cl[0] ** 2, "multiplicity": len(cl)} for cl in clusters],
        "rank": int(np.sum(sv > 1.0e-9)),
        "gauge_residual_linear": gauge_lin,
        "gauge_residual_nonlinear": gauge_nl,
        "orbit_recovers": {
            "metric_fine_rms": orbit["metric_fine_rms"],
            "metric_coarse": orbit["metric_coarse"],
            "eps12_fine_rms": orbit["eps12_fine_rms"],
            "eps12_coarse": orbit["eps12_coarse"],
        },
        "mc_iso": {
            "samples": MC_SAMPLES,
            "rms_lambda": float(np.sqrt(np.mean(lam**2))),
            "mean_lambda": float(np.mean(lam)),
            "max_lambda": float(np.max(lam)),
        },
    }


def run_two_level() -> dict:
    """K1: 1->5->25. L2 = L1 ∘ blockdiag(L^{(a)}), 중간 중심화 없음, λ2 = λ_iso(L2, 25)."""
    parent = regular_simplex_vertices()
    l1 = linear_map(parent)
    blocks = [linear_map(cell) for cell in sub_cells(parent)]
    l2 = np.zeros((10, 250))
    for a, la in enumerate(blocks):
        l2 += l1[:, 10 * a : 10 * a + 10] @ np.pad(la, ((0, 0), (50 * a, 250 - 50 * a - 50)))
    s1 = lambda_stats(l1, 5)
    s2 = lambda_stats(l2, 25)
    sub_iso = [lambda_stats(la, 5)["lambda_iso"] for la in blocks]
    return {
        "mode": "two_level",
        "lambda_1": s1["lambda_iso"],
        "lambda_2": s2["lambda_iso"],
        "lambda_2_over_lambda_1_squared": s2["lambda_iso"] / s1["lambda_iso"] ** 2,
        "lambda_max_2": s2["lambda_max"],
        "sub_cell_lambda_iso": sub_iso,
        "gamma_geo_2": math.log(s2["lambda_iso"]) / math.log(25.0),
        "kill_window_lambda2_over_lambda1_squared": [0.75, 1.25],
    }


def run_irregular() -> dict:
    """K2: 제곱 변 길이 2(1±0.1), 부호는 seed 20260902의 처음 10개 choice."""
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1.0, 1.0], size=10)
    squared = 2.0 * (1.0 + 0.1 * signs)
    parent = vertices_from_squared_lengths(squared)
    reg = lambda_stats(linear_map(regular_simplex_vertices()), 5)
    irr = lambda_stats(linear_map(parent), 5)
    return {
        "mode": "irregular",
        "signs": signs.tolist(),
        "squared_lengths": squared.tolist(),
        "lambda_iso_regular": reg["lambda_iso"],
        "lambda_iso_irregular": irr["lambda_iso"],
        "lambda_max_regular": reg["lambda_max"],
        "lambda_max_irregular": irr["lambda_max"],
        "ratio_iso": irr["lambda_iso"] / reg["lambda_iso"],
        "ratio_max": irr["lambda_max"] / reg["lambda_max"],
        "sign_flip_iso": (irr["lambda_iso"] < 1.0) != (reg["lambda_iso"] < 1.0),
        "spectrum_irregular": irr["singular_values_times_sqrt_n"],
        "kill_window_ratio": [0.8, 1.25],
    }


def run_delta() -> dict:
    """K3: 12.4 쌍 잔차 비 λ12(δ)의 δ 수렴, 64 방향, δ∈{0.02,0.01,0.005}."""
    parent = regular_simplex_vertices()
    rng = np.random.default_rng(SEED)
    h = centering(5)
    deltas = (0.02, 0.01, 0.005)
    table = []
    for _ in range(64):
        x = h @ rng.normal(size=50)
        x = x / (np.linalg.norm(x) / math.sqrt(5.0))
        row = {}
        for d in deltas:
            tetrads = [np.eye(4) + d * vec_to_sym(x[10 * a : 10 * a + 10]) for a in range(5)]
            blk = nonlinear_block(parent, tetrads)
            row[str(d)] = blk["eps12_coarse"] / blk["eps12_fine_rms"]
        table.append(row)
    change_fine = [abs(r["0.01"] / r["0.005"] - 1.0) for r in table]
    change_coarse = [abs(r["0.02"] / r["0.01"] - 1.0) for r in table]
    return {
        "mode": "delta",
        "directions": 64,
        "deltas": list(deltas),
        "max_rel_change_0p01_to_0p005": max(change_fine),
        "max_rel_change_0p02_to_0p01": max(change_coarse),
        "lambda12_mean_at_0p005": float(np.mean([r["0.005"] for r in table])),
        "lambda12_max_at_0p005": float(np.max([r["0.005"] for r in table])),
        "kill_threshold": 0.01,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("predict", "two_level", "irregular", "delta"), default="predict")
    args = parser.parse_args()
    runner = {"predict": run_predict, "two_level": run_two_level, "irregular": run_irregular, "delta": run_delta}[args.mode]
    result = runner()
    out = HERE / ("predictions.json" if args.mode == "predict" else f"result_{args.mode}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
