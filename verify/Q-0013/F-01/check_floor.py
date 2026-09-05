"""Q-0013 F-01 사전등록 시험: 비등방 바닥 법칙

    eps(n)^2 = delta^4 [ (tr H kappa)^2 ||tl G(Sigma)||_F^2 + 2 ||H kappa H||_F^2 T(Sigma) ] / (12 n^2)

카드: derivations/Q-0013/F-01.formula.md.  자유 파라미터 0개 — 아래 PRED의 숫자는 모두
`--mode constants`의 정확 선형대수(구조상수 M^{ab})에서 나오며 MC 적합이 아니다.

사전등록 상수(결과를 본 뒤 바꾸지 않는다):
    DELTA = 0.005,  SIZES = (4, 8, 16, 32, 64),  TRIALS = 512,  SEED = 20260913
    영모드 시험만 delta in (0.005, 0.3)   — delta=0.3은 O(delta^2) 바닥 해석과 정확 영모드를 가른다.
    통계는 12.4 정규화 잔차 ||tl G||_F/||G||_F 의 trial RMS.

모드:
    constants  구조상수·정확 예측값(무작위 없음) -> structure_constants.json
    zero       K4: e_00 / e_11 단일 성분 라벨의 정확 영모드
    rank1      K1: (0,1) 단일 성분 라벨의 바닥 진폭과 sqrt(n^2-1)/n 형상
    iso        K2: Sigma = I_16 등방 라벨의 파라미터 없는 절대값 (eps_star = sqrt(10) delta^2)
    piso       K3: (0,1),(0,2),(0,3) 등가중 -> w=(1,1,1) -> 바닥 정확히 0
    axis       K5: (2,3) vs (0,1) 축 클래스 등가 (공통 난수)
    mix        K6: w=(2,1,1) -> n_x = 28 교차 곡선
    all        위 여섯 -> result.json
    smoke      작은 크기·적은 trial의 배관 점검(판정에 쓰지 않는다) -> smoke.json

사용: .claude\\hooks\\python.cmd python verify\\Q-0013\\F-01\\check_floor.py --mode all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
    wedge_scalar,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

OUT = ROOT / "verify" / "Q-0013" / "F-01"

# ---------------------------------------------------------------- 사전등록 상수
DELTA = 0.005
ZERO_DELTAS = (0.005, 0.3)
SIZES = (4, 8, 16, 32, 64)
TRIALS = 512
SEED = 20260913

# 카드의 사전등록 값과 창 (derivations/Q-0013/F-01.formula.md predicts/kill)
PRED = {
    "zero_max_residual": 0.0,
    "r01_eps64_over_delta2": 0.11783674,
    "r01_ratio_64_over_4": 1.03266948,
    "iso_eps64_over_delta2": 0.39218439,
    "piso_eps64_over_delta2": 0.05660694,
    "piso_slope": -0.45342619,
    "axis_ratio_23_over_01": 1.0,
    "mix_eps64_over_delta2": 0.13865812,
    "mix_slope": -0.25640403,
}
WINDOW = {
    "zero_max_residual": (-1.0e-12, 1.0e-12),
    "r01_eps64_over_delta2": (0.10370, 0.13197),
    "r01_ratio_64_over_4": (0.899, 1.167),
    "iso_eps64_over_delta2": (0.34512, 0.43925),
    "piso_eps64_over_delta2": (0.04981, 0.06340),
    "piso_slope": (-0.563, -0.343),
    "axis_ratio_23_over_01": (0.95, 1.05),
    "mix_eps64_over_delta2": (0.12202, 0.15530),
    "mix_slope": (-0.366, -0.146),
}

REF = geometric_self_dual_triple(np.eye(4))
NORM_G0 = 2.0 * math.sqrt(3.0)  # ||G(Sigma_0)||_F


# ---------------------------------------------------------------- 블록 잔차
def cell(label: np.ndarray, delta: float) -> np.ndarray:
    """polar 정렬된 한 cell의 자기쌍대 삼중항 (F-02 check_modes와 같은 규약)."""
    triple = geometric_self_dual_triple(np.eye(4) + delta * label)
    return optimal_internal_alignment(REF, triple).aligned_candidate


def block_residual(labels: np.ndarray, delta: float = DELTA) -> float:
    return simplicity_residual(sum(cell(lab, delta) for lab in labels))


def rms(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def loglog_slope(sizes, values) -> float:
    return float(np.polyfit(np.log(np.asarray(sizes, float)), np.log(np.asarray(values, float)), 1)[0])


# ---------------------------------------------------------------- 구조상수 (정확)
def traceless(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.trace(matrix) / 3.0 * np.eye(3)


def gram(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array([[wedge_scalar(first[i], second[j]) for j in range(3)] for i in range(3)])


def aligned_derivative(label: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    """L~(label) = d/d delta [ R_delta Sigma(I + delta label) ] at 0 (중심차분, 오차 O(step^2))."""
    return (cell(label, step) - cell(label, -step)) / (2.0 * step)


def structure_constants() -> tuple[np.ndarray, np.ndarray]:
    basis = []
    for a in range(16):
        e = np.zeros((4, 4))
        e[a // 4, a % 4] = 1.0
        basis.append(e)
    L = [aligned_derivative(e) for e in basis]
    M = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            g = gram(L[a], L[b])
            M[a, b] = 0.5 * (g + g.T)
    Mt = np.array([[traceless(M[a, b]) for b in range(16)] for a in range(16)])
    return M, Mt


def floor_amplitude(sigma: np.ndarray, M: np.ndarray) -> float:
    """F = ||tl G(Sigma)||_F."""
    return float(np.linalg.norm(traceless(np.einsum("ab,abij->ij", sigma, M))))


def fluctuation_amplitude(sigma: np.ndarray, Mt: np.ndarray) -> float:
    """T(Sigma) = sum_ij sum_abcd tlM^ab_ij Sigma_ac Sigma_bd tlM^cd_ij (Wick 4점)."""
    return float(np.einsum("abij,ac,bd,cdij->", Mt, sigma, sigma, Mt))


def predicted_eps_over_delta2(n: int, F: float, T: float) -> float:
    """i.i.d. cell(kappa=I): tr(H)=||HIH||_F^2=n-1."""
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


# ---------------------------------------------------------------- 라벨 생성기
def labels_component(rng, n: int, comps, scales) -> np.ndarray:
    lab = np.zeros((n, 4, 4))
    for (mu, nu), s in zip(comps, scales):
        lab[:, mu, nu] = s * rng.normal(size=n)
    return lab


# ---------------------------------------------------------------- 모드
def mode_constants() -> dict:
    M, Mt = structure_constants()
    axis_of = {}
    norms = {}
    for a in range(16):
        t = traceless(M[a, a])
        norms[f"{a // 4}{a % 4}"] = float(np.linalg.norm(t))
        axis_of[f"{a // 4}{a % 4}"] = int(np.argmax(np.diag(t))) + 1 if np.linalg.norm(t) > 1e-9 else 0
    K = np.einsum("abij,abij->ab", Mt, Mt)

    def sigma_from(comps, scales):
        s = np.zeros((16, 16))
        for (mu, nu), sc in zip(comps, scales):
            a = 4 * mu + nu
            s[a, a] = sc * sc
        return s

    cases = {
        "zero_00": sigma_from([(0, 0)], [1.0]),
        "zero_11": sigma_from([(1, 1)], [1.0]),
        "rank1_01": sigma_from([(0, 1)], [1.0]),
        "rank1_23": sigma_from([(2, 3)], [1.0]),
        "iso": np.eye(16),
        "piso": sigma_from([(0, 1), (0, 2), (0, 3)], [1.0, 1.0, 1.0]),
        "mix": sigma_from([(0, 1), (0, 2), (0, 3)], [math.sqrt(2.0), 1.0, 1.0]),
    }
    exact = {}
    for name, sigma in cases.items():
        F = floor_amplitude(sigma, M)
        T = fluctuation_amplitude(sigma, Mt)
        exact[name] = {
            "F": F,
            "T": T,
            "floor_over_delta2": F / NORM_G0,
            "eps_star_over_delta2": math.sqrt(2.0 * T) / NORM_G0,
            "n_cross": (1.0 + 2.0 * T / (F * F)) if F > 1e-9 else None,
            "eps_over_delta2": {str(n): predicted_eps_over_delta2(n, F, T) for n in SIZES},
            "slope": (
                loglog_slope(SIZES, [max(predicted_eps_over_delta2(n, F, T), 1e-300) for n in SIZES])
                if (F > 1e-9 or T > 1e-9)
                else None
            ),
        }

    # SO(3) 공변성: l -> R~ l R~^T (R~ = diag(1,R)) 아래 V(Sigma) = 2 tl G(Sigma) 가 R V R^T 로 간다
    rng = np.random.default_rng(SEED)
    cov_err = 0.0
    for _ in range(10):
        A = rng.normal(size=(3, 3))
        Q, R_ = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R_)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        Rt = np.eye(4)
        Rt[1:, 1:] = Q
        P = np.kron(Rt, Rt)
        sigma = cases["mix"]
        V = 2.0 * traceless(np.einsum("ab,abij->ij", sigma, M))
        Vr = 2.0 * traceless(np.einsum("ab,abij->ij", P @ sigma @ P.T, M))
        cov_err = max(cov_err, float(np.linalg.norm(Vr - Q @ V @ Q.T)))

    result = {
        "norm_G0": NORM_G0,
        "tl_M_aa_norm": norms,
        "axis_class": axis_of,
        "sum_a_tl_M_aa_norm": float(np.linalg.norm(sum(traceless(M[a, a]) for a in range(16)))),
        "sum_ab_K": float(K.sum()),
        "eps_star_isotropic_over_delta2": math.sqrt(2.0 * 60.0) / NORM_G0,
        "so3_covariance_max_error": cov_err,
        "exact": exact,
    }
    (OUT / "structure_constants.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def mode_zero(sizes, trials) -> dict:
    worst = 0.0
    table = {}
    for delta in ZERO_DELTAS:
        for comp in ((0, 0), (1, 1)):
            rng = np.random.default_rng(SEED + 101 * comp[0] + 7 * comp[1])
            for n in sizes:
                vals = [
                    abs(block_residual(labels_component(rng, n, [comp], [1.0]), delta))
                    for _ in range(trials)
                ]
                key = f"d{delta}_e{comp[0]}{comp[1]}_n{n}"
                table[key] = {"rms": rms(vals), "max": float(max(vals))}
                worst = max(worst, float(max(vals)))
    return {"zero_max_residual": worst, "zero_table": table}


def _sweep(rng_seed: int, sizes, trials, comps, scales) -> dict:
    rng = np.random.default_rng(rng_seed)
    out = {}
    for n in sizes:
        vals = [block_residual(labels_component(rng, n, comps, scales)) for _ in range(trials)]
        out[n] = rms(vals)
    return out


def mode_rank1(sizes, trials) -> dict:
    curve = _sweep(SEED + 1, sizes, trials, [(0, 1)], [1.0])
    top, bot = max(sizes), min(sizes)
    return {
        "r01_curve_over_delta2": {str(n): v / DELTA**2 for n, v in curve.items()},
        "r01_eps64_over_delta2": curve[top] / DELTA**2,
        "r01_ratio_64_over_4": curve[top] / curve[bot],
        "r01_slope": loglog_slope(sizes, [curve[n] for n in sizes]),
    }


def mode_iso(sizes, trials) -> dict:
    rng = np.random.default_rng(SEED + 2)
    curve = {}
    for n in sizes:
        curve[n] = rms([block_residual(rng.normal(size=(n, 4, 4))) for _ in range(trials)])
    top = max(sizes)
    return {
        "iso_curve_over_delta2": {str(n): v / DELTA**2 for n, v in curve.items()},
        "iso_eps64_over_delta2": curve[top] / DELTA**2,
        "iso_slope": loglog_slope(sizes, [curve[n] for n in sizes]),
    }


def mode_piso(sizes, trials) -> dict:
    curve = _sweep(SEED + 3, sizes, trials, [(0, 1), (0, 2), (0, 3)], [1.0, 1.0, 1.0])
    top = max(sizes)
    return {
        "piso_curve_over_delta2": {str(n): v / DELTA**2 for n, v in curve.items()},
        "piso_eps64_over_delta2": curve[top] / DELTA**2,
        "piso_slope": loglog_slope(sizes, [curve[n] for n in sizes]),
    }


def mode_axis(sizes, trials) -> dict:
    """공통 난수: 같은 진폭 표본을 (0,1)과 (2,3)에 각각 실어 비를 만든다."""
    v01, v23 = [], []
    rng = np.random.default_rng(SEED + 4)
    for n in sizes:
        a, b = [], []
        for _ in range(trials):
            g = rng.normal(size=n)
            lab1 = np.zeros((n, 4, 4))
            lab1[:, 0, 1] = g
            lab2 = np.zeros((n, 4, 4))
            lab2[:, 2, 3] = g
            a.append(block_residual(lab1))
            b.append(block_residual(lab2))
        v01.append(rms(a))
        v23.append(rms(b))
    return {
        "axis_01_over_delta2": {str(n): v / DELTA**2 for n, v in zip(sizes, v01)},
        "axis_23_over_delta2": {str(n): v / DELTA**2 for n, v in zip(sizes, v23)},
        "axis_ratio_23_over_01": rms(v23) / rms(v01),
    }


def mode_mix(sizes, trials) -> dict:
    curve = _sweep(SEED + 5, sizes, trials, [(0, 1), (0, 2), (0, 3)], [math.sqrt(2.0), 1.0, 1.0])
    top = max(sizes)
    return {
        "mix_curve_over_delta2": {str(n): v / DELTA**2 for n, v in curve.items()},
        "mix_eps64_over_delta2": curve[top] / DELTA**2,
        "mix_slope": loglog_slope(sizes, [curve[n] for n in sizes]),
    }


RUNNERS = {
    "zero": mode_zero,
    "rank1": mode_rank1,
    "iso": mode_iso,
    "piso": mode_piso,
    "axis": mode_axis,
    "mix": mode_mix,
}


def verdict(stats: dict) -> dict:
    fired = []
    for key, (low, high) in WINDOW.items():
        if key in stats and not (low <= float(stats[key]) <= high):
            fired.append(key)
    return {"kill_fired": fired, "status": "refuted" if fired else "consistent"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["constants", "all", "smoke", *RUNNERS.keys()])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode == "constants":
        result = mode_constants()
        print(json.dumps(result["exact"], ensure_ascii=False, indent=2)[:2000])
        print("sum_ab K =", result["sum_ab_K"], " sum_a tl M^aa =", result["sum_a_tl_M_aa_norm"],
              " SO(3) cov err =", result["so3_covariance_max_error"])
        return 0

    if args.mode == "smoke":
        sizes, trials = (4, 8), 48
        stats = {}
        stats.update(mode_zero(sizes, 16))
        stats.update(mode_rank1(sizes, trials))
        stats.update(mode_iso(sizes, trials))
        stats.update(mode_piso(sizes, trials))
        stats.update(mode_axis(sizes, trials))
        stats.update(mode_mix(sizes, trials))
        # 사전등록 키(…_eps64_…)는 크기 64를 뜻하므로 smoke에서는 이름을 바꿔 오독을 막는다
        stats = {("smoke_" + k.replace("eps64", f"eps{max(sizes)}")): v for k, v in stats.items()}
        payload = {
            "note": "SMOKE ONLY — 사전등록 크기(SIZES/TRIALS/SEED)가 아니므로 kill 판정에 쓰지 않는다",
            "sizes": list(sizes), "trials": trials, "delta": DELTA,
            "stats": stats,
            "predicted_at_smoke_sizes": {
                "rank1": {str(n): predicted_eps_over_delta2(n, math.sqrt(1 / 6), 1 / 6) for n in sizes},
                "iso": {str(n): predicted_eps_over_delta2(n, 0.0, 60.0) for n in sizes},
                "piso": {str(n): predicted_eps_over_delta2(n, 0.0, 1.25) for n in sizes},
                "mix": {str(n): predicted_eps_over_delta2(n, math.sqrt(1 / 6), 2.25) for n in sizes},
            },
        }
        (OUT / "smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload["stats"], ensure_ascii=False, indent=2)[:3000])
        return 0

    modes = list(RUNNERS) if args.mode == "all" else [args.mode]
    stats = {}
    for name in modes:
        stats.update(RUNNERS[name](SIZES, TRIALS))
    payload = {
        "card": "derivations/Q-0013/F-01.formula.md",
        "modes": modes,
        "delta": DELTA, "sizes": list(SIZES), "trials": TRIALS, "seed": SEED,
        "predicted": PRED, "window": WINDOW,
        "stats": stats,
        "verdict": verdict(stats),
    }
    path = OUT / "result.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(old.get("stats", {}))
        merged.update(stats)
        payload["stats"] = merged
        payload["modes"] = sorted(set(old.get("modes", [])) | set(modes))
        payload["verdict"] = verdict(merged)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"],
                      "stats": {k: v for k, v in payload["stats"].items() if k in WINDOW}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
