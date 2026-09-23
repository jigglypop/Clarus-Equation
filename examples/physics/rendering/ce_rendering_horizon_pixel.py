"""지평선의 화소 — 나이퀴스트 화소 하나에 1 nat. 원장: 43장 §43.71. 예측값을 바꾸지 않는다.

§43.33 반례: 넓이 칸 aℓ_P²마다 정수 d 상태면 ln d = a/4, a = 4에서 d = e(정수 아님). 계산 전에 적은 판본과 규칙:

HP(가설, 목격 후). (N) 해상도 ℓ_P의 화면이 독립으로 가르는 가장 작은 무늬는 두 칸(나이퀴스트): 화소 (2ℓ_P)².
   (E) 화소마다 기록 시계 하나. 평균 한 틱만 아는 양의 시계의 최대 엔트로피 분포는 지수분포이고 엔트로피는 정확히 1 nat.
   S = A/(2ℓ_P)² × 1 nat = A/4ℓ_P². §43.33의 d = e는 개수가 아니라 e^{1 nat}이다.
유일성(목격 후, 증거 아님): 화소 변 {1, 2}ℓ_P × 화소당 엔트로피 {ln 2, 1, ln 3, H(Poisson(D)), ½ln(2πe)}에서 정확히 1/4인 조합.
(E) 검산: 평균 1인 양의 분포(지수, 반정규, 균등[0,2], 감마(2))의 미분 엔트로피 비교.
따름 ①(이론): 질량 고정 = 시계 합 고정(제약 하나). N개 지수 시계의 합이 N인 단체의 엔트로피 ln(N^{N−1}/(N−1)!)
   = N − ½ln(2πN) + …이므로 S = A/4 − ½ ln A + 상수. 루프 양자중력은 −3/2.
따름 ②(관측): 넓이 양자 ΔA = 4ℓ_P²(Bekenstein–Mukhanov α = 4) → 선 간격 ω = α/(32πM) = 1/(8πM).
   Laghi+ 2021(CQG 38, 095005): α = 15.6 (+20.5 −13.3), 로그 승산비 0.1 ± 0.6(무정보). kill: 링다운이 α = 4를 3σ 넘게 배제.

python -B -m examples.physics.rendering.ce_rendering_horizon_pixel
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import gammaln

from examples.physics.rendering import ce_rendering_registry as R

QNM_220 = 0.3737
LAGHI_ALPHA = (15.6, 13.3, 20.5)


def _poisson_entropy(lam: float, kmax: int = 200) -> float:
    k = np.arange(kmax)
    p = stats.poisson.pmf(k, lam)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def uniqueness() -> dict:
    d = R.core(R.calibrated_alpha_s()[0])["D"]
    ent = {"bit ln2": math.log(2), "nat (exp clock)": 1.0, "ln3": math.log(3),
           "H(Poisson(D))": _poisson_entropy(d), "gauss clock 1/2 ln(2 pi e)": 0.5 * math.log(2 * math.pi * math.e)}
    rows = {f"side {s} | {k}": v / s ** 2 for s in (1, 2) for k, v in ent.items()}
    exact = sorted(k for k, v in rows.items() if abs(v - 0.25) < 1e-12)
    return {"rows": rows, "exact_quarter": exact}


def clock_entropies() -> dict:
    """평균 1인 양의 분포의 미분 엔트로피: 지수가 최대(= 1 nat)."""
    return {"exponential(1)": float(stats.expon(scale=1).entropy()),
            "half-normal(mean 1)": float(stats.halfnorm(scale=math.sqrt(math.pi / 2)).entropy()),
            "uniform[0,2]": float(stats.uniform(0, 2).entropy()),
            "gamma(2, mean 1)": float(stats.gamma(2, scale=0.5).entropy())}


def log_correction(ns=(10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6)) -> dict:
    """제약 하나(시계 합 = N)의 단체 엔트로피 − N을 ln N에 대해 맞춘 기울기."""
    x = np.log(np.array(ns, dtype=float))
    y = np.array([(n - 1) * math.log(n) - gammaln(n) - n for n in ns])
    slope = float(np.polyfit(x, y, 1)[0])
    return {"slope_vs_lnN": slope, "CE": -0.5, "LQG": -1.5}


def ringdown() -> dict:
    alpha = 4.0
    spacing = alpha / (32 * math.pi)
    n_near = QNM_220 / spacing
    a0, lo, hi = LAGHI_ALPHA
    return {"alpha": alpha, "line_spacing_M_omega": spacing, "QNM_in_units_of_spacing": n_near,
            "laghi_interval": (a0 - lo, a0 + hi), "alpha_inside": a0 - lo <= alpha <= a0 + hi,
            "excluded": False}


def main() -> None:
    u = uniqueness()
    print("exact 1/4:", u["exact_quarter"])
    print({k: round(v, 4) for k, v in u["rows"].items()})
    print("clock entropies:", {k: round(v, 4) for k, v in clock_entropies().items()})
    print("log correction:", log_correction())
    print("ringdown:", ringdown())


if __name__ == "__main__":
    main()
