"""질량 = 빠르기(MR)의 독립 시험 — 경주 사다리의 다음 칸. 원장: 43장 §43.68. 예측값을 바꾸지 않는다.

MR(§43.67): 경주로 이어진 두 상태의 질량비는 기록 빠르기의 비(확률)다. 계산 전에 적은 판본과 규칙:

범위 확인(구성상, 증거 아님). 경주 쌍(Higgs/Z, 렙톤 계단 m_μ/m_τ = (ŝ²/4)w)은 확률 꼴이 진폭 꼴을 이기고,
   회전 쌍(W/Z, 와인버그 섞임)은 진폭 꼴이 이긴다. MR은 기록의 경주에만 적용된다.
TL(사다리, 목격 후 가설). Z(자기) → Higgs(한 걸음) → top(Higgs를 자기로 한 걸음 더): m_t = M_Z F².
   α_s 없는 형태: M_H² = M_Z m_t(Higgs는 Z와 top의 기하평균).
   목격: 계획 중 암산으로 먼저 보았다. 가족 = 밑 {M_Z, M_W, M_H} × 지수 {½, 1, 3/2, 2, 5/2, 3}(18개)를 m_t에 대고
   1σ 적중과 우연 확률(과녁을 가족 범위에서 로그 균등)을 보고한다. 순증거 = −log₂(우연 확률).
m_t는 PDG 2024 직접 측정 평균 172.57 ± 0.29 GeV(극 질량으로 읽음; 극 질량의 고유 모호성 ~0.1 GeV는 오차에 넣지 않음).

python -B -m examples.physics.rendering.ce_rendering_mass_rate
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R

M_Z, M_W, M_H = 91.1876, 80.3692, (125.20, 0.11)
M_T = (172.57, 0.29)
BASES = {"M_Z": M_Z, "M_W": M_W, "M_H": M_H[0]}
EXPONENTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _f() -> float:
    return R.core(R.calibrated_alpha_s()[0])["F"]


def scope_checks() -> dict:
    """확률 꼴 대 진폭 꼴: 경주 쌍과 회전 쌍."""
    f = _f()
    s2 = R.SZ2
    rows = {r.key: r for r in R.rows("SK") if r.key in ("m_mu/m_tau",)}
    mmt = rows["m_mu/m_tau"].obs
    w = 1 + s2 * (1 - s2) / (2 * math.pi)
    c_os = M_W / M_Z
    return {"H/Z probability 1/F": (1 / f) / (M_Z / M_H[0]) - 1, "H/Z amplitude 1/sqrt(F)": (1 / math.sqrt(f)) / (M_Z / M_H[0]) - 1,
            "mu/tau probability s2/4 w": (s2 / 4 * w) / mmt - 1, "mu/tau amplitude sqrt(s2)/2 w": (math.sqrt(s2) / 2 * w) / mmt - 1,
            "W/Z amplitude c": math.sqrt(1 - 0.22348) / c_os - 1, "W/Z probability c^2": (1 - 0.22348) / c_os - 1}


def top_ladder() -> dict:
    f = _f()
    pred = M_Z * f ** 2
    geo = math.sqrt(M_Z * M_T[0])
    geo_sig = 0.5 * M_T[1] / M_T[0] * geo
    return {"m_t_pred": pred, "pull_m_t": (pred - M_T[0]) / M_T[1],
            "M_H_geometric_mean": geo, "pull_M_H_geo": (geo - M_H[0]) / math.hypot(geo_sig, M_H[1]),
            "m_t_from_measured_M_H": M_H[0] ** 2 / M_Z}


def family(n_draw: int = 400000, seed: int = 5) -> dict:
    f = _f()
    cands = {f"{b}*F^{k}": v * f ** k for b, v in BASES.items() for k in EXPONENTS}
    hits = sorted(k for k, v in cands.items() if abs(v - M_T[0]) <= M_T[1])
    lo, hi = min(cands.values()), max(cands.values())
    rng = np.random.default_rng(seed)
    draws = np.exp(rng.uniform(math.log(lo), math.log(hi), n_draw))
    vals = np.array(list(cands.values()))
    p_any = float(np.mean(np.any(np.abs(draws[:, None] - vals[None, :]) <= M_T[1] * draws[:, None] / M_T[0], axis=1)))
    return {"n": len(cands), "hits": hits, "range": (lo, hi), "p_chance": p_any, "bits": -math.log2(p_any)}


def main() -> None:
    print("scope:", {k: f"{v:+.2%}" for k, v in scope_checks().items()})
    print("top ladder:", {k: round(v, 3) for k, v in top_ladder().items()})
    print("family:", family())


if __name__ == "__main__":
    main()
