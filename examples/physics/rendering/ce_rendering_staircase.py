"""보강 4의 계속 — 렌더링 계단(τ → μ → e)과 옥탄트. 원장: 43장 §43.43(계단 분석). 예측값을 바꾸지 않는다.

렌더링은 무거운 세대에서 가벼운 세대로 진행하고 마지막 축 e가 “나”다(사용자 확인). 단계 m의 새 축은 τ, μ, e이고
S2의 대응 θ₁₃ ↔ 1, θ₁₂ ↔ 2, θ₂₃ ↔ 3에서 옥탄트는 “나”가 렌더링되는 마지막 계단이다. 계단 높이 h_m(단위 δ/8)로
s₁₃² = h₁δ/8, s₁₂² = (1 − h₂δ/8)/3, s₂₃² = (1 − h₃δ/8)/2. 계산 전에 적은 후보와 kill:

후보. 두 배(S2) 2^{m−1}, 선형 m, 경계만 2^{m−1} − 1, S1 2^m − 1, 순수 1.
K1. 공리 B(TM1: s₁₂² ≈ (1 − 2s₁₃²)/3)가 요구하는 h₂ = 2h₁, 1–2단계 자료 |pull| < 2. 어기면 기각.
K2. 부호 규칙 “계단 정렬”(ν₃는 τ, ν₂는 μ 쪽으로 기운다)은 같은 자료에서 두 열이 함께 정렬될 때만 유지.
K3. 살아남은 규칙의 h₃를 현재 자료가 가를 수 있는지 보고. P01은 바꾸지 않는다.

python -B -m examples.physics.rendering.ce_rendering_staircase
"""

from __future__ import annotations

import itertools
import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_boundary as BD
from examples.physics.rendering import ce_rendering_registry as R

RULES = {
    "doubling 2^(m-1) (S2)": (1, 2, 4),
    "linear m": (1, 2, 3),
    "boundary-only 2^(m-1)-1": (0, 1, 3),
    "S1 2^m-1": (1, 3, 7),
    "pure 1": (1, 1, 1),
}
S13_OBS = (0.02215, 0.00056, 0.00058)
S12_JUNO = (0.3092, 0.0087, 0.0087)
S23_OBS = {"SK": (0.470, 0.017, 0.013), "noSK": (0.561, 0.012, 0.015)}
DL_OBS = {"SK": (212.0, 26.0, 41.0), "noSK": (177.0, 19.0, 20.0)}
NUFIT_BEST = {"SK": (0.308, 0.470, 0.02215, 212.0), "noSK": (0.307, 0.561, 0.02195, 177.0)}


def angles(h: tuple[int, int, int], d: float) -> dict:
    return {"s13sq": h[0] * d / 8, "s12sq": (1 - h[1] * d / 8) / 3, "s23sq": (1 - h[2] * d / 8) / 2}


def cos_phi_for(s13sq: float, s23sq: float) -> float:
    """TM1 = U_TBM·R23(θ, φ)에서 주어진 s13², s23²를 주는 cos φ(옥탄트 = 부호)."""
    theta = math.asin(math.sqrt(3 * s13sq))
    f = lambda p: BD.tm1_angles(BD.tm1_matrix(theta, p))["s23sq"] - s23sq
    return math.cos(brentq(f, 1e-9, math.pi - 1e-9))


def score_rules(c: dict) -> dict:
    d = c["d"]
    out = {}
    for name, h in RULES.items():
        a = angles(h, d)
        b_ok = h[1] == 2 * h[0]
        row = {"h": h, **a, "B_consistent": b_ok, "pull_s13": BD.pull(a["s13sq"], S13_OBS),
               "pull_s12_JUNO": BD.pull(a["s12sq"], S12_JUNO)}
        row["K1_pass"] = b_ok and abs(row["pull_s13"]) < 2 and abs(row["pull_s12_JUNO"]) < 2
        for key, obs in S23_OBS.items():
            row[f"pull_s23_{key}"] = BD.pull(a["s23sq"], obs)
        if row["K1_pass"]:
            dl = BD.tm1_delta_for_octant(c, a["s23sq"])
            row["delta_TM1_M1"] = dl
            row.update({f"pull_delta_{k}": BD.pull(dl, o) for k, o in DL_OBS.items()})
            row["cos_phi_over_half_sqrt_d"] = cos_phi_for(a["s13sq"], a["s23sq"]) / (math.sqrt(d) / 2)
        out[name] = row
    return out


def data_step_height(c: dict) -> dict:
    """자료의 옥탄트 계단 높이 h₃ = (1 − 2s₂₃²)·8/δ와 한 계단의 폭 δ/16."""
    d = c["d"]
    k = 16 / d
    return {"stair_width_s23sq": d / 16,
            **{key: (k * (0.5 - v), k * dn, k * up) for key, (v, up, dn) in S23_OBS.items()}}


def alignment(key: str) -> dict:
    """K2: NuFIT 최적값에서 ν₃의 τ − μ, ν₂의 μ − τ(둘 다 양이면 계단 정렬)."""
    s12, s23, s13, dl = NUFIT_BEST[key]
    u = R.pmns_matrix(s12, s23, s13, math.radians(dl))
    p = [[abs(u[i][j]) ** 2 for j in range(3)] for i in range(3)]
    return {"nu3_tau_minus_mu": p[2][2] - p[1][2], "nu2_mu_minus_tau": p[1][1] - p[2][1]}


def world_heights(m_max: int = 3) -> tuple[int, ...]:
    """Bool: 단계 m의 세계(참·거짓 배정 = 부분집합) 가운데 “나(m번째 명제)”가 참인 세계의 수."""
    return tuple(sum(1 for k in range(m + 1) for s in itertools.combinations(range(m), k) if m - 1 in s)
                 for m in range(1, m_max + 1))


def truth_bias_for_height(h3: float, n: int = 3) -> float:
    """최종 단계에서 P(나 = 참) = h₃/2ⁿ. 무차별(참 = 거짓)이면 1/2 → h₃ = 4. 선형 h₃ = 3은 P = 3/8."""
    return h3 / 2 ** n


def not_me_pair(c: dict) -> dict:
    """여집합(호지 쌍대)은 나 ↔ 아닌 나, s₂₃² ↔ 1 − s₂₃²를 맞바꾼다. 두 NuFIT 판본과의 거리."""
    me = angles(RULES["doubling 2^(m-1) (S2)"], c["d"])["s23sq"]
    notme = 1 - me
    return {"me": me, "not_me": notme, "me_vs_SK": BD.pull(me, S23_OBS["SK"]),
            "not_me_vs_noSK": BD.pull(notme, S23_OBS["noSK"])}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    print("Bool world heights:", world_heights(), "| P(me) for doubling / linear:",
          truth_bias_for_height(4), truth_bias_for_height(3))
    print("me / not-me pair:", {k: round(v, 4) for k, v in not_me_pair(c).items()})
    for name, r in score_rules(c).items():
        keep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "h"}
        print(f"{name:26s} h={r['h']} {keep}")
    print("data octant step:", {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else round(v, 5))
                                for k, v in data_step_height(c).items()})
    for key in ("SK", "noSK"):
        print(f"K2 alignment {key}:", {k: round(v, 4) for k, v in alignment(key).items()})


if __name__ == "__main__":
    main()
