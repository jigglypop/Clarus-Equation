"""E4의 모양 — 기록은 곱하고 진폭은 더한다. 원장: 43장 §43.70. 예측값을 바꾸지 않는다.

E4: α_s = a³(색 축 셋이 동시에 빔), sin θ_W = 2a²(약력 축 둘에 대해 a²를 축 수만큼 더함). 계산 전에 적은 판본과 규칙:

AP(원리, 읽기). 결합 상수는 기록(경주, §43.67)의 양이므로 축들의 결합 확률의 곱 a^m이다. 섞임각은 회전(§43.68)의
   양이므로 진폭이 축마다 더해져 k·a^k다. 가족 AP: α_s = a^m, sin θ_W = k a^k, m, k ∈ {1..5}(25개). E4 = (3, 2).
대조군: 기록에 합 α_s = m a^m, 진폭에서 합 제거 sin θ_W = a^k, 진폭 대신 확률 ŝ² = k a^k(각 25개).
적중: ŝ² = 0.23129에서 예측한 α_s가 세계 평균 0.1180 ± 0.0009와 |pull| ≤ 1.
kill: AP 가족에서 E4보다 나은 후보가 있거나 E4가 적중하지 않으면 AP 기각.
목격: E4는 알던 식이고 AP도 E4를 보며 세웠다. 적중은 증거가 아니라 읽기다. 요점은 대조군이 떨어지는가다.
범위: AP가 약력 결합에도 통하는지(α₂ = a²) 확인한다(E1: 렌더링 연산은 V₃에만 작용).

python -B -m examples.physics.rendering.ce_rendering_e4_shape
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_registry as R

S2 = R.SZ2
AS_WORLD = (0.1180, 0.0009)
AEM_MZ = 1 / 127.951


def _alpha(alpha_form: str, m: int, weak_form: str, k: int) -> float:
    s = math.sqrt(S2)
    if weak_form == "amp_sum":
        a = (s / k) ** (1 / k)
    elif weak_form == "amp_bare":
        a = s ** (1 / k)
    else:  # prob_sum: s^2 = k a^k
        a = (S2 / k) ** (1 / k)
    return a ** m if alpha_form == "prob" else m * a ** m


FAMILIES = {"AP (prob x amp_sum)": ("prob", "amp_sum"), "record summed (m a^m)": ("sum", "amp_sum"),
            "amplitude unsummed (a^k)": ("prob", "amp_bare"), "probability for mixing (s2 = k a^k)": ("prob", "prob_sum")}


def scan() -> dict:
    out = {}
    for name, (af, wf) in FAMILIES.items():
        rows = {(m, k): _alpha(af, m, wf, k) for m in range(1, 6) for k in range(1, 6)}
        pulls = {mk: (v - AS_WORLD[0]) / AS_WORLD[1] for mk, v in rows.items()}
        hits = sorted(mk for mk, p in pulls.items() if abs(p) <= 1)
        best = min(pulls, key=lambda mk: abs(pulls[mk]))
        out[name] = {"hits": hits, "best": best, "best_pull": pulls[best],
                     "near_misses": sorted(mk for mk, p in pulls.items() if 1 < abs(p) <= 3)}
    ap = out["AP (prob x amp_sum)"]
    out["E4_is_unique_AP_hit"] = ap["hits"] == [(3, 2)]
    return out


def e4_pull() -> float:
    return (_alpha("prob", 3, "amp_sum", 2) - AS_WORLD[0]) / AS_WORLD[1]


def scope_weak_coupling() -> dict:
    """AP를 약력 결합에 적용: α₂ = a²(약력 축 둘의 결합 확률)?"""
    a = R.calibrated_alpha_s()[0] ** (1 / 3)
    alpha2_obs = AEM_MZ / S2
    return {"alpha2_AP": a ** 2, "alpha2_obs": alpha2_obs, "ratio": a ** 2 / alpha2_obs}


def main() -> None:
    s = scan()
    for k, v in s.items():
        print(k, v)
    print("E4 pull:", round(e4_pull(), 3))
    print("scope alpha_2:", {k: round(v, 4) for k, v in scope_weak_coupling().items()})


if __name__ == "__main__":
    main()
