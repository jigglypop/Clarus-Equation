"""기록 갱신 규칙 R1: 중력의 원천은 과거 빛원뿔 안의 기록에 조건을 건 국소 ρ다. 원장: 43장 §43.31.

QG1(§43.30)과 L(§43.29)에서: 원격 기록이 국소 원천을 즉시 바꾸면 비선형 되먹임으로 신호가 생긴다(가지 무게와 같은 이유).
따라서 확정은 빛원뿔 안에서만 국소 원천을 갱신한다.
시험:
1. Page–Geilker형: 실험실 안의 양자 난수 기록이 질량 위치를 정할 때 중력 원천이 결과를 따르는가(상관 1).
2. 빛원뿔 밖 기록: 밥의 평균 상태가 앨리스의 기저 선택과 무관한가(거리 0).
3. 자발 중력 붕괴 판본(디오시–펜로즈, R0 = ħ/(M_Z c), E5 사건 척도): 길이 하한(문헌 약 1e-10 m, 기억 기반 차수) 대비.

python -B -m examples.physics.rendering.ce_rendering_record_update
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_probability_weight as PW

HBAR_C_GEV_M = 1.973269804e-16   # ħc [GeV·m]
M_Z_GEV = 91.1876
DP_R0_LOWER_BOUND_M = 1e-10      # 차수만: Donadi et al. 2021 (Nat. Phys.) 계열의 하한, 원문 수치 확인 필요


def page_geilker_correlation(rule: str, trials: int = 20000, seed: int = 7) -> float:
    """큐비트 결과 k(=0/1)가 질량을 왼/오른쪽에 둔다. 중력 원천 위치와 실제 위치의 상관.

    rule 'record': 실험실 안 기록(빛원뿔 안)에 조건을 건 ρ → 원천 = 실제 위치.
    rule 'average': 기록을 무시한 평균 ρ → 원천 = 0(가운데).
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 2, trials)
    actual = 2 * k - 1
    source = actual.astype(float) if rule == "record" else np.zeros(trials)
    if np.std(source) == 0:
        return 0.0
    return float(np.corrcoef(actual, source)[0, 1])


def outside_light_cone_signalling() -> float:
    """원격 기록이 빛원뿔 밖이면 밥의 원천은 조건 없는 국소 ρ → 확률 무게 규칙과 같다."""
    return PW.signalling("weight")


def instant_remote_update_signalling() -> float:
    """원격 기록이 밥의 원천을 즉시 조건화하면 가지 무게와 같아진다."""
    return PW.signalling("branch")


def dp_event_scale_length() -> dict:
    r0 = HBAR_C_GEV_M / M_Z_GEV
    return {"R0_event_scale_m": r0, "lower_bound_m": DP_R0_LOWER_BOUND_M,
            "orders_below_bound": math.log10(DP_R0_LOWER_BOUND_M / r0)}


def main() -> None:
    print(f"Page-Geilker correlation: record={page_geilker_correlation('record'):.3f}, "
          f"average={page_geilker_correlation('average'):.3f}")
    print(f"signalling: outside light cone={outside_light_cone_signalling():.6f}, "
          f"instant remote update={instant_remote_update_signalling():.6f}")
    print("DP with CE event scale:", {k: f"{v:.3g}" for k, v in dp_event_scale_length().items()})


if __name__ == "__main__":
    main()
