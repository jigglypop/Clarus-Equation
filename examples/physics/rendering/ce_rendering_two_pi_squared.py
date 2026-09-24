"""지평선 선도항의 (2π)² — CE 공리로 세울 수 있는 원환에서 나오는가. 원장: 43장 §43.96. 예측값을 바꾸지 않는다.

§43.95: 지평선 선도항 (π²/2) N_e = (2π)² × S_TT, S_TT = N_e/8은 열적 시간의 자연 원환(열적 원 2π, 공간 원 N_e e-fold)에서
세 NS 마요라나 탑의 엔트로피다. x = βω는 두 원의 길이 비로만 정해지므로, (2π)²가 구조에서 나오려면 원환이 달라지거나
두 번째 구조가 (2π)²를 곱해야 한다. 계산 전에 적은 후보(모두 기존 공리에서 정해지며 맞출 수 있는 수가 없다).
사전 목격: 대수라 대부분의 계수를 암산으로 먼저 보았다. 판정은 보고로만 쓴다.

A1 자연 원환(기준): 열적 원 β = 2π(유클리드 주기, C2·TT), 공간 원 L = N_e(e-fold = 빠르기), 주기 모드.
A2 모듈러 쌍대: 두 원의 역할을 바꾼다(x → 4π²/x).
A3 인과 창을 열적 원으로: 기록은 |φ| < π/4에서만 생긴다(정리 C′). 창의 폭 π/2를 열적 원으로 둔다.
A4 허블 온도: 탑의 온도를 기브스–호킹 H/2π 대신 H로 둔다(β = 1 e-fold).
A5 TT 감김 작용: TT에서 기록 위상은 열적 원 한 바퀴에 2π 감긴다. 그 유클리드 작용 S_w = (K/2)(2π/β)²·βL = 2π²K·L/β.
   강성 K는 탑과 같은 자유도의 자유 페르미온 값(디랙 하나 = 마요라나 둘 = K 1/4π, 정규화 S = (1/8π)∫(∂φ)²)이다.
A6 구간 모드: 자연 원환에서 모드를 구간 경계로 양자화한다(ω = π/L).
판정: 선도 계수가 π²/2와 1e-9 안이면 (2π)²의 근원, 아니면 기록. 필요한 열적 원(β = L/(2πN_e))도 적는다.

python -B -m examples.physics.rendering.ce_rendering_two_pi_squared
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_registry as R

PI = math.pi
TARGET = PI ** 2 / 2.0
NS_CHIRAL_S_COEF = PI ** 2 / 6.0            # 키랄 NS 마요라나 하나의 S = (π²/6)/x
TOWERS = 3
K_DIRAC = 1.0 / (4.0 * PI)                  # 자유 페르미온 반지름의 강성(디랙 하나)


def _ne() -> float:
    return R.core(R.calibrated_alpha_s()[0])["Ne"]


def tower_coefficient(x_times_ne: float) -> float:
    """x = (x·N_e)/N_e일 때 세 탑의 엔트로피 / N_e."""
    return TOWERS * NS_CHIRAL_S_COEF / x_times_ne


def candidates() -> dict:
    ne = _ne()
    beta_nat, length = 2.0 * PI, ne
    x_nat = 2.0 * PI * beta_nat / length
    out = {
        "A1 자연 원환": tower_coefficient(x_nat * ne),
        "A2 모듈러 쌍대": TOWERS * NS_CHIRAL_S_COEF / (4.0 * PI ** 2 / x_nat) / ne,
        "A3 인과 창 = 열적 원": tower_coefficient(2.0 * PI * (PI / 2.0) / length * ne),
        "A4 허블 온도": tower_coefficient(2.0 * PI * 1.0 / length * ne),
        "A5 TT 감김 작용": 2.0 * PI ** 2 * (K_DIRAC * TOWERS / 2.0) * (length / beta_nat) / ne,
        "A6 구간 모드": tower_coefficient(PI * beta_nat / length * ne),
    }
    return {name: {"coef": c, "missing_factor": TARGET / c if c > 0 else math.inf,
                   "hit": abs(c - TARGET) < 1e-9} for name, c in out.items()}


def requirement() -> dict:
    """PH5가 요구하는 열적 원: x = 2πβ/L = 1/N_e, L = N_e → β = 1/(2π) e-fold, 곧 온도 2πH = (2π)² T_GH."""
    beta_needed = 1.0 / (2.0 * PI)
    return {"beta_needed_efolds": beta_needed, "T_over_T_GH": (2.0 * PI) / beta_needed, "T_over_H": 1.0 / beta_needed}


def equivalent_forms() -> dict:
    """같은 선도항의 다른 표기(정의상 항등). N_e = 18D."""
    ne = _ne()
    d = ne / 18.0
    lead = TARGET * ne
    return {"(2π)²·N_e/8": (2 * PI) ** 2 * ne / 8, "3ζ(2)·N_e": 3 * (PI ** 2 / 6) * ne, "(3π)²·D": (3 * PI) ** 2 * d,
            "lead": lead}


def main() -> None:
    for name, d in candidates().items():
        print(f"{name:18s} coef {d['coef']:.5f}  missing factor {d['missing_factor']:.4f}  hit {d['hit']}")
    print("target π²/2 =", round(TARGET, 5), "| requirement:", {k: round(v, 4) for k, v in requirement().items()})
    print("equivalent forms:", {k: round(v, 6) for k, v in equivalent_forms().items()})


if __name__ == "__main__":
    main()
