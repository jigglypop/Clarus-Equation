"""증명 가능한 것의 정리와 증명 불가한 것의 예측. 원장: 43장 §43.36, 사전 등록 v12.

정리 E(나이테 = 무게 평균, 1차): 정리 A로 기하는 Tr(ρT)에 선형 반응한다. 기울기 손실 ε_i = 1 − cos θ_i ≤ 0.044에서
  나이테 판독의 1차 변화는 Σ w_i ε_i이고 진공은 ε = 0(틀 불변)이므로 1 − Ω_m(1 − cos θ). 2차 오차는 O(ε²).
규칙(증명 아님): 초기 기록(r_d, θ*)으로 교정한 판독 = 나이테, 교정 없는 오늘의 절대 판독 = 시계(O1 전체 기울기).
예측:
 P26 중력파 표준 사이렌 H0 = 100 h / cos(π/8)(시계 판독). 규칙이 틀리면 100 h.
 P27 Higgs 자기결합 κ_λ = 1, 추가 스칼라 없음(Higgs = Λ¹ 약 통로 하나, §43.22).
 P28 M_H / M_Z = F = 1 + α_s D.
증명 불가 항목의 판정 실험 표: S2(P01), T1(P02), ξ²(P21), ⑤(α_em 행), 지평선 미시 상태(관측 수단 없음).

python -B -m examples.physics.rendering.ce_rendering_open_predictions
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

DECISIVE = (
    ("P01 S2 octant s23^2", "JUNO + DUNE + Hyper-K", "~2030-2035"),
    ("P02 T1 delta_PMNS", "DUNE + Hyper-K", "~2032-2037"),
    ("P17 neutrino mass sum", "CMB-S4 + DESI/Euclid", "~2030"),
    ("P21-22 w(z) (amplitude xi^2)", "DESI final + Euclid", "~2028-2030"),
    ("P23 gravity-mediated entanglement", "BMV/QGEM", "unscheduled"),
    ("P24 Omega_k", "Euclid + 21 cm", "~2030+"),
    ("P25 BAO scale drift", "DESI final + Euclid", "~2028-2030"),
    ("P26 siren H0", "LIGO-Virgo-KAGRA O5, Einstein Telescope", "~2028-2035+"),
    ("P27 kappa_lambda", "HL-LHC (+-50%), FCC-hh (+-5%)", "~2040, ~2070"),
    ("P28 M_H/M_Z", "FCC-ee (M_H +-0.01 GeV)", "~2045"),
    ("horizon microstates", "none", "no prediction"),
)


def ring_first_order_error(c: dict) -> dict:
    """정확한 무게 평균(cos)과 1차 선형 응답의 차이, 그리고 최대 ε."""
    z = np.linspace(0.0, 2.5, 251)
    theta = np.array([VT.cycle_phase(1 / (1 + zi), c["Om"]) for zi in z]) / 2
    eps = 1 - np.cos(theta)
    exact = 1 - c["Om"] * eps
    linear = 1 - c["Om"] * (theta ** 2 / 2)
    return {"max_eps": float(eps.max()), "max_second_order_gap": float(np.abs(exact - linear).max())}


def predictions(c: dict) -> dict:
    h = PL.h_rings(c)
    return {"P26_siren_H0": 100 * h / math.cos(R.TIME_AXIS_TILT), "P26_if_rule_wrong": 100 * h,
            "P27_kappa_lambda": 1.0, "P28_MH_over_MZ": c["F"]}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    print("ring = weight average, first order:", {k: f"{v:.2e}" for k, v in ring_first_order_error(c).items()})
    print("predictions:", {k: round(v, 5) for k, v in predictions(c).items()})
    for row in DECISIVE:
        print("  ", " | ".join(row))


if __name__ == "__main__":
    main()
