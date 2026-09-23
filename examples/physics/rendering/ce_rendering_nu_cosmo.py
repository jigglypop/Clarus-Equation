"""중성미자 질량 합과 우주론 상한 — P17의 기각 조건 점검. 원장: 43장 §43.74. 예측값을 바꾸지 않는다.

P17(v7): Σm_ν = 59.16 meV, kill “cosmological upper bound below 55 meV at 95% CL, or inverted ordering ≥ 3σ”.
등록 문구는 모형·사전분포를 지정하지 않았다. 판정 규칙: 문구 그대로 읽는다(어느 발표 상한이든 < 55 meV이면 충족).

자료(arXiv:2606.17994, 2026-06): CMB-SPA(Planck 2018, ACT DR6, SPT-3G) + DESI DR2 BAO + DES Y5 SN.
   ΛCDM 0부터의 사전분포: 단열 52, 등곡률 57 meV. 위계 사전분포 92 meV. CPL 111 meV.
   DESI DR2 + Planck PR4(공식 기준): 64.2 meV.
JUNO 첫 결과(arXiv:2511.14593): Δm²₂₁ = (7.50 ± 0.12)×10⁻⁵ eV², sin²θ₁₂ = 0.3092 ± 0.0087.
맥락(구제 아님, 결과를 본 뒤의 어림): G1m(§43.49)은 BAO의 Ω_m 판독을 ΔΩ_m = −0.0099만큼 낮춘다. θ*를 고정한 ΛCDM에서
   Σ가 Ω_m을 올리는 기울기로 이 이동을 Σ로 환산한다(중성미자는 z < z*에서 물질로 근사, r_s 불변 근사).

python -B -m examples.physics.rendering.ce_rendering_nu_cosmo
"""

from __future__ import annotations

import math

from scipy.integrate import quad
from scipy.optimize import brentq

CE = {"sum_meV": 59.16, "m1_meV": 0.307, "dm21": 7.403e-5, "s12sq": 0.318517}
DM31_NO = 2.513e-3
BOUNDS = {"LCDM prior>=0 adiabatic (CMB-SPA+DESI DR2+DESY5)": 52.0,
          "LCDM prior>=0 isocurvature": 57.0,
          "LCDM hierarchy prior": 92.0,
          "CPL dark energy": 111.0,
          "DESI DR2 + Planck PR4 (official)": 64.2}
JUNO = {"dm21": (7.50e-5, 0.12e-5), "s12sq": (0.3092, 0.0087)}
G1M_DOM = -0.0099
OMEGA_B, OMEGA_C, OMEGA_G, Z_STAR = 0.02237, 0.1200, 2.47e-5, 1090.0


def oscillation_minimum() -> float:
    return (math.sqrt(JUNO["dm21"][0]) + math.sqrt(DM31_NO)) * 1e3


def kill_check() -> dict:
    triggered = sorted(k for k, v in BOUNDS.items() if v < 55.0)
    osc_min = oscillation_minimum()
    return {"triggered_by": triggered, "P17_killed_as_registered": bool(triggered),
            "oscillation_minimum_meV": osc_min, "CE_above_minimum_meV": CE["sum_meV"] - osc_min,
            "minimum_excluded_by": sorted(k for k, v in BOUNDS.items() if v < osc_min)}


def juno_pulls() -> dict:
    return {"dm21 (P19)": (CE["dm21"] - JUNO["dm21"][0]) / JUNO["dm21"][1],
            "s12sq (P03)": (CE["s12sq"] - JUNO["s12sq"][0]) / JUNO["s12sq"][1]}


def _dm_star(h: float, sum_ev: float) -> float:
    om = (OMEGA_B + OMEGA_C + sum_ev / 93.14) / h ** 2
    orad = OMEGA_G * 1.6913 / h ** 2 if sum_ev == 0 else OMEGA_G / h ** 2
    ol = 1 - om - orad
    return quad(lambda z: 1 / math.sqrt(om * (1 + z) ** 3 + orad * (1 + z) ** 4 + ol), 0, Z_STAR, limit=200)[0] / h


def g1m_equivalent_shift(ref_sum_ev: float = 0.06, h_ref: float = 0.6736, d_sum: float = 0.02) -> dict:
    target = _dm_star(h_ref, ref_sum_ev)

    def om_at(s):
        h = brentq(lambda x: _dm_star(x, s) - target, 0.5, 0.9)
        return (OMEGA_B + OMEGA_C + s / 93.14) / h ** 2, h
    om_lo, _ = om_at(ref_sum_ev - d_sum)
    om_hi, _ = om_at(ref_sum_ev + d_sum)
    slope = (om_hi - om_lo) / (2 * d_sum)
    return {"dOm_dSum_per_eV": slope, "G1m_dOm": G1M_DOM, "equivalent_dSum_eV": G1M_DOM / slope}


def main() -> None:
    print("kill check:", kill_check())
    print("JUNO pulls:", {k: round(v, 2) for k, v in juno_pulls().items()})
    print("G1m shift (rough, post hoc):", {k: round(v, 4) for k, v in g1m_equivalent_shift().items()})


if __name__ == "__main__":
    main()
