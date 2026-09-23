"""우주의 순환(생겨남 → 흩어짐 → 다시 생겨남)과 시간축 기울기. 원장: 43장 §43.14.

잠긴 모듈(v1–v4)은 import만 한다. 이 모듈은 예측값을 바꾸지 않는 일관성 검사다.

순환은 유클리드 드 시터의 원이며 반지름은 c/H_Λ(H_Λ = H0 √Ω_Λ), 한 바퀴의 허수 시간 주기는 2π/H_Λ다.
생겨난 뒤 지난 위상은 H_Λ t0 = (2/3) arsinh √(Ω_Λ/Ω_m)로 H0와 무관하다. 출발점의 접선(기록 틀)과
현재 지점으로 그은 현(현재 틀)의 각은 접선–현 정리로 지난 호의 절반이다.

python -B -m examples.physics.rendering.ce_rendering_cycle
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
KMS_MPC_PER_GYR = 1.0 / 977.79
HBAR, K_B, MPC_M = 1.054571817e-34, 1.380649e-23, 3.0856775814913673e22


def elapsed_phase(omega_m: float) -> float:
    """H_Λ t0: 평탄 물질+진공 우주가 생겨난 뒤 지난 순환 위상(복사 무시)."""
    return (2.0 / 3.0) * math.asinh(math.sqrt((1.0 - omega_m) / omega_m))


def tangent_chord_tilt(omega_m: float) -> float:
    """C3: 접선–현 정리로 얻는 시간축 기울기 = 지난 위상의 절반."""
    return elapsed_phase(omega_m) / 2.0


def cycle_geometry(c: dict) -> dict:
    h = PL.h_rings(c)
    h_lambda = 100.0 * h * math.sqrt(1.0 - c["Om"])
    radius = C_KM_S / h_lambda
    period_gyr = 2 * math.pi / (h_lambda * KMS_MPC_PER_GYR)
    temperature = HBAR * (h_lambda * 1000 / MPC_M) / (2 * math.pi * K_B)
    return {"H_lambda": h_lambda, "radius_mpc": radius, "circumference_mpc": 2 * math.pi * radius,
            "period_gyr_imaginary": period_gyr, "temperature_k": temperature,
            "age_gyr": elapsed_phase(c["Om"]) / (h_lambda * KMS_MPC_PER_GYR)}


def omega_m_for_phase(target: float = math.pi / 4) -> float:
    return brentq(lambda om: elapsed_phase(om) - target, 0.2, 0.4)


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    g = cycle_geometry(c)
    print({k: (round(v, 4) if v > 1e-6 else f"{v:.2e}") for k, v in g.items()})
    print(f"tilt from age {tangent_chord_tilt(c['Om']):.5f} vs pi/8 {math.pi / 8:.5f} "
          f"({100 * (tangent_chord_tilt(c['Om']) / (math.pi / 8) - 1):+.2f}%)")
    om = omega_m_for_phase()
    cb = dict(c)
    cb["Om"] = om
    print(f"phase = pi/4 needs Omega_m = {om:.5f}; with R-Pl h the acoustic angle moves to "
          f"{DV.ce_theta_pull(cb, PL.h_rings(c)):+.1f} sigma (rejected)")


if __name__ == "__main__":
    main()
