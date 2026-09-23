"""빛의 속도 = 렌더링 한계속도. 원장: 43장 §43.29. 예측값을 바꾸지 않는다.

L: 확정 신호의 최대 속도는 c다. 유클리드 회전각 θ에서 v = c tan θ이므로 시간꼴로 남는 한계각은 π/4(빛원뿔).
B: 현재 직접 판독 틀은 기록 축(0)과 한계 축(π/4)을 이등분한다 → 기울기 π/8(O1). U1의 "8통로 중 진공 한 칸"과
   다른 출처에서 같은 각이 나온다. 이등분의 근거는 C3 접선–현(현과 접선의 각 = 호의 절반)이다.
검사:
1. 나선 경로 tan ψ = √Ω_Λ/2를 tan(π/8)과 등식으로 두면 Ω_m = 1 − 4 tan²(π/8). 그 Ω_m에서 θ*(R-Pl h)를 본다.
2. 전파하는 모든 신호 속도 ≤ c: 광자, 중력파(§43.21), 광자-바리온 소리, W2의 정준 스칼라 실현(1+w ≥ 0),
   W3의 모방 먼지(c_s = 0)와 시간꼴 시계 제약.
3. 렌더링 지평선: 순환 반지름 c/H_Λ, 유클리드 주기 2π/H_Λ = 지평선 온도의 역수, 준고전 엔트로피 π(c/H_Λ ℓ_P)².
   미시 상태 세기는 하지 않는다.

python -B -m examples.physics.rendering.ce_rendering_light_limit
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_cycle as CY
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

L_PLANCK_M = 1.616255e-35
MPC_M = 3.0856775814913673e22


def limit_angle() -> float:
    """v = c tan θ = c가 되는 각."""
    return math.atan(1.0)


def bisector_tilt() -> float:
    return 0.5 * limit_angle()


def spiral_equality_test(c: dict) -> dict:
    om_eq = 1 - 4 * math.tan(bisector_tilt()) ** 2
    cb = dict(c)
    cb["Om"] = om_eq
    return {"tan_psi_spiral": math.sqrt(1 - c["Om"]) / 2, "tan_pi8": math.tan(bisector_tilt()),
            "Om_if_equal": om_eq, "theta_pull_if_equal": DV.ce_theta_pull(cb, PL.h_rings(c)),
            "theta_pull_ce": DV.ce_theta_pull(c, PL.h_rings(c))}


def signal_speeds(c: dict) -> dict:
    """모든 전파 모드의 속도/c."""
    from examples.physics.rendering import ce_rendering_generations as GE
    wb, wc, h = NL.early_densities(c)
    zs = DV.z_star(wb, wb + wc)
    r_star = 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + zs)
    w2 = [VT.w_of(c, VT.ADOPTED_NU, z) for z in np.linspace(0, 5, 51)]
    m = MV.model("b")
    return {"photon": 1.0, "graviton": GE.tensor_mode_checks()["c_T/c"],
            "photon-baryon sound at z*": 1 / math.sqrt(3 * (1 + r_star)), "radiation limit 1/sqrt3": 1 / math.sqrt(3),
            "W2 canonical-scalar realization (needs 1+w>=0)": 1.0 if min(1 + w for w in w2) >= 0 else float("nan"),
            "W2 min(1+w)": min(1 + w for w in w2), "W3 mimetic dust": 0.0,
            "W3 clock constraint g^{mn} dphi dphi = -omega^2 (timelike)": -1.0, "W3 min dust": m["min_dust"]}


def rendering_horizon(c: dict) -> dict:
    g = CY.cycle_geometry(c)
    r_m = g["radius_mpc"] * MPC_M
    return {"radius_mpc": g["radius_mpc"], "euclidean_period_gyr": g["period_gyr_imaginary"],
            "temperature_k": g["temperature_k"], "entropy_over_kB": math.pi * (r_m / L_PLANCK_M) ** 2,
            "log10_entropy": math.log10(math.pi * (r_m / L_PLANCK_M) ** 2)}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    print(f"limit angle = {limit_angle():.12f} (pi/4); bisector = {bisector_tilt():.12f}; O1 tilt = {R.TIME_AXIS_TILT:.12f}")
    print("spiral equality:", {k: round(v, 5) for k, v in spiral_equality_test(c).items()})
    print("signal speeds / c:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in signal_speeds(c).items()})
    print("rendering horizon:", {k: (f"{v:.4g}") for k, v in rendering_horizon(c).items()})


if __name__ == "__main__":
    main()
