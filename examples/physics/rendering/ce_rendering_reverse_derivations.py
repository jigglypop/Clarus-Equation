"""가설 규칙의 역유도. 원장: 43장 §43.34. 예측값을 바꾸지 않는다.

1. G1m의 k = Ω_m: BAO 나이테 판독 = 확률 무게(QG1)로 평균한 틀 투영. 물질 cos θ, 진공 1(T = −ρg는 틀 불변).
   Ω_m cos θ + Ω_Λ = 1 − Ω_m(1 − cos θ). 무게 시점: 오늘(판독 시점, 주) / 기록 시기 Ω_m(z)(대안).
2. W2 부호: c_1 < 0이면 w < −1(팬텀) → 널 에너지 조건 위반(유령). 주파수는 C6 최저 조화(§43.27).
3. U1 = O1: 시간축 ⊥ 공간축의 직각을 빛원뿔 이등분선(π/8)이 π/8, 3π/8로 나눈 직각삼각형 = CKM 삼각형.
4. R-Pl = ⑤ × 4: α_s/(4π) = 4 · α_s/(16π) (중력은 T_μν의 네 방향과 결합).

python -B -m examples.physics.rendering.ce_rendering_reverse_derivations
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_harmonic as VH
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT


def vacuum_frame_invariance(rapidity: float = 0.7, angle: float = 0.4) -> float:
    """T = −ρ g의 에너지 밀도를 부스트·회전된 관측자 틀에서 잰 값과 ρ의 차(0이어야 한다)."""
    rho = 1.0
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    t = -rho * g
    u = np.array([math.cosh(rapidity), math.sinh(rapidity) * math.cos(angle), math.sinh(rapidity) * math.sin(angle), 0.0])
    t_low = g @ t @ g
    return float(abs(u @ t_low @ u - rho))


def weight_average_factor(z: np.ndarray, om: float, epoch_weight: bool = False) -> np.ndarray:
    theta = np.array([VT.cycle_phase(1 / (1 + zi), om) for zi in np.atleast_1d(z)]) / 2
    w_m = om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om) if epoch_weight else om
    return w_m * np.cos(theta) + (1 - w_m) * 1.0


def g1m_derivation_check() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    z = R.BAO_Z
    same = float(np.abs(weight_average_factor(z, c["Om"]) - GD.factor("G1m", z, c["Om"])).max())
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b0 = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU))
    a_ce = GD.C_KM_S / (100 * h * rd)
    base = VT.score(VT.ADOPTED_NU)
    chi_rows = base["chi2"] - base["branch"]["bao_chi2_fixed"]
    out = {"max_diff_to_G1m": same}
    for tag, ep in (("present weights", False), ("epoch weights", True)):
        r = a_ce * b0 * weight_average_factor(z, c["Om"], ep) - R.BAO_Y
        chi = float(r @ R.BAO_CINV @ r)
        out[tag] = {"bao_chi2_fixed": chi, "V39": math.sqrt((chi_rows + chi) / base["N"])}
    return out


def w2_sign_by_null_energy() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    minus = VH.harmonic_density(c, 1, -1.0)
    plus = VH.harmonic_density(c, 1, 1.0)
    def w0(f, eps=1e-5):
        return -1 - (math.log(f(1 + eps)) - math.log(f(1 - eps))) / (3 * (math.log1p(eps) - math.log1p(-eps)))
    return {"w0_plus": w0(plus), "w0_minus": w0(minus), "minus_is_phantom": w0(minus) < -1}


def unitarity_triangle_from_bisector() -> dict:
    beta = R.TIME_AXIS_TILT                      # 이등분선과 시간축
    gamma = math.pi / 2 - beta                   # 이등분선과 공간축
    alpha = math.pi / 2                          # 시간축 ⊥ 공간축
    tri = R.ckm_triangle(R.calibrated_alpha_s()[0])
    return {"angles_deg": (math.degrees(beta), math.degrees(gamma), math.degrees(alpha)),
            "sum_deg": math.degrees(alpha + beta + gamma), "registry_U1": tri}


def rpl_from_channel_loop() -> dict:
    a = R.calibrated_alpha_s()[0]
    c = R.core(a)
    g2 = 4 * math.pi * a
    return {"4 * a/(16 pi)": 4 * a / (16 * math.pi), "a/(4 pi)": a / (4 * math.pi), "g^2/(16 pi^2)": g2 / (16 * math.pi ** 2),
            "planck_unit_factor - 1": PL.planck_unit_factor(c) - 1}


def main() -> None:
    print(f"vacuum energy density frame-invariance residual: {vacuum_frame_invariance():.2e}")
    print("G1m from probability-weight average:", g1m_derivation_check())
    print("W2 sign by null energy:", w2_sign_by_null_energy())
    print("U1 from light-cone bisector:", unitarity_triangle_from_bisector())
    print("R-Pl = 4 x channel loop:", rpl_from_channel_loop())


if __name__ == "__main__":
    main()
