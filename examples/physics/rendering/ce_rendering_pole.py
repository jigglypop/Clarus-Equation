"""분별 = 무분별 — 극에서 운동과 질량이 같아지고 기록만 남는다. 원장: 43장 §43.63. 예측값을 바꾸지 않는다.

사용자 가설: “분별은 무분별과 같아서.” 계산 전에 적은 판본과 kill:

(i) 동전. 닻에서 Z는 나를 부호로만 가르고 평균 분별은 0이다(⟨γ5⟩ = 0, §43.60). 확인만 한다.
(ii) 척도(RS). 운반자의 전파 인자 1/(q² − M² + iMΓ)에서 운동 항 q²(무분별: 대칭, 질량 없음)와 질량 항 M²(분별: 깨짐)는
   극 q² = M²에서만 같고, 거기서 실수부가 사라져 진폭은 순수한 흡수부(기록, 붕괴 폭)만 남는다. 동전은 기록에서 정해지므로
   (R1) Z 동전(DZ)의 관계 E4는 Z 극의 기록에 들어가는 결합, 곧 μ = M_Z에서 성립한다. OE(유일 교차)와 함께 E5를 대신한다.
(iii) 결과. E4의 α_s는 Z 극 기록에서 뽑은 α_s(R_ℓ, Γ_Z, σ_had)와 같아야 한다. 판정 값: PDG 2024 전약 리뷰의
   Z 극 α_s = 0.1221 ± 0.0027(방향은 값을 보기 전에 적었다). 참고: 고에너지 0.1211 ± 0.0025, 전약 전체 0.1187 ± 0.0017,
   세계 평균 0.1180 ± 0.0009.
kill. Z 극 α_s가 E4 값을 3σ 넘게 배제하면 RS 읽기 기각(E5는 다시 공리).

python -B -m examples.physics.rendering.ce_rendering_pole
"""

from __future__ import annotations

import cmath
import math

from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_distinction as DST
from examples.physics.rendering import ce_rendering_registry as R

M_Z, GAMMA_Z = 91.1876, 2.4955
ALPHA_S_RECORDS = {"Z pole (R_l, Gamma_Z, sigma_had)": (0.1221, 0.0027),
                   "high-energy incl. W decays": (0.1211, 0.0025),
                   "EW global fit": (0.1187, 0.0017),
                   "world average": (0.1180, 0.0009)}


def propagator(q2: float, m: float = M_Z, g: float = GAMMA_Z) -> complex:
    return 1 / (q2 - m * m + 1j * m * g)


def pole_reading() -> dict:
    """(ii): 운동 항과 질량 항이 같아지는 곳에서 실수부 0, 위상 90°."""
    at = propagator(M_Z ** 2)
    off = {dq: propagator((M_Z + dq) ** 2) for dq in (-GAMMA_Z / 2, GAMMA_Z / 2)}
    return {"real_at_pole": at.real, "phase_deg_at_pole": math.degrees(cmath.phase(at)),
            "phase_deg_half_width": {k: math.degrees(cmath.phase(v)) for k, v in off.items()},
            "kinetic_minus_mass_at_pole": M_Z ** 2 - M_Z ** 2}


def coin_reading() -> dict:
    """(i): 닻에서 Z의 순수 분별은 평균 분별 0(P_R − P_L = 0)."""
    z = DST.z_couplings("e", 0.25)
    return {"mean_distinction": (1 - z["P_L"]) - z["P_L"], "g_V": z["g_V"]}


def record_alpha_check() -> dict:
    """(iii): E4·렙톤 규칙·FP의 α_s를 Z 극 기록의 α_s와 비교."""
    ce = {"E4": R.calibrated_alpha_s()[0], "lepton rule": BLP.alpha_from_lepton_rule()["alpha_s_lepton"],
          "FP": BLP.fixed_point_alpha("linear")}
    out = {k: {src: (a - v) / s for src, a in ce.items()} for k, (v, s) in ALPHA_S_RECORDS.items()}
    z = ALPHA_S_RECORDS["Z pole (R_l, Gamma_Z, sigma_had)"]
    out["killed"] = abs(ce["E4"] - z[0]) / z[1] >= 3
    out["ce_values"] = ce
    return out


def main() -> None:
    print("coin:", coin_reading())
    print("pole:", pole_reading())
    r = record_alpha_check()
    for k, v in r.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
