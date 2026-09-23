"""θ*의 중성미자 처리 경로 차이 해결: 정확한 페르미–디랙 중성미자 밀도. 원장: 43장 §43.38.

§43.28에서 θ*를 계산하는 두 경로가 약 2.3σ 달랐다. 원인은 중성미자다.
 경로 D(유도 모듈): 늦은 우주 물질에서 ν를 빼고 그 몫을 진공에 넣으며 ν를 항상 무질량 복사로 둔다.
 경로 M(모방 모듈): CE 쪽은 ν를 늦은 물질에 넣지만 Planck 기준은 경로 D로 계산한다.
계산 전에 고정한 규칙:
 - CE와 Planck 기준을 같은 코드로 계산하고, 두 쪽 모두 질량 있는 ν를 FD 적분으로 넣는다(N_eff = 3.044).
 - 질량: CE는 중성미자 모듈의 정상 순서 (m1, m2, m3), Planck 기준은 (0, 0, 60 meV).
 - z*는 Hu–Sugiyama 식에 차가운 물질(ω_b + ω_c)을 넣는다(재결합 시 ν는 상대론적).
 - 판본 L0·W2·W3(b)의 θ* 잔차를 다시 계산해 이전 판정(W2 채택, W3 (a) 기각·(b) 통과, W1 기각)이 뒤집히는지 본다.

python -B -m examples.physics.rendering.ce_rendering_theta_nu
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import quad

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_neutrino as NU
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458
T_CMB = 2.7255
K_B_EV = 8.617333262e-5
T_NU0_EV = K_B_EV * T_CMB * (4.0 / 11.0) ** (1.0 / 3.0)
PLANCK_MASSES_EV = (0.0, 0.0, 0.06)
# 한 종의 무질량 ν 에너지 밀도(ω), N_eff = 3.044를 세 종에 고르게 나눔
OMEGA_NU1_MASSLESS_H2 = DV.OMEGA_GAMMA_H2 * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * DV.N_EFF / 3.0


@lru_cache(maxsize=65536)
def _fd_ratio(y: float) -> float:
    """ρ(m)/ρ(m=0) for one FD species, y = m / T_ν."""
    if y == 0.0:
        return 1.0
    num = quad(lambda x: x * x * math.sqrt(x * x + y * y) / (math.exp(x) + 1.0), 0, 60, limit=200)[0]
    return num / (7.0 * math.pi ** 4 / 120.0)


def omega_nu_h2(z: float, masses_ev: tuple[float, ...]) -> float:
    """ν 에너지 밀도 ω_ν(z) = ρ_ν(z)/ρ_crit,100 (h² 단위)."""
    t = T_NU0_EV * (1 + z)
    y_round = lambda m: round(m / t, 6)
    return OMEGA_NU1_MASSLESS_H2 * (1 + z) ** 4 * sum(_fd_ratio(y_round(m)) for m in masses_ev)


def _hub_factory(wb: float, wc: float, h: float, masses: tuple[float, ...],
                 dark_shape: Callable[[float], float] | None) -> Callable[[float], float]:
    """E(z)·100h. dark_shape(a)는 오늘 1로 규격화된 어두운 부문 밀도 모양(None이면 상수)."""
    today = (wb + wc + DV.OMEGA_GAMMA_H2 + omega_nu_h2(0.0, masses)) / h ** 2
    ode0 = 1.0 - today
    shape = dark_shape or (lambda a: 1.0)

    def hub(z: float) -> float:
        rest = ((wb + wc) * (1 + z) ** 3 + DV.OMEGA_GAMMA_H2 * (1 + z) ** 4 + omega_nu_h2(z, masses)) / h ** 2
        return 100 * h * math.sqrt(rest + ode0 * shape(1 / (1 + z)))
    return hub


def theta_star_100(wb: float, wc: float, h: float, masses: tuple[float, ...],
                   dark_shape: Callable[[float], float] | None = None) -> float:
    hub = _hub_factory(wb, wc, h, masses, dark_shape)
    zs = DV.z_star(wb, wb + wc)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + z)))
    rs = quad(lambda z: cs(z) / hub(z), zs, np.inf, limit=400)[0]
    dm = quad(lambda z: C_KM_S / hub(z), 0, 10, limit=400)[0] + quad(lambda z: C_KM_S / hub(z), 10, zs, limit=400)[0]
    return 100 * rs / dm


@lru_cache(maxsize=1)
def planck_reference() -> float:
    wb, wc, h = DV.PLANCK_BEST
    return theta_star_100(wb, wc, h, PLANCK_MASSES_EV)


def ce_masses_ev(c: dict) -> tuple[float, ...]:
    return tuple(m / 1000.0 for m in NU.neutrino_masses_mev(c))


def pulls() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    wb, wc, h = NL.early_densities(c)
    ms = ce_masses_ev(c)
    ref = planck_reference()
    f2 = VT.density(c, VT.ADOPTED_NU)
    m3 = MV.model("b")
    d3 = lambda a: float(m3["dark"](a)) / float(m3["dark"](1.0))
    out = {"planck_ref": ref, "planck_ref_pathD": DV.theta_star_100(*DV.PLANCK_BEST)}
    for name, shape in (("L0", None), ("W2", f2), ("W3b", d3)):
        out[name] = (theta_star_100(wb, wc, h, ms, shape) - ref) / DV.THETA_SIGMA
    # 옛 경로 D(ν를 늦은 물질에서 빼고 진공에 넣음)의 L0 값과 비교
    out["L0_pathD"] = (DV.theta_star_100(wb, wc, h) - DV.theta_star_100(*DV.PLANCK_BEST)) / DV.THETA_SIGMA
    # W3 (a) 읽기: 차가운 물질이 생긴 먼지만큼 적다
    m3a = MV.model("a")
    wca = (m3a["prim"] - c["q"]) * h * h - NL.omega_nu_h2(c)
    d3a = lambda a: float(m3a["dark"](a)) / float(m3a["dark"](1.0))
    out["W3a"] = (theta_star_100(wb, wca, h, ms, d3a) - ref) / DV.THETA_SIGMA
    return out


def variant_v_with_fd_theta() -> float:
    """현재 판정(G1m 사슬, θ* 행은 W2와 같음)에서 θ* 행만 FD 값으로 바꾼 판본 V 공동 RMSE."""
    from examples.physics.rendering import ce_rendering_gradient as GD
    base = VT.score(VT.ADOPTED_NU)
    old_theta = next(o["pull"] for o in base["rows"] if o["key"] == "100 theta*")
    chi2 = GD.joint_v("G1m") ** 2 * base["N"] - old_theta ** 2 + pulls()["W2"] ** 2
    return math.sqrt(chi2 / base["N"])


def main() -> None:
    p = pulls()
    print(f"Planck reference 100theta*: FD {p['planck_ref']:.6f}  path D {p['planck_ref_pathD']:.6f}")
    for k in ("L0_pathD", "L0", "W2", "W3b", "W3a"):
        print(f"  {k:9s} theta* pull {p[k]:+.2f} sigma")


if __name__ == "__main__":
    main()
