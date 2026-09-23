"""인과 잠금 — 90°의 반의 반, 과거의 나가 현재를 끈다. 원장: 43장 §43.51. 예측값을 바꾸지 않는다.

사용자 가설: “90도의 반의 반이네. 과거의 나가 끄는 거지. 인과율.” 계산 전에 적은 판본과 kill:

H1 (두 번의 반). 기록 위상 Φ = 2θ(기록 에너지 e^{iΦ})의 인과 창은 [0, π/2]다(C′). 무차별(Ind)을 “현재는 과거로 정해졌는가”에
   적용하면 Re e^{iΦ} = Im e^{iΦ}, 곧 Φ* = π/4(첫째 반). 기울기는 진폭이라 θ = Φ/2 = π/8(둘째 반).
H2 (방향). 과거가 현재를 끌면(인과) 현재 틀이 기운 쪽이고 팽창은 현재 틀에 있다 → 직접 판독 = 나이테/cos(π/8).
   현재가 과거를 끌면(반인과) 방향이 뒤집혀 나이테·cos(π/8). kill: 자료가 반인과 쪽을 고르면 H2 기각.
H3 (한 방향 결합). 기록은 되돌릴 수 없다(RL). 두 위상 모형에서 뒤로 가는 결합이 0이면 과거 위상은 자유 회전 그대로이고
   어긋남은 모두 현재에 실린다. 뒤로 가는 결합이 있으면 과거 기록이 밀린다. kill: 한 방향 결합에서도 과거가 밀림.

python -B -m examples.physics.rendering.ce_rendering_causal_lock
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_open_checks as OC
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT


def two_halvings() -> dict:
    """H1: 인과 창의 가운데(Re = Im)와 그 절반."""
    phi_star = math.atan2(1.0, 1.0)                     # cos Φ = sin Φ
    return {"window_deg": 90.0, "phi_star_deg": math.degrees(phi_star), "theta_deg": math.degrees(phi_star / 2),
            "theta_is_pi_8": abs(phi_star / 2 - math.pi / 8) < 1e-15, "load": math.sin(phi_star)}


def direction() -> dict:
    """H2: 인과(현재가 기움) 대 반인과(과거가 기움)의 직접 판독과 SH0ES pull."""
    o = OC.o1_direction()
    return {"causal": o["projection"]["H0"], "causal_SH0ES": o["projection"]["SH0ES"],
            "anti_causal": o["clock_dilation"]["H0"], "anti_causal_SH0ES": o["clock_dilation"]["SH0ES"]}


def two_phase_lock(k_back: float, k_fwd: float = 1.0, detune: float = 1 / math.sqrt(2) / 2,
                   omega: float = 0.3, dt: float = 1e-3, steps: int = 200000) -> dict:
    """H3: 과거 위상 a, 현재 위상 b. da = ω + k_back sin 2(b − a), db = ω + Δ − (k_fwd/2) sin 2(b − a).

    k_back = 0이면 한 방향(인과). 적재율 L = 2Δ/k_fwd = 1/√2로 두면 잠긴 차이는 π/8이 되어야 한다.
    반환: 잠긴 차이, 과거 위상이 자유 회전 ωt에서 벗어난 양."""
    a = b = 0.0
    for _ in range(steps):
        d = b - a
        da = omega + k_back * math.sin(2 * d)
        db = omega + detune - 0.5 * k_fwd * math.sin(2 * d)
        a += da * dt
        b += db * dt
    t = steps * dt
    return {"lock": b - a, "past_drift": a - omega * t}


def why_now() -> dict:
    """읽기: 기록 위상 Φ₀ = H_Λ t₀(접선–현)와 π/4. 정확한 등식은 §43.14에서 기각(θ* +14σ)."""
    c = R.core(R.calibrated_alpha_s()[0])
    phi0 = VT.cycle_phase(1.0, c["Om"])
    return {"phi0": phi0, "pi_4": math.pi / 4, "rel_gap": phi0 / (math.pi / 4) - 1,
            "ring_H0": 100 * NL.early_densities(c)[2]}


def main() -> None:
    print("H1:", two_halvings())
    print("H2:", {k: round(v, 2) for k, v in direction().items()})
    for kb in (0.0, 0.2, 1.0):
        r = two_phase_lock(kb)
        print(f"H3 k_back={kb}: lock {r['lock']:.6f} (pi/8 {math.pi / 8:.6f}), past drift {r['past_drift']:+.6f}")
    print("why now:", {k: round(v, 4) for k, v in why_now().items()})


if __name__ == "__main__":
    main()
