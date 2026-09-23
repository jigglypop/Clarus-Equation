"""돌면서 퍼지는 시공간의 나선 긴장. 원장: 43장 §43.15.

잠긴 모듈(v1–v4)은 import만 한다.

C4  시간축은 접선–현 정리에 따라 ω = H_Λ/2로 돌고, 우주는 H로 퍼진다. 퍼지는 방향과 실제 진행 방향의
    각은 tan ψ = ω/H(로그 나선의 기울기)다. 직접 판독은 두 속도를 합친 크기로 읽는다:
        H_direct² = H_rings² + (H_Λ/2)²,  H_direct/H_rings = √(1 + Ω_Λ/4).
    π/8을 쓰지 않는다. 이 회전은 관측된 우주 소용돌이 한계 때문에 실제 공간의 회전이 아니라
    관계공간(허수 시간) 안의 회전으로만 읽는다.

python -B -m examples.physics.rendering.ce_rendering_spiral
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_cycle as CY
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R


def spiral_pitch(omega_lambda: float, h_ratio: float = 1.0) -> float:
    """tan ψ = ω/H = √Ω_Λ H0 / (2H). h_ratio = H/H0 (오늘 1)."""
    return math.atan(math.sqrt(omega_lambda) / (2.0 * h_ratio))


def direct_over_rings(omega_lambda: float) -> float:
    return math.sqrt(1.0 + omega_lambda / 4.0)


def direct_readout(c: dict) -> float:
    return 100.0 * PL.h_rings(c) * direct_over_rings(1.0 - c["Om"])


def three_routes(c: dict) -> dict:
    return {"vacuum channel pi/8": math.pi / 8,
            "tangent-chord of cycle age": CY.tangent_chord_tilt(c["Om"]),
            "spiral pitch arctan(omega/H)": spiral_pitch(1.0 - c["Om"])}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    for k, v in three_routes(c).items():
        print(f"{k:32s} {v:.5f} rad ({math.degrees(v):.3f} deg)")
    hd = direct_readout(c)
    print(f"H_direct = sqrt(H_rings^2 + (H_L/2)^2) = {hd:.2f}; ratio {direct_over_rings(1 - c['Om']):.5f} "
          f"vs 1/cos(pi/8) {1 / math.cos(math.pi / 8):.5f}")
    for name, direct, v, up, dn in R.H0_READOUTS:
        if direct:
            print(f"   {name:10s} pull {(hd - v) / (up if hd >= v else dn):+.2f}")
    print(f"asymptotic pitch (H -> H_L): {math.degrees(spiral_pitch(1.0 - c['Om'], math.sqrt(1 - c['Om']))):.2f} deg")


if __name__ == "__main__":
    main()
