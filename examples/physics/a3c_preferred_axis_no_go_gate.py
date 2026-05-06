"""Preferred-axis no-go audit for scalar A3c/GER readout.

A3c/GER currently uses only scalar CE data:
    D_eff, x, sigma, gamma, P_GER, Q_A3c.

Those quantities can set amplitudes for A_s and large-angle handles, but they
cannot choose a direction on S^2.  This gate makes that limitation explicit so
the theory does not over-claim CMB preferred-axis closure.

Conclusion:
    scalar GER closes amplitude handles;
    preferred-axis closure needs a new vector/tensor boundary bridge.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)
    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_a3c = p_ger * x * sigma
    quadrupole_power = p_ger * p_ger
    hemispherical_amplitude = 2.0 * q_a3c / sigma

    scalar_inputs = {
        "alpha_s": ALPHA_S,
        "sin2_theta_w": sin2_theta_w,
        "delta": delta,
        "D_eff": d_eff,
        "x": x,
        "sigma": sigma,
        "gamma_eff": gamma_eff,
        "P_GER": p_ger,
        "Q_A3c": q_a3c,
    }

    vector_inputs: list[str] = []
    tensor_inputs: list[str] = []
    can_select_axis = bool(vector_inputs or tensor_inputs)

    print("# A3c Preferred-Axis No-Go Gate")
    print()
    print("## Scalar GER inputs")
    print()
    print("| quantity | value | SO(3) type |")
    print("|---|---:|---|")
    for name, value in scalar_inputs.items():
        print(f"| {name} | {value:.8f} | scalar |")
    print()

    print("## Amplitude handles still available")
    print()
    print("| handle | equation | value | status |")
    print("|---|---|---:|---|")
    print(f"| quadrupole power scale | `P_GER^2` | {quadrupole_power:.8f} | amplitude only |")
    print(f"| hemispherical contrast scale | `2 Q_A3c/sigma` | {hemispherical_amplitude:.8f} | amplitude only |")
    print()

    print("## Axis-selection audit")
    print()
    print("| required object | present in scalar GER? | reason |")
    print("|---|---|---|")
    print("| preferred unit vector `n_i` | no | all GER inputs are SO(3) scalars |")
    print("| quadrupole orientation tensor `T_ij` | no | no trace-free rank-2 tensor is generated |")
    print("| phase map `phi_lm` | no | no sky-basis or boundary vector is supplied |")
    print("| full likelihood | no | no CMB map/covariance enters this gate |")
    print()

    print("## Minimal bridge needed")
    print()
    print("A future axis closure must add one of:")
    print("1. a boundary vector n_i from an early-late horizon gradient,")
    print("2. a trace-free tensor T_ij from anisotropic recursive defect flow, or")
    print("3. a data-facing CMB map/covariance bridge that estimates the axis and tests the fixed GER amplitude.")
    print()

    print("## Verdict")
    print()
    print("Scalar A3c/GER cannot derive a preferred CMB axis by itself.")
    print("This is a no-go result, not a failure of the amplitude readout.")
    print("The current closure level is: amplitude handle yes; axis/phase/likelihood no.")

    if can_select_axis:
        raise SystemExit("unexpected vector/tensor axis source present")
    if not (0.0 < quadrupole_power < 1.0):
        raise SystemExit("quadrupole amplitude handle should be a suppression")
    if not (0.0 < hemispherical_amplitude < 0.2):
        raise SystemExit("hemispherical handle out of expected broad range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
