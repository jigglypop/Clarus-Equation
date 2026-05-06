"""Conditional axis bridge for A3c/GER large-angle CMB handles.

Scalar GER fixes amplitudes but cannot choose a sky direction.  This gate shows
the minimal conditional bridge: if an external/theoretical symmetry-breaking
bridge supplies a unit vector n_i, CE fixes the modulation amplitude and the
traceless quadrupole tensor normalization.

No observed CMB axis is fitted here.  The default n=(0,0,1) is a gauge choice
used only to verify tensor algebra.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable


ALPHA_S = 0.11789
D_SPATIAL = 3.0


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def norm3(v: Iterable[float]) -> tuple[float, float, float]:
    a, b, c = [float(x) for x in v]
    n = math.sqrt(a * a + b * b + c * c)
    if n <= 0.0:
        raise ValueError("axis vector must be nonzero")
    return a / n, b / n, c / n


def quadrupole_tensor(axis: tuple[float, float, float], amplitude: float) -> list[list[float]]:
    return [
        [amplitude * (axis[i] * axis[j] - (1.0 / 3.0 if i == j else 0.0)) for j in range(3)]
        for i in range(3)
    ]


def trace(matrix: list[list[float]]) -> float:
    return sum(matrix[i][i] for i in range(3))


def frobenius_norm(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(matrix[i][j] ** 2 for i in range(3) for j in range(3)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
        help="conditional unit-axis source; default is gauge z-axis",
    )
    args = parser.parse_args()

    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)

    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_a3c = p_ger * x * sigma
    quadrupole_power_handle = p_ger * p_ger
    hemispherical_amplitude = 2.0 * q_a3c / sigma
    large_angle_fractional = q_a3c / sigma

    axis = norm3(args.axis)
    t_quad = quadrupole_tensor(axis, quadrupole_power_handle)
    tensor_norm = frobenius_norm(t_quad)
    expected_norm = quadrupole_power_handle * math.sqrt(2.0 / 3.0)
    dipole_vector = [hemispherical_amplitude * component for component in axis]
    dipole_norm = math.sqrt(sum(component * component for component in dipole_vector))

    print("# A3c Conditional Axis Bridge Gate")
    print()
    print("## Scalar GER amplitudes")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"P_GER = (2/pi)sigma^[D/(D+1)] = {p_ger:.8f}")
    print(f"Q_A3c = P_GER x(1-x) = {q_a3c:.8f}")
    print(f"quadrupole power handle S_Q = P_GER^2 = {quadrupole_power_handle:.8f}")
    print(f"hemispherical modulation A_H = 2Q_A3c/sigma = {hemispherical_amplitude:.8f}")
    print(f"large-angle fractional residual = Q_A3c/sigma = {large_angle_fractional:.8f}")
    print()

    print("## Conditional axis object")
    print()
    print(f"input axis normalized n = ({axis[0]:+.8f}, {axis[1]:+.8f}, {axis[2]:+.8f})")
    print("dipole/modulation vector m_i = A_H n_i")
    print(
        "m = "
        f"({dipole_vector[0]:+.8f}, {dipole_vector[1]:+.8f}, {dipole_vector[2]:+.8f})"
    )
    print(f"|m| = {dipole_norm:.8f}")
    print()

    print("## Conditional quadrupole tensor")
    print()
    print("T_ij = S_Q (n_i n_j - delta_ij/3)")
    print("| i | T_i1 | T_i2 | T_i3 |")
    print("|---:|---:|---:|---:|")
    for i, row in enumerate(t_quad, start=1):
        print(f"| {i} | {row[0]:+.8f} | {row[1]:+.8f} | {row[2]:+.8f} |")
    print()
    print(f"trace(T) = {trace(t_quad):+.3e}")
    print(f"||T||_F = {tensor_norm:.8f}")
    print(f"expected ||T||_F = S_Q sqrt(2/3) = {expected_norm:.8f}")
    print()

    print("## Closure status")
    print()
    print("| layer | closed by CE? | object |")
    print("|---|---|---|")
    print("| scalar amplitudes | yes/candidate | P_GER, Q_A3c, S_Q, A_H |")
    print("| axis direction | no | requires n_i bridge |")
    print("| tensor algebra after n_i | yes/conditional | T_ij traceless by construction |")
    print("| data likelihood | no | requires observed CMB map/covariance |")
    print()

    print("## Verdict")
    print()
    print("Given a unit axis n_i, A3c/GER fixes the large-angle modulation and quadrupole tensor.")
    print("The missing part is now sharply isolated: derive or ingest n_i.")

    if abs(trace(t_quad)) > 1e-12:
        raise SystemExit("quadrupole tensor should be traceless")
    if abs(tensor_norm - expected_norm) > 1e-12:
        raise SystemExit("quadrupole tensor norm check failed")
    if abs(dipole_norm - hemispherical_amplitude) > 1e-12:
        raise SystemExit("dipole vector amplitude check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
