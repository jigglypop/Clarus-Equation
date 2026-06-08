"""
Primordial spectrum readout gate for the CE recursive fixed point.

This gate separates two mathematically different quantities:

1. total fixed-point response, dx/dD, which measures how the solved fixed
   point moves when D changes;
2. local residual drive, partial_D r(x; D), which measures the source that
   pushes the recursive constraint before the fixed point readjusts.

The A_s readout rule used here is conditional on the bridge claim that CMB
density perturbations observe the projected residual drive, not the total
fixed-point response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


ALPHA_S = 0.11789
SIN2_THETA_W = 4 * ALPHA_S ** (4 / 3)
DELTA = SIN2_THETA_W * (1 - SIN2_THETA_W)
D_EFF = 3 + DELTA
N_GAUGE = 12
N_E = 3 * D_EFF * N_GAUGE / 2
OBS_AS_1E9 = 2.1056
OBS_AS_SIGMA_1E9 = 0.0034


def bootstrap_fixed_point(d_eff: float = D_EFF) -> float:
    """Solve x = exp(-(1-x)D) by stable fixed-point iteration."""
    x = 0.05
    for _ in range(200):
        x_next = math.exp(-(1 - x) * d_eff)
        if abs(x_next - x) < 1e-16:
            return x_next
        x = x_next
    return x


X = bootstrap_fixed_point()
SIGMA = 1 - X


@dataclass(frozen=True)
class SpectrumReadout:
    name: str
    source_amplitude: float
    as_1e9: float
    status: str
    proof_note: str

    @property
    def sigma_offset(self) -> float:
        return (self.as_1e9 - OBS_AS_1E9) / OBS_AS_SIGMA_1E9


def as_from_source(source_amplitude: float) -> float:
    """Return A_s scaled by 1e9 from a dimensionless source amplitude."""
    value = (source_amplitude**2 / SIGMA**2) * X / (2 * math.pi * N_E**2)
    return value * 1e9


def total_fixed_point_response() -> SpectrumReadout:
    source = abs(-X * SIGMA / (1 - D_EFF * X))
    return SpectrumReadout(
        "total fixed-point response",
        source,
        as_from_source(source),
        "reject",
        "dx/dD is the full solved-point displacement and overpredicts A_s.",
    )


def residual_drive() -> SpectrumReadout:
    source = X * SIGMA
    return SpectrumReadout(
        "local residual drive",
        source,
        as_from_source(source),
        "reject",
        "partial_D r removes fixed-point readjustment but is still too large.",
    )


def phase_projected_drive() -> SpectrumReadout:
    source = (2 / math.pi) * X * SIGMA
    return SpectrumReadout(
        "phase projected drive",
        source,
        as_from_source(source),
        "candidate",
        "Half-cycle projection gives the right scale but remains high.",
    )


def integer_geometry_drive() -> SpectrumReadout:
    source = (2 / math.pi) * (SIGMA ** (3 / 4)) * X * SIGMA
    return SpectrumReadout(
        "integer geometry projected drive",
        source,
        as_from_source(source),
        "pass",
        "3 spatial dimensions projected through 4 spacetime dimensions.",
    )


def effective_geometry_drive() -> SpectrumReadout:
    gamma = D_EFF / (D_EFF + 1)
    source = (2 / math.pi) * (SIGMA**gamma) * X * SIGMA
    return SpectrumReadout(
        "effective geometry projected drive",
        source,
        as_from_source(source),
        "pass",
        "Effective dimension D_eff projected through D_eff+1 spacetime depth.",
    )


def readouts() -> list[SpectrumReadout]:
    return [
        total_fixed_point_response(),
        residual_drive(),
        phase_projected_drive(),
        integer_geometry_drive(),
        effective_geometry_drive(),
    ]


def inferred_geometry_exponent() -> float:
    """Exponent that would be inferred from the observed amplitude."""
    required_source = SIGMA * math.sqrt((OBS_AS_1E9 / 1e9) * 2 * math.pi * N_E**2 / X)
    required_projection = required_source / (X * SIGMA)
    return math.log(required_projection / (2 / math.pi)) / math.log(SIGMA)


def main() -> None:
    print("=" * 100)
    print("CE PRIMORDIAL SPECTRUM READOUT GATE")
    print("=" * 100)
    print(f"D_eff={D_EFF:.8f}, N_e={N_E:.8f}, x={X:.10f}, sigma={SIGMA:.10f}")
    print(f"gamma_eff={D_EFF/(D_EFF+1):.8f}, gamma_obs={inferred_geometry_exponent():.8f}")
    print("-" * 100)
    print(f"{'readout':38s} {'status':10s} {'source':>14s} {'A_s x 1e9':>14s} {'sigma':>10s}")
    print("-" * 100)
    for item in readouts():
        print(
            f"{item.name:38s} {item.status:10s} "
            f"{item.source_amplitude:14.8g} {item.as_1e9:14.8g} {item.sigma_offset:10.2f}"
        )
    print("-" * 100)
    print("Boundary rule:")
    print("- reject total response as an observable density source; it is a solved-point displacement.")
    print("- accept projected residual drive only as a Bridge/Phenomenology readout, not as Exact.")
    print("=" * 100)


if __name__ == "__main__":
    main()
