"""
Proof-completion attempt ledger for currently open CE constants.

The script is intentionally conservative:
- candidate_pass means an input-dependent bridge/readout is numerically viable
  but still needs an independent derivation;
- obstruction means the current formula cannot be counted as proven.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

if __package__:
    from .primordial_spectrum_readout_gate import (
        OBS_AS_1E9,
        OBS_AS_SIGMA_1E9,
        effective_geometry_drive,
        total_fixed_point_response,
    )
else:
    from primordial_spectrum_readout_gate import (  # type: ignore[no-redef]
        OBS_AS_1E9,
        OBS_AS_SIGMA_1E9,
        effective_geometry_drive,
        total_fixed_point_response,
    )


ALPHA_S = 0.11789
SIN2_THETA_W = 4 * ALPHA_S ** (4 / 3)
DELTA = SIN2_THETA_W * (1 - SIN2_THETA_W)
D_EFF = 3 + DELTA
EPS2 = 0.0486466333
F = 1 + ALPHA_S * D_EFF


@dataclass(frozen=True)
class Attempt:
    name: str
    formula: str
    value: float
    observed: float | None
    sigma: float | None
    status: str
    proof_status: str

    @property
    def sigma_offset(self) -> float | None:
        if self.observed is None or self.sigma is None or self.sigma <= 0:
            return None
        return (self.value - self.observed) / self.sigma


def vcb_lo() -> Attempt:
    value = ALPHA_S ** (3 / 2)
    return Attempt(
        "|V_cb| LO",
        "alpha_s^(3/2)",
        value,
        0.04153,
        0.00016,
        "obstruction",
        "Strict local average gives a 6.58 sigma failure.",
    )


def vcb_nlo_candidate() -> Attempt:
    value = ALPHA_S ** (3 / 2) * (1 + DELTA / (2 * math.pi))
    return Attempt(
        "|V_cb| NLO candidate",
        "alpha_s^(3/2) * (1 + delta/(2*pi))",
        value,
        0.04153,
        0.00016,
        "candidate_pass",
        "Accepted under the one-loop electroweak projector bridge gate.",
    )


def vus_tree() -> Attempt:
    return Attempt(
        "|V_us| tree",
        "sin^2(theta_W)",
        SIN2_THETA_W,
        0.22650,
        0.00048,
        "obstruction",
        "Tree bridge is 9.84 sigma high under the local strict reference.",
    )


def vus_one_loop() -> Attempt:
    value = SIN2_THETA_W / (1 + ALPHA_S / (2 * math.pi))
    return Attempt(
        "|V_us| one-loop candidate",
        "sin^2(theta_W)/(1 + alpha_s/(2*pi))",
        value,
        0.22650,
        0.00048,
        "candidate_pass",
        "Fixed-form readout is within 1 sigma; alpha_s remains an external input.",
    )


def ns_transition_count() -> Attempt:
    n_eff = 3 * D_EFF * 12 / 2
    value = 1 - 2 / n_eff
    return Attempt(
        "n_s transition-count candidate",
        "1 - 2/(d*D_eff*12/2)",
        value,
        0.9649,
        0.0042,
        "candidate_pass",
        "Numerically close if the CE transition count 12 is accepted; inflationary dynamics remain external.",
    )


def as_raw() -> Attempt:
    return Attempt(
        "A_s raw",
        "raw recursive derivative amplitude",
        total_fixed_point_response().as_1e9 * 1e-9,
        OBS_AS_1E9 * 1e-9,
        OBS_AS_SIGMA_1E9 * 1e-9,
        "obstruction",
        "Raw value is far outside the observed amplitude; the readout candidate is post-hoc until selected.",
    )


def as_readout_candidate() -> Attempt:
    return Attempt(
        "A_s readout candidate",
        "(2/pi)*sigma^(D_eff/(D_eff+1))*eps2*(1-eps2)",
        effective_geometry_drive().as_1e9 * 1e-9,
        OBS_AS_1E9 * 1e-9,
        OBS_AS_SIGMA_1E9 * 1e-9,
        "candidate_pass",
        "Numerically viable under the projected residual-drive readout gate.",
    )


def attempts() -> list[Attempt]:
    return [
        vcb_lo(),
        vcb_nlo_candidate(),
        vus_tree(),
        vus_one_loop(),
        ns_transition_count(),
        as_raw(),
        as_readout_candidate(),
    ]


def main() -> None:
    print("=" * 96)
    print("CE PROOF-COMPLETION ATTEMPTS")
    print("=" * 96)
    print(f"alpha_s={ALPHA_S:.8f}, sin2_theta_W={SIN2_THETA_W:.8f}, delta={DELTA:.8f}, D_eff={D_EFF:.8f}")
    print("-" * 96)
    print(f"{'name':28s} {'status':18s} {'value':>14s} {'obs':>14s} {'sigma':>10s}")
    print("-" * 96)
    for item in attempts():
        obs = "N/A" if item.observed is None else f"{item.observed:.8g}"
        sig = "N/A" if item.sigma_offset is None else f"{item.sigma_offset:+.2f}"
        print(f"{item.name:28s} {item.status:18s} {item.value:14.8g} {obs:>14s} {sig:>10s}")
    print("-" * 96)
    for item in attempts():
        print(f"- {item.name}: {item.formula}; {item.proof_status}")
    print("=" * 96)


if __name__ == "__main__":
    main()
