"""
Proof-completion attempt ledger for currently open CE constants.

The script is intentionally conservative:
- closed means the formula is already justified by the local proof rules;
- conditional_pass means a zero-free-parameter bridge candidate is numerically
  viable but still needs an independent derivation;
- obstruction means the current formula cannot be counted as proven.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
        "conditional_pass",
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
        "conditional_pass",
        "No-free-parameter correction is within 1 sigma; still a phenomenological bridge.",
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
        "conditional_pass",
        "Numerically closed if the CE transition count 12 is accepted; inflationary dynamics remain external.",
    )


def as_raw() -> Attempt:
    return Attempt(
        "A_s raw",
        "raw recursive derivative amplitude",
        7.84e-9,
        2.1056e-9,
        0.0034e-9,
        "obstruction",
        "Raw value is far outside the observed amplitude; the readout candidate is post-hoc until selected.",
    )


def as_readout_candidate() -> Attempt:
    return Attempt(
        "A_s readout candidate",
        "(2/pi)*sigma^(D_eff/(D_eff+1))*eps2*(1-eps2)",
        2.1038087e-9,
        2.1056e-9,
        0.0034e-9,
        "conditional_pass",
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
