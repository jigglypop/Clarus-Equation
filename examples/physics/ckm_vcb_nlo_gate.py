"""
CKM |V_cb| NLO bridge gate.

The leading CE readout treats the 2->3 generation transition as an isotropic
QCD tunneling amplitude:

    |V_cb|_LO = alpha_s^(d/2)

That is numerically low under the strict local average.  This gate tests the
minimal electroweak projector correction:

    Z_cb^(1) = 1 + delta/(2*pi),
    delta = sin^2(theta_W) cos^2(theta_W)

Interpretation: the 2->3 transition is QCD-dominant at LO, while the observed
charged-current CKM element includes one electroweak-mixing projector averaged
over a closed phase loop.  This is a Bridge/Phenomenology closure, not an Exact
theorem.

Cross-reference: among the wrong-phase controls below, "1 + delta/pi" is
REJECTED as a multiplicative NLO phase correction to this tunneling amplitude.
hubble_tension.py uses the same numerical ratio delta/pi as an ADDITIVE offset
delta_eps_0 = -delta/pi in the eps-flow sector -- mathematically a different
object (different layer, different role), so this rejection does not transfer
there; but neither does this gate provide any support for it.  The cosmology
usage remains an underived closure hypothesis (Open/Phenomenology).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


ALPHA_S = 0.11789
D = 3
SIN2_THETA_W = 4 * ALPHA_S ** (4 / 3)
DELTA = SIN2_THETA_W * (1 - SIN2_THETA_W)
OBS_VCB = 0.04153
OBS_SIGMA = 0.00016


@dataclass(frozen=True)
class VcbReadout:
    name: str
    formula: str
    value: float
    status: str
    note: str

    @property
    def sigma_offset(self) -> float:
        return (self.value - OBS_VCB) / OBS_SIGMA


def vcb_lo() -> VcbReadout:
    value = ALPHA_S ** (D / 2)
    return VcbReadout(
        "LO QCD tunneling",
        "alpha_s^(d/2)",
        value,
        "reject",
        "Strict local average rejects the leading readout.",
    )


def vcb_nlo_electroweak_projector() -> VcbReadout:
    value = ALPHA_S ** (D / 2) * (1 + DELTA / (2 * math.pi))
    return VcbReadout(
        "NLO electroweak projector",
        "alpha_s^(d/2) * (1 + delta/(2*pi))",
        value,
        "pass",
        "No-free-parameter one-loop projector bridge.",
    )


def vcb_wrong_phase_controls() -> list[VcbReadout]:
    lo = ALPHA_S ** (D / 2)
    controls = [
        ("half phase", "1 + delta/pi", lo * (1 + DELTA / math.pi)),
        ("quarter phase", "1 + delta/(4*pi)", lo * (1 + DELTA / (4 * math.pi))),
        ("QCD phase", "1 + alpha_s/(2*pi)", lo * (1 + ALPHA_S / (2 * math.pi))),
        ("dimension averaged QCD", "1 + alpha_s/(2*D_eff)", lo * (1 + ALPHA_S / (2 * (3 + DELTA)))),
    ]
    return [
        VcbReadout(name, formula, value, "control", "Alternative phase/control correction.")
        for name, formula, value in controls
    ]


def readouts() -> list[VcbReadout]:
    return [vcb_lo(), vcb_nlo_electroweak_projector(), *vcb_wrong_phase_controls()]


def main() -> None:
    print("=" * 100)
    print("CE CKM |V_cb| NLO GATE")
    print("=" * 100)
    print(f"alpha_s={ALPHA_S:.8f}, sin2_theta_W={SIN2_THETA_W:.8f}, delta={DELTA:.8f}")
    print(f"obs={OBS_VCB:.8f}, sigma={OBS_SIGMA:.8f}")
    print("-" * 100)
    print(f"{'readout':28s} {'status':10s} {'value':>14s} {'sigma':>10s} {'formula'}")
    print("-" * 100)
    for item in readouts():
        print(
            f"{item.name:28s} {item.status:10s} "
            f"{item.value:14.8g} {item.sigma_offset:10.2f} {item.formula}"
        )
    print("-" * 100)
    print("Boundary rule:")
    print("- LO remains rejected under the strict average.")
    print("- NLO is accepted only as a one-loop electroweak projector Bridge.")
    print("=" * 100)


if __name__ == "__main__":
    main()
