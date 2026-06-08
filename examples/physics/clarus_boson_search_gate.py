"""
Clarus-field pole/correlation search gate.

The 29.65 MeV object is registered here as the inverse correlation length /
two-point-function pole of the Clarus field, then read in particle language as
a local scalar bridge.  A resonance-like signal would support that pole bridge;
a null result constrains or rejects the particle-language readout, not the core
definition of the Clarus field.
"""

from __future__ import annotations

from dataclasses import dataclass


ALPHA_S = 0.11789
PROTON_MASS_MEV = 938.2720813
SIN2_THETA_W = 4 * ALPHA_S ** (4 / 3)
DELTA = SIN2_THETA_W * (1 - SIN2_THETA_W)
LAMBDA_HP = DELTA**2
M_PHI_MEV = PROTON_MASS_MEV * LAMBDA_HP
HBAR_C_MEV_FM = 197.3269804
COMPTON_FM = HBAR_C_MEV_FM / M_PHI_MEV

# From the local docs: alpha_s uncertainty propagates to about 0.42 MeV.
MASS_SIGMA_MEV = 0.42
DISCOVERY_SIGMA = 3
MASS_WINDOW = (
    M_PHI_MEV - DISCOVERY_SIGMA * MASS_SIGMA_MEV,
    M_PHI_MEV + DISCOVERY_SIGMA * MASS_SIGMA_MEV,
)


@dataclass(frozen=True)
class ExperimentalResult:
    """Minimal normalized experimental input for the field-pole gate."""

    experiment: str
    mass_mev: float | None
    significance_sigma: float | None
    pole_compatible: bool
    excludes_mass_window: bool = False
    excludes_bridge_coupling: bool = False
    notes: str = ""


@dataclass(frozen=True)
class GateDecision:
    status: str
    reason: str


def in_mass_window(mass_mev: float | None) -> bool:
    if mass_mev is None:
        return False
    lo, hi = MASS_WINDOW
    return lo <= mass_mev <= hi


def classify_result(result: ExperimentalResult) -> GateDecision:
    """Classify a single search result against the registered CE gate."""
    if result.excludes_mass_window and result.excludes_bridge_coupling:
        return GateDecision(
            "bridge_rejected",
            "The local particle-language pole bridge is excluded at the benchmark coupling; the core Clarus field is not falsified.",
        )

    if (
        in_mass_window(result.mass_mev)
        and result.pole_compatible
        and result.significance_sigma is not None
        and result.significance_sigma >= 5
    ):
        return GateDecision(
            "pole_confirmed",
            "Five-sigma pole-compatible structure inside the CE mass window.",
        )

    if in_mass_window(result.mass_mev) and result.pole_compatible:
        return GateDecision(
            "pole_candidate",
            "Pole-compatible excess in the CE mass window, below discovery threshold.",
        )

    if result.excludes_mass_window:
        return GateDecision(
            "bridge_constrained",
            "The registered mass window is constrained, but the pole bridge is not fully excluded.",
        )

    return GateDecision("open_test", "No decisive field-pole hit or bridge exclusion.")


def registered_search_channels() -> list[dict[str, str]]:
    return [
        {
            "channel": "e+e- or missing-mass resonance scan",
            "experiments": "PADME / NA64",
            "pass_condition": "pole-compatible structure in the registered mass window",
        },
        {
            "channel": "K+ -> pi+ + invisible or displaced hidden mediator",
            "experiments": "NA62",
            "pass_condition": "missing-energy or displaced signature compatible with the 29.65 MeV field-pole bridge",
        },
        {
            "channel": "precision lepton-proton scattering",
            "experiments": "PRad-II / MUSE",
            "pass_condition": "correlation-length deviation consistent with lambda_C = 6.66 fm",
        },
    ]


def main() -> None:
    print("=" * 96)
    print("CE CLARUS FIELD-POLE SEARCH GATE")
    print("=" * 96)
    print(f"delta={DELTA:.8f}")
    print(f"lambda_HP=delta^2={LAMBDA_HP:.8f}")
    print(f"m_phi=m_p*delta^2={M_PHI_MEV:.5f} MeV")
    print(f"3-sigma mass window=[{MASS_WINDOW[0]:.3f}, {MASS_WINDOW[1]:.3f}] MeV")
    print(f"Compton length={COMPTON_FM:.5f} fm")
    print("-" * 96)
    for channel in registered_search_channels():
        print(f"- {channel['experiments']}: {channel['channel']}")
        print(f"  pass: {channel['pass_condition']}")
    print("-" * 96)
    examples = [
        ExperimentalResult("PADME-like X17", 17.0, 5.0, True),
        ExperimentalResult("CE-window pole excess", 29.7, 3.0, True),
        ExperimentalResult("CE-window pole hit", 29.7, 5.2, True),
        ExperimentalResult("Bridge exclusion", None, None, False, True, True),
    ]
    for item in examples:
        decision = classify_result(item)
        print(f"{item.experiment:22s} -> {decision.status:11s} {decision.reason}")
    print("=" * 96)


if __name__ == "__main__":
    main()
