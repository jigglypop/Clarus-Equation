"""First-pass falsification gates for CE wormhole stress-source candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .spatial_folding import NEWTON_G_M3_KG_S2, SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class ResonanceSourceAudit:
    throat_radius_m: float
    required_nec_magnitude_j_m3: float
    base_negative_density_j_m3: float
    base_correlation_length_m: float
    density_gap: float
    coherence_q_required: float
    density_q_required: float
    combined_q_required: float
    density_gain_exponent: float
    assumed_density_at_combined_q_j_m3: float
    assumed_correlation_at_combined_q_m: float
    numerical_gates_pass_under_ansatz: bool
    density_scaling_law_derived_from_ce: bool
    renormalized_stress_tensor_derived: bool


@dataclass(frozen=True)
class ScalarNullEnergyAudit:
    kinetic_sign: float
    null_directional_derivative: float
    null_projection: float
    violates_nec: bool
    ghost_free_kinetic_term: bool
    supports_throat_and_is_ghost_free: bool


@dataclass(frozen=True)
class CandidateStatus:
    name: str
    nec_or_anec_violation: str
    renormalized_full_stress: str
    conservation: str
    quantum_inequality: str
    backreaction_solution: str
    stability: str
    derived_from_ce: str
    first_failed_or_open_gate: str


def resonance_source_audit(
    *,
    throat_radius_m: float,
    base_negative_density_j_m3: float,
    base_correlation_length_m: float,
    density_gain_exponent: float,
    shape_derivative: float = -1.0,
) -> ResonanceSourceAudit:
    """Screen a hypothetical ``density(Q)=density(1)*Q**p`` resonance law.

    CE currently supplies no such stress-density scaling law.  The calculation
    therefore reports what a chosen exponent would require, not a derivation.
    """

    radius = float(throat_radius_m)
    density = float(base_negative_density_j_m3)
    correlation = float(base_correlation_length_m)
    exponent = float(density_gain_exponent)
    b_prime = float(shape_derivative)
    for value, name in (
        (radius, "throat_radius_m"),
        (density, "base_negative_density_j_m3"),
        (correlation, "base_correlation_length_m"),
        (exponent, "density_gain_exponent"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(b_prime) or b_prime >= 1.0:
        raise ValueError("shape_derivative must be finite and below one")

    required = (
        SPEED_OF_LIGHT_M_S**4
        * (1.0 - b_prime)
        / (8.0 * math.pi * NEWTON_G_M3_KG_S2 * radius**2)
    )
    density_gap = required / density
    coherence_q = max(1.0, radius / correlation)
    density_q = max(1.0, density_gap ** (1.0 / exponent))
    combined_q = max(coherence_q, density_q)
    assumed_density = density * combined_q**exponent
    assumed_correlation = correlation * combined_q
    return ResonanceSourceAudit(
        throat_radius_m=radius,
        required_nec_magnitude_j_m3=required,
        base_negative_density_j_m3=density,
        base_correlation_length_m=correlation,
        density_gap=density_gap,
        coherence_q_required=coherence_q,
        density_q_required=density_q,
        combined_q_required=combined_q,
        density_gain_exponent=exponent,
        assumed_density_at_combined_q_j_m3=assumed_density,
        assumed_correlation_at_combined_q_m=assumed_correlation,
        numerical_gates_pass_under_ansatz=(
            assumed_density >= required and assumed_correlation >= radius
        ),
        density_scaling_law_derived_from_ce=False,
        renormalized_stress_tensor_derived=False,
    )


def scalar_null_energy_audit(
    *,
    null_directional_derivative: float,
    kinetic_sign: float = 1.0,
) -> ScalarNullEnergyAudit:
    """Audit ``T_kk = sign * (k^mu partial_mu sigma)^2``.

    ``kinetic_sign=+1`` is a canonical scalar and ``-1`` is a phantom kinetic
    term.  The latter can violate NEC but fails the minimal ghost-free gate.
    """

    derivative = float(null_directional_derivative)
    sign = float(kinetic_sign)
    if not math.isfinite(derivative):
        raise ValueError("null_directional_derivative must be finite")
    if sign not in (-1.0, 1.0):
        raise ValueError("kinetic_sign must be +1 or -1")

    projection = sign * derivative**2
    violates = projection < 0.0
    ghost_free = sign > 0.0
    return ScalarNullEnergyAudit(
        kinetic_sign=sign,
        null_directional_derivative=derivative,
        null_projection=projection,
        violates_nec=violates,
        ghost_free_kinetic_term=ghost_free,
        supports_throat_and_is_ghost_free=violates and ghost_free,
    )


def source_candidate_catalog() -> tuple[CandidateStatus, ...]:
    """Return an auditable status matrix; literature controls are not CE proofs."""

    return (
        CandidateStatus(
            name="CE static Casimir cell",
            nec_or_anec_violation="CONDITIONAL",
            renormalized_full_stress="OPEN",
            conservation="OPEN",
            quantum_inequality="OPEN",
            backreaction_solution="OPEN",
            stability="OPEN",
            derived_from_ce="PARTIAL",
            first_failed_or_open_gate="1 m density and coherence FAIL",
        ),
        CandidateStatus(
            name="CE resonance-Q ansatz",
            nec_or_anec_violation="OPEN",
            renormalized_full_stress="OPEN",
            conservation="OPEN",
            quantum_inequality="OPEN",
            backreaction_solution="OPEN",
            stability="OPEN",
            derived_from_ce="NO",
            first_failed_or_open_gate="negative-stress Q scaling OPEN",
        ),
        CandidateStatus(
            name="CE minimally coupled canonical scalar channel",
            nec_or_anec_violation="FAIL",
            renormalized_full_stress="CLASSICAL FORM ONLY",
            conservation="CONDITIONAL ON SHELL",
            quantum_inequality="NOT REACHED",
            backreaction_solution="NOT REACHED",
            stability="CANONICAL KINETIC SIGN",
            derived_from_ce="ACTION ANSATZ",
            first_failed_or_open_gate="classical T_kk is non-negative",
        ),
        CandidateStatus(
            name="phantom scalar control",
            nec_or_anec_violation="PASS",
            renormalized_full_stress="MODEL-DEPENDENT",
            conservation="CONDITIONAL ON SHELL",
            quantum_inequality="OPEN",
            backreaction_solution="OPEN",
            stability="FAIL GHOST GATE",
            derived_from_ce="NO",
            first_failed_or_open_gate="wrong-sign kinetic instability",
        ),
        CandidateStatus(
            name="quantum negative-energy state control",
            nec_or_anec_violation="LOCAL PASS / ANEC RESTRICTED",
            renormalized_full_stress="STATE-DEPENDENT",
            conservation="PASS FOR SPECIFIED QFT",
            quantum_inequality="RESTRICTED IN DURATION/MAGNITUDE",
            backreaction_solution="OPEN",
            stability="TRANSIENT/OPEN",
            derived_from_ce="NO",
            first_failed_or_open_gate="sustained averaged stress and backreaction OPEN",
        ),
        CandidateStatus(
            name="Gao-Jafferis-Wall control",
            nec_or_anec_violation="PASS IN ADS MODEL",
            renormalized_full_stress="PASS IN MODEL",
            conservation="PASS IN MODEL",
            quantum_inequality="MODEL-CONSISTENT",
            backreaction_solution="PASS PERTURBATIVELY",
            stability="CONDITIONAL",
            derived_from_ce="NO",
            first_failed_or_open_gate="CE action/boundary coupling mapping OPEN",
        ),
        CandidateStatus(
            name="Maldacena-Milekhin-Popov control",
            nec_or_anec_violation="PASS IN 4D MODEL",
            renormalized_full_stress="PASS IN MODEL",
            conservation="PASS IN MODEL",
            quantum_inequality="MODEL-SPECIFIC",
            backreaction_solution="PASS IN MODEL",
            stability="CONDITIONAL",
            derived_from_ce="NO",
            first_failed_or_open_gate="CE field-content mapping OPEN",
        ),
    )
