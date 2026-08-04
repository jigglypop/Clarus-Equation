"""Candidate funnel for negative null stress sourced by the Clarus sector."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .spatial_folding import SPEED_OF_LIGHT_M_S


HBAR_J_S = 1.054571817e-34


@dataclass(frozen=True)
class NonminimalNullAudit:
    nonminimal_coupling: float
    field_value_planck: float
    affine_first_derivative: float
    affine_second_derivative: float
    second_derivative_field_squared: float
    null_numerator: float
    effective_planck_factor: float
    effective_null_source: float
    locally_violates_effective_nec: bool
    canonical_kinetic_sign_retained: bool
    positive_effective_planck_mass: bool
    local_candidate_survives: bool
    global_solution_derived: bool


@dataclass(frozen=True)
class AveragedNonminimalAudit:
    gradient_squared_integral: float
    endpoint_field_squared_derivative_jump: float
    averaged_null_numerator: float
    averaged_nec_violated: bool
    localized_vacuum_boundary_conditions: bool
    boundary_or_topology_support_required: bool


@dataclass(frozen=True)
class NegativeSourceCandidate:
    name: str
    frontier: str
    ce_native_components: str
    local_null_gate: str
    averaged_null_gate: str
    ghost_gradient_gate: str
    renormalized_stress_gate: str
    backreaction_gate: str
    decisive_next_calculation: str


@dataclass(frozen=True)
class CasimirPlateScaleAudit:
    required_null_magnitude_j_m3: float
    plate_separation_m: float
    ce_correlation_length_m: float
    separation_to_ce_correlation_ratio: float
    normal_null_projection_factor: float
    macroscopic_throat_source_established: bool


@dataclass(frozen=True)
class EffectivePlanckAmplificationAudit:
    required_amplification: float
    required_effective_planck_factor: float
    nonminimal_coupling: float
    critical_field_planck: float
    field_for_required_factor_planck: float
    relative_distance_below_critical: float
    algebraically_closes_density_gap: bool
    regular_effective_gravity_limit_established: bool


def nonminimal_scalar_null_audit(
    *,
    nonminimal_coupling: float,
    field_value_planck: float,
    affine_first_derivative: float,
    affine_second_derivative: float,
) -> NonminimalNullAudit:
    """Audit the local Jordan-frame null numerator in Planck-normalized units.

    For an action containing ``(1 - xi*phi**2) R / 2`` and a canonical scalar,
    the rearranged null Einstein equation has numerator

    ``(phi')**2 - xi * (phi**2)''``.

    This necessary local gate does not solve the field equations or perturbative
    stability problem.
    """

    xi = float(nonminimal_coupling)
    field = float(field_value_planck)
    first = float(affine_first_derivative)
    second = float(affine_second_derivative)
    if not all(math.isfinite(value) for value in (xi, field, first, second)):
        raise ValueError("all nonminimal scalar inputs must be finite")
    if xi < 0.0:
        raise ValueError("nonminimal_coupling must be non-negative")

    field_squared_second = 2.0 * (first**2 + field * second)
    numerator = first**2 - xi * field_squared_second
    planck_factor = 1.0 - xi * field**2
    planck_positive = planck_factor > 0.0
    effective = numerator / planck_factor if planck_positive else math.nan
    violates = planck_positive and effective < 0.0
    return NonminimalNullAudit(
        nonminimal_coupling=xi,
        field_value_planck=field,
        affine_first_derivative=first,
        affine_second_derivative=second,
        second_derivative_field_squared=field_squared_second,
        null_numerator=numerator,
        effective_planck_factor=planck_factor,
        effective_null_source=effective,
        locally_violates_effective_nec=violates,
        canonical_kinetic_sign_retained=True,
        positive_effective_planck_mass=planck_positive,
        local_candidate_survives=violates and planck_positive,
        global_solution_derived=False,
    )


def averaged_nonminimal_null_audit(
    *,
    nonminimal_coupling: float,
    gradient_squared_integral: float,
    endpoint_field_squared_derivative_jump: float,
) -> AveragedNonminimalAudit:
    """Integrate the unrearranged local numerator along an affine null curve."""

    xi = float(nonminimal_coupling)
    gradient = float(gradient_squared_integral)
    endpoint_jump = float(endpoint_field_squared_derivative_jump)
    if not all(math.isfinite(value) for value in (xi, gradient, endpoint_jump)):
        raise ValueError("all averaged-null inputs must be finite")
    if xi < 0.0 or gradient < 0.0:
        raise ValueError("coupling and gradient_squared_integral must be non-negative")

    averaged = gradient - xi * endpoint_jump
    localized = endpoint_jump == 0.0
    violates = averaged < 0.0
    return AveragedNonminimalAudit(
        gradient_squared_integral=gradient,
        endpoint_field_squared_derivative_jump=endpoint_jump,
        averaged_null_numerator=averaged,
        averaged_nec_violated=violates,
        localized_vacuum_boundary_conditions=localized,
        boundary_or_topology_support_required=not violates and localized,
    )


def casimir_plate_scale_audit(
    *,
    required_null_magnitude_j_m3: float,
    ce_correlation_length_m: float,
    normal_null_projection_factor: float = 4.0,
) -> CasimirPlateScaleAudit:
    """Invert the ideal electromagnetic parallel-plate Casimir null stress.

    The ideal energy density magnitude is ``pi^2 hbar c / (720 a^4)``.  For a
    null direction normal to the plates, the standard stress has
    ``|rho+p_normal| = 4 |rho|``.  This is a scale control, not a realizable
    spherical source or a Clarus-field stress derivation.
    """

    required = float(required_null_magnitude_j_m3)
    correlation = float(ce_correlation_length_m)
    factor = float(normal_null_projection_factor)
    if not all(math.isfinite(value) and value > 0.0 for value in (required, correlation, factor)):
        raise ValueError("Casimir scale inputs must be finite and positive")

    coefficient = factor * math.pi**2 * HBAR_J_S * SPEED_OF_LIGHT_M_S / 720.0
    separation = (coefficient / required) ** 0.25
    return CasimirPlateScaleAudit(
        required_null_magnitude_j_m3=required,
        plate_separation_m=separation,
        ce_correlation_length_m=correlation,
        separation_to_ce_correlation_ratio=separation / correlation,
        normal_null_projection_factor=factor,
        macroscopic_throat_source_established=False,
    )


def effective_planck_amplification_audit(
    *,
    required_amplification: float,
    nonminimal_coupling: float,
) -> EffectivePlanckAmplificationAudit:
    """Invert amplification by ``1 / (1 - xi*phi^2)`` near its zero."""

    gain = float(required_amplification)
    xi = float(nonminimal_coupling)
    if not math.isfinite(gain) or gain <= 1.0:
        raise ValueError("required_amplification must be finite and greater than one")
    if not math.isfinite(xi) or xi <= 0.0:
        raise ValueError("nonminimal_coupling must be finite and positive")

    factor = 1.0 / gain
    critical = 1.0 / math.sqrt(xi)
    root = math.sqrt(1.0 - factor)
    field = critical * root
    # Stable form of 1 - sqrt(1 - factor) for factor far below machine epsilon.
    relative_distance = factor / (1.0 + root)
    return EffectivePlanckAmplificationAudit(
        required_amplification=gain,
        required_effective_planck_factor=factor,
        nonminimal_coupling=xi,
        critical_field_planck=critical,
        field_for_required_factor_planck=field,
        relative_distance_below_critical=relative_distance,
        algebraically_closes_density_gap=True,
        regular_effective_gravity_limit_established=False,
    )


def clarus_negative_source_funnel() -> tuple[NegativeSourceCandidate, ...]:
    """Rank candidate mechanisms by CE proximity, without assigning fake odds."""

    return (
        NegativeSourceCandidate(
            name="CE Casimir boundary + general-redshift throat",
            frontier="DEFERRED_PHYSICAL_BOUNDARY",
            ce_native_components="CE Casimir estimate and GR throat geometry",
            local_null_gate="PASS IN GLOBAL VARIABLE-ANISOTROPY TARGET",
            averaged_null_gate="GLOBAL CONSERVATION PASS FOR TARGET",
            ghost_gradient_gate="NO SUBNUCLEAR REFLECTOR MODEL",
            renormalized_stress_gate="OPEN",
            backreaction_gate="GLOBAL INVERSE-GEOMETRY CONTROL PASS",
            decisive_next_calculation="derive non-material CE boundary before further geometry work",
        ),
        NegativeSourceCandidate(
            name="CE nonminimal scalar + Casimir completion",
            frontier="FRONTIER_A",
            ce_native_components="xi R Phi^2 and CE Casimir estimate",
            local_null_gate="CONDITIONAL PASS",
            averaged_null_gate="OPEN WITH BOUNDARY TERMS",
            ghost_gradient_gate="CANONICAL SIGN; FULL MODES OPEN",
            renormalized_stress_gate="OPEN",
            backreaction_gate="OPTIONAL COMPLETION IF GLOBAL CASIMIR ODE FAILS",
            decisive_next_calculation="coupled scalar-boundary throat series",
        ),
        NegativeSourceCandidate(
            name="CE specific-wavelength multi-mode boundary resonance",
            frontier="DEFERRED_PHYSICAL_RESONANCE",
            ce_native_components="Casimir boundary ansatz; corrected target is 153 GeV",
            local_null_gate="PASS IN IDEAL CASIMIR CONTROL",
            averaged_null_gate="VARIABLE ANISOTROPY REQUIRED",
            ghost_gradient_gate="NO PHYSICAL 4.05 AM BOUNDARY",
            renormalized_stress_gate="OPEN",
            backreaction_gate="GLOBAL TARGET PASS / PHYSICAL SCALE FAIL",
            decisive_next_calculation="derive CE renormalized stress without material mirrors",
        ),
        NegativeSourceCandidate(
            name="CE nonminimal scalar vacuum polarization",
            frontier="DEFERRED_MACRO",
            ce_native_components="xi R Phi^2 with xi about 0.49",
            local_null_gate="CONDITIONAL PASS",
            averaged_null_gate="CURVED-SPACE STATE REQUIRED",
            ghost_gradient_gate="FULL QUADRATIC MODES OPEN",
            renormalized_stress_gate="OPEN",
            backreaction_gate="1 m LARGE-MASS SCALE FAILS",
            decisive_next_calculation="seek massless/collective sector; heavy CE pole is insufficient",
        ),
        NegativeSourceCandidate(
            name="CE+SM charged-fermion magnetic Casimir mapping",
            frontier="DEFERRED_HUMAN_SCALE",
            ce_native_components="SM charged fermions plus gravity sectors",
            local_null_gate="PASS IN MMP CONTROL",
            averaged_null_gate="PASS IN MMP CONTROL",
            ghost_gradient_gate="CONDITIONAL IN CONTROL",
            renormalized_stress_gate="EXTERNAL LONG-WORMHOLE CONTROL ONLY",
            backreaction_gate="PASS IN EXTERNAL 4D MODEL",
            decisive_next_calculation="requires a new sub-micro-eV charged CE fermion and flux action",
        ),
        NegativeSourceCandidate(
            name="CE two-boundary/double-trace state",
            frontier="FRONTIER_B",
            ce_native_components="path boundary language only",
            local_null_gate="PASS IN GJW CONTROL",
            averaged_null_gate="PASS IN ADS CONTROL",
            ghost_gradient_gate="CONDITIONAL IN CONTROL",
            renormalized_stress_gate="NOT MAPPED TO CE",
            backreaction_gate="PASS PERTURBATIVELY IN ADS",
            decisive_next_calculation="derive a CE two-boundary interaction",
        ),
        NegativeSourceCandidate(
            name="CE resonance-Q only",
            frontier="DEFERRED",
            ce_native_components="correlation-length ansatz",
            local_null_gate="SIGN NOT IDENTIFIED",
            averaged_null_gate="OPEN",
            ghost_gradient_gate="OPEN",
            renormalized_stress_gate="OPEN",
            backreaction_gate="OPEN",
            decisive_next_calculation="derive spectral residue before using Q",
        ),
        NegativeSourceCandidate(
            name="beyond-Horndeski extension",
            frontier="EXTERNAL_EXTENSION",
            ce_native_components="not present in current CE action",
            local_null_gate="MODEL-DEPENDENT PASS",
            averaged_null_gate="MODEL-DEPENDENT",
            ghost_gradient_gate="PARTIAL EXTERNAL PASS",
            renormalized_stress_gate="CLASSICAL MODIFIED GRAVITY",
            backreaction_gate="EXTERNAL EXAMPLES; STABILITY INCOMPLETE",
            decisive_next_calculation="justify adding degenerate higher derivatives",
        ),
        NegativeSourceCandidate(
            name="phantom Clarus scalar",
            frontier="REJECTED",
            ce_native_components="requires kinetic-sign replacement",
            local_null_gate="PASS",
            averaged_null_gate="POSSIBLE",
            ghost_gradient_gate="FAIL GHOST",
            renormalized_stress_gate="NOT REACHED",
            backreaction_gate="NOT REACHED",
            decisive_next_calculation="none without ghost-free completion",
        ),
    )
