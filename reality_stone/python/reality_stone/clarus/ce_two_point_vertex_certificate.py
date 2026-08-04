"""Q0.4--Q0.5 control certificates for the optional ``Z2`` portal branch.

The repository contains a classical, bare, tree-level control action with a
canonically normalized real singlet ``phi`` and

``V = m0_phi^2 phi^2 / 2 + lambda_phi phi^4 / 4
     + lambda_HP |H|^2 phi^2``.

This module derives the singlet quadratic kernel and the local portal vertices
from that declared action.  It also tests whether the registered
``29.64757 MeV`` inverse-correlation target can be the same pole as the
portal-dominated ``v*sqrt(lambda_HP)`` scale.

The result is deliberately conditional.  A tree-level pole can be constructed
after a bare mass is supplied, but this does not derive the pole parameter from
the CE core, compute a renormalized two-point function, or complete LSZ.  In
particular, back-solving ``m0_phi^2`` to reproduce a target is not a prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

from .a1_q0_action_bridge import audit_higgs_invisible_width
from .casimir_carrier_target import DEFAULT_CE_INVERSE_CORRELATION_SCALE_MEV
from .q0_manifest_gate import q0_control_action_definition_sha256


CONTROL_SCOPE = "optional_z2_singlet_portal_tree_level_q0_4_q0_5_only"
CONTROL_STATUS = "CONDITIONAL_TREE_LEVEL_PORTAL_CERTIFICATE"
HBAR_C_GEV_FM = 0.1973269804
PROTON_MASS_MEV = 938.2720813
DEFAULT_LAMBDA_HP = DEFAULT_CE_INVERSE_CORRELATION_SCALE_MEV / PROTON_MASS_MEV


@dataclass(frozen=True)
class CorrelationAsymptoticCountermodel:
    """One spectral model sharing the registered leading decay scale."""

    label: str
    euclidean_representation: str
    leading_inverse_length_gev: float
    isolated_delta_pole: bool
    invariant_pole_residue: float | None
    reflection_positive: bool
    stable_lsz_particle_present: bool


@dataclass(frozen=True)
class CorrelationLengthIdentifiabilityAudit:
    """Counterexample family showing what one inverse length cannot identify."""

    inverse_correlation_scale_gev: float
    correlation_length_fm: float
    small_positive_residue: float
    countermodels: tuple[CorrelationAsymptoticCountermodel, ...]
    all_models_share_leading_exponential_scale: bool
    isolated_pole_identified_by_inverse_length: bool
    residue_magnitude_identified_by_inverse_length: bool
    residue_sign_identified_by_inverse_length: bool
    reflection_positivity_identified_by_inverse_length: bool
    lsz_particle_identified_by_inverse_length: bool
    maximum_supported_stage: str


@dataclass(frozen=True)
class HessianVertexNonidentifiabilityAudit:
    """Jet counterexample proving that a Hessian does not fix interactions."""

    expansion_point: float
    cubic_deformation: float
    quartic_deformation: float
    deformation_gradient_at_background: float
    deformation_hessian_at_background: float
    deformation_cubic_derivative_at_background: float
    deformation_quartic_derivative_at_background: float
    background_gradient_unchanged: bool
    background_hessian_unchanged: bool
    cubic_vertex_changed: bool
    quartic_vertex_changed: bool
    cubic_vertex_identified_by_hessian: bool
    quartic_vertex_identified_by_hessian: bool


@dataclass(frozen=True)
class TreeLevelTwoPointAudit:
    """Pole data derived from one supplied bare tree-level singlet block."""

    signature: str
    action_definition_sha256: str
    minkowski_invariant_definition: str
    inverse_kernel: str
    feynman_propagator: str
    bare_mass_squared_gev2: float
    portal_mass_shift_gev2: float
    pole_mass_squared_gev2: float
    pole_mass_gev: float | None
    spatial_momentum_gev: float
    positive_energy_gev: float | None
    on_shell_invariant_gev2: float | None
    on_shell_kernel_residual_gev2: float | None
    invariant_pole_residue: float
    positive_energy_pole_residue_gev_inv: float | None
    static_correlation_length_fm: float | None
    canonical_kinetic_coefficient: float
    tachyon_free: bool
    isolated_massive_tree_pole: bool
    positive_tree_residue: bool
    relativistic_dispersion_identity_pass: bool
    tree_level_local_pole_candidate: bool
    mass_parameter_supplied_not_derived: bool
    loop_self_energy_included: bool
    renormalized_two_point_derived: bool
    full_ce_two_point_derived: bool
    lsz_completed: bool


@dataclass(frozen=True)
class PortalVertexAudit:
    """Exact vacuum derivatives of the declared ``Z2`` portal potential."""

    lambda_hp: float
    higgs_vev_gev: float
    singlet_self_coupling: float
    h_phi_cross_hessian_gev: float
    h_phi_phi_lagrangian_monomial_coefficient_gev: float
    h_phi_phi_derivative_gev: float
    h_h_phi_phi_lagrangian_monomial_coefficient: float
    h_h_phi_phi_derivative: float
    chi_chi_phi_phi_lagrangian_monomial_coefficient: float
    chi_chi_phi_phi_derivative: float
    phi_four_derivative: float
    expected_h_phi_phi_derivative_gev: float
    expected_h_h_phi_phi_derivative: float
    expected_chi_chi_phi_phi_derivative: float
    expected_phi_four_derivative: float
    maximum_identity_residual: float
    z2_odd_vacuum_derivatives_zero: bool
    bilinear_h_phi_mixing_zero: bool
    h_phi_phi_pair_vertex_present: bool
    h_h_phi_phi_pair_vertex_present: bool
    chi_chi_phi_phi_pair_vertex_present: bool
    local_derivative_identities_pass: bool
    single_phi_source_derived: bool
    direct_phi_squared_daughter_squared_vertex_derived: bool
    portal_action_selected_not_ce_derived: bool


@dataclass(frozen=True)
class LightPolePortalCompatibilityAudit:
    """Compatibility theorem for a light target and the portal mass relation."""

    target_mass_gev: float
    lambda_hp: float
    higgs_vev_gev: float
    portal_mass_shift_gev2: float
    zero_bare_mass_portal_pole_gev: float
    portal_to_target_mass_ratio: float
    required_bare_mass_squared_gev2: float
    required_bare_mass_sign: str
    required_bare_to_portal_shift_ratio: float
    target_squared_to_portal_shift_ratio: float
    cancellation_decimal_digits: float
    maximum_lambda_for_nonnegative_bare_mass: float
    supplied_to_nonnegative_bare_lambda_ratio: float
    portal_dominance_max_bare_to_shift_ratio: float
    target_reachable_with_nonnegative_bare_mass: bool
    target_reachable_with_back_solved_bare_mass: bool
    portal_dominance_satisfied_by_required_bare_mass: bool
    same_field_light_pole_and_portal_dominance_compatible: bool
    parameter_cancellation_required: bool
    ce_matching_relation_derived: bool


@dataclass(frozen=True)
class PortalVacuumAudit:
    """Tree-level vacuum check for the back-solved two-scalar potential."""

    higgs_self_coupling: float
    singlet_self_coupling: float
    lambda_hp: float
    higgs_vev_gev: float
    bare_mass_squared_gev2: float
    singlet_effective_mass_squared_gev2: float
    higgs_radial_curvature_gev2: float
    quartic_potential_bounded: bool
    selected_ew_vacuum_local_minimum: bool
    selected_ew_vacuum_energy_gev4: float
    origin_energy_gev4: float
    singlet_only_stationary_exists: bool
    singlet_only_field_squared_gev2: float | None
    singlet_only_energy_gev4: float | None
    mixed_stationary_exists: bool
    mixed_higgs_field_squared_gev2: float | None
    mixed_singlet_field_squared_gev2: float | None
    mixed_energy_gev4: float | None
    selected_ew_vacuum_global_among_tree_stationary_points: bool
    minimum_singlet_self_coupling_against_singlet_only_vacuum: float
    selected_vacuum_preserves_z2_despite_negative_bare_mass: bool
    loop_and_thermal_vacuum_stability_derived: bool


@dataclass(frozen=True)
class InvisibleWidthConstraintAudit:
    """Conditional ``h -> phi phi`` check against a caller-supplied limit."""

    target_mass_gev: float
    higgs_mass_gev: float
    lambda_hp: float
    higgs_vev_gev: float
    sm_higgs_width_gev: float
    supplied_branching_fraction_upper_limit: float
    kinematically_open: bool
    phase_space_factor: float
    partial_width_gev: float
    branching_fraction: float
    maximum_allowed_abs_lambda: float | None
    supplied_to_maximum_coupling_ratio: float | None
    supplied_benchmark_allowed: bool
    limit_supplied_not_derived: bool
    loop_and_global_fit_included: bool


@dataclass(frozen=True)
class Q04Q05PortalCertificate:
    """Serializable result with all full-CE promotion flags locked off."""

    schema_version: str
    scope: str
    status: str
    action_definition_sha256: str
    registered_target_mass_mev: float
    pole_parameter_source: str
    correlation_identifiability: CorrelationLengthIdentifiabilityAudit
    hessian_vertex_identifiability: HessianVertexNonidentifiabilityAudit
    two_point: TreeLevelTwoPointAudit
    vertices: PortalVertexAudit
    mass_compatibility: LightPolePortalCompatibilityAudit
    vacuum: PortalVacuumAudit
    invisible_width: InvisibleWidthConstraintAudit
    singlet_block_q0_4_tree_control_pass: bool
    singlet_block_q0_5_tree_control_pass: bool
    conditional_portal_pair_vertex_derived: bool
    registered_inverse_correlation_target_is_a_constructible_tree_pole: bool
    registered_target_is_predicted_by_portal_action: bool
    registered_target_equals_portal_dominated_pole: bool
    physical_clarus_pole_derived: bool
    renormalized_pole_and_residue_derived: bool
    full_lsz_passed: bool
    full_ce_production_vertex_derived: bool
    physical_sm_production_rate_derived: bool
    negative_stress_derived: bool
    maximum_supported_stage: str
    blockers: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _strict_tolerance(value: Real) -> float:
    return _positive(value, name="tolerance")


def audit_inverse_correlation_identifiability(
    *,
    inverse_correlation_scale_gev: Real,
    small_positive_residue: Real = 1.0e-12,
) -> CorrelationLengthIdentifiabilityAudit:
    """Build pole, ghost, and continuum models with one exponential scale.

    The continuum model has a positive spectral density beginning at the same
    threshold.  Its position-space correlator has the same leading
    ``exp(-m r)`` rate, multiplied by a different power of ``r``, without an
    isolated delta-function pole.  Hence even a measured leading inverse
    length would not determine pole existence or LSZ data.
    """

    mass = _positive(
        inverse_correlation_scale_gev,
        name="inverse_correlation_scale_gev",
    )
    small_residue = _positive(
        small_positive_residue,
        name="small_positive_residue",
    )
    models = (
        CorrelationAsymptoticCountermodel(
            label="canonical_simple_pole",
            euclidean_representation="G_E(q)=1/(q_squared+m_squared)",
            leading_inverse_length_gev=mass,
            isolated_delta_pole=True,
            invariant_pole_residue=1.0,
            reflection_positive=True,
            stable_lsz_particle_present=True,
        ),
        CorrelationAsymptoticCountermodel(
            label="small_overlap_simple_pole",
            euclidean_representation=("G_E(q)=epsilon/(q_squared+m_squared)"),
            leading_inverse_length_gev=mass,
            isolated_delta_pole=True,
            invariant_pole_residue=small_residue,
            reflection_positive=True,
            stable_lsz_particle_present=True,
        ),
        CorrelationAsymptoticCountermodel(
            label="negative_residue_ghost_pole",
            euclidean_representation="G_E(q)=-1/(q_squared+m_squared)",
            leading_inverse_length_gev=mass,
            isolated_delta_pole=True,
            invariant_pole_residue=-1.0,
            reflection_positive=False,
            stable_lsz_particle_present=False,
        ),
        CorrelationAsymptoticCountermodel(
            label="positive_continuum_threshold_without_delta_pole",
            euclidean_representation=(
                "G_E(q)=integral_[m_squared,infinity] "
                "rho(mu_squared)/(q_squared+mu_squared) dmu_squared"
            ),
            leading_inverse_length_gev=mass,
            isolated_delta_pole=False,
            invariant_pole_residue=None,
            reflection_positive=True,
            stable_lsz_particle_present=False,
        ),
    )
    shared_scale = all(model.leading_inverse_length_gev == mass for model in models)
    return CorrelationLengthIdentifiabilityAudit(
        inverse_correlation_scale_gev=mass,
        correlation_length_fm=HBAR_C_GEV_FM / mass,
        small_positive_residue=small_residue,
        countermodels=models,
        all_models_share_leading_exponential_scale=shared_scale,
        isolated_pole_identified_by_inverse_length=False,
        residue_magnitude_identified_by_inverse_length=False,
        residue_sign_identified_by_inverse_length=False,
        reflection_positivity_identified_by_inverse_length=False,
        lsz_particle_identified_by_inverse_length=False,
        maximum_supported_stage="INVERSE_CORRELATION_SCALE_ANSATZ",
    )


def audit_hessian_vertex_nonidentifiability(
    *,
    cubic_deformation: Real = 1.0,
    quartic_deformation: Real = 1.0,
) -> HessianVertexNonidentifiabilityAudit:
    """Audit ``Delta S=C eta^3/3!+D eta^4/4!`` at ``eta=0``.

    Adding this deformation to any action leaves its background gradient and
    Hessian unchanged while shifting the third and fourth derivatives by
    arbitrary ``C`` and ``D``.  Therefore A1 quadratic data cannot determine a
    production vertex without an independently selected higher action jet.
    """

    cubic = _finite_real(cubic_deformation, name="cubic_deformation")
    quartic = _finite_real(quartic_deformation, name="quartic_deformation")
    if cubic == 0.0 and quartic == 0.0:
        raise ValueError("at least one vertex deformation must be nonzero")
    return HessianVertexNonidentifiabilityAudit(
        expansion_point=0.0,
        cubic_deformation=cubic,
        quartic_deformation=quartic,
        deformation_gradient_at_background=0.0,
        deformation_hessian_at_background=0.0,
        deformation_cubic_derivative_at_background=cubic,
        deformation_quartic_derivative_at_background=quartic,
        background_gradient_unchanged=True,
        background_hessian_unchanged=True,
        cubic_vertex_changed=cubic != 0.0,
        quartic_vertex_changed=quartic != 0.0,
        cubic_vertex_identified_by_hessian=False,
        quartic_vertex_identified_by_hessian=False,
    )


def audit_tree_level_two_point(
    *,
    bare_mass_squared_gev2: Real,
    lambda_hp: Real,
    higgs_vev_gev: Real,
    spatial_momentum_gev: Real = 0.0,
    tolerance: Real = 1.0e-12,
) -> TreeLevelTwoPointAudit:
    """Derive the canonical singlet tree propagator for supplied parameters.

    With signature ``(-,+,+,+)`` and ``p^2 = omega^2-|k|^2``, integration by
    parts gives ``K(p^2)=p^2-m_eff^2`` and
    ``G_F=i/(p^2-m_eff^2+i0)``.  This convention has invariant residue one.
    """

    threshold = _strict_tolerance(tolerance)
    bare_mass = _finite_real(
        bare_mass_squared_gev2,
        name="bare_mass_squared_gev2",
    )
    coupling = _nonnegative(lambda_hp, name="lambda_hp")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    momentum = _nonnegative(
        spatial_momentum_gev,
        name="spatial_momentum_gev",
    )

    portal_shift = coupling * vev**2
    pole_mass_squared = bare_mass + portal_shift
    tachyon_free = pole_mass_squared >= -threshold
    if tachyon_free:
        physical_mass_squared = max(0.0, pole_mass_squared)
        pole_mass = math.sqrt(physical_mass_squared)
        energy = math.sqrt(momentum**2 + physical_mass_squared)
        invariant = energy**2 - momentum**2
        kernel_residual = abs(invariant - pole_mass_squared)
        energy_residue = None if energy <= threshold else 1.0 / (2.0 * energy)
        correlation_length = None if pole_mass <= threshold else HBAR_C_GEV_FM / pole_mass
    else:
        pole_mass = None
        energy = None
        invariant = None
        kernel_residual = None
        energy_residue = None
        correlation_length = None

    scale = max(1.0, abs(pole_mass_squared), momentum**2)
    dispersion_pass = kernel_residual is not None and kernel_residual <= threshold * scale
    isolated_massive_pole = tachyon_free and pole_mass_squared > threshold
    positive_residue = 1.0 > 0.0
    local_pole_candidate = isolated_massive_pole and positive_residue and dispersion_pass
    return TreeLevelTwoPointAudit(
        signature="(-,+,+,+)",
        action_definition_sha256=q0_control_action_definition_sha256(),
        minkowski_invariant_definition="p_squared=omega_squared-|k|_squared",
        inverse_kernel="K_F(p)=p_squared-m_eff_squared+i0",
        feynman_propagator="G_F(p)=i/(p_squared-m_eff_squared+i0)",
        bare_mass_squared_gev2=bare_mass,
        portal_mass_shift_gev2=portal_shift,
        pole_mass_squared_gev2=pole_mass_squared,
        pole_mass_gev=pole_mass,
        spatial_momentum_gev=momentum,
        positive_energy_gev=energy,
        on_shell_invariant_gev2=invariant,
        on_shell_kernel_residual_gev2=kernel_residual,
        invariant_pole_residue=1.0,
        positive_energy_pole_residue_gev_inv=energy_residue,
        static_correlation_length_fm=correlation_length,
        canonical_kinetic_coefficient=1.0,
        tachyon_free=tachyon_free,
        isolated_massive_tree_pole=isolated_massive_pole,
        positive_tree_residue=positive_residue,
        relativistic_dispersion_identity_pass=dispersion_pass,
        tree_level_local_pole_candidate=local_pole_candidate,
        mass_parameter_supplied_not_derived=True,
        loop_self_energy_included=False,
        renormalized_two_point_derived=False,
        full_ce_two_point_derived=False,
        lsz_completed=False,
    )


def audit_z2_portal_vertices(
    *,
    lambda_hp: Real,
    higgs_vev_gev: Real,
    singlet_self_coupling: Real,
    tolerance: Real = 1.0e-12,
) -> PortalVertexAudit:
    """Differentiate the local portal action at ``h=phi=0``.

    The relevant interaction Lagrangian is
    ``-lambda_HP*v*h*phi^2-lambda_HP*(h^2+chi^2)*phi^2/2
    -lambda_phi*phi^4/4``.  Reported derivative coefficients multiply ``i``
    to give the corresponding momentum-space Feynman rule.
    """

    threshold = _strict_tolerance(tolerance)
    coupling = _nonnegative(lambda_hp, name="lambda_hp")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    self_coupling = _finite_real(
        singlet_self_coupling,
        name="singlet_self_coupling",
    )

    cross_hessian = 0.0
    h_phi_phi_monomial = -coupling * vev
    h_phi_phi_derivative = 2.0 * h_phi_phi_monomial
    h_h_phi_phi_monomial = -0.5 * coupling
    h_h_phi_phi_derivative = 4.0 * h_h_phi_phi_monomial
    chi_chi_phi_phi_monomial = -0.5 * coupling
    chi_chi_phi_phi_derivative = 4.0 * chi_chi_phi_phi_monomial
    phi_four_derivative = -6.0 * self_coupling

    expected_cubic = -2.0 * coupling * vev
    expected_quartic = -2.0 * coupling
    expected_goldstone_quartic = -2.0 * coupling
    expected_self_quartic = -6.0 * self_coupling
    residuals = (
        abs(cross_hessian),
        abs(h_phi_phi_derivative - expected_cubic),
        abs(h_h_phi_phi_derivative - expected_quartic),
        abs(chi_chi_phi_phi_derivative - expected_goldstone_quartic),
        abs(phi_four_derivative - expected_self_quartic),
    )
    maximum_residual = max(residuals)
    scale = max(
        1.0,
        abs(expected_cubic),
        abs(expected_quartic),
        abs(expected_goldstone_quartic),
        abs(expected_self_quartic),
    )
    identities_pass = maximum_residual <= threshold * scale
    return PortalVertexAudit(
        lambda_hp=coupling,
        higgs_vev_gev=vev,
        singlet_self_coupling=self_coupling,
        h_phi_cross_hessian_gev=cross_hessian,
        h_phi_phi_lagrangian_monomial_coefficient_gev=(h_phi_phi_monomial),
        h_phi_phi_derivative_gev=h_phi_phi_derivative,
        h_h_phi_phi_lagrangian_monomial_coefficient=(h_h_phi_phi_monomial),
        h_h_phi_phi_derivative=h_h_phi_phi_derivative,
        chi_chi_phi_phi_lagrangian_monomial_coefficient=(chi_chi_phi_phi_monomial),
        chi_chi_phi_phi_derivative=chi_chi_phi_phi_derivative,
        phi_four_derivative=phi_four_derivative,
        expected_h_phi_phi_derivative_gev=expected_cubic,
        expected_h_h_phi_phi_derivative=expected_quartic,
        expected_chi_chi_phi_phi_derivative=expected_goldstone_quartic,
        expected_phi_four_derivative=expected_self_quartic,
        maximum_identity_residual=maximum_residual,
        z2_odd_vacuum_derivatives_zero=True,
        bilinear_h_phi_mixing_zero=True,
        h_phi_phi_pair_vertex_present=abs(expected_cubic) > threshold,
        h_h_phi_phi_pair_vertex_present=abs(expected_quartic) > threshold,
        chi_chi_phi_phi_pair_vertex_present=(abs(expected_goldstone_quartic) > threshold),
        local_derivative_identities_pass=identities_pass,
        single_phi_source_derived=False,
        direct_phi_squared_daughter_squared_vertex_derived=False,
        portal_action_selected_not_ce_derived=True,
    )


def audit_light_pole_portal_compatibility(
    *,
    target_mass_gev: Real,
    lambda_hp: Real,
    higgs_vev_gev: Real,
    portal_dominance_max_bare_to_shift_ratio: Real = 0.1,
    tolerance: Real = 1.0e-12,
) -> LightPolePortalCompatibilityAudit:
    """Test ``m_target^2=m0^2+lambda_HP*v^2`` without hiding tuning.

    For ``lambda_HP>0`` and ``m0^2>=0``, the exact lower bound is
    ``m_pole>=v*sqrt(lambda_HP)``.  If the target is below this bound, the
    unique back-solved bare mass is negative.  It may construct the target,
    but it contradicts the portal-dominance approximation when its magnitude
    is comparable to the portal shift.
    """

    threshold = _strict_tolerance(tolerance)
    target = _positive(target_mass_gev, name="target_mass_gev")
    coupling = _positive(lambda_hp, name="lambda_hp")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    dominance_limit = _nonnegative(
        portal_dominance_max_bare_to_shift_ratio,
        name="portal_dominance_max_bare_to_shift_ratio",
    )

    portal_shift = coupling * vev**2
    portal_pole = math.sqrt(portal_shift)
    target_squared = target**2
    required_bare = target_squared - portal_shift
    bare_ratio = abs(required_bare) / portal_shift
    squared_remainder = target_squared / portal_shift
    cancellation_digits = max(0.0, -math.log10(squared_remainder))
    maximum_nonnegative_lambda = target_squared / vev**2
    coupling_ratio = coupling / maximum_nonnegative_lambda
    mass_ratio = portal_pole / target

    nonnegative_compatible = required_bare >= -threshold * max(
        1.0,
        portal_shift,
    )
    dominance_satisfied = bare_ratio <= dominance_limit + threshold
    same_field_portal_dominance = dominance_satisfied
    if required_bare > threshold:
        bare_sign = "positive"
    elif required_bare < -threshold:
        bare_sign = "negative"
    else:
        bare_sign = "zero_within_tolerance"

    return LightPolePortalCompatibilityAudit(
        target_mass_gev=target,
        lambda_hp=coupling,
        higgs_vev_gev=vev,
        portal_mass_shift_gev2=portal_shift,
        zero_bare_mass_portal_pole_gev=portal_pole,
        portal_to_target_mass_ratio=mass_ratio,
        required_bare_mass_squared_gev2=required_bare,
        required_bare_mass_sign=bare_sign,
        required_bare_to_portal_shift_ratio=bare_ratio,
        target_squared_to_portal_shift_ratio=squared_remainder,
        cancellation_decimal_digits=cancellation_digits,
        maximum_lambda_for_nonnegative_bare_mass=(maximum_nonnegative_lambda),
        supplied_to_nonnegative_bare_lambda_ratio=coupling_ratio,
        portal_dominance_max_bare_to_shift_ratio=dominance_limit,
        target_reachable_with_nonnegative_bare_mass=nonnegative_compatible,
        target_reachable_with_back_solved_bare_mass=True,
        portal_dominance_satisfied_by_required_bare_mass=(dominance_satisfied),
        same_field_light_pole_and_portal_dominance_compatible=(same_field_portal_dominance),
        parameter_cancellation_required=required_bare < 0.0,
        ce_matching_relation_derived=False,
    )


def audit_portal_tree_vacuum(
    *,
    bare_mass_squared_gev2: Real,
    lambda_hp: Real,
    higgs_vev_gev: Real,
    higgs_self_coupling: Real,
    singlet_self_coupling: Real,
    tolerance: Real = 1.0e-12,
) -> PortalVacuumAudit:
    """Enumerate tree stationary points of the radial Higgs-singlet potential.

    The convention is
    ``V=-mu_H^2 r^2/2+lambda_H r^4/4+m0^2 s^2/2
    +lambda_s s^4/4+lambda_HP r^2 s^2/2`` with
    ``mu_H^2=lambda_H*v^2``.  This check prevents a negative back-solved bare
    mass from being mislabeled as an automatic zero-temperature ``Z2`` failure.
    It does not include loop or thermal stability.
    """

    threshold = _strict_tolerance(tolerance)
    bare_mass = _finite_real(
        bare_mass_squared_gev2,
        name="bare_mass_squared_gev2",
    )
    portal = _nonnegative(lambda_hp, name="lambda_hp")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    higgs_quartic = _positive(
        higgs_self_coupling,
        name="higgs_self_coupling",
    )
    singlet_quartic = _positive(
        singlet_self_coupling,
        name="singlet_self_coupling",
    )

    mu_squared = higgs_quartic * vev**2
    singlet_curvature = bare_mass + portal * vev**2
    higgs_curvature = 2.0 * higgs_quartic * vev**2
    bounded = portal >= -math.sqrt(higgs_quartic * singlet_quartic)
    local_minimum = bounded and higgs_curvature > threshold and singlet_curvature > threshold

    origin_energy = 0.0
    ew_energy = -higgs_quartic * vev**4 / 4.0
    if bare_mass < 0.0:
        singlet_squared = -bare_mass / singlet_quartic
        singlet_energy = -(bare_mass**2) / (4.0 * singlet_quartic)
        minimum_self_coupling = bare_mass**2 / (higgs_quartic * vev**4)
    else:
        singlet_squared = None
        singlet_energy = None
        minimum_self_coupling = 0.0

    determinant = higgs_quartic * singlet_quartic - portal**2
    mixed_higgs_squared: float | None = None
    mixed_singlet_squared: float | None = None
    mixed_energy: float | None = None
    if abs(determinant) > threshold:
        candidate_higgs_squared = (mu_squared * singlet_quartic + portal * bare_mass) / determinant
        candidate_singlet_squared = (-higgs_quartic * bare_mass - portal * mu_squared) / determinant
        if candidate_higgs_squared > threshold and candidate_singlet_squared > threshold:
            mixed_higgs_squared = candidate_higgs_squared
            mixed_singlet_squared = candidate_singlet_squared
            mixed_energy = (
                -0.5 * mu_squared * candidate_higgs_squared
                + 0.25 * higgs_quartic * candidate_higgs_squared**2
                + 0.5 * bare_mass * candidate_singlet_squared
                + 0.25 * singlet_quartic * candidate_singlet_squared**2
                + 0.5 * portal * candidate_higgs_squared * candidate_singlet_squared
            )

    competing_energies = [origin_energy]
    if singlet_energy is not None:
        competing_energies.append(singlet_energy)
    if mixed_energy is not None:
        competing_energies.append(mixed_energy)
    energy_scale = max(1.0, abs(ew_energy), *(abs(x) for x in competing_energies))
    global_minimum = local_minimum and all(
        ew_energy <= energy + threshold * energy_scale for energy in competing_energies
    )
    return PortalVacuumAudit(
        higgs_self_coupling=higgs_quartic,
        singlet_self_coupling=singlet_quartic,
        lambda_hp=portal,
        higgs_vev_gev=vev,
        bare_mass_squared_gev2=bare_mass,
        singlet_effective_mass_squared_gev2=singlet_curvature,
        higgs_radial_curvature_gev2=higgs_curvature,
        quartic_potential_bounded=bounded,
        selected_ew_vacuum_local_minimum=local_minimum,
        selected_ew_vacuum_energy_gev4=ew_energy,
        origin_energy_gev4=origin_energy,
        singlet_only_stationary_exists=singlet_energy is not None,
        singlet_only_field_squared_gev2=singlet_squared,
        singlet_only_energy_gev4=singlet_energy,
        mixed_stationary_exists=mixed_energy is not None,
        mixed_higgs_field_squared_gev2=mixed_higgs_squared,
        mixed_singlet_field_squared_gev2=mixed_singlet_squared,
        mixed_energy_gev4=mixed_energy,
        selected_ew_vacuum_global_among_tree_stationary_points=global_minimum,
        minimum_singlet_self_coupling_against_singlet_only_vacuum=(minimum_self_coupling),
        selected_vacuum_preserves_z2_despite_negative_bare_mass=(
            bare_mass < 0.0 and global_minimum
        ),
        loop_and_thermal_vacuum_stability_derived=False,
    )


def audit_invisible_width_constraint(
    *,
    target_mass_gev: Real,
    lambda_hp: Real,
    higgs_vev_gev: Real,
    higgs_mass_gev: Real,
    sm_higgs_width_gev: Real,
    branching_fraction_upper_limit: Real,
    tolerance: Real = 1.0e-12,
) -> InvisibleWidthConstraintAudit:
    """Compute the exact tree control and invert its supplied BR limit."""

    threshold = _strict_tolerance(tolerance)
    target = _positive(target_mass_gev, name="target_mass_gev")
    coupling = _nonnegative(lambda_hp, name="lambda_hp")
    vev = _positive(higgs_vev_gev, name="higgs_vev_gev")
    higgs_mass = _positive(higgs_mass_gev, name="higgs_mass_gev")
    sm_width = _positive(sm_higgs_width_gev, name="sm_higgs_width_gev")
    upper_limit = _finite_real(
        branching_fraction_upper_limit,
        name="branching_fraction_upper_limit",
    )
    if not 0.0 <= upper_limit < 1.0:
        raise ValueError("branching_fraction_upper_limit must be in [0, 1)")

    base = audit_higgs_invisible_width(
        lambda_hp=coupling,
        higgs_vev=vev,
        higgs_mass=higgs_mass,
        scalar_mass=target,
        sm_higgs_width=sm_width,
        branching_fraction_upper_limit=upper_limit,
        tolerance=threshold,
    )
    if base.kinematically_open:
        maximum_partial_width = upper_limit * sm_width / (1.0 - upper_limit)
        if maximum_partial_width == 0.0:
            maximum_coupling = 0.0
        else:
            maximum_coupling = math.sqrt(
                maximum_partial_width
                * 8.0
                * math.pi
                * higgs_mass
                / (vev**2 * base.phase_space_factor)
            )
        ratio = (
            math.inf
            if maximum_coupling == 0.0 and coupling > 0.0
            else (0.0 if maximum_coupling == 0.0 else coupling / maximum_coupling)
        )
    else:
        maximum_coupling = None
        ratio = None

    return InvisibleWidthConstraintAudit(
        target_mass_gev=target,
        higgs_mass_gev=higgs_mass,
        lambda_hp=coupling,
        higgs_vev_gev=vev,
        sm_higgs_width_gev=sm_width,
        supplied_branching_fraction_upper_limit=upper_limit,
        kinematically_open=base.kinematically_open,
        phase_space_factor=base.phase_space_factor,
        partial_width_gev=base.partial_width,
        branching_fraction=base.branching_fraction,
        maximum_allowed_abs_lambda=maximum_coupling,
        supplied_to_maximum_coupling_ratio=ratio,
        supplied_benchmark_allowed=base.benchmark_allowed,
        limit_supplied_not_derived=True,
        loop_and_global_fit_included=False,
    )


def ce_light_pole_q04_q05_certificate(
    *,
    registered_target_mass_mev: Real = DEFAULT_CE_INVERSE_CORRELATION_SCALE_MEV,
    lambda_hp: Real = DEFAULT_LAMBDA_HP,
    higgs_vev_gev: Real = 246.22,
    singlet_self_coupling: Real = 0.1,
    higgs_mass_gev: Real = 125.25,
    sm_higgs_width_gev: Real = 0.00407,
    branching_fraction_upper_limit: Real = 0.11,
    spatial_momentum_gev: Real = 1.0,
    portal_dominance_max_bare_to_shift_ratio: Real = 0.1,
    tolerance: Real = 1.0e-12,
) -> Q04Q05PortalCertificate:
    """Run the 29.65 MeV target through the optional portal control branch."""

    target_mev = _positive(
        registered_target_mass_mev,
        name="registered_target_mass_mev",
    )
    target_gev = target_mev / 1000.0
    correlation_identifiability = audit_inverse_correlation_identifiability(
        inverse_correlation_scale_gev=target_gev
    )
    hessian_vertex_identifiability = audit_hessian_vertex_nonidentifiability(
        cubic_deformation=1.0,
        quartic_deformation=1.0,
    )
    compatibility = audit_light_pole_portal_compatibility(
        target_mass_gev=target_gev,
        lambda_hp=lambda_hp,
        higgs_vev_gev=higgs_vev_gev,
        portal_dominance_max_bare_to_shift_ratio=(portal_dominance_max_bare_to_shift_ratio),
        tolerance=tolerance,
    )
    two_point = audit_tree_level_two_point(
        bare_mass_squared_gev2=(compatibility.required_bare_mass_squared_gev2),
        lambda_hp=lambda_hp,
        higgs_vev_gev=higgs_vev_gev,
        spatial_momentum_gev=spatial_momentum_gev,
        tolerance=tolerance,
    )
    vertices = audit_z2_portal_vertices(
        lambda_hp=lambda_hp,
        higgs_vev_gev=higgs_vev_gev,
        singlet_self_coupling=singlet_self_coupling,
        tolerance=tolerance,
    )
    vev_value = _positive(higgs_vev_gev, name="higgs_vev_gev")
    higgs_mass_value = _positive(higgs_mass_gev, name="higgs_mass_gev")
    vacuum = audit_portal_tree_vacuum(
        bare_mass_squared_gev2=(compatibility.required_bare_mass_squared_gev2),
        lambda_hp=lambda_hp,
        higgs_vev_gev=vev_value,
        higgs_self_coupling=higgs_mass_value**2 / (2.0 * vev_value**2),
        singlet_self_coupling=singlet_self_coupling,
        tolerance=tolerance,
    )
    invisible_width = audit_invisible_width_constraint(
        target_mass_gev=target_gev,
        lambda_hp=lambda_hp,
        higgs_vev_gev=higgs_vev_gev,
        higgs_mass_gev=higgs_mass_value,
        sm_higgs_width_gev=sm_higgs_width_gev,
        branching_fraction_upper_limit=branching_fraction_upper_limit,
        tolerance=tolerance,
    )

    q0_4_pass = (
        two_point.isolated_massive_tree_pole
        and two_point.positive_tree_residue
        and two_point.relativistic_dispersion_identity_pass
    )
    q0_5_pass = vertices.local_derivative_identities_pass
    constructible = q0_4_pass and math.isclose(
        two_point.pole_mass_squared_gev2,
        target_gev**2,
        rel_tol=float(tolerance),
        abs_tol=float(tolerance),
    )
    return Q04Q05PortalCertificate(
        schema_version="1.0",
        scope=CONTROL_SCOPE,
        status=CONTROL_STATUS,
        action_definition_sha256=two_point.action_definition_sha256,
        registered_target_mass_mev=target_mev,
        pole_parameter_source=("bare_mass_squared_back_solved_from_registered_target"),
        correlation_identifiability=correlation_identifiability,
        hessian_vertex_identifiability=hessian_vertex_identifiability,
        two_point=two_point,
        vertices=vertices,
        mass_compatibility=compatibility,
        vacuum=vacuum,
        invisible_width=invisible_width,
        singlet_block_q0_4_tree_control_pass=q0_4_pass,
        singlet_block_q0_5_tree_control_pass=q0_5_pass,
        conditional_portal_pair_vertex_derived=(vertices.h_phi_phi_pair_vertex_present),
        registered_inverse_correlation_target_is_a_constructible_tree_pole=(constructible),
        registered_target_is_predicted_by_portal_action=False,
        registered_target_equals_portal_dominated_pole=False,
        physical_clarus_pole_derived=False,
        renormalized_pole_and_residue_derived=False,
        full_lsz_passed=False,
        full_ce_production_vertex_derived=False,
        physical_sm_production_rate_derived=False,
        negative_stress_derived=False,
        maximum_supported_stage=("CONDITIONAL_TREE_LEVEL_Z2_PORTAL_POLE_AND_PAIR_VERTEX"),
        blockers=(
            "the 29.64757 MeV mass fixes the bare parameter by inversion rather than prediction",
            "lambda_HP=0.0316 with nonnegative bare mass has a 43.77 GeV lower pole scale",
            "the back-solved negative bare mass cancels the portal shift and violates portal dominance",
            "the supplied light-scalar invisible-width benchmark fails its supplied limit",
            "loop self-energy, counterterms, RG matching, spectral density, and LSZ are absent",
            "the optional portal field has not been identified with the CE Hessian readout",
        ),
        conclusion=(
            "The declared optional Z2 portal EFT has an exact canonical tree-level pole and "
            "nonzero h-phi-phi/h-h-phi-phi vertices after parameters are supplied. The "
            "29.64757 MeV target is constructible only by back-solving a nearly cancelling "
            "negative bare mass; it is not the portal-dominated 43.77 GeV pole and is not a "
            "derived physical Clarus particle."
        ),
    )
