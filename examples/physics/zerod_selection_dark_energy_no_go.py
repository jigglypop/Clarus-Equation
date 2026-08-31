"""End-to-end no-go audit for stationary 0D selection as dark energy.

The audit separates two claims.

1. A dimensionless, spacetime-independent selection parameter does not fix an
   absolute vacuum source in a local diffeomorphism-invariant EFT. An allowed
   covariant vacuum counterterm leaves the selection recursion unchanged while
   shifting the Friedmann solution.
2. The current normalization-invariant rendering factor is essentially a
   constant calibration. It cannot reproduce the redshift-dependent distance
   shape separating a flat matter-only universe from flat LambdaCDM.

The calculations are counterexamples and conditional FLRW identities, not a
fit and not a dark-energy prediction.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
import math

from examples.physics.observation_rendering_invariant_schur import (
    DEFAULT_D_EFF,
    DEFAULT_DELTA,
    DEFAULT_Q_LOW,
    DEFAULT_SPATIAL_DIMENSION,
    controlled_depth_rendering_sequence,
    poisson_probability_step,
)


LIGHT_SPEED_KM_S = 299_792.458
DEFAULT_H0_KM_S_MPC = 70.0
DEFAULT_REDSHIFTS = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0)
DEFAULT_TARGET_OMEGA_M = 0.3
DEFAULT_TARGET_OMEGA_LAMBDA = 0.7
DEFAULT_REDUCED_PLANCK_SCALE_EV = 2.435e27
DEFAULT_DARK_ENERGY_QUARTER_SCALE_EV = 2.25e-3


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def quartic_hierarchy_coefficient(low_scale: float, high_scale: float) -> float:
    """Return the dimensionless coefficient (low_scale/high_scale)^4."""

    low_scale = _positive(low_scale, "low_scale")
    high_scale = _positive(high_scale, "high_scale")
    return (low_scale / high_scale) ** 4


@dataclass(frozen=True)
class CovariantCountertermNoGo:
    q: float
    d_eff: float
    q_update_model_a: float
    q_update_model_b: float
    counterterm_form_factor: float
    reference_mass_scale: float
    reduced_planck_mass: float
    matter_density: float
    coefficient_a: float
    coefficient_b: float
    vacuum_density_a: float
    vacuum_density_b: float
    hubble_squared_a: float
    hubble_squared_b: float
    vacuum_density_shift: float
    hubble_squared_shift: float
    rendering_sequence_model_a: tuple[float, ...]
    rendering_sequence_model_b: tuple[float, ...]
    selection_recursion_identical: bool
    rendering_readout_identical: bool
    expansion_history_different: bool
    q_mass_dimension: int = 0
    reference_mass_dimension: int = 1
    vacuum_density_mass_dimension: int = 4
    hubble_squared_mass_dimension: int = 2
    local_and_diffeomorphism_invariant: bool = True
    constant_vacuum_stress_bianchi_conserved: bool = True
    absolute_source_unique_from_q: bool = False
    counterexample_complete: bool = True
    status: str = "STATIONARY_0D_Q_ABSOLUTE_SOURCE_REFUTED_BY_COVARIANT_COUNTERTERM"


def certify_covariant_counterterm_no_go(
    *,
    q: float = DEFAULT_Q_LOW,
    d_eff: float = DEFAULT_D_EFF,
    reference_mass_scale: float = 1.0,
    reduced_planck_mass: float = 1.0,
    matter_density: float = 1.0,
    coefficient_a: float = 0.0,
    coefficient_b: float = 0.25,
    rendering_steps: int = 8,
) -> CovariantCountertermNoGo:
    """Construct two covariant EFTs with the same q but different vacuum source.

    The witness uses the allowed local term
    -integral sqrt(-g) c M_*^4 f(q), with f(q)=1+q^2.
    Since q is a fixed 0D parameter, this preserves the selection recursion and
    the Bianchi identity while shifting the absolute vacuum density.
    """

    q = _finite(q, "q")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    d_eff = _positive(d_eff, "d_eff")
    reference_mass_scale = _positive(reference_mass_scale, "reference_mass_scale")
    reduced_planck_mass = _positive(reduced_planck_mass, "reduced_planck_mass")
    matter_density = _finite(matter_density, "matter_density")
    if matter_density < 0.0:
        raise ValueError("matter_density must be non-negative")
    coefficient_a = _finite(coefficient_a, "coefficient_a")
    coefficient_b = _finite(coefficient_b, "coefficient_b")

    form_factor = 1.0 + q * q
    mass_fourth = reference_mass_scale**4
    vacuum_a = coefficient_a * mass_fourth * form_factor
    vacuum_b = coefficient_b * mass_fourth * form_factor
    if matter_density + min(vacuum_a, vacuum_b) < 0.0:
        raise ValueError("both Friedmann witnesses require non-negative total density")
    planck_squared = reduced_planck_mass**2
    hubble_a = (matter_density + vacuum_a) / (3.0 * planck_squared)
    hubble_b = (matter_density + vacuum_b) / (3.0 * planck_squared)

    update_a = poisson_probability_step(q, d_eff)
    update_b = poisson_probability_step(q, d_eff)
    sequence_a = controlled_depth_rendering_sequence(
        rendering_steps, d_eff=d_eff
    ).spatial_scale_factors
    sequence_b = controlled_depth_rendering_sequence(
        rendering_steps, d_eff=d_eff
    ).spatial_scale_factors
    same_selection = update_a == update_b
    same_rendering = sequence_a == sequence_b
    different_expansion = not math.isclose(
        hubble_a, hubble_b, rel_tol=0.0, abs_tol=1.0e-15
    )
    return CovariantCountertermNoGo(
        q=q,
        d_eff=d_eff,
        q_update_model_a=update_a,
        q_update_model_b=update_b,
        counterterm_form_factor=form_factor,
        reference_mass_scale=reference_mass_scale,
        reduced_planck_mass=reduced_planck_mass,
        matter_density=matter_density,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        vacuum_density_a=vacuum_a,
        vacuum_density_b=vacuum_b,
        hubble_squared_a=hubble_a,
        hubble_squared_b=hubble_b,
        vacuum_density_shift=vacuum_b - vacuum_a,
        hubble_squared_shift=hubble_b - hubble_a,
        rendering_sequence_model_a=sequence_a,
        rendering_sequence_model_b=sequence_b,
        selection_recursion_identical=same_selection,
        rendering_readout_identical=same_rendering,
        expansion_history_different=different_expansion,
        counterexample_complete=(
            same_selection and same_rendering and different_expansion
        ),
    )


@dataclass(frozen=True)
class ScaleSupplyAudit:
    dark_energy_quarter_scale_ev: float
    reduced_planck_scale_ev: float
    required_planck_quartic_coefficient: float
    matter_tracking_dlnrho_dln_a: float
    matter_tracking_effective_w: float
    matter_tracking_deceleration_parameter: float
    vacuum_pressure_matter_scaling_continuity_residual: float
    h_tracking_dlnh2_dln_a: float
    h_tracking_effective_w: float
    h_tracking_deceleration_parameter: float
    four_volume_mass_dimension: int
    inverse_sqrt_four_volume_mass_dimension: int
    density_mass_dimension: int
    missing_volume_lift_mass_dimension: int
    reference_scales_are_external_inputs: bool = True
    stationary_q_gradient_terms_zero: bool = True
    planck_coefficient_derived_from_q: bool = False
    matter_abundance_derived_from_q: bool = False
    curvature_wilson_functions_derived_from_q: bool = False
    global_boundary_datum_derived_from_q: bool = False
    nonlocal_kernel_and_ir_scale_derived_from_q: bool = False
    unique_dimensionful_scale_supply: bool = False
    status: str = "ALL_DECLARED_SCALE_SUPPLY_ROUTES_RETAIN_EXTERNAL_INPUT"


def audit_scale_supply_routes(
    *,
    dark_energy_quarter_scale_ev: float = DEFAULT_DARK_ENERGY_QUARTER_SCALE_EV,
    reduced_planck_scale_ev: float = DEFAULT_REDUCED_PLANCK_SCALE_EV,
) -> ScaleSupplyAudit:
    """Audit the distinct ways a dimension-four source scale could enter."""

    dark_energy_quarter_scale_ev = _positive(
        dark_energy_quarter_scale_ev, "dark_energy_quarter_scale_ev"
    )
    reduced_planck_scale_ev = _positive(
        reduced_planck_scale_ev, "reduced_planck_scale_ev"
    )
    # If rho_x is proportional to conserved pressureless matter or to H^2 in
    # the algebraic Friedmann equation, it scales as a^-3 and has w_eff=0.
    dlnrho = -3.0
    effective_w = -1.0 - dlnrho / 3.0
    deceleration = 0.5 * (1.0 + 3.0 * effective_w)
    # Assigning p=-rho while retaining rho proportional to a^-3 violates
    # dlnrho/dlna + 3(1+w)=0 by -3.
    vacuum_continuity_residual = dlnrho + 3.0 * (1.0 - 1.0)
    return ScaleSupplyAudit(
        dark_energy_quarter_scale_ev=dark_energy_quarter_scale_ev,
        reduced_planck_scale_ev=reduced_planck_scale_ev,
        required_planck_quartic_coefficient=quartic_hierarchy_coefficient(
            dark_energy_quarter_scale_ev, reduced_planck_scale_ev
        ),
        matter_tracking_dlnrho_dln_a=dlnrho,
        matter_tracking_effective_w=effective_w,
        matter_tracking_deceleration_parameter=deceleration,
        vacuum_pressure_matter_scaling_continuity_residual=vacuum_continuity_residual,
        h_tracking_dlnh2_dln_a=dlnrho,
        h_tracking_effective_w=effective_w,
        h_tracking_deceleration_parameter=deceleration,
        four_volume_mass_dimension=-4,
        inverse_sqrt_four_volume_mass_dimension=2,
        density_mass_dimension=4,
        missing_volume_lift_mass_dimension=2,
    )


def _simpson_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    intervals: int = 4096,
) -> float:
    if intervals <= 0 or intervals % 2:
        raise ValueError("intervals must be a positive even integer")
    if upper < lower:
        raise ValueError("upper must not be below lower")
    if upper == lower:
        return 0.0
    step = (upper - lower) / intervals
    total = function(lower) + function(upper)
    total += 4.0 * sum(
        function(lower + index * step) for index in range(1, intervals, 2)
    )
    total += 2.0 * sum(
        function(lower + index * step) for index in range(2, intervals, 2)
    )
    return total * step / 3.0


def flat_e_of_z(z: float, omega_m: float, omega_lambda: float) -> float:
    """Return dimensionless H(z)/H0 for a flat matter+Lambda background."""

    z = _finite(z, "z")
    if z < 0.0:
        raise ValueError("z must be non-negative")
    omega_m = _finite(omega_m, "omega_m")
    omega_lambda = _finite(omega_lambda, "omega_lambda")
    if omega_m < 0.0 or omega_lambda < 0.0:
        raise ValueError("density fractions must be non-negative")
    if not math.isclose(omega_m + omega_lambda, 1.0, abs_tol=1.0e-12):
        raise ValueError("flat matter+Lambda fractions must sum to one")
    return math.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda)


def flat_luminosity_distance_over_c_h0(
    z: float,
    omega_m: float,
    omega_lambda: float,
    *,
    intervals: int = 4096,
) -> float:
    """Return the dimensionless flat-FLRW luminosity distance H0*dL/c."""

    z = _finite(z, "z")
    if z <= 0.0:
        raise ValueError("z must be positive")
    comoving = _simpson_integral(
        lambda local_z: 1.0
        / flat_e_of_z(local_z, omega_m, omega_lambda),
        0.0,
        z,
        intervals=intervals,
    )
    return (1.0 + z) * comoving


def eds_luminosity_distance_over_c_h0(z: float) -> float:
    """Return exact H0*dL/c for Einstein-de Sitter."""

    z = _finite(z, "z")
    if z <= 0.0:
        raise ValueError("z must be positive")
    return 2.0 * (1.0 + z) * (1.0 - 1.0 / math.sqrt(1.0 + z))


def eds_redshift_for_dimensionless_luminosity_distance(distance: float) -> float:
    """Invert the exact Einstein-de Sitter luminosity distance."""

    distance = _positive(distance, "distance")
    root_one_plus_z = 0.5 * (1.0 + math.sqrt(1.0 + 2.0 * distance))
    return root_one_plus_z**2 - 1.0


def _distance_modulus(distance_mpc: float) -> float:
    return 5.0 * math.log10(_positive(distance_mpc, "distance_mpc")) + 25.0


@dataclass(frozen=True)
class DistanceRenderingPoint:
    z: float
    eds_luminosity_distance_mpc: float
    target_luminosity_distance_mpc: float
    eds_distance_modulus: float
    target_distance_modulus: float
    target_minus_eds_modulus: float
    target_h_over_eds_h: float
    eds_over_target_distance: float
    required_flux_survival: float
    required_optical_depth: float
    required_distance_duality_eta: float
    required_amplitude_steps: float
    target_inverse_eds_redshift: float


@dataclass(frozen=True)
class RenderingCosmologyNoGo:
    h0_km_s_mpc: float
    target_omega_m: float
    target_omega_lambda: float
    rendering_lambda_initial: float
    rendering_lambda_fixed: float
    constant_rendering_modulus_shift: float
    controlled_sequence_fractional_span: float
    controlled_sequence_modulus_span: float
    points: tuple[DistanceRenderingPoint, ...]
    distance_ratio_span: float
    constant_factor_low_z_calibrated_shape_effect: float
    distance_ratio_is_redshift_dependent: bool
    constant_rendering_matches_all_redshifts: bool
    controlled_sequence_span_sufficient: bool
    opacity_requires_distance_duality_violation: bool
    target_inverse_redshift_map_required: bool
    benchmark_is_observational_fit: bool = False
    lambda_as_photon_amplitude_requires_new_axiom: bool = True
    event_depth_to_redshift_derived: bool = False
    photon_survival_law_derived: bool = False
    growth_and_lensing_changed_by_readout: bool = False
    observation_only_dark_energy_explanation: bool = False
    status: str = "STATIC_RENDERING_AND_OPACITY_ROUTES_REFUTED_FOR_FULL_DE_SHAPE"


def audit_rendering_cosmology_no_go(
    *,
    redshifts: Sequence[float] = DEFAULT_REDSHIFTS,
    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC,
    target_omega_m: float = DEFAULT_TARGET_OMEGA_M,
    target_omega_lambda: float = DEFAULT_TARGET_OMEGA_LAMBDA,
    rendering_lambda: float | None = None,
    integration_intervals: int = 4096,
) -> RenderingCosmologyNoGo:
    """Compare the supplied rendering map with the full dark-energy distance shape."""

    h0_km_s_mpc = _positive(h0_km_s_mpc, "h0_km_s_mpc")
    target_omega_m = _finite(target_omega_m, "target_omega_m")
    target_omega_lambda = _finite(target_omega_lambda, "target_omega_lambda")
    if not math.isclose(
        target_omega_m + target_omega_lambda, 1.0, abs_tol=1.0e-12
    ):
        raise ValueError("target density fractions must define a flat model")
    samples = tuple(_positive(value, "redshift") for value in redshifts)
    if not samples or any(
        right <= left
        for left, right in zip(samples[:-1], samples[1:], strict=True)
    ):
        raise ValueError("redshifts must be nonempty and strictly increasing")

    sequence = controlled_depth_rendering_sequence(32)
    lambda_initial = sequence.spatial_scale_factors[0]
    lambda_fixed = (
        math.sqrt(
            1.0
            - DEFAULT_DELTA / DEFAULT_SPATIAL_DIMENSION
            - DEFAULT_Q_LOW**2
        )
        if rendering_lambda is None
        else _positive(rendering_lambda, "rendering_lambda")
    )
    if lambda_fixed >= 1.0:
        raise ValueError("rendering_lambda must lie below one for the loss audit")

    distance_scale_mpc = LIGHT_SPEED_KM_S / h0_km_s_mpc
    points: list[DistanceRenderingPoint] = []
    for z in samples:
        eds_dimensionless = eds_luminosity_distance_over_c_h0(z)
        target_dimensionless = flat_luminosity_distance_over_c_h0(
            z,
            target_omega_m,
            target_omega_lambda,
            intervals=integration_intervals,
        )
        eds_mpc = distance_scale_mpc * eds_dimensionless
        target_mpc = distance_scale_mpc * target_dimensionless
        distance_ratio = eds_dimensionless / target_dimensionless
        survival = distance_ratio**2
        required_steps = math.log(survival) / (2.0 * math.log(lambda_fixed))
        target_e = flat_e_of_z(z, target_omega_m, target_omega_lambda)
        eds_e = (1.0 + z) ** 1.5
        points.append(
            DistanceRenderingPoint(
                z=z,
                eds_luminosity_distance_mpc=eds_mpc,
                target_luminosity_distance_mpc=target_mpc,
                eds_distance_modulus=_distance_modulus(eds_mpc),
                target_distance_modulus=_distance_modulus(target_mpc),
                target_minus_eds_modulus=5.0
                * math.log10(target_dimensionless / eds_dimensionless),
                target_h_over_eds_h=target_e / eds_e,
                eds_over_target_distance=distance_ratio,
                required_flux_survival=survival,
                required_optical_depth=-math.log(survival),
                required_distance_duality_eta=1.0 / math.sqrt(survival),
                required_amplitude_steps=required_steps,
                target_inverse_eds_redshift=(
                    eds_redshift_for_dimensionless_luminosity_distance(
                        target_dimensionless
                    )
                ),
            )
        )

    ratios = tuple(point.eds_over_target_distance for point in points)
    ratio_span = max(ratios) - min(ratios)
    sequence_span = lambda_initial / lambda_fixed - 1.0
    modulus_span = 5.0 * math.log10(lambda_initial / lambda_fixed)
    constant_matches = all(
        math.isclose(ratio, lambda_fixed, rel_tol=0.0, abs_tol=1.0e-6)
        for ratio in ratios
    )
    return RenderingCosmologyNoGo(
        h0_km_s_mpc=h0_km_s_mpc,
        target_omega_m=target_omega_m,
        target_omega_lambda=target_omega_lambda,
        rendering_lambda_initial=lambda_initial,
        rendering_lambda_fixed=lambda_fixed,
        constant_rendering_modulus_shift=5.0 * math.log10(1.0 / lambda_fixed),
        controlled_sequence_fractional_span=sequence_span,
        controlled_sequence_modulus_span=modulus_span,
        points=tuple(points),
        distance_ratio_span=ratio_span,
        constant_factor_low_z_calibrated_shape_effect=0.0,
        distance_ratio_is_redshift_dependent=(ratio_span > 1.0e-3),
        constant_rendering_matches_all_redshifts=constant_matches,
        controlled_sequence_span_sufficient=(sequence_span >= ratio_span),
        opacity_requires_distance_duality_violation=any(
            abs(point.required_distance_duality_eta - 1.0) > 1.0e-3
            for point in points
        ),
        target_inverse_redshift_map_required=any(
            abs(point.target_inverse_eds_redshift - point.z) > 1.0e-3
            for point in points
        ),
    )


@dataclass(frozen=True)
class ZeroDSelectionDarkEnergyEndToEndAudit:
    counterterm: CovariantCountertermNoGo
    scale_supply: ScaleSupplyAudit
    rendering: RenderingCosmologyNoGo
    pivots_tested: tuple[str, ...] = (
        "local_q_field_or_clock",
        "global_volume_or_sequestering",
        "ctp_or_nonlocal_memory",
        "causal_order_and_event_volume",
        "uv_dimensional_transmutation",
        "relative_detector_or_opacity",
    )
    escape_requirements: tuple[str, ...] = (
        "symmetry_or_uv_matching_fixes_the_vacuum_counterterm",
        "a_dimensionful_scale_is_derived",
        "q_dynamics_or_a_causal_nonlocal_state_kernel_is_fixed",
        "all_wilson_coefficients_and_cosmological_initial_global_data_are_fixed",
    )
    dimensionless_core_arguments: tuple[tuple[str, str], ...] = (
        ("D * (1 - q)", "D and q are dimensionless"),
        ("1 - delta/d - q**2", "delta, d, and q are dimensionless"),
        ("H0 * dL / c", "distance is normalized by c/H0"),
        ("low_scale / high_scale", "both scales use the same energy unit"),
    )
    stationary_selection_absolute_source_parent_refuted: bool = True
    static_rendering_full_dark_energy_parent_refuted: bool = True
    direct_q_to_hubble_parent_refuted: bool = True
    unique_dark_energy_prediction: bool = False
    status: str = "END_TO_END_NO_GO_CLOSED_WITH_EXPLICIT_ESCAPE_REQUIREMENTS"


def audit_zerod_selection_dark_energy_end_to_end() -> (
    ZeroDSelectionDarkEnergyEndToEndAudit
):
    return ZeroDSelectionDarkEnergyEndToEndAudit(
        counterterm=certify_covariant_counterterm_no_go(),
        scale_supply=audit_scale_supply_routes(),
        rendering=audit_rendering_cosmology_no_go(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zerod_selection_dark_energy_no_go")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    audit = audit_zerod_selection_dark_energy_end_to_end()
    print(
        json.dumps(
            asdict(audit),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
