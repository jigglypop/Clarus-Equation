"""Executable proof/counterexample loop for the remaining Chapter-2 bridges.

The checks in this module have a deliberately narrow meaning.  They either
verify a conditional identity or exhibit two models with the same upstream
data and different downstream outputs.  A passing run therefore closes an
*unconditional implication*; it does not validate the missing physical action.
"""

from __future__ import annotations

import cmath
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable


BRIDGE_LEDGER = {
    "B0": ("depth_rate_identifiability", "counterexample"),
    "B1": ("coherent_amplitude_to_positive_path_measure", "counterexample"),
    "B2": ("branching_extinction_to_path_survival", "counterexample"),
    "B3": ("path_fraction_to_energy_fraction", "iff_condition"),
    "B4": ("energy_fraction_to_baryon_observables", "iff_and_counterexample"),
    "B5": ("algebraic_overlap_to_physical_projector", "construction_nonunique"),
    "B6": ("alpha_s_to_physical_weak_angle", "counterexample"),
    "B7": ("background_expansion_to_dark_split", "counterexample"),
    "B8": ("closed_action_to_gradient_flow", "counterexample"),
    "B9": ("ger_susceptibility_to_primordial_amplitude", "counterexample"),
    "B10": ("local_field_equations_to_global_causality", "counterexample"),
    "B11": ("low_energy_inflation_action_to_uv_robustness", "counterexample"),
    "B12": ("low_energy_axion_action_to_quality", "counterexample"),
    "B13": ("neutrino_mass_matrix_to_uv_yukawa", "counterexample"),
    "B14": ("transport_existence_to_baryon_asymmetry", "counterexample"),
    "B15": ("permutation_symmetry_to_koide_cone", "counterexample"),
    "B16": ("spatial_dimension_to_internal_gauge_group", "counterexample"),
    "B17": ("formal_dimension_to_operational_depth", "counterexample"),
    "B18": ("registered_function_to_physical_coupling", "counterexample"),
    "B19": ("scalar_fixed_point_to_multitype_matrix_and_no_memory", "counterexample"),
    "B20": ("waveform_and_environment_split_to_unique_ger", "counterexample"),
    "B21": ("inflation_potential_to_pivot_efolds", "counterexample"),
    "B22": ("mass_spectrum_to_flavour_texture", "counterexample"),
    "B23": ("planar_wall_source_to_cosmic_asymmetry", "counterexample"),
    "B24": ("euclidean_or_lorentzian_weight_to_energy_or_attenuation", "counterexample"),
    "B25": ("entropy_ansatz_to_absolute_vacuum_scale", "counterexample"),
    "B26": ("pole_mass_to_precision_observable", "counterexample"),
    "B27": ("background_expansion_to_growth", "counterexample"),
    "B28": ("one_scale_coupling_relation_to_rg_unification", "counterexample"),
    "B29": ("tree_hierarchy_to_radiative_naturalness", "counterexample"),
    "B30": ("folding_kinematics_to_einstein_gravity", "counterexample"),
    "B31": ("kinematic_exotic_geometry_to_dynamical_solution", "counterexample"),
    "B32": ("bao_ratios_to_absolute_hubble_scale", "counterexample"),
    "B33": ("dimensionless_recombination_data_to_cmb_temperature", "counterexample"),
    "B34": ("registered_scalar_to_field_content_and_domain_wall_number", "counterexample"),
    "B35": (
        "single_calibration_input_to_independent_predictions",
        "linearized_rank_bound",
    ),
}


def coherent_weight(amplitudes: Iterable[complex]) -> float:
    """Return the coherent Born weight of one coarse-grained alternative."""

    return abs(sum(amplitudes, 0j)) ** 2


def coherent_additivity_defect(left: complex, right: complex) -> float:
    """Return W({left,right}) - W({left}) - W({right})."""

    return (
        coherent_weight((left, right))
        - coherent_weight((left,))
        - coherent_weight((right,))
    )


def poisson_extinction(mean_offspring: float) -> float:
    """Return the minimal fixed point using a cancellation-safe survival root."""

    if not math.isfinite(mean_offspring) or mean_offspring < 0.0:
        raise ValueError("mean_offspring must be finite and nonnegative")
    if mean_offspring <= 1.0:
        return 1.0

    first_extinction = math.exp(-mean_offspring)
    if first_extinction == 0.0:
        return 0.0

    def survival_residual(survival: float) -> float:
        return -math.expm1(-mean_offspring * survival) - survival

    lower = 0.0
    upper = 1.0 - first_extinction
    assert survival_residual(upper) < 0.0
    for _ in range(256):
        midpoint = 0.5 * (lower + upper)
        if survival_residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 1.0 - 0.5 * (lower + upper)


def path_energy_readout(
    probabilities: tuple[float, ...],
    survivor_weights: tuple[float, ...],
    energies: tuple[float, ...],
) -> tuple[float, float, float]:
    """Return path fraction, energy fraction, and their covariance numerator."""

    if not probabilities or not (
        len(probabilities) == len(survivor_weights) == len(energies)
    ):
        raise ValueError("the three finite vectors must have one common nonzero size")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(
        sum(probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ValueError("probabilities must sum to one")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in survivor_weights
    ):
        raise ValueError("survivor weights must lie in [0,1]")
    if any(not math.isfinite(value) or value <= 0.0 for value in energies):
        raise ValueError("energies must be finite and positive")

    path_fraction = sum(
        probability * survivor
        for probability, survivor in zip(
            probabilities, survivor_weights, strict=True
        )
    )
    mean_energy = sum(
        probability * energy
        for probability, energy in zip(probabilities, energies, strict=True)
    )
    survivor_energy = sum(
        probability * survivor * energy
        for probability, survivor, energy in zip(
            probabilities, survivor_weights, energies, strict=True
        )
    )
    covariance_numerator = survivor_energy - path_fraction * mean_energy
    return (
        path_fraction,
        survivor_energy / mean_energy,
        covariance_numerator,
    )


def omega_b_from_energy_fraction(
    energy_fraction: float,
    total_density_parameter: float,
) -> float:
    """Return Omega_b=x_E*Omega_tot at one epoch."""

    if (
        not math.isfinite(energy_fraction)
        or not 0.0 <= energy_fraction <= 1.0
        or not math.isfinite(total_density_parameter)
        or total_density_parameter < 0.0
    ):
        raise ValueError("invalid density data")
    return energy_fraction * total_density_parameter


def baryon_to_photon_ratio(
    energy_fraction: float,
    total_density: float,
    mean_baryon_mass: float,
    photon_density: float,
    asymmetry_fraction: float,
) -> float:
    """Evaluate eta_b under the explicit nonrelativistic rest-mass readout."""

    if (
        not 0.0 <= energy_fraction <= 1.0
        or total_density < 0.0
        or mean_baryon_mass <= 0.0
        or photon_density <= 0.0
        or not -1.0 <= asymmetry_fraction <= 1.0
    ):
        raise ValueError("invalid baryon readout data")
    return (
        energy_fraction
        * total_density
        * asymmetry_fraction
        / (mean_baryon_mass * photon_density)
    )


def required_baryon_transfer(
    total_density: float,
    target_fraction: float,
    target_fraction_rate: float,
    hubble_rate: float,
    baryon_equation_of_state: float,
    effective_equation_of_state: float,
) -> float:
    """Return the transfer source required by a specified fraction history."""

    if total_density <= 0.0:
        raise ValueError("total_density must be positive")
    return total_density * (
        target_fraction_rate
        + 3.0
        * hubble_rate
        * target_fraction
        * (baryon_equation_of_state - effective_equation_of_state)
    )


def fixed_point_map(depth: float, kappa: float, state: float) -> float:
    """Evaluate the CE map using only the invariant optical depth kappa*depth."""

    return math.exp(-kappa * depth * (1.0 - state))


def poisson_vector_map(
    mean_matrix: tuple[tuple[float, ...], ...],
    state: tuple[float, ...],
) -> tuple[float, ...]:
    """Evaluate the componentwise multitype Poisson generating map."""

    size = len(state)
    if size == 0 or len(mean_matrix) != size:
        raise ValueError("matrix and state must have one common nonzero size")
    if any(len(row) != size for row in mean_matrix):
        raise ValueError("mean_matrix must be square")
    if any(
        not math.isfinite(entry) or entry < 0.0
        for row in mean_matrix
        for entry in row
    ):
        raise ValueError("mean_matrix must be finite and nonnegative")
    if any(
        not math.isfinite(component) or not 0.0 <= component <= 1.0
        for component in state
    ):
        raise ValueError("state components must lie in [0,1]")
    return tuple(
        math.exp(
            -sum(
                row[column] * (1.0 - state[column])
                for column in range(size)
            )
        )
        for row in mean_matrix
    )


def projector_overlap_realization(overlap: float) -> tuple[float, float]:
    """Realize any overlap in [0,1] by two rank-one projectors in C^2."""

    if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0,1]")
    component_parallel = math.sqrt(overlap)
    component_orthogonal = math.sqrt(1.0 - overlap)
    norm_residual = component_parallel**2 + component_orthogonal**2 - 1.0
    realized_overlap = component_parallel**2
    return realized_overlap, norm_residual


def weak_mixing_angle(gauge_su2: float, gauge_u1: float) -> float:
    """Return the tree-level canonical neutral mixing parameter."""

    if (
        not math.isfinite(gauge_su2)
        or not math.isfinite(gauge_u1)
        or gauge_su2 <= 0.0
        or gauge_u1 <= 0.0
    ):
        raise ValueError("gauge couplings must be finite and positive")
    return gauge_u1**2 / (gauge_su2**2 + gauge_u1**2)


def dark_split(total_remainder: float, ratio: float) -> tuple[float, float]:
    """Split one nonnegative total using a supplied positive ratio."""

    if (
        not math.isfinite(total_remainder)
        or total_remainder < 0.0
        or not math.isfinite(ratio)
        or ratio <= 0.0
    ):
        raise ValueError("the total must be nonnegative and the ratio positive")
    dark_energy = total_remainder / (1.0 + ratio)
    dark_matter = total_remainder - dark_energy
    return dark_matter, dark_energy


def vacuum_variance(normalization: float, frequency: float) -> float:
    """Ground-state variance for Z/2*(qdot^2-omega^2*q^2)."""

    if (
        not math.isfinite(normalization)
        or normalization <= 0.0
        or not math.isfinite(frequency)
        or frequency <= 0.0
    ):
        raise ValueError("normalization and frequency must be finite and positive")
    return 1.0 / (2.0 * normalization * frequency)


def oscillator_potential_rate(
    coordinate: float,
    velocity: float,
    frequency: float,
) -> float:
    """Return dV/dt for V=omega^2*q^2/2 along arbitrary initial data."""

    return frequency**2 * coordinate * velocity


def gradient_potential_rate(gradient: float, inverse_metric: float = 1.0) -> float:
    """Return dV/dt along qdot=-G^{-1} grad(V) in one dimension."""

    if inverse_metric <= 0.0:
        raise ValueError("inverse_metric must be positive")
    return -(gradient**2) * inverse_metric


def registered_coupling_family(
    alpha: float,
    calibration_alpha: float,
    deformation: float,
) -> float:
    """A family agreeing with 4*alpha^(4/3) at one calibration point."""

    if alpha <= 0.0 or calibration_alpha <= 0.0:
        raise ValueError("couplings must be positive")
    return 4.0 * alpha ** (4.0 / 3.0) * math.exp(
        deformation * (alpha - calibration_alpha)
    )


def allocation_exponent(optical_depth: float, environment_weight: float) -> float:
    """Return the additive log-action share carried by the internal sector."""

    if optical_depth < 0.0 or environment_weight <= 0.0:
        raise ValueError("invalid allocation weights")
    return optical_depth / (optical_depth + environment_weight)


def reheating_expansion(
    end_density: float,
    reheating_density: float,
    equation_of_state: float,
) -> float:
    """Return ln(a_re/a_end) for a constant-w reheating stage."""

    if (
        end_density <= 0.0
        or reheating_density <= 0.0
        or reheating_density > end_density
        or equation_of_state <= -1.0
    ):
        raise ValueError("invalid reheating parameters")
    return math.log(end_density / reheating_density) / (
        3.0 * (1.0 + equation_of_state)
    )


def oriented_domain_average(local_source: float, positive_fraction: float) -> float:
    """Average equal-magnitude wall sources with opposite orientations."""

    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("positive_fraction must lie in [0,1]")
    return (2.0 * positive_fraction - 1.0) * local_source


def portal_proxy(coupling: float, pole_mass: float) -> float:
    """Minimal coupling-over-pole proxy showing residue non-identifiability."""

    if pole_mass <= 0.0:
        raise ValueError("pole_mass must be positive")
    return coupling**2 / pole_mass**2


def inverse_coupling_run(
    inverse_coupling: float,
    beta_coefficient: float,
    log_scale_ratio: float,
) -> float:
    """One-loop affine running of an inverse gauge coupling."""

    return inverse_coupling - beta_coefficient * log_scale_ratio / (
        2.0 * math.pi
    )


def threshold_mass_shift(coupling: float, heavy_mass: float) -> float:
    """Generic one-loop scalar threshold scale."""

    if heavy_mass < 0.0:
        raise ValueError("heavy_mass must be nonnegative")
    return coupling * heavy_mass**2 / (16.0 * math.pi**2)


def higher_operator_ratio(
    coefficient: float,
    quartic: float,
    field_coordinate: float,
    operator_power: int = 6,
) -> float:
    """Return |V_n/V_4| for c_n Phi^n/Mp^(n-4)."""

    if quartic <= 0.0 or field_coordinate < 0.0 or operator_power < 6:
        raise ValueError("invalid EFT hierarchy data")
    return (
        4.0
        * abs(coefficient)
        * field_coordinate ** (operator_power - 4)
        / quartic
    )


def axion_phase_shift(explicit_breaking_ratio: float) -> float:
    """Principal minimum shift for 1-cos(theta)+epsilon*sin(theta)."""

    return -math.atan(explicit_breaking_ratio)


def linear_transport(source: float, operator: float) -> float:
    """Solve the one-dimensional invertible transport equation L*n=S."""

    if operator == 0.0:
        raise ValueError("operator must be invertible")
    return source / operator


def koide_quadratic_selector(
    singlet_coefficient: float,
    doublet_coefficient: float,
) -> float:
    """Minimizing singlet norm squared on the unit sphere."""

    if singlet_coefficient == doublet_coefficient:
        raise ValueError("degenerate coefficients do not select a direction")
    return 1.0 if singlet_coefficient < doublet_coefficient else 0.0


def euclidean_energy(action: float, circle_length: float) -> float:
    """Return E=S_E/beta for a constant-energy Euclidean saddle."""

    if circle_length <= 0.0:
        raise ValueError("circle_length must be positive")
    return action / circle_length


def constant_vacuum_shift(
    potential_value: float,
    potential_force: float,
    counterterm: float,
) -> tuple[float, float]:
    """Shift a potential by a constant without changing its force."""

    return potential_value + counterterm, potential_force


def growth_driving_term(
    matter_fraction: float,
    effective_clustering: float,
    density_contrast: float,
) -> float:
    """Return the source coefficient 3*mu*Omega_m*delta/2."""

    return (
        1.5 * matter_fraction * effective_clustering * density_contrast
    )


def newton_coupling_proxy(einstein_coefficient: float) -> float:
    """Return the inverse Einstein-Hilbert coefficient up to constants."""

    if einstein_coefficient <= 0.0:
        raise ValueError("einstein_coefficient must be positive")
    return 1.0 / einstein_coefficient


def bao_distance_ratio(distance: float, sound_horizon: float) -> float:
    """Return a dimensionless BAO distance ratio."""

    if sound_horizon <= 0.0:
        raise ValueError("sound_horizon must be positive")
    return distance / sound_horizon


def photon_temperature(entropy_constant: float, scale_factor: float) -> float:
    """Return T_gamma=C_gamma/a in a fixed-entropy-degree interval."""

    if entropy_constant <= 0.0 or scale_factor <= 0.0:
        raise ValueError("thermal scales must be positive")
    return entropy_constant / scale_factor


def pq_anomaly_coefficient(
    charges: tuple[float, ...],
    dynkin_indices: tuple[float, ...],
) -> float:
    """Return 2*sum(X_f*T(R_f)) in a fixed primitive charge convention."""

    if not charges or len(charges) != len(dynkin_indices):
        raise ValueError("charge and representation data must align")
    return 2.0 * sum(
        charge * index
        for charge, index in zip(charges, dynkin_indices, strict=True)
    )


def linearized_covariance(
    jacobian: tuple[float, ...],
    input_variance: float,
) -> tuple[tuple[float, ...], ...]:
    """Return J sigma^2 J^T for one scalar input."""

    if not jacobian or input_variance < 0.0:
        raise ValueError("invalid linearized covariance data")
    return tuple(
        tuple(input_variance * left * right for right in jacobian)
        for left in jacobian
    )


def covariance_rank_one_certificate(
    covariance: tuple[tuple[float, ...], ...],
    tolerance: float = 1e-14,
) -> int:
    """Certify rank zero or one by testing entries and all two-by-two minors."""

    size = len(covariance)
    if size == 0 or any(len(row) != size for row in covariance):
        raise ValueError("covariance must be a nonempty square matrix")
    if all(abs(entry) <= tolerance for row in covariance for entry in row):
        return 0
    for row_a in range(size):
        for row_b in range(row_a + 1, size):
            for column_a in range(size):
                for column_b in range(column_a + 1, size):
                    determinant = (
                        covariance[row_a][column_a]
                        * covariance[row_b][column_b]
                        - covariance[row_a][column_b]
                        * covariance[row_b][column_a]
                    )
                    if abs(determinant) > tolerance:
                        raise ValueError("matrix is not rank one")
    return 1


def wormhole_nec_at_throat(
    shape_derivative: float,
    throat_radius: float,
    newton_constant: float = 1.0,
) -> float:
    """rho+p_r at a static Morris--Thorne throat in Einstein gravity."""

    if throat_radius <= 0.0 or newton_constant <= 0.0:
        raise ValueError("scales must be positive")
    return (shape_derivative - 1.0) / (
        8.0 * math.pi * newton_constant * throat_radius**2
    )


def _matmul2(
    left: tuple[tuple[complex, complex], tuple[complex, complex]],
    right: tuple[tuple[complex, complex], tuple[complex, complex]],
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def _transpose2(
    matrix: tuple[tuple[complex, complex], tuple[complex, complex]],
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    return ((matrix[0][0], matrix[1][0]), (matrix[0][1], matrix[1][1]))


def casas_ibarra_rank2(
    mass_1: float,
    mass_2: float,
    angle: complex,
) -> tuple[
    tuple[tuple[complex, complex], tuple[complex, complex]],
    float,
]:
    """Return a rank-two Yukawa representative and its mass residual."""

    if mass_1 <= 0.0 or mass_2 <= 0.0:
        raise ValueError("masses must be positive")
    cosine = cmath.cos(angle)
    sine = cmath.sin(angle)
    rotation = ((cosine, sine), (-sine, cosine))
    root_mass = ((math.sqrt(mass_1), 0j), (0j, math.sqrt(mass_2)))
    yukawa = _matmul2(root_mass, rotation)
    reconstructed = _matmul2(yukawa, _transpose2(yukawa))
    target = ((complex(mass_1), 0j), (0j, complex(mass_2)))
    residual = max(
        abs(reconstructed[row][column] - target[row][column])
        for row in range(2)
        for column in range(2)
    )
    return yukawa, residual


def matrix_distance2(
    left: tuple[tuple[complex, complex], tuple[complex, complex]],
    right: tuple[tuple[complex, complex], tuple[complex, complex]],
) -> float:
    return math.sqrt(
        sum(
            abs(left[row][column] - right[row][column]) ** 2
            for row in range(2)
            for column in range(2)
        )
    )


@dataclass(frozen=True)
class OpenBridgeAudit:
    bridge_count: int
    destructive_interference_additivity_defect: float
    decoherent_pair_additivity_defect: float
    branching_extinction: float
    branching_nonextinction: float
    branching_semantic_gap: float
    common_path_fraction: float
    low_survivor_energy_fraction: float
    high_survivor_energy_fraction: float
    low_energy_covariance_identity_residual: float
    high_energy_covariance_identity_residual: float
    curved_omega_b: float
    symmetric_baryon_to_photon_ratio: float
    asymmetric_baryon_to_photon_ratio: float
    required_constant_fraction_transfer: float
    depth_reparameterization_residual: float
    realized_projector_overlap: float
    projector_normalization_residual: float
    projector_complement_delta_residual: float
    weak_angle_first: float
    weak_angle_second: float
    same_alpha_s_weak_angle_gap: float
    first_dark_ratio: float
    second_dark_ratio: float
    first_dark_total_residual: float
    second_dark_total_residual: float
    ger_vacuum_variance_ratio: float
    conservative_positive_potential_rate: float
    gradient_negative_potential_rate: float
    ci_first_mass_residual: float
    ci_second_mass_residual: float
    ci_yukawa_nonuniqueness_distance: float
    zero_source_transport_output: float
    doubled_source_transport_ratio: float
    axion_quality_shift: float
    koide_singlet_selected_cosine_squared: float
    koide_doublet_selected_cosine_squared: float
    flat_time_circle_interval_squared: float
    same_spatial_dimension_gauge_algebra_dimensions: tuple[int, int, int]
    dimension_six_to_quartic_per_c6: float
    dimension_depth_same_tau_residual: float
    registered_function_calibration_residual: float
    registered_function_off_point_gap: float
    same_row_sum_branching_residual: float
    multitype_hidden_mode_gap: float
    waveform_average_gap: float
    environment_allocation_exponent_gap: float
    reheating_expansion_gap: float
    same_mass_flavour_angle_gap: float
    symmetric_wall_average: float
    aligned_wall_average: float
    euclidean_energy_scale_ratio: float
    unitary_attenuation_defect: float
    vacuum_counterterm_force_gap: float
    vacuum_counterterm_energy_gap: float
    zero_residue_observable: float
    nonzero_residue_observable: float
    same_background_growth_source_gap: float
    one_scale_relation_rg_drift: float
    hierarchy_threshold_to_tree_ratio: float
    same_fold_newton_coupling_ratio: float
    wormhole_throat_nec: float
    bao_scaling_ratio_residual: float
    cmb_temperature_integration_constant_ratio: float
    same_registered_delta_portal_gap: float
    same_low_energy_axion_domain_wall_gap: int
    linearized_single_input_covariance_rank: int
    unconditional_implications_valid: bool
    physical_realizations_validated: bool


def build_audit() -> OpenBridgeAudit:
    destructive_defect = coherent_additivity_defect(1.0 + 0j, -1.0 + 0j)
    decoherent_defect = coherent_additivity_defect(1.0 + 0j, 0.0 + 1.0j)

    branching_q = poisson_extinction(2.0)
    branching_p = 1.0 - branching_q

    low_path, low_energy, low_covariance = path_energy_readout(
        (0.5, 0.5),
        (1.0, 0.0),
        (1.0, 9.0),
    )
    high_path, high_energy, high_covariance = path_energy_readout(
        (0.5, 0.5),
        (1.0, 0.0),
        (9.0, 1.0),
    )
    low_covariance_identity = (
        low_energy - low_path - low_covariance / 5.0
    )
    high_covariance_identity = (
        high_energy - high_path - high_covariance / 5.0
    )

    raw_depth = 3.1779129995
    raw_kappa = 1.0
    scale = 7.0
    sample_state = 0.2
    reparameterization_residual = fixed_point_map(
        raw_depth, raw_kappa, sample_state
    ) - fixed_point_map(
        scale * raw_depth,
        raw_kappa / scale,
        sample_state,
    )

    overlap = 0.2315097758
    realized_overlap, projector_norm_residual = projector_overlap_realization(overlap)
    target_delta = overlap * (1.0 - overlap)
    realized_delta = realized_overlap * (1.0 - realized_overlap)

    weak_first = weak_mixing_angle(1.0, 0.5)
    weak_second = weak_mixing_angle(1.0, 2.0)

    first_dark = dark_split(0.95, 0.25)
    second_dark = dark_split(0.95, 4.0)

    variance_small_z = vacuum_variance(1.0, 1.0)
    variance_large_z = vacuum_variance(100.0, 1.0)

    first_yukawa, first_mass_residual = casas_ibarra_rank2(0.01, 0.05, 0j)
    second_yukawa, second_mass_residual = casas_ibarra_rank2(
        0.01, 0.05, 0.4 + 0.2j
    )

    curved_omega_b = omega_b_from_energy_fraction(0.05, 1.2)
    symmetric_eta = baryon_to_photon_ratio(0.05, 1.0, 1.0, 2.0, 0.0)
    asymmetric_eta = baryon_to_photon_ratio(0.05, 1.0, 1.0, 2.0, 1.0)
    constant_fraction_transfer = required_baryon_transfer(
        1.0,
        0.05,
        0.0,
        1.0,
        0.0,
        -1.0,
    )

    # A one-dimensional invertible transport operator 2*n=S already suffices:
    # existence is automatic, S=0 gives no asymmetry, and scaling S scales n.
    zero_transport = linear_transport(0.0, 2.0)
    unit_transport = linear_transport(1.0, 2.0)
    double_transport = linear_transport(2.0, 2.0)

    # V(theta)=chi*(1-cos(theta))+epsilon*sin(theta) has
    # tan(theta_vac)=-epsilon/chi on its principal minimum branch.
    axion_shift = axion_phase_shift(0.01)

    quartic = 1.3434991214e-10
    inflaton_coordinate = 11.0974588093
    dimension_six_ratio = higher_operator_ratio(
        1.0,
        quartic,
        inflaton_coordinate,
    )

    # The same optical depth can be assigned to different formal dimensions.
    formal_delta = 0.18
    target_tau = 3.0 + formal_delta
    alternative_depth = 2.0 + formal_delta
    alternative_kappa = target_tau / alternative_depth

    calibration_alpha = 0.118
    calibrated_first = registered_coupling_family(
        calibration_alpha, calibration_alpha, 0.0
    )
    calibrated_second = registered_coupling_family(
        calibration_alpha, calibration_alpha, 7.0
    )
    off_point_alpha = 0.12
    off_point_first = registered_coupling_family(
        off_point_alpha, calibration_alpha, 0.0
    )
    off_point_second = registered_coupling_family(
        off_point_alpha, calibration_alpha, 7.0
    )

    # A=tau*I and A=tau/2*11^T have the same row sums and hence the same
    # uniform branching equation, but their non-uniform eigenvalues differ.
    uniform_state = (0.4, 0.4)
    diagonal_matrix = ((2.0, 0.0), (0.0, 2.0))
    coupled_matrix = ((1.0, 1.0), (1.0, 1.0))
    diagonal_output = poisson_vector_map(diagonal_matrix, uniform_state)
    coupled_output = poisson_vector_map(coupled_matrix, uniform_state)
    same_row_sum_residual = max(
        abs(first - second)
        for first, second in zip(
            diagonal_output,
            coupled_output,
            strict=True,
        )
    )
    hidden_mode_gap = 2.0

    average_sine = 2.0 / math.pi
    average_sine_cubed = 4.0 / (3.0 * math.pi)

    reheating_matterlike = reheating_expansion(1.0, 1e-12, 0.0)
    reheating_radiationlike = reheating_expansion(1.0, 1e-12, 1.0 / 3.0)

    # Same singular values can be combined with arbitrary left rotation.
    first_flavour_angle = 0.0
    second_flavour_angle = 0.4

    euclidean_action = 2.0
    energy_short_circle = euclidean_energy(euclidean_action, 1.0)
    energy_long_circle = euclidean_energy(euclidean_action, 2.0)

    first_vacuum_energy, first_vacuum_force = constant_vacuum_shift(
        0.0,
        3.0,
        0.0,
    )
    second_vacuum_energy, second_vacuum_force = constant_vacuum_shift(
        0.0,
        3.0,
        1.0,
    )

    pole_mass = 0.025
    zero_portal = portal_proxy(0.0, pole_mass)
    nonzero_portal = portal_proxy(1e-4, pole_mass)

    # Same H and delta at one instant, different effective clustering strength.
    growth_standard = growth_driving_term(0.3, 1.0, 1.0)
    growth_modified = growth_driving_term(0.3, 1.2, 1.0)

    first_inverse = 25.0
    second_inverse = 25.0
    scale_log = 1.0
    first_run = inverse_coupling_run(first_inverse, 1.0, scale_log)
    second_run = inverse_coupling_run(second_inverse, -2.0, scale_log)

    tree_mass_squared = 1e-12
    radiative_shift = threshold_mass_shift(1.0, 1.0)

    # A common kinematic folding label does not fix the Einstein-Hilbert
    # coefficient M^2, hence G_N proportional to 1/M^2 remains arbitrary.
    first_planck_coefficient = 1.0
    second_planck_coefficient = 4.0

    distance = 14.0
    sound_horizon = 0.147
    distance_rescaled = 3.0 * distance
    sound_horizon_rescaled = 3.0 * sound_horizon

    first_photon_temperature = photon_temperature(1.0, 1.0)
    second_photon_temperature = photon_temperature(2.0, 1.0)
    registered_delta = 0.18
    first_portal_coupling = 0.0
    second_portal_coupling = registered_delta**2
    first_domain_wall_number = abs(
        pq_anomaly_coefficient((1.0,), (0.5,))
    )
    second_domain_wall_number = abs(
        pq_anomaly_coefficient((1.0, 1.0), (0.5, 0.5))
    )
    local_covariance = linearized_covariance((1.0, 2.0, 3.0), 0.04)

    return OpenBridgeAudit(
        bridge_count=len(BRIDGE_LEDGER),
        destructive_interference_additivity_defect=destructive_defect,
        decoherent_pair_additivity_defect=decoherent_defect,
        branching_extinction=branching_q,
        branching_nonextinction=branching_p,
        branching_semantic_gap=branching_p - branching_q,
        common_path_fraction=0.5 * (low_path + high_path),
        low_survivor_energy_fraction=low_energy,
        high_survivor_energy_fraction=high_energy,
        low_energy_covariance_identity_residual=low_covariance_identity,
        high_energy_covariance_identity_residual=high_covariance_identity,
        curved_omega_b=curved_omega_b,
        symmetric_baryon_to_photon_ratio=symmetric_eta,
        asymmetric_baryon_to_photon_ratio=asymmetric_eta,
        required_constant_fraction_transfer=constant_fraction_transfer,
        depth_reparameterization_residual=reparameterization_residual,
        realized_projector_overlap=realized_overlap,
        projector_normalization_residual=projector_norm_residual,
        projector_complement_delta_residual=realized_delta - target_delta,
        weak_angle_first=weak_first,
        weak_angle_second=weak_second,
        same_alpha_s_weak_angle_gap=weak_second - weak_first,
        first_dark_ratio=first_dark[0] / first_dark[1],
        second_dark_ratio=second_dark[0] / second_dark[1],
        first_dark_total_residual=sum(first_dark) - 0.95,
        second_dark_total_residual=sum(second_dark) - 0.95,
        ger_vacuum_variance_ratio=variance_small_z / variance_large_z,
        conservative_positive_potential_rate=oscillator_potential_rate(
            1.0,
            1.0,
            1.0,
        ),
        gradient_negative_potential_rate=gradient_potential_rate(1.0),
        ci_first_mass_residual=first_mass_residual,
        ci_second_mass_residual=second_mass_residual,
        ci_yukawa_nonuniqueness_distance=matrix_distance2(
            first_yukawa, second_yukawa
        ),
        zero_source_transport_output=zero_transport,
        doubled_source_transport_ratio=double_transport / unit_transport,
        axion_quality_shift=axion_shift,
        koide_singlet_selected_cosine_squared=koide_quadratic_selector(
            0.0,
            1.0,
        ),
        koide_doublet_selected_cosine_squared=koide_quadratic_selector(
            1.0,
            0.0,
        ),
        flat_time_circle_interval_squared=-1.0,
        same_spatial_dimension_gauge_algebra_dimensions=(1, 3, 8),
        dimension_six_to_quartic_per_c6=dimension_six_ratio,
        dimension_depth_same_tau_residual=(
            alternative_kappa * alternative_depth - target_tau
        ),
        registered_function_calibration_residual=(
            calibrated_second - calibrated_first
        ),
        registered_function_off_point_gap=off_point_second - off_point_first,
        same_row_sum_branching_residual=same_row_sum_residual,
        multitype_hidden_mode_gap=hidden_mode_gap,
        waveform_average_gap=average_sine - average_sine_cubed,
        environment_allocation_exponent_gap=(
            allocation_exponent(2.0, 1.0) - allocation_exponent(2.0, 2.0)
        ),
        reheating_expansion_gap=(
            reheating_matterlike - reheating_radiationlike
        ),
        same_mass_flavour_angle_gap=(
            second_flavour_angle - first_flavour_angle
        ),
        symmetric_wall_average=oriented_domain_average(1.0, 0.5),
        aligned_wall_average=oriented_domain_average(1.0, 1.0),
        euclidean_energy_scale_ratio=(
            energy_short_circle / energy_long_circle
        ),
        unitary_attenuation_defect=abs(cmath.exp(1.0j * 3.0)) - 1.0,
        vacuum_counterterm_force_gap=(
            second_vacuum_force - first_vacuum_force
        ),
        vacuum_counterterm_energy_gap=(
            second_vacuum_energy - first_vacuum_energy
        ),
        zero_residue_observable=zero_portal,
        nonzero_residue_observable=nonzero_portal,
        same_background_growth_source_gap=(
            growth_modified - growth_standard
        ),
        one_scale_relation_rg_drift=first_run - second_run,
        hierarchy_threshold_to_tree_ratio=radiative_shift / tree_mass_squared,
        same_fold_newton_coupling_ratio=(
            newton_coupling_proxy(second_planck_coefficient)
            / newton_coupling_proxy(first_planck_coefficient)
        ),
        wormhole_throat_nec=wormhole_nec_at_throat(0.5, 1.0),
        bao_scaling_ratio_residual=(
            bao_distance_ratio(distance_rescaled, sound_horizon_rescaled)
            - bao_distance_ratio(distance, sound_horizon)
        ),
        cmb_temperature_integration_constant_ratio=(
            second_photon_temperature / first_photon_temperature
        ),
        same_registered_delta_portal_gap=(
            second_portal_coupling - first_portal_coupling
        ),
        same_low_energy_axion_domain_wall_gap=int(
            second_domain_wall_number - first_domain_wall_number
        ),
        linearized_single_input_covariance_rank=(
            covariance_rank_one_certificate(local_covariance)
        ),
        unconditional_implications_valid=False,
        physical_realizations_validated=False,
    )


def validate(audit: OpenBridgeAudit) -> None:
    assert set(BRIDGE_LEDGER) == {f"B{index}" for index in range(36)}
    assert audit.bridge_count == 36
    assert math.isclose(audit.destructive_interference_additivity_defect, -2.0)
    assert abs(audit.decoherent_pair_additivity_defect) < 1e-15
    assert math.isclose(audit.branching_extinction, 0.20318786998, abs_tol=5e-12)
    assert audit.branching_semantic_gap > 0.5
    assert math.isclose(audit.common_path_fraction, 0.5)
    assert math.isclose(audit.low_survivor_energy_fraction, 0.1)
    assert math.isclose(audit.high_survivor_energy_fraction, 0.9)
    assert abs(audit.low_energy_covariance_identity_residual) < 1e-15
    assert abs(audit.high_energy_covariance_identity_residual) < 1e-15
    assert math.isclose(audit.curved_omega_b, 0.06)
    assert audit.symmetric_baryon_to_photon_ratio == 0.0
    assert audit.asymmetric_baryon_to_photon_ratio > 0.0
    assert math.isclose(audit.required_constant_fraction_transfer, 0.15)
    assert abs(audit.depth_reparameterization_residual) < 1e-15
    assert math.isclose(audit.realized_projector_overlap, 0.2315097758)
    assert abs(audit.projector_normalization_residual) < 1e-15
    assert abs(audit.projector_complement_delta_residual) < 1e-15
    assert math.isclose(audit.weak_angle_first, 0.2)
    assert math.isclose(audit.weak_angle_second, 0.8)
    assert audit.same_alpha_s_weak_angle_gap > 0.5
    assert math.isclose(audit.first_dark_ratio, 0.25)
    assert math.isclose(audit.second_dark_ratio, 4.0)
    assert abs(audit.first_dark_total_residual) < 1e-15
    assert abs(audit.second_dark_total_residual) < 1e-15
    assert math.isclose(audit.ger_vacuum_variance_ratio, 100.0)
    assert audit.conservative_positive_potential_rate > 0.0
    assert audit.gradient_negative_potential_rate < 0.0
    assert audit.ci_first_mass_residual < 1e-15
    assert audit.ci_second_mass_residual < 1e-15
    assert audit.ci_yukawa_nonuniqueness_distance > 0.05
    assert audit.zero_source_transport_output == 0.0
    assert audit.doubled_source_transport_ratio == 2.0
    assert audit.axion_quality_shift != 0.0
    assert audit.koide_singlet_selected_cosine_squared == 1.0
    assert audit.koide_doublet_selected_cosine_squared == 0.0
    assert audit.flat_time_circle_interval_squared < 0.0
    assert audit.same_spatial_dimension_gauge_algebra_dimensions == (1, 3, 8)
    assert audit.dimension_six_to_quartic_per_c6 > 1e12
    assert abs(audit.dimension_depth_same_tau_residual) < 1e-15
    assert abs(audit.registered_function_calibration_residual) < 1e-15
    assert audit.registered_function_off_point_gap > 0.001
    assert abs(audit.same_row_sum_branching_residual) < 1e-15
    assert audit.multitype_hidden_mode_gap > 1.0
    assert audit.waveform_average_gap > 0.2
    assert audit.environment_allocation_exponent_gap > 0.1
    assert audit.reheating_expansion_gap > 2.0
    assert audit.same_mass_flavour_angle_gap > 0.3
    assert audit.symmetric_wall_average == 0.0
    assert audit.aligned_wall_average == 1.0
    assert audit.euclidean_energy_scale_ratio == 2.0
    assert abs(audit.unitary_attenuation_defect) < 1e-15
    assert audit.vacuum_counterterm_force_gap == 0.0
    assert audit.vacuum_counterterm_energy_gap == 1.0
    assert audit.zero_residue_observable == 0.0
    assert audit.nonzero_residue_observable > 0.0
    assert audit.same_background_growth_source_gap > 0.0
    assert abs(audit.one_scale_relation_rg_drift) > 0.1
    assert audit.hierarchy_threshold_to_tree_ratio > 1e8
    assert audit.same_fold_newton_coupling_ratio == 0.25
    assert audit.wormhole_throat_nec < 0.0
    assert abs(audit.bao_scaling_ratio_residual) < 1e-12
    assert audit.cmb_temperature_integration_constant_ratio == 2.0
    assert audit.same_registered_delta_portal_gap > 0.0
    assert audit.same_low_energy_axion_domain_wall_gap == 1
    assert audit.linearized_single_input_covariance_rank == 1
    assert not audit.unconditional_implications_valid
    assert not audit.physical_realizations_validated


def main() -> None:
    audit = build_audit()
    validate(audit)
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
    print("bridge_implications_proved_or_refuted: true")
    print("physical_realizations_validated: false")


if __name__ == "__main__":
    main()
