"""Single-clock perturbation and EFT-cutoff gate for the kinetic dark sector.

This promotes the reproducible, low-data part of the former R2 diagnostic into
the main examples tree.  It tests the clock+Einstein subsystem only.  The
fixed-background ``pi`` growth below is a diagnostic of the clock coordinate,
not the observable matter-growth function or an ``f sigma_8`` likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.kinetic_dark_sector_gate import (
    BackgroundSolution,
    KineticClockConfig,
    _densities,
    _rhs,
    solve_background,
)


H0_KM_S_MPC = 67.4
MPC_IN_M = 3.0856775814913673e22
HBAR_EV_S = 6.582119569e-16
HBAR_C_EV_M = 1.973269804e-7
REDUCED_MPL_EV = 2.435e27
H0_EV = H0_KM_S_MPC * 1000.0 / MPC_IN_M * HBAR_EV_S
RHO_CRIT0_EV4 = 3.0 * REDUCED_MPL_EV**2 * H0_EV**2
MPC_INV_EV = HBAR_C_EV_M / MPC_IN_M


@dataclass(frozen=True)
class PerturbationNode:
    n: float
    e2: float
    friction: float
    tachyon_ratio: float
    cs2: float
    q_s_over_mpl2: float
    pump_slope: float
    zeta_decay_slope: float
    energy_cutoff_ev: float
    wavenumber_cutoff_ev: float


@dataclass(frozen=True)
class SingleClockGate:
    gamma: float
    min_friction: float
    max_tachyon_ratio: float
    fixed_coordinate_growth_minus_one: float
    max_log_growth_bound: float
    min_pump_slope: float
    min_zeta_decay_slope: float
    min_energy_cutoff_over_h: float
    min_wavenumber_cutoff_over_k_1mpc: float
    status: str = "PASS_SINGLE_CLOCK_ONLY"
    failed_gates: tuple[str, ...] = ()
    matter_growth_likelihood: str = "NOT_IMPLEMENTED_COUPLED_EQUATIONS_REQUIRED"


@dataclass(frozen=True)
class QuasiStaticGrowthDiagnostic:
    redshift: float
    predicted_fsigma8: float
    observed_fsigma8: float
    observed_sigma: float
    pull: float
    sigma8_0: float
    closure: str = "KINETIC_CLUSTERS_VACUUM_SMOOTH_GR_SUBHORIZON"
    role: str = "APPROXIMATE_DIAGNOSTIC_NOT_FULL_COUPLED_LIKELIHOOD"


@dataclass(frozen=True)
class KappaSensitivityRow:
    kappa: float
    min_cs2: float
    min_friction: float
    max_log_growth_bound: float
    min_energy_cutoff_over_h: float
    status: str
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class FiniteProductGaussianStateDensityAudit:
    """Exact finite-mode state-to-energy receipt for the retained action.

    Natural units ``hbar=c=1`` and ``a0=1`` are fixed.  The homogeneous
    canonical coordinates are ``q=sqrt(Vc)*(phi, chi)`` and
    ``p=sqrt(Vc)*(dot(phi), dot(chi))``.  The supplied boundary state is an
    uncoupled ``phi`` coherent state times a ``chi`` thermal state.  It is a
    generally correlated Gaussian state after the *same* orthogonal normal
    mode rotation is applied to both the position and momentum blocks.

    The finite-mode subtraction ``E_i-mu_i/2`` is normal ordering relative to
    the interacting two-oscillator vacuum.  It is not covariant QFT stress
    renormalization and it does not select a primordial cosmological state.
    """

    action_parameter_manifest: tuple[float, float, float, float]
    state_boundary_manifest: tuple[float, float, float, float, float]
    environment_thermal_marginal_manifest: tuple[float, float, float, float]
    covariance_ordering: tuple[str, str, str, str]
    canonical_mean: tuple[float, float, float, float]
    centered_symmetrized_covariance: tuple[tuple[float, ...], ...]
    symplectic_eigenvalues: tuple[float, float]
    expected_symplectic_eigenvalues: tuple[float, float]
    dimensionless_uncertainty_minimum_eigenvalue: float
    normal_mode_rotation: tuple[tuple[float, float], ...]
    phase_space_symplectic_transform: tuple[tuple[float, ...], ...]
    symplectic_residual: float
    normal_mode_diagonalization_relative_residual: float
    normal_mode_mass_squared: tuple[float, float]
    normal_mode_masses: tuple[float, float]
    relative_spectral_gap: float
    normal_mode_mean: tuple[float, float, float, float]
    normal_mode_centered_covariance: tuple[tuple[float, ...], ...]
    normal_mode_position_cross_covariance: float
    normal_mode_momentum_cross_covariance: float
    finite_mode_raw_energies_ev: tuple[float, float]
    finite_mode_vacuum_energies_ev: tuple[float, float]
    finite_mode_vacuum_subtracted_energies_ev: tuple[float, float]
    finite_mode_density_constants_ev4: tuple[float, float]
    uncoupled_coherent_preparation_energy_ev: float
    uncoupled_thermal_preparation_energy_ev: float
    bare_product_vacuum_energy_ev: float
    interacting_vacuum_energy_ev: float
    vacuum_mismatch_quench_energy_ev: float
    finite_mode_vacuum_subtracted_total_energy_ev: float
    finite_mode_vacuum_subtracted_total_density_ev4: float
    vacuum_cell_energy_ev: float
    thermal_occupation_relative_residual: float
    raw_energy_rotation_relative_residual: float
    excitation_energy_ledger_relative_residual: float
    mode_sign_flip_energy_relative_residual: float
    uncertainty_principle_pass: bool
    covariance_physicality_pass: bool
    canonical_transform_pass: bool
    mass_matrix_stable: bool
    nondegenerate_mode_allocation_pass: bool
    finite_mode_excitation_nonnegative: bool
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str
    representation: str = "RETAINED_TWO_FIELD_STATE_ENERGY_ONLY"
    boundary_condition: str = (
        "SUPPLIED_UNCOUPLED_COHERENT_X_THERMAL_PRODUCT_AT_A0_EQ_1"
    )
    normal_mode_state_role: str = (
        "CORRELATED_GAUSSIAN_AFTER_CANONICAL_BASIS_ROTATION_NOT_MODE_THERMAL"
    )
    same_state_finite_mode_energy_map_derived: bool = True
    ctp_to_cosmological_state_map_derived: bool = False
    finite_mode_vacuum_subtraction_only: bool = True
    covariant_qft_stress_renormalized: bool = False
    preparation_battery_dynamics_derived: bool = False
    cosmological_initial_state_derived: bool = False
    absolute_abundance_predicted: bool = False
    vacuum_energy_from_state_derived: bool = False
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def _real_matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _relative_residual(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0.0 else abs(left - right) / scale


def audit_finite_product_gaussian_state_densities(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    comoving_volume_ev_minus3: float,
    system_field_mean_ev: float,
    system_field_velocity_ev2: float,
    environment_mean_occupation: float,
    environment_inverse_temperature_ev_minus1: float,
    tolerance: float = 2.0e-11,
) -> FiniteProductGaussianStateDensityAudit:
    """Map one supplied product Gaussian state to exact normal-mode energies.

    For ``R=(q_phi,q_chi,p_phi,p_chi)``, the centered covariance is

    ``Sigma=diag(1/(2m), (n+1/2)/M, m/2, M(n+1/2))``.

    If ``O.T@K@O=diag(mu_-^2,mu_+^2)``, the block transform
    ``S=diag(O.T,O.T)`` is symplectic and the excitation of mode ``i`` is

    ``E_i=1/2[Pbar_i^2+mu_i^2 Qbar_i^2``
    ``    +Sigma_PP[ii]+mu_i^2 Sigma_QQ[ii]-mu_i]``.

    No negative value is clipped.  A physicality or non-negativity violation
    fails closed.  Individual mode densities also fail closed at degeneracy,
    where only their sum would be basis invariant.
    """

    finite_inputs = {
        "system_mass_ev": system_mass_ev,
        "environment_mass_ev": environment_mass_ev,
        "bilinear_coupling_ev2": bilinear_coupling_ev2,
        "vacuum_energy_density_ev4": vacuum_energy_density_ev4,
        "comoving_volume_ev_minus3": comoving_volume_ev_minus3,
        "system_field_mean_ev": system_field_mean_ev,
        "system_field_velocity_ev2": system_field_velocity_ev2,
        "environment_mean_occupation": environment_mean_occupation,
        "tolerance": tolerance,
    }
    checked: dict[str, float] = {}
    for name, value in finite_inputs.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        checked[name] = converted

    system_mass = checked["system_mass_ev"]
    environment_mass = checked["environment_mass_ev"]
    coupling = checked["bilinear_coupling_ev2"]
    vacuum_density = checked["vacuum_energy_density_ev4"]
    volume = checked["comoving_volume_ev_minus3"]
    field_mean = checked["system_field_mean_ev"]
    field_velocity = checked["system_field_velocity_ev2"]
    occupation = checked["environment_mean_occupation"]
    tolerance = checked["tolerance"]
    beta = float(environment_inverse_temperature_ev_minus1)

    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError("system_mass_ev and environment_mass_ev must be positive")
    if volume <= 0.0:
        raise ValueError("comoving_volume_ev_minus3 must be positive")
    if occupation < 0.0:
        raise ValueError("environment_mean_occupation must be nonnegative")
    if math.isnan(beta) or beta <= 0.0:
        raise ValueError(
            "environment_inverse_temperature_ev_minus1 must be positive"
        )
    if tolerance <= 0.0 or tolerance > 1.0e-8:
        raise ValueError("tolerance must lie in (0, 1e-8]")

    if math.isinf(beta):
        expected_occupation = 0.0
    else:
        beta_omega = beta * environment_mass
        expected_occupation = (
            0.0 if beta_omega > 700.0 else 1.0 / math.expm1(beta_omega)
        )
    thermal_residual = abs(occupation - expected_occupation) / max(
        1.0,
        occupation,
        expected_occupation,
    )
    if thermal_residual > tolerance:
        raise ValueError(
            "environment_mean_occupation is inconsistent with beta*M"
        )

    mass_matrix = np.array(
        ((system_mass**2, coupling), (coupling, environment_mass**2)),
        dtype=float,
    )
    determinant = float(np.linalg.det(mass_matrix))
    determinant_stable = determinant > 0.0
    if not determinant_stable:
        raise ValueError(
            "bilinear mass matrix must satisfy m^2*M^2-kappa^2 > 0"
        )

    mass_squared, rotation = np.linalg.eigh(mass_matrix)
    mass_matrix_stable = bool(
        determinant_stable and float(mass_squared[0]) > 0.0
    )
    if not mass_matrix_stable:
        raise ValueError("bilinear mass matrix must be positive definite")
    for column in range(2):
        pivot = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[pivot, column] < 0.0:
            rotation[:, column] *= -1.0
    gap = float(mass_squared[1] - mass_squared[0])
    relative_gap = gap / float(np.max(np.abs(mass_squared)))
    if relative_gap <= tolerance:
        raise ValueError(
            "normal-mode spectrum is degenerate; only aggregate energy is "
            "basis invariant"
        )
    mode_masses_array = np.sqrt(mass_squared)

    root_volume = math.sqrt(volume)
    canonical_mean = np.array(
        (root_volume * field_mean, 0.0, root_volume * field_velocity, 0.0),
        dtype=float,
    )
    centered_covariance = np.diag(
        (
            1.0 / (2.0 * system_mass),
            (occupation + 0.5) / environment_mass,
            system_mass / 2.0,
            environment_mass * (occupation + 0.5),
        )
    )
    identity = np.eye(2)
    zero = np.zeros((2, 2))
    symplectic_form = np.block([[zero, identity], [-identity, zero]])

    # A local symplectic rescaling makes all quadratures dimensionless before
    # the Robertson matrix is diagonalized.  Thus beta*M and every eigenvalue
    # used as a physicality gate are dimensionless.
    quadrature_rescaling = np.diag(
        (
            math.sqrt(system_mass),
            math.sqrt(environment_mass),
            1.0 / math.sqrt(system_mass),
            1.0 / math.sqrt(environment_mass),
        )
    )
    dimensionless_covariance = (
        quadrature_rescaling @ centered_covariance @ quadrature_rescaling.T
    )
    uncertainty_matrix = (
        dimensionless_covariance + 0.5j * symplectic_form
    )
    uncertainty_minimum = float(np.min(np.linalg.eigvalsh(uncertainty_matrix)))
    symplectic_spectrum = np.sort(
        np.abs(np.linalg.eigvals(1j * symplectic_form @ centered_covariance))
    )[::2]
    expected_symplectic_spectrum = np.array((0.5, occupation + 0.5))
    symplectic_spectrum_residual = float(
        np.max(np.abs(symplectic_spectrum - expected_symplectic_spectrum))
    )
    uncertainty_pass = bool(
        uncertainty_minimum >= -tolerance
        and float(np.min(symplectic_spectrum)) >= 0.5 - tolerance
    )
    covariance_physicality_pass = bool(
        uncertainty_pass and symplectic_spectrum_residual <= tolerance
    )
    if not covariance_physicality_pass:
        raise ValueError("Gaussian covariance violates the uncertainty gate")

    phase_space_transform = np.block(
        [[rotation.T, zero], [zero, rotation.T]]
    )
    symplectic_residual = float(
        np.max(
            np.abs(
                phase_space_transform
                @ symplectic_form
                @ phase_space_transform.T
                - symplectic_form
            )
        )
    )
    diagonalized_mass_matrix = rotation.T @ mass_matrix @ rotation
    diagonalization_scale = float(np.max(np.abs(mass_squared)))
    diagonalization_residual = float(
        np.max(
            np.abs(diagonalized_mass_matrix - np.diag(mass_squared))
        )
        / diagonalization_scale
    )
    canonical_transform_pass = bool(
        symplectic_residual <= tolerance
        and diagonalization_residual <= tolerance
    )
    if not canonical_transform_pass:
        raise ValueError("normal-mode transform failed the canonical gate")

    normal_mean = phase_space_transform @ canonical_mean
    normal_covariance = (
        phase_space_transform
        @ centered_covariance
        @ phase_space_transform.T
    )

    def mode_energies(
        transformed_mean: np.ndarray,
        transformed_covariance: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            tuple(
                0.5
                * (
                    transformed_mean[2 + index] ** 2
                    + transformed_covariance[2 + index, 2 + index]
                    + mass_squared[index]
                    * (
                        transformed_mean[index] ** 2
                        + transformed_covariance[index, index]
                    )
                )
                for index in range(2)
            ),
            dtype=float,
        )

    raw_mode_energies = mode_energies(normal_mean, normal_covariance)
    vacuum_mode_energies = 0.5 * mode_masses_array
    excitation_energies = raw_mode_energies - vacuum_mode_energies
    excitation_scale = float(np.max(vacuum_mode_energies))
    if float(np.min(excitation_energies)) < -tolerance * excitation_scale:
        raise ValueError(
            "physical Gaussian state produced a negative mode excitation"
        )
    excitation_nonnegative = bool(float(np.min(excitation_energies)) >= 0.0)
    if not excitation_nonnegative:
        raise ValueError(
            "mode excitation is numerically negative; no clipping is permitted"
        )

    original_raw_energy = float(
        0.5
        * (
            canonical_mean[2:] @ canonical_mean[2:]
            + np.trace(centered_covariance[2:, 2:])
            + canonical_mean[:2] @ mass_matrix @ canonical_mean[:2]
            + np.trace(mass_matrix @ centered_covariance[:2, :2])
        )
    )
    rotated_raw_energy = float(np.sum(raw_mode_energies))
    raw_rotation_residual = _relative_residual(
        original_raw_energy, rotated_raw_energy
    )

    sign_flip = np.diag((-1.0, 1.0))
    flipped_rotation = rotation @ sign_flip
    flipped_transform = np.block(
        [[flipped_rotation.T, zero], [zero, flipped_rotation.T]]
    )
    flipped_energies = mode_energies(
        flipped_transform @ canonical_mean,
        flipped_transform @ centered_covariance @ flipped_transform.T,
    )
    sign_flip_residual = max(
        _relative_residual(float(left), float(right))
        for left, right in zip(raw_mode_energies, flipped_energies)
    )

    coherent_preparation_energy = float(
        0.5
        * (
            canonical_mean[2] ** 2
            + system_mass**2 * canonical_mean[0] ** 2
        )
    )
    thermal_preparation_energy = environment_mass * occupation
    bare_vacuum_energy = 0.5 * (system_mass + environment_mass)
    interacting_vacuum_energy = float(np.sum(vacuum_mode_energies))
    mismatch_energy = bare_vacuum_energy - interacting_vacuum_energy
    if mismatch_energy < -tolerance * bare_vacuum_energy:
        raise ValueError("vacuum mismatch energy unexpectedly became negative")
    total_excitation_energy = float(np.sum(excitation_energies))
    ledger_right = (
        coherent_preparation_energy
        + thermal_preparation_energy
        + mismatch_energy
    )
    ledger_residual = _relative_residual(total_excitation_energy, ledger_right)
    density_constants = excitation_energies / volume

    # Symbolic mass-dimension propagation in natural units.  These values are
    # calculated from the declared input dimensions rather than setting a
    # receipt flag directly.
    input_dimensions = {
        "mass": 1.0,
        "coupling": 2.0,
        "vacuum_density": 4.0,
        "comoving_volume": -3.0,
        "field": 1.0,
        "field_velocity": 2.0,
        "inverse_temperature": -1.0,
        "scale_factor": 0.0,
        "comoving_wavenumber": 1.0,
        "hubble": 1.0,
    }
    derived_dimensions = {
        "beta_times_environment_mass": (
            input_dimensions["inverse_temperature"]
            + input_dimensions["mass"]
        ),
        "canonical_q": (
            0.5 * input_dimensions["comoving_volume"]
            + input_dimensions["field"]
        ),
        "canonical_p": (
            0.5 * input_dimensions["comoving_volume"]
            + input_dimensions["field_velocity"]
        ),
    }
    derived_dimensions.update(
        {
            "p_squared_energy": 2.0 * derived_dimensions["canonical_p"],
            "mass_squared_q_squared_energy": (
                2.0 * input_dimensions["mass"]
                + 2.0 * derived_dimensions["canonical_q"]
            ),
            "mode_density_constant": (
                2.0 * derived_dimensions["canonical_p"]
                - input_dimensions["comoving_volume"]
            ),
            "vacuum_cell_energy": (
                input_dimensions["vacuum_density"]
                + input_dimensions["comoving_volume"]
            ),
            "hubble_over_mode_mass": (
                input_dimensions["hubble"] - input_dimensions["mass"]
            ),
            "physical_wavenumber_over_mode_mass": (
                input_dimensions["comoving_wavenumber"]
                - input_dimensions["scale_factor"]
                - input_dimensions["mass"]
            ),
        }
    )
    expected_dimensions = {
        "beta_times_environment_mass": 0.0,
        "canonical_q": -0.5,
        "canonical_p": 0.5,
        "p_squared_energy": 1.0,
        "mass_squared_q_squared_energy": 1.0,
        "mode_density_constant": 4.0,
        "vacuum_cell_energy": 1.0,
        "hubble_over_mode_mass": 0.0,
        "physical_wavenumber_over_mode_mass": 0.0,
    }
    dimensions_pass = bool(derived_dimensions == expected_dimensions)
    dimensionless_core_dimensions = {
        name: derived_dimensions[name]
        for name in (
            "beta_times_environment_mass",
            "hubble_over_mode_mass",
            "physical_wavenumber_over_mode_mass",
        )
    }
    invariant_pass = bool(
        raw_rotation_residual <= tolerance
        and ledger_residual <= tolerance
        and sign_flip_residual <= tolerance
    )
    if not invariant_pass:
        raise ValueError("state energy failed a rotation or ledger invariant")

    return FiniteProductGaussianStateDensityAudit(
        action_parameter_manifest=(
            system_mass,
            environment_mass,
            coupling,
            vacuum_density,
        ),
        state_boundary_manifest=(
            volume,
            field_mean,
            field_velocity,
            occupation,
            beta,
        ),
        environment_thermal_marginal_manifest=(
            environment_mass,
            volume,
            occupation,
            beta,
        ),
        covariance_ordering=("q_phi", "q_chi", "p_phi", "p_chi"),
        canonical_mean=tuple(float(value) for value in canonical_mean),
        centered_symmetrized_covariance=_real_matrix_tuple(centered_covariance),
        symplectic_eigenvalues=tuple(
            float(value) for value in symplectic_spectrum
        ),
        expected_symplectic_eigenvalues=tuple(
            float(value) for value in expected_symplectic_spectrum
        ),
        dimensionless_uncertainty_minimum_eigenvalue=uncertainty_minimum,
        normal_mode_rotation=_real_matrix_tuple(rotation),
        phase_space_symplectic_transform=_real_matrix_tuple(
            phase_space_transform
        ),
        symplectic_residual=symplectic_residual,
        normal_mode_diagonalization_relative_residual=diagonalization_residual,
        normal_mode_mass_squared=tuple(float(value) for value in mass_squared),
        normal_mode_masses=tuple(float(value) for value in mode_masses_array),
        relative_spectral_gap=relative_gap,
        normal_mode_mean=tuple(float(value) for value in normal_mean),
        normal_mode_centered_covariance=_real_matrix_tuple(normal_covariance),
        normal_mode_position_cross_covariance=float(normal_covariance[0, 1]),
        normal_mode_momentum_cross_covariance=float(normal_covariance[2, 3]),
        finite_mode_raw_energies_ev=tuple(
            float(value) for value in raw_mode_energies
        ),
        finite_mode_vacuum_energies_ev=tuple(
            float(value) for value in vacuum_mode_energies
        ),
        finite_mode_vacuum_subtracted_energies_ev=tuple(
            float(value) for value in excitation_energies
        ),
        finite_mode_density_constants_ev4=tuple(
            float(value) for value in density_constants
        ),
        uncoupled_coherent_preparation_energy_ev=coherent_preparation_energy,
        uncoupled_thermal_preparation_energy_ev=thermal_preparation_energy,
        bare_product_vacuum_energy_ev=bare_vacuum_energy,
        interacting_vacuum_energy_ev=interacting_vacuum_energy,
        vacuum_mismatch_quench_energy_ev=mismatch_energy,
        finite_mode_vacuum_subtracted_total_energy_ev=total_excitation_energy,
        finite_mode_vacuum_subtracted_total_density_ev4=(
            total_excitation_energy / volume
        ),
        vacuum_cell_energy_ev=vacuum_density * volume,
        thermal_occupation_relative_residual=thermal_residual,
        raw_energy_rotation_relative_residual=raw_rotation_residual,
        excitation_energy_ledger_relative_residual=ledger_residual,
        mode_sign_flip_energy_relative_residual=sign_flip_residual,
        uncertainty_principle_pass=uncertainty_pass,
        covariance_physicality_pass=covariance_physicality_pass,
        canonical_transform_pass=canonical_transform_pass,
        mass_matrix_stable=mass_matrix_stable,
        nondegenerate_mode_allocation_pass=True,
        finite_mode_excitation_nonnegative=excitation_nonnegative,
        mass_dimension_manifest=tuple(derived_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        status="PASS_CONDITIONAL_FINITE_GAUSSIAN_STATE_DENSITY_MAP",
    )


@dataclass(frozen=True)
class GaussianNormalModePerturbationAudit:
    """Conditional WKB perturbation receipt for the retained Gaussian action.

    This audit uses the same supplied parameter manifest
    ``(m, M, kappa, V0)`` as the finite Gaussian CTP witness, but switches
    representation: both canonical fields are retained and diagonalized.
    The integrated-out influence Gram is never inserted as a gravity source.
    """

    action_parameter_manifest: tuple[float, float, float, float]
    normal_mode_mass_squared: tuple[float, float]
    normal_mode_masses: tuple[float, float]
    mass_matrix_determinant_ev4: float
    scale_factor: float
    hubble_ev: float
    comoving_wavenumber_ev: float
    reduced_planck_mass_ev: float
    comoving_mode_density_constants_ev4: tuple[float, float]
    mode_densities_ev4: tuple[float, float]
    background_density_ev4: float
    background_pressure_ev4: float
    vacuum_equation_of_state: float
    vacuum_density_perturbation_ev4: float
    linear_anisotropic_stress: float
    microscopic_characteristic_speed_squared: tuple[float, float]
    effective_sound_speed_squared: tuple[float, float]
    wkb_hubble_ratios: tuple[float, float]
    nonrelativistic_momentum_ratios: tuple[float, float]
    subhorizon_ratio: float
    pressure_frequency_squared_ev2: tuple[float, float]
    four_pi_g_density_sources_ev2: tuple[float, float]
    coupled_density_contrast_matrix_ev2: tuple[tuple[float, float], ...]
    jeans_comoving_wavenumbers_ev: tuple[float, float]
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    mass_matrix_stable: bool
    wkb_domain_pass: bool
    nonrelativistic_domain_pass: bool
    subhorizon_domain_pass: bool
    positive_vacuum_pass: bool
    background_dm_de_limit: bool
    perturbation_discriminant_derived: bool
    status: str
    failed_gates: tuple[str, ...]
    representation: str = "RETAINED_TWO_FIELD_ACTION_ONLY"
    closure: str = "WKB_NONRELATIVISTIC_GR_SUBHORIZON"
    same_action_metric_variation_declared: bool = True
    einstein_gravity_supplied: bool = True
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    ctp_to_cosmological_state_map_derived: bool = False
    initial_conditions_derived: bool = False
    absolute_abundance_derived: bool = False
    growth_history_derived: bool = False
    lensing_likelihood_derived: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


@dataclass(frozen=True)
class ProductGaussianWKBPerturbationAudit:
    """Composition of one retained Gaussian state with the WKB gate."""

    state_density_audit: FiniteProductGaussianStateDensityAudit
    perturbation_audit: GaussianNormalModePerturbationAudit
    action_parameter_manifest_match: bool
    derived_density_constants_match: bool
    same_state_finite_mode_energy_map_derived: bool
    perturbation_discriminant_derived: bool
    status: str
    representation: str = "RETAINED_TWO_FIELD_STATE_AND_WKB_ONLY"
    ctp_to_cosmological_state_map_derived: bool = False
    cosmological_initial_state_derived: bool = False
    absolute_abundance_predicted: bool = False
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def audit_gaussian_normal_mode_perturbations(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    scale_factor: float,
    hubble_ev: float,
    comoving_wavenumber_ev: float,
    comoving_mode_density_constants_ev4: tuple[float, float],
    reduced_planck_mass_ev: float = REDUCED_MPL_EV,
    validity_limit: float = 0.1,
) -> GaussianNormalModePerturbationAudit:
    """Compute the two-mode WKB sound and Jeans discriminants.

    The retained action is

    ``L = -1/2 (d phi)^2 -m^2 phi^2/2``
    ``    -1/2 (d chi)^2 -M^2 chi^2/2-kappa phi chi-V0``.

    Its exact normal masses are diagonalized before coarse graining.  For each
    rapidly oscillating nonrelativistic mode,

    ``c_eff^2 = k^2/(4*a^2*m_mode^2)`` and
    ``omega_pressure^2 = k^4/(4*a^4*m_mode^2)``.

    With supplied Einstein gravity, the subhorizon density-contrast system is
    represented by

    ``delta_i'' + 2H delta_i' + sum_j A_ij delta_j = 0`` with
    ``A_ij = omega_pressure_i^2 delta_ij - 4*pi*G*rho_j``.

    Initial amplitudes are deliberately not selected or evolved, so this is a
    scale-dependent perturbation discriminator rather than a growth or lensing
    prediction.
    """

    values = {
        "system_mass_ev": system_mass_ev,
        "environment_mass_ev": environment_mass_ev,
        "bilinear_coupling_ev2": bilinear_coupling_ev2,
        "vacuum_energy_density_ev4": vacuum_energy_density_ev4,
        "scale_factor": scale_factor,
        "hubble_ev": hubble_ev,
        "comoving_wavenumber_ev": comoving_wavenumber_ev,
        "reduced_planck_mass_ev": reduced_planck_mass_ev,
        "validity_limit": validity_limit,
    }
    checked: dict[str, float] = {}
    for name, value in values.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        checked[name] = converted

    system_mass = checked["system_mass_ev"]
    environment_mass = checked["environment_mass_ev"]
    coupling = checked["bilinear_coupling_ev2"]
    vacuum_density = checked["vacuum_energy_density_ev4"]
    a = checked["scale_factor"]
    hubble = checked["hubble_ev"]
    wavenumber = checked["comoving_wavenumber_ev"]
    planck_mass = checked["reduced_planck_mass_ev"]
    limit = checked["validity_limit"]
    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError("system_mass_ev and environment_mass_ev must be positive")
    if a <= 0.0 or hubble < 0.0 or wavenumber <= 0.0:
        raise ValueError(
            "scale_factor and comoving_wavenumber_ev must be positive and "
            "hubble_ev must be nonnegative"
        )
    if planck_mass <= 0.0:
        raise ValueError("reduced_planck_mass_ev must be positive")
    if limit <= 0.0 or limit > 0.25:
        raise ValueError("validity_limit must lie in (0, 0.25]")

    densities = tuple(float(value) for value in comoving_mode_density_constants_ev4)
    if len(densities) != 2 or any(not math.isfinite(value) for value in densities):
        raise ValueError(
            "comoving_mode_density_constants_ev4 must contain two finite values"
        )
    if any(value < 0.0 for value in densities) or sum(densities) <= 0.0:
        raise ValueError(
            "comoving_mode_density_constants_ev4 must be nonnegative with "
            "positive total"
        )

    determinant = system_mass**2 * environment_mass**2 - coupling**2
    if determinant <= 0.0:
        raise ValueError(
            "bilinear mass matrix must satisfy m^2*M^2-kappa^2 > 0"
        )
    discriminant = math.sqrt(
        (environment_mass**2 - system_mass**2) ** 2 + 4.0 * coupling**2
    )
    mass_minus_squared = 0.5 * (
        system_mass**2 + environment_mass**2 - discriminant
    )
    mass_plus_squared = 0.5 * (
        system_mass**2 + environment_mass**2 + discriminant
    )
    mode_mass_squared = (mass_minus_squared, mass_plus_squared)
    mode_masses = tuple(math.sqrt(value) for value in mode_mass_squared)

    physical_wavenumber = wavenumber / a
    wkb_ratios = tuple(hubble / mass for mass in mode_masses)
    momentum_ratios = tuple(
        physical_wavenumber / mass for mass in mode_masses
    )
    subhorizon_ratio = a * hubble / wavenumber
    sound_speeds = tuple(
        wavenumber**2 / (4.0 * a**2 * mass_squared)
        for mass_squared in mode_mass_squared
    )
    pressure_frequencies = tuple(
        wavenumber**4 / (4.0 * a**4 * mass_squared)
        for mass_squared in mode_mass_squared
    )

    mode_densities = tuple(value / a**3 for value in densities)
    density_sources = tuple(
        value / (2.0 * planck_mass**2) for value in mode_densities
    )
    total_gravity_source = sum(density_sources)
    contrast_matrix = (
        (
            pressure_frequencies[0] - density_sources[0],
            -density_sources[1],
        ),
        (
            -density_sources[0],
            pressure_frequencies[1] - density_sources[1],
        ),
    )
    jeans_wavenumbers = tuple(
        (
            4.0 * mass_squared * a**4 * total_gravity_source
        ) ** 0.25
        for mass_squared in mode_mass_squared
    )

    wkb_pass = max(wkb_ratios) <= limit
    nonrelativistic_pass = max(momentum_ratios) <= limit
    subhorizon_pass = subhorizon_ratio <= limit
    positive_vacuum = vacuum_density > 0.0
    gates = {
        "wkb": wkb_pass,
        "nonrelativistic": nonrelativistic_pass,
        "subhorizon": subhorizon_pass,
        "positive_vacuum": positive_vacuum,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    input_dimensions = {
        "mode_mass": 1.0,
        "mode_mass_squared": 2.0,
        "scale_factor": 0.0,
        "hubble": 1.0,
        "comoving_wavenumber": 1.0,
        "mode_density": 4.0,
        "reduced_planck_mass": 1.0,
    }
    derived_dimensions = {
        "wkb_hubble_ratio": (
            input_dimensions["hubble"] - input_dimensions["mode_mass"]
        ),
        "nonrelativistic_momentum_ratio": (
            input_dimensions["comoving_wavenumber"]
            - input_dimensions["scale_factor"]
            - input_dimensions["mode_mass"]
        ),
        "subhorizon_ratio": (
            input_dimensions["scale_factor"]
            + input_dimensions["hubble"]
            - input_dimensions["comoving_wavenumber"]
        ),
        "effective_sound_speed_squared": (
            2.0 * input_dimensions["comoving_wavenumber"]
            - 2.0 * input_dimensions["scale_factor"]
            - input_dimensions["mode_mass_squared"]
        ),
        "pressure_frequency_squared": (
            4.0 * input_dimensions["comoving_wavenumber"]
            - 4.0 * input_dimensions["scale_factor"]
            - input_dimensions["mode_mass_squared"]
        ),
        "four_pi_g_density_source": (
            input_dimensions["mode_density"]
            - 2.0 * input_dimensions["reduced_planck_mass"]
        ),
    }
    derived_dimensions["jeans_wavenumber_fourth_power"] = (
        input_dimensions["mode_mass_squared"]
        + 4.0 * input_dimensions["scale_factor"]
        + derived_dimensions["four_pi_g_density_source"]
    )
    derived_dimensions["jeans_wavenumber"] = (
        derived_dimensions["jeans_wavenumber_fourth_power"] / 4.0
    )
    expected_dimensions = {
        "wkb_hubble_ratio": 0.0,
        "nonrelativistic_momentum_ratio": 0.0,
        "subhorizon_ratio": 0.0,
        "effective_sound_speed_squared": 0.0,
        "pressure_frequency_squared": 2.0,
        "four_pi_g_density_source": 2.0,
        "jeans_wavenumber_fourth_power": 4.0,
        "jeans_wavenumber": 1.0,
    }
    dimensions_pass = bool(derived_dimensions == expected_dimensions)
    dimensionless_core_dimensions = {
        name: derived_dimensions[name]
        for name in (
            "wkb_hubble_ratio",
            "nonrelativistic_momentum_ratio",
            "subhorizon_ratio",
            "effective_sound_speed_squared",
        )
    }
    approximation_pass = not failed and dimensions_pass

    return GaussianNormalModePerturbationAudit(
        action_parameter_manifest=(
            system_mass,
            environment_mass,
            coupling,
            vacuum_density,
        ),
        normal_mode_mass_squared=mode_mass_squared,
        normal_mode_masses=mode_masses,
        mass_matrix_determinant_ev4=determinant,
        scale_factor=a,
        hubble_ev=hubble,
        comoving_wavenumber_ev=wavenumber,
        reduced_planck_mass_ev=planck_mass,
        comoving_mode_density_constants_ev4=densities,
        mode_densities_ev4=mode_densities,
        background_density_ev4=vacuum_density + sum(mode_densities),
        background_pressure_ev4=-vacuum_density,
        vacuum_equation_of_state=-1.0,
        vacuum_density_perturbation_ev4=0.0,
        linear_anisotropic_stress=0.0,
        microscopic_characteristic_speed_squared=(1.0, 1.0),
        effective_sound_speed_squared=sound_speeds,
        wkb_hubble_ratios=wkb_ratios,
        nonrelativistic_momentum_ratios=momentum_ratios,
        subhorizon_ratio=subhorizon_ratio,
        pressure_frequency_squared_ev2=pressure_frequencies,
        four_pi_g_density_sources_ev2=density_sources,
        coupled_density_contrast_matrix_ev2=contrast_matrix,
        jeans_comoving_wavenumbers_ev=jeans_wavenumbers,
        mass_dimension_manifest=tuple(derived_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        mass_matrix_stable=True,
        wkb_domain_pass=wkb_pass,
        nonrelativistic_domain_pass=nonrelativistic_pass,
        subhorizon_domain_pass=subhorizon_pass,
        positive_vacuum_pass=positive_vacuum,
        background_dm_de_limit=wkb_pass and positive_vacuum,
        perturbation_discriminant_derived=approximation_pass,
        status=(
            "PASS_CONDITIONAL_GAUSSIAN_WKB_PERTURBATIONS"
            if approximation_pass
            else "FAIL_CONDITIONAL_APPROXIMATION_GATE"
        ),
        failed_gates=failed,
    )


def audit_product_gaussian_state_wkb_perturbations(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    comoving_volume_ev_minus3: float,
    system_field_mean_ev: float,
    system_field_velocity_ev2: float,
    environment_mean_occupation: float,
    environment_inverse_temperature_ev_minus1: float,
    scale_factor: float,
    hubble_ev: float,
    comoving_wavenumber_ev: float,
    reduced_planck_mass_ev: float = REDUCED_MPL_EV,
    validity_limit: float = 0.1,
    tolerance: float = 2.0e-11,
) -> ProductGaussianWKBPerturbationAudit:
    """Feed exact finite-state energies into the conditional WKB receipt.

    The composition is deliberately one-way and representation-exclusive:
    the retained-state energy constants are used as the WKB matter input, but
    no CTP influence Gram or integrated-out environment stress is added.
    """

    state_audit = audit_finite_product_gaussian_state_densities(
        system_mass_ev=system_mass_ev,
        environment_mass_ev=environment_mass_ev,
        bilinear_coupling_ev2=bilinear_coupling_ev2,
        vacuum_energy_density_ev4=vacuum_energy_density_ev4,
        comoving_volume_ev_minus3=comoving_volume_ev_minus3,
        system_field_mean_ev=system_field_mean_ev,
        system_field_velocity_ev2=system_field_velocity_ev2,
        environment_mean_occupation=environment_mean_occupation,
        environment_inverse_temperature_ev_minus1=(
            environment_inverse_temperature_ev_minus1
        ),
        tolerance=tolerance,
    )
    perturbation_audit = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=system_mass_ev,
        environment_mass_ev=environment_mass_ev,
        bilinear_coupling_ev2=bilinear_coupling_ev2,
        vacuum_energy_density_ev4=vacuum_energy_density_ev4,
        scale_factor=scale_factor,
        hubble_ev=hubble_ev,
        comoving_wavenumber_ev=comoving_wavenumber_ev,
        comoving_mode_density_constants_ev4=(
            state_audit.finite_mode_density_constants_ev4
        ),
        reduced_planck_mass_ev=reduced_planck_mass_ev,
        validity_limit=validity_limit,
    )
    manifest_match = bool(
        state_audit.action_parameter_manifest
        == perturbation_audit.action_parameter_manifest
    )
    density_match = bool(
        np.allclose(
            state_audit.finite_mode_density_constants_ev4,
            perturbation_audit.comoving_mode_density_constants_ev4,
            rtol=tolerance,
            atol=0.0,
        )
    )
    if not manifest_match or not density_match:
        raise ValueError("state-to-WKB composition failed its manifest gate")
    perturbation_derived = bool(
        perturbation_audit.perturbation_discriminant_derived
    )
    return ProductGaussianWKBPerturbationAudit(
        state_density_audit=state_audit,
        perturbation_audit=perturbation_audit,
        action_parameter_manifest_match=manifest_match,
        derived_density_constants_match=density_match,
        same_state_finite_mode_energy_map_derived=(
            state_audit.same_state_finite_mode_energy_map_derived
        ),
        perturbation_discriminant_derived=perturbation_derived,
        status=(
            "PASS_CONDITIONAL_SAME_STATE_GAUSSIAN_WKB_BRIDGE"
            if perturbation_derived
            else "FAIL_CONDITIONAL_SAME_STATE_WKB_DOMAIN"
        ),
    )


def _nodes(solution: BackgroundSolution) -> tuple[PerturbationNode, ...]:
    config = solution.config
    rho_inf_ev4 = solution.amplitude * RHO_CRIT0_EV4
    lambda3_ev = 2.0 * (config.kappa * rho_inf_ev4) ** 0.25
    result: list[PerturbationNode] = []
    for node in solution.nodes:
        data = _densities(node.n, node.tau, node.u, config, solution.amplitude)
        rho_b, rho_r, _, rho_k, p_k, _ = data
        _, u_prime = _rhs(
            node.n, (node.tau, node.u), config, solution.amplitude
        )
        delta = node.u / config.kappa
        delta_prime = u_prime / config.kappa
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        kinetic_prime = 3.0 * delta_prime / (2.0 + 3.0 * delta)
        friction = 3.0 + h_prime_over_h + kinetic_prime
        tachyon_ratio = (
            config.gamma**2
            * config.x_star
            * math.exp(-config.gamma * node.tau)
            / (config.kappa * (2.0 + 3.0 * delta) * node.e2)
        )
        pump = (
            3.0
            + delta_prime / (1.0 + delta)
            + 3.0 * delta_prime / (2.0 + 3.0 * delta)
            - 2.0 * h_prime_over_h
        )
        energy_cutoff = lambda3_ev * node.cs2 ** (7.0 / 8.0)
        wavenumber_cutoff = lambda3_ev * node.cs2 ** (3.0 / 8.0)
        result.append(
            PerturbationNode(
                n=node.n,
                e2=node.e2,
                friction=friction,
                tachyon_ratio=tachyon_ratio,
                cs2=node.cs2,
                q_s_over_mpl2=node.q_s_over_mpl2,
                pump_slope=pump,
                zeta_decay_slope=pump + h_prime_over_h,
                energy_cutoff_ev=energy_cutoff,
                wavenumber_cutoff_ev=wavenumber_cutoff,
            )
        )
    return tuple(result)


def _fixed_coordinate_growth(nodes: tuple[PerturbationNode, ...]) -> float:
    y = 0.0
    velocity = 0.0
    for left, right in zip(nodes, nodes[1:]):
        step = right.n - left.n
        friction = 0.5 * (left.friction + right.friction)
        ratio = 0.5 * (left.tachyon_ratio + right.tachyon_ratio)

        def derivative(yy: float, vv: float) -> tuple[float, float]:
            return vv, -friction * vv + ratio * (1.0 + yy)

        k1 = derivative(y, velocity)
        k2 = derivative(y + step * k1[0] / 2.0, velocity + step * k1[1] / 2.0)
        k3 = derivative(y + step * k2[0] / 2.0, velocity + step * k2[1] / 2.0)
        k4 = derivative(y + step * k3[0], velocity + step * k3[1])
        y += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    return y


def evaluate_single_clock_gate(
    solution: BackgroundSolution | None = None,
) -> SingleClockGate:
    selected = solution or solve_background(KineticClockConfig())
    nodes = _nodes(selected)
    min_friction = min(node.friction for node in nodes)
    max_ratio = max(node.tachyon_ratio for node in nodes)
    interval = nodes[-1].n - nodes[0].n
    positive_root = 2.0 * max_ratio / (
        math.sqrt(min_friction**2 + 4.0 * max_ratio) + min_friction
    )
    bound = positive_root * interval
    min_energy_over_h = min(
        node.energy_cutoff_ev / (H0_EV * math.sqrt(node.e2)) for node in nodes
    )
    min_wavenumber_over_k = min(
        node.wavenumber_cutoff_ev / (MPC_INV_EV * math.exp(-node.n))
        for node in nodes
    )
    gate_values = {
        "positive_friction": min_friction > 0.0,
        "sub_hubble_tachyon": max_ratio < 1.0,
        "sub_order_one_growth_bound": bound < 1.0,
        "positive_pump": min(node.pump_slope for node in nodes) > 0.0,
        "decaying_zeta_integrand": min(node.zeta_decay_slope for node in nodes) > 0.0,
        "energy_cutoff_above_h": min_energy_over_h > 1.0,
        "momentum_cutoff_above_1mpc": min_wavenumber_over_k > 1.0,
    }
    failed = tuple(name for name, passed in gate_values.items() if not passed)
    return SingleClockGate(
        gamma=selected.config.gamma,
        min_friction=min_friction,
        max_tachyon_ratio=max_ratio,
        fixed_coordinate_growth_minus_one=_fixed_coordinate_growth(nodes),
        max_log_growth_bound=bound,
        min_pump_slope=min(node.pump_slope for node in nodes),
        min_zeta_decay_slope=min(node.zeta_decay_slope for node in nodes),
        min_energy_cutoff_over_h=min_energy_over_h,
        min_wavenumber_cutoff_over_k_1mpc=min_wavenumber_over_k,
        status=("PASS_SINGLE_CLOCK_ONLY" if not failed else "FAIL_SINGLE_CLOCK_GATE"),
        failed_gates=failed,
    )


def quasi_static_growth_diagnostic(
    solution: BackgroundSolution | None = None,
    *,
    redshift: float = 0.07,
    observed_fsigma8: float = 0.4497,
    observed_sigma: float = 0.0548,
    sigma8_0: float = 0.811,
) -> QuasiStaticGrowthDiagnostic:
    """Solve a declared subhorizon closure and compare one compact datum.

    The kinetic inventory and baryons source the Poisson term, the saturated
    readout is smooth, GR is retained, and radiation affects only H(a).  This
    is the strongest low-data diagnostic available before deriving the full
    multi-component perturbation system.  ``sigma8_0`` remains external.
    """

    if redshift < 0.0 or observed_sigma <= 0.0 or sigma8_0 <= 0.0:
        raise ValueError("growth diagnostic inputs are outside their domain")
    selected = solution or solve_background(KineticClockConfig())
    target_n = -math.log1p(redshift)
    if target_n < selected.nodes[0].n:
        raise ValueError("growth redshift is outside the solved window")

    # Growing-mode matter-era seed at a>=0.01.  This deliberately avoids
    # pretending that the closure supplies an adiabatic radiation-era transfer.
    growth_nodes = tuple(node for node in selected.nodes if node.n >= math.log(0.01))
    if len(growth_nodes) < 10:
        raise ValueError("background grid is too sparse for the growth closure")
    d_value = math.exp(growth_nodes[0].n)
    velocity = d_value
    history: list[tuple[float, float, float]] = [
        (growth_nodes[0].n, d_value, velocity)
    ]

    def coefficients(n: float) -> tuple[float, float]:
        node = selected.at_n(n)
        rho_b, rho_r, _, rho_k, p_k, _ = _densities(
            n, node.tau, node.u, selected.config, selected.amplitude
        )
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        omega_cluster = (rho_b + rho_k) / node.e2
        return 2.0 + h_prime_over_h, 1.5 * omega_cluster

    def derivative(n: float, d: float, v: float) -> tuple[float, float]:
        drag, source = coefficients(n)
        return v, -drag * v + source * d

    for left, right in zip(growth_nodes, growth_nodes[1:]):
        step = right.n - left.n
        n = left.n
        k1 = derivative(n, d_value, velocity)
        k2 = derivative(
            n + step / 2.0,
            d_value + step * k1[0] / 2.0,
            velocity + step * k1[1] / 2.0,
        )
        k3 = derivative(
            n + step / 2.0,
            d_value + step * k2[0] / 2.0,
            velocity + step * k2[1] / 2.0,
        )
        k4 = derivative(
            n + step,
            d_value + step * k3[0],
            velocity + step * k3[1],
        )
        d_value += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        history.append((right.n, d_value, velocity))

    normalization = history[-1][1]
    for left, right in zip(history, history[1:]):
        if left[0] <= target_n <= right[0]:
            weight = (target_n - left[0]) / (right[0] - left[0])
            d_target = left[1] + weight * (right[1] - left[1])
            v_target = left[2] + weight * (right[2] - left[2])
            break
    else:
        d_target, v_target = history[-1][1], history[-1][2]
    # f*sigma8 = (D'/D) * sigma8_0*(D/D0) = sigma8_0*D'/D0.
    prediction = sigma8_0 * v_target / normalization
    return QuasiStaticGrowthDiagnostic(
        redshift=redshift,
        predicted_fsigma8=prediction,
        observed_fsigma8=observed_fsigma8,
        observed_sigma=observed_sigma,
        pull=(prediction - observed_fsigma8) / observed_sigma,
        sigma8_0=sigma8_0,
    )


def scan_kappa_sensitivity(
    kappa_values: tuple[float, ...] = (1.0e10, 3.0e11, 1.0e12, 1.0e14, 1.0e17, 1.0e20),
    *,
    gamma: float = 10.0,
    steps: int = 1200,
) -> tuple[KappaSensitivityRow, ...]:
    """Expose which role the otherwise external stiffness scale plays."""

    if not kappa_values or any(value <= 0.0 for value in kappa_values):
        raise ValueError("kappa scan values must be non-empty and positive")
    rows: list[KappaSensitivityRow] = []
    for kappa in kappa_values:
        solution = solve_background(
            KineticClockConfig(gamma=gamma, kappa=kappa, steps=steps)
        )
        gate = evaluate_single_clock_gate(solution)
        rows.append(
            KappaSensitivityRow(
                kappa=kappa,
                min_cs2=solution.min_cs2,
                min_friction=gate.min_friction,
                max_log_growth_bound=gate.max_log_growth_bound,
                min_energy_cutoff_over_h=gate.min_energy_cutoff_over_h,
                status=gate.status,
                failed_gates=gate.failed_gates,
            )
        )
    return tuple(rows)


def main() -> int:
    gate = evaluate_single_clock_gate()
    for name, value in gate.__dict__.items():
        print(name, value)
    growth = quasi_static_growth_diagnostic()
    for name, value in growth.__dict__.items():
        print(f"growth_{name}", value)
    for row in scan_kappa_sensitivity():
        print(
            "kappa_scan",
            row.kappa,
            row.min_cs2,
            row.min_friction,
            row.max_log_growth_bound,
            row.min_energy_cutoff_over_h,
            row.status,
            ",".join(row.failed_gates) or "none",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
