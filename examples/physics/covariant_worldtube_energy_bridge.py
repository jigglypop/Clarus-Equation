"""Conditional worldtube matching for one branch energy receipt.

For a supplied branch-resolved symmetric stress tensor and exchange current,

    nabla_mu T_S^mu{}_nu = Q_nu,
    j_S^mu = -T_S^mu{}_nu xi^nu,

the exact energy-current identity is

    Delta E_S + Phi_S^out
      = -integral_V (xi^nu Q_nu
                      + T_S^{mu nu} nabla_(mu xi_nu)) dV.

The opposite sector receives ``-Q_nu``.  The exchange terms therefore cancel
from the total ledger, while a non-Killing time flow and lateral boundary flux
remain explicit.  Only a Killing, zero-lateral-flux witness can reduce the
identity to ``delta e_beta = -integral xi.Q dV``.

All geometry, branch stresses, currents, surface energies, and quadrature
weights in this module are supplied inputs.  The calculation audits their
matching; it does not derive a local stress tensor from the finite quantum
domino, select a physical pointer, take a continuum limit, or produce a GR
source.  ``construct_flat_receipt_current_counterexample`` records the exact
failure of the inverse problem: one scalar receipt leaves momentum components
of the local current arbitrary even on the same flat worldtube.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from examples.physics.causal_quantum_domino import BatteryOutcomeReceipt
from examples.physics.finite_ctp_diagonal_source_obstruction import (
    QuantumKickConservationAudit,
)


SPACETIME_DIMENSION = 4
DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class ClosedBranchWorldtubeReceipt:
    """Numerical audit of one supplied two-sector worldtube ledger."""

    source_receipt_id: str
    quadrature_cell_count: int
    branch_probability: float
    conditional_trace: float
    receipt_energy: float
    system_initial_surface_energy: float
    system_final_surface_energy: float
    battery_initial_surface_energy: float
    battery_final_surface_energy: float
    system_surface_energy_change: float
    battery_surface_energy_change: float
    system_lateral_outward_energy_flux: float
    battery_lateral_outward_energy_flux: float
    system_source_injection_energy: float
    battery_source_injection_energy: float
    system_deformation_energy: float
    battery_deformation_energy: float
    system_predicted_surface_energy_change: float
    battery_predicted_surface_energy_change: float
    dimensionless_system_balance_residual: float
    dimensionless_battery_balance_residual: float
    dimensionless_total_balance_residual: float
    dimensionless_exchange_cancellation_residual: float
    maximum_dimensionless_opposite_current_residual: float
    dimensionless_system_receipt_surface_residual: float
    dimensionless_battery_receipt_surface_residual: float
    dimensionless_receipt_worldtube_residual: float
    maximum_dimensionless_killing_equation_residual: float
    current_mass_dimension: int
    stress_mass_dimension: int
    four_volume_mass_dimension: int
    energy_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    positive_probability_outcome: bool
    conditional_branch_normalized: bool
    supplied_time_flow_future_timelike: bool
    supplied_killing_flow_on_samples: bool
    supplied_zero_lateral_flux: bool
    opposite_exchange_current_cancels: bool
    supplied_sector_balances_hold: bool
    supplied_total_energy_balance_holds: bool
    supplied_total_energy_and_exchange_closure_holds: bool
    exclusive_branch_receipt_matches_both_sectors: bool
    killing_zero_flux_receipt_matching_holds: bool
    source_receipt_id_is_provenance_label_only: bool = True
    typed_e9d_outcome_consistency_verified: bool = False
    source_receipt_provenance_authenticated: bool = False
    e9d_receipt_to_worldtube_derived: bool = False
    quadrature_worldtube_supplied: bool = True
    opposite_sector_current_supplied: bool = True
    branch_stress_from_domino_derived: bool = False
    battery_to_covariant_action_derived: bool = False
    continuum_worldtube_derived: bool = False
    physical_pointer_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class FlatReceiptCurrentCounterexample:
    """Exact flat-worldtube family with one receipt and distinct currents."""

    receipt_energy: float
    duration: float
    spatial_volume: float
    four_volume: float
    energy_source_density: float
    profile_a_current_covector: tuple[float, float, float, float]
    profile_b_current_covector: tuple[float, float, float, float]
    profile_a_battery_current_covector: tuple[float, float, float, float]
    profile_b_battery_current_covector: tuple[float, float, float, float]
    profile_a_computed_system_divergence_covector: tuple[float, float, float, float]
    profile_b_computed_system_divergence_covector: tuple[float, float, float, float]
    profile_a_computed_battery_divergence_covector: tuple[float, float, float, float]
    profile_b_computed_battery_divergence_covector: tuple[float, float, float, float]
    profile_a_integrated_energy: float
    profile_b_integrated_energy: float
    complement_constant_energy_density: float
    minimum_complement_energy_density: float
    dimensionless_profile_a_receipt_residual: float
    dimensionless_profile_b_receipt_residual: float
    dimensionless_current_difference: float
    maximum_dimensionless_divergence_identity_residual: float
    maximum_dimensionless_total_divergence_residual: float
    maximum_dimensionless_lateral_energy_flux_density: float
    current_mass_dimension: int
    four_volume_mass_dimension: int
    energy_mass_dimension: int
    dimensions_pass: bool
    same_flat_worldtube: bool
    same_scalar_receipt: bool
    current_profiles_distinct: bool
    lateral_energy_flux_zero: bool
    opposite_sector_closes_total_stress: bool
    unique_current_from_receipt_claim_refuted: bool
    worldtube_selected_by_receipt: bool = False
    branch_stress_from_receipt_derived: bool = False
    covariant_action_from_receipt_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class FlatQuantumKickWorldtubeReceipt:
    """Conditional match of a conserved finite kick to a flat worldtube.

    The supplied shared inertial basis uses signature ``(-,+,+,+)``.  For
    each closed sector ``s`` the audited discrete identity is

        Delta P_s^nu + Phi_s^nu
          = sum_i Delta V_i eta^(nu alpha) Q_(s,i,alpha).

    This is the integrated worldtube consequence of a local Ward identity.
    The routine does not derive the localization ``Q``, a local stress whose
    divergence is ``Q``, or the identification of the Hilbert-space operator
    components with a physical Lorentz four-vector.
    """

    sector_count: int
    component_count: int
    quadrature_cell_count: int
    quantum_mean_kicks: tuple[tuple[float, ...], ...]
    integrated_exchange_four_momenta: tuple[tuple[float, ...], ...]
    lateral_outward_four_momentum_fluxes: tuple[tuple[float, ...], ...]
    predicted_worldtube_kicks: tuple[tuple[float, ...], ...]
    maximum_dimensionless_sector_matching_residual: float
    maximum_dimensionless_local_exchange_residual: float
    dimensionless_integrated_exchange_residual: float
    dimensionless_total_quantum_kick_residual: float
    current_mass_dimension: int
    four_volume_mass_dimension: int
    integrated_source_mass_dimension: int
    lateral_flux_mass_dimension: int
    kick_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    quantum_operator_conservation_certified: bool
    all_receivers_included: bool
    local_exchange_currents_cancel: bool
    integrated_exchange_cancels: bool
    numerical_integrated_worldtube_matching_holds: bool
    same_local_action_identification_supplied: bool
    shared_inertial_four_vector_basis_supplied: bool
    conditional_quantum_to_worldtube_bridge_holds: bool
    worldtube_localization_supplied: bool = True
    lateral_flux_supplied: bool = True
    operator_components_as_physical_four_vector_derived: bool = False
    exchange_current_from_quantum_dynamics_derived: bool = False
    local_stress_from_quantum_kick_derived: bool = False
    general_curved_spacetime_transport_derived: bool = False
    physical_clarus_source_derived: bool = False


@dataclass(frozen=True)
class FlatChargeStressKernelCounterexample:
    """Two conserved local stresses with exactly the same four charges."""

    spatial_volume: float
    energy_density: float
    shear_amplitude: float
    profile_a_stress_contravariant: tuple[tuple[float, ...], ...]
    profile_b_stress_contravariant: tuple[tuple[float, ...], ...]
    profile_a_four_momentum: tuple[float, float, float, float]
    profile_b_four_momentum: tuple[float, float, float, float]
    dimensionless_four_momentum_residual: float
    dimensionless_local_stress_difference: float
    maximum_dimensionless_ward_residual: float
    stress_mass_dimension: int
    spatial_volume_mass_dimension: int
    four_momentum_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    both_profiles_symmetric: bool
    both_profiles_divergence_free: bool
    both_profiles_satisfy_dominant_energy_condition: bool
    same_complete_four_momentum: bool
    local_stresses_distinct: bool
    finite_charge_to_local_stress_nonuniqueness_certified: bool
    periodic_spatial_cell_supplied: bool = True
    local_action_for_profiles_derived: bool = False
    local_stress_selected_by_finite_charges: bool = False
    cosmological_perturbations_selected_by_background_charges: bool = False


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array(
    values: Sequence[object],
    name: str,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != len(trailing_shape) + 1 or result.shape[1:] != trailing_shape:
        expected = "N-by-" + "-by-".join(str(size) for size in trailing_shape)
        raise ValueError(f"{name} must be a finite {expected} array")
    if result.shape[0] == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty array")
    return result


def _validate_geometry(
    metrics_covariant: np.ndarray,
    orientation_observers_contravariant: np.ndarray,
    time_flows_contravariant: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    cell_count = metrics_covariant.shape[0]
    if orientation_observers_contravariant.shape[0] != cell_count:
        raise ValueError("orientation observers must have one row per quadrature cell")
    if time_flows_contravariant.shape[0] != cell_count:
        raise ValueError("time flows must have one row per quadrature cell")

    positive_inverse_metrics: list[np.ndarray] = []
    future_timelike = True
    for index in range(cell_count):
        metric = metrics_covariant[index]
        if not np.allclose(metric, metric.T, rtol=0.0, atol=tolerance):
            raise ValueError("each metric_covariant sample must be symmetric")
        eigenvalues = np.linalg.eigvalsh(metric)
        metric_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        negative = int(np.count_nonzero(eigenvalues < -tolerance * metric_scale))
        positive = int(np.count_nonzero(eigenvalues > tolerance * metric_scale))
        if negative != 1 or positive != 3:
            raise ValueError("each metric sample must have signature (-,+,+,+)")

        observer = orientation_observers_contravariant[index]
        observer_norm = float(observer @ metric @ observer)
        if not math.isclose(observer_norm, -1.0, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError("each orientation observer must be unit timelike")

        time_flow = time_flows_contravariant[index]
        time_flow_norm = float(time_flow @ metric @ time_flow)
        relative_orientation = float(observer @ metric @ time_flow)
        if time_flow_norm >= -tolerance or relative_orientation >= -tolerance:
            future_timelike = False

        inverse = np.linalg.inv(metric)
        positive_inverse = inverse + 2.0 * np.outer(observer, observer)
        positive_eigenvalues = np.linalg.eigvalsh(positive_inverse)
        if float(np.min(positive_eigenvalues)) <= tolerance:
            raise ArithmeticError("observer-induced covector metric is not positive")
        positive_inverse_metrics.append(positive_inverse)

    if not future_timelike:
        raise ValueError("each supplied time flow must be future timelike")
    return np.stack(positive_inverse_metrics), future_timelike


def _validate_symmetric_samples(
    samples: np.ndarray,
    name: str,
    tolerance: float,
) -> None:
    if not np.allclose(
        samples,
        np.swapaxes(samples, 1, 2),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(f"{name} must be symmetric at every quadrature cell")


def _maximum_positive_tensor_norm(
    tensors_covariant: np.ndarray,
    positive_inverse_metrics: np.ndarray,
) -> float:
    squared = np.einsum(
        "nij,nik,njl,nkl->n",
        tensors_covariant,
        positive_inverse_metrics,
        positive_inverse_metrics,
        tensors_covariant,
    )
    if float(np.min(squared)) < -1.0e-12:
        raise ArithmeticError("observer-induced tensor norm became negative")
    return math.sqrt(max(0.0, float(np.max(squared))))


def _maximum_positive_covector_norm(
    covectors: np.ndarray,
    positive_inverse_metrics: np.ndarray,
) -> float:
    squared = np.einsum(
        "ni,nij,nj->n",
        covectors,
        positive_inverse_metrics,
        covectors,
    )
    if float(np.min(squared)) < -1.0e-12:
        raise ArithmeticError("observer-induced covector norm became negative")
    return math.sqrt(max(0.0, float(np.max(squared))))


def audit_flat_quantum_kick_worldtube_receipt(
    *,
    kick_audit: QuantumKickConservationAudit,
    exchange_currents_covariant: Sequence[Sequence[Sequence[float]]],
    proper_four_volume_weights: Sequence[float],
    lateral_outward_four_momentum_fluxes: Sequence[Sequence[float]],
    same_local_action_identification_supplied: bool,
    shared_inertial_four_vector_basis_supplied: bool,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatQuantumKickWorldtubeReceipt:
    """Match a finite four-kick to a supplied flat-worldtube Ward receipt.

    The equality tested here is the discrete, flat-frame version of

        Delta P_s^nu + Phi_s^nu = integral_W Q_s^nu dV.

    ``same_local_action_identification_supplied`` records the indispensable
    premise that the quantum unitary and the local current are two
    descriptions of the same interaction.  A true value is a caller contract,
    not an independent derivation of that premise.
    """

    if not isinstance(kick_audit, QuantumKickConservationAudit):
        raise TypeError("kick_audit must be a QuantumKickConservationAudit")
    for value, name in (
        (same_local_action_identification_supplied, "same_local_action_identification_supplied"),
        (shared_inertial_four_vector_basis_supplied, "shared_inertial_four_vector_basis_supplied"),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    if kick_audit.component_count != SPACETIME_DIMENSION:
        raise ValueError("kick_audit must contain exactly four declared components")

    currents = np.asarray(exchange_currents_covariant, dtype=float)
    expected_prefix = (kick_audit.sector_count,)
    if (
        currents.ndim != 3
        or currents.shape[0:1] != expected_prefix
        or currents.shape[1] == 0
        or currents.shape[2] != SPACETIME_DIMENSION
        or not np.all(np.isfinite(currents))
    ):
        raise ValueError(
            "exchange_currents_covariant must have shape "
            "(sector_count, cell_count, 4)"
        )
    cell_count = currents.shape[1]
    volume_weights = np.asarray(proper_four_volume_weights, dtype=float)
    if (
        volume_weights.shape != (cell_count,)
        or not np.all(np.isfinite(volume_weights))
        or np.any(volume_weights <= 0.0)
    ):
        raise ValueError(
            "proper_four_volume_weights must contain one positive finite value per cell"
        )
    lateral_fluxes = np.asarray(
        lateral_outward_four_momentum_fluxes,
        dtype=float,
    )
    if (
        lateral_fluxes.shape
        != (kick_audit.sector_count, SPACETIME_DIMENSION)
        or not np.all(np.isfinite(lateral_fluxes))
    ):
        raise ValueError(
            "lateral_outward_four_momentum_fluxes must have shape "
            "(sector_count, 4)"
        )

    mean_kicks = np.asarray(kick_audit.mean_kicks, dtype=float)
    if mean_kicks.shape != lateral_fluxes.shape:
        raise ValueError("kick_audit mean_kicks have an inconsistent shape")
    minkowski_inverse = np.diag((-1.0, 1.0, 1.0, 1.0))
    integrated_sources = np.einsum(
        "n,sna,ab->sb",
        volume_weights,
        currents,
        minkowski_inverse,
        optimize=True,
    )
    predicted_kicks = integrated_sources - lateral_fluxes
    matching_residuals = mean_kicks - predicted_kicks
    local_exchange_residuals = np.sum(currents, axis=0)
    integrated_exchange_residual = np.sum(integrated_sources, axis=0)
    total_quantum_kick = np.sum(mean_kicks, axis=0)

    momentum_scale = reference_mass_scale
    current_scale = reference_mass_scale**5
    maximum_matching_residual = float(
        np.max(np.linalg.norm(matching_residuals, axis=1)) / momentum_scale
    )
    maximum_local_exchange_residual = float(
        np.max(np.linalg.norm(local_exchange_residuals, axis=1)) / current_scale
    )
    dimensionless_integrated_exchange_residual = float(
        np.linalg.norm(integrated_exchange_residual) / momentum_scale
    )
    dimensionless_total_quantum_kick_residual = float(
        np.linalg.norm(total_quantum_kick) / momentum_scale
    )

    current_mass_dimension = 5
    four_volume_mass_dimension = -4
    integrated_source_mass_dimension = 1
    lateral_flux_mass_dimension = 1
    kick_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == integrated_source_mass_dimension
        and integrated_source_mass_dimension == lateral_flux_mass_dimension
        and lateral_flux_mass_dimension == kick_mass_dimension
        and kick_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    local_exchange_cancels = maximum_local_exchange_residual <= tolerance
    integrated_exchange_cancels = (
        dimensionless_integrated_exchange_residual <= tolerance
    )
    numerical_matching = maximum_matching_residual <= tolerance
    conditional_bridge = (
        kick_audit.operator_conservation_certified
        and kick_audit.all_receivers_included
        and local_exchange_cancels
        and integrated_exchange_cancels
        and dimensionless_total_quantum_kick_residual <= tolerance
        and numerical_matching
        and bool(same_local_action_identification_supplied)
        and bool(shared_inertial_four_vector_basis_supplied)
        and dimensions_pass
    )

    return FlatQuantumKickWorldtubeReceipt(
        sector_count=kick_audit.sector_count,
        component_count=kick_audit.component_count,
        quadrature_cell_count=cell_count,
        quantum_mean_kicks=tuple(
            tuple(float(item) for item in row) for row in mean_kicks
        ),
        integrated_exchange_four_momenta=tuple(
            tuple(float(item) for item in row) for row in integrated_sources
        ),
        lateral_outward_four_momentum_fluxes=tuple(
            tuple(float(item) for item in row) for row in lateral_fluxes
        ),
        predicted_worldtube_kicks=tuple(
            tuple(float(item) for item in row) for row in predicted_kicks
        ),
        maximum_dimensionless_sector_matching_residual=maximum_matching_residual,
        maximum_dimensionless_local_exchange_residual=(
            maximum_local_exchange_residual
        ),
        dimensionless_integrated_exchange_residual=(
            dimensionless_integrated_exchange_residual
        ),
        dimensionless_total_quantum_kick_residual=(
            dimensionless_total_quantum_kick_residual
        ),
        current_mass_dimension=current_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        integrated_source_mass_dimension=integrated_source_mass_dimension,
        lateral_flux_mass_dimension=lateral_flux_mass_dimension,
        kick_mass_dimension=kick_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        quantum_operator_conservation_certified=(
            kick_audit.operator_conservation_certified
        ),
        all_receivers_included=kick_audit.all_receivers_included,
        local_exchange_currents_cancel=local_exchange_cancels,
        integrated_exchange_cancels=integrated_exchange_cancels,
        numerical_integrated_worldtube_matching_holds=numerical_matching,
        same_local_action_identification_supplied=bool(
            same_local_action_identification_supplied
        ),
        shared_inertial_four_vector_basis_supplied=bool(
            shared_inertial_four_vector_basis_supplied
        ),
        conditional_quantum_to_worldtube_bridge_holds=conditional_bridge,
    )


def construct_flat_charge_stress_kernel_counterexample(
    *,
    spatial_volume: float,
    energy_density: float,
    shear_amplitude: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatChargeStressKernelCounterexample:
    """Construct a Ward-preserving kernel of the charge-to-stress map.

    On a flat periodic spatial cell, compare constant stresses

        T_A = diag(rho, 0, 0, 0),
        T_B = diag(rho, s, -s, 0).

    Both are symmetric and divergence-free.  Their complete surface charges
    ``P^nu = integral T^(0 nu) d^3x`` agree, while their local spatial stresses
    differ whenever ``s != 0``.  Requiring ``|s| <= rho`` keeps both type-I
    tensors inside the dominant-energy bound for this explicit witness.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    spatial_volume = _positive_scalar(spatial_volume, "spatial_volume")
    energy_density = _positive_scalar(energy_density, "energy_density")
    shear_amplitude = _finite_scalar(shear_amplitude, "shear_amplitude")
    if abs(shear_amplitude) / reference_mass_scale**4 <= tolerance:
        raise ValueError("shear_amplitude must be nonzero at the declared tolerance")
    if abs(shear_amplitude) > energy_density:
        raise ValueError("the explicit dominant-energy witness requires |shear| <= rho")

    stress_a = np.diag((energy_density, 0.0, 0.0, 0.0))
    stress_b = np.diag(
        (energy_density, shear_amplitude, -shear_amplitude, 0.0)
    )
    momentum_a = spatial_volume * stress_a[0]
    momentum_b = spatial_volume * stress_b[0]
    momentum_scale = reference_mass_scale
    stress_scale = reference_mass_scale**4
    momentum_residual = float(np.linalg.norm(momentum_a - momentum_b) / momentum_scale)
    stress_difference = float(np.linalg.norm(stress_a - stress_b, ord=2) / stress_scale)
    ward_residual = 0.0

    stress_mass_dimension = 4
    spatial_volume_mass_dimension = -3
    four_momentum_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        stress_mass_dimension + spatial_volume_mass_dimension
        == four_momentum_mass_dimension
        and four_momentum_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    symmetric = bool(
        np.array_equal(stress_a, stress_a.T)
        and np.array_equal(stress_b, stress_b.T)
    )
    divergence_free = ward_residual <= tolerance
    dominant_energy = bool(abs(shear_amplitude) <= energy_density)
    same_momentum = momentum_residual <= tolerance
    distinct = stress_difference > tolerance
    witness = (
        symmetric
        and divergence_free
        and dominant_energy
        and same_momentum
        and distinct
        and dimensions_pass
    )

    return FlatChargeStressKernelCounterexample(
        spatial_volume=spatial_volume,
        energy_density=energy_density,
        shear_amplitude=shear_amplitude,
        profile_a_stress_contravariant=tuple(
            tuple(float(item) for item in row) for row in stress_a
        ),
        profile_b_stress_contravariant=tuple(
            tuple(float(item) for item in row) for row in stress_b
        ),
        profile_a_four_momentum=tuple(float(item) for item in momentum_a),
        profile_b_four_momentum=tuple(float(item) for item in momentum_b),
        dimensionless_four_momentum_residual=momentum_residual,
        dimensionless_local_stress_difference=stress_difference,
        maximum_dimensionless_ward_residual=ward_residual,
        stress_mass_dimension=stress_mass_dimension,
        spatial_volume_mass_dimension=spatial_volume_mass_dimension,
        four_momentum_mass_dimension=four_momentum_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        both_profiles_symmetric=symmetric,
        both_profiles_divergence_free=divergence_free,
        both_profiles_satisfy_dominant_energy_condition=dominant_energy,
        same_complete_four_momentum=same_momentum,
        local_stresses_distinct=distinct,
        finite_charge_to_local_stress_nonuniqueness_certified=witness,
    )


def audit_closed_branch_worldtube(
    *,
    source_receipt_id: str,
    branch_probability: float,
    conditional_trace: float,
    receipt_energy: float,
    metrics_covariant: Sequence[Sequence[Sequence[float]]],
    orientation_observers_contravariant: Sequence[Sequence[float]],
    time_flows_contravariant: Sequence[Sequence[float]],
    exchange_currents_system_covariant: Sequence[Sequence[float]],
    exchange_currents_battery_covariant: Sequence[Sequence[float]],
    system_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    battery_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    symmetrized_time_flow_gradients_covariant: Sequence[
        Sequence[Sequence[float]]
    ],
    proper_four_volume_weights: Sequence[float],
    system_initial_surface_energy: float,
    system_final_surface_energy: float,
    battery_initial_surface_energy: float,
    battery_final_surface_energy: float,
    system_lateral_outward_energy_flux: float,
    battery_lateral_outward_energy_flux: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ClosedBranchWorldtubeReceipt:
    """Audit the supplied system/battery divergence-theorem balance.

    Natural units are used.  The current has mass dimension five, stress has
    dimension four, the symmetrized gradient of the dimensionless time flow has
    dimension one, and each proper four-volume weight has dimension minus four.
    Both sector currents are independent supplied arrays.  The routine checks
    their pointwise opposite-current relation; it does not impose or derive that
    relation from an action.
    """

    if not isinstance(source_receipt_id, str) or not source_receipt_id.strip():
        raise ValueError("source_receipt_id must be a non-empty string")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    branch_probability = _finite_scalar(branch_probability, "branch_probability")
    if not 0.0 <= branch_probability <= 1.0:
        raise ValueError("branch_probability must lie in [0, 1]")
    conditional_trace = _finite_scalar(conditional_trace, "conditional_trace")
    if conditional_trace < 0.0:
        raise ValueError("conditional_trace must be non-negative")
    receipt_energy = _finite_scalar(receipt_energy, "receipt_energy")
    if receipt_energy < 0.0:
        raise ValueError("receipt_energy must be non-negative for an E9-D receipt")

    metrics = _array(
        metrics_covariant,
        "metrics_covariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    observers = _array(
        orientation_observers_contravariant,
        "orientation_observers_contravariant",
        (SPACETIME_DIMENSION,),
    )
    time_flows = _array(
        time_flows_contravariant,
        "time_flows_contravariant",
        (SPACETIME_DIMENSION,),
    )
    exchange_currents = _array(
        exchange_currents_system_covariant,
        "exchange_currents_system_covariant",
        (SPACETIME_DIMENSION,),
    )
    battery_exchange_currents = _array(
        exchange_currents_battery_covariant,
        "exchange_currents_battery_covariant",
        (SPACETIME_DIMENSION,),
    )
    system_stresses = _array(
        system_stresses_contravariant,
        "system_stresses_contravariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    battery_stresses = _array(
        battery_stresses_contravariant,
        "battery_stresses_contravariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    time_flow_gradients = _array(
        symmetrized_time_flow_gradients_covariant,
        "symmetrized_time_flow_gradients_covariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    cell_count = metrics.shape[0]
    arrays = (
        observers,
        time_flows,
        exchange_currents,
        battery_exchange_currents,
        system_stresses,
        battery_stresses,
        time_flow_gradients,
    )
    if any(array.shape[0] != cell_count for array in arrays):
        raise ValueError("all sampled fields must share the quadrature cell count")
    volume_weights = np.asarray(proper_four_volume_weights, dtype=float)
    if volume_weights.shape != (cell_count,) or not np.all(np.isfinite(volume_weights)):
        raise ValueError("proper_four_volume_weights must have one finite value per cell")
    if np.any(volume_weights <= 0.0):
        raise ValueError("proper_four_volume_weights must be positive")

    positive_inverse_metrics, future_timelike = _validate_geometry(
        metrics,
        observers,
        time_flows,
        tolerance,
    )
    _validate_symmetric_samples(system_stresses, "system stresses", tolerance)
    _validate_symmetric_samples(battery_stresses, "battery stresses", tolerance)
    _validate_symmetric_samples(
        time_flow_gradients,
        "symmetrized time-flow gradients",
        tolerance,
    )

    system_initial_surface_energy = _finite_scalar(
        system_initial_surface_energy,
        "system_initial_surface_energy",
    )
    system_final_surface_energy = _finite_scalar(
        system_final_surface_energy,
        "system_final_surface_energy",
    )
    battery_initial_surface_energy = _finite_scalar(
        battery_initial_surface_energy,
        "battery_initial_surface_energy",
    )
    battery_final_surface_energy = _finite_scalar(
        battery_final_surface_energy,
        "battery_final_surface_energy",
    )
    system_lateral_outward_energy_flux = _finite_scalar(
        system_lateral_outward_energy_flux,
        "system_lateral_outward_energy_flux",
    )
    battery_lateral_outward_energy_flux = _finite_scalar(
        battery_lateral_outward_energy_flux,
        "battery_lateral_outward_energy_flux",
    )

    source_contractions = np.einsum("ni,ni->n", time_flows, exchange_currents)
    battery_source_contractions = np.einsum(
        "ni,ni->n",
        time_flows,
        battery_exchange_currents,
    )
    system_deformation_contractions = np.einsum(
        "nij,nij->n",
        system_stresses,
        time_flow_gradients,
    )
    battery_deformation_contractions = np.einsum(
        "nij,nij->n",
        battery_stresses,
        time_flow_gradients,
    )
    system_source_injection = -float(np.dot(volume_weights, source_contractions))
    battery_source_injection = -float(
        np.dot(volume_weights, battery_source_contractions)
    )
    system_deformation_energy = -float(
        np.dot(volume_weights, system_deformation_contractions)
    )
    battery_deformation_energy = -float(
        np.dot(volume_weights, battery_deformation_contractions)
    )

    system_change = system_final_surface_energy - system_initial_surface_energy
    battery_change = battery_final_surface_energy - battery_initial_surface_energy
    system_predicted_change = (
        system_source_injection
        + system_deformation_energy
        - system_lateral_outward_energy_flux
    )
    battery_predicted_change = (
        battery_source_injection
        + battery_deformation_energy
        - battery_lateral_outward_energy_flux
    )
    system_balance_difference = system_change - system_predicted_change
    battery_balance_difference = battery_change - battery_predicted_change
    total_balance_difference = (
        system_change
        + battery_change
        + system_lateral_outward_energy_flux
        + battery_lateral_outward_energy_flux
        - system_deformation_energy
        - battery_deformation_energy
    )
    exchange_cancellation_difference = (
        system_source_injection + battery_source_injection
    )
    opposite_current_residuals = exchange_currents + battery_exchange_currents
    system_receipt_surface_difference = receipt_energy - system_change
    battery_receipt_surface_difference = -receipt_energy - battery_change
    receipt_worldtube_difference = (
        receipt_energy
        + system_lateral_outward_energy_flux
        - system_source_injection
        - system_deformation_energy
    )

    energy_scale = reference_mass_scale
    dimensionless_system_balance_residual = abs(system_balance_difference) / energy_scale
    dimensionless_battery_balance_residual = abs(battery_balance_difference) / energy_scale
    dimensionless_total_balance_residual = abs(total_balance_difference) / energy_scale
    dimensionless_exchange_cancellation_residual = (
        abs(exchange_cancellation_difference) / energy_scale
    )
    dimensionless_system_receipt_surface_residual = (
        abs(system_receipt_surface_difference) / energy_scale
    )
    dimensionless_battery_receipt_surface_residual = (
        abs(battery_receipt_surface_difference) / energy_scale
    )
    dimensionless_receipt_worldtube_residual = (
        abs(receipt_worldtube_difference) / energy_scale
    )
    maximum_dimensionless_killing_equation_residual = (
        _maximum_positive_tensor_norm(
            time_flow_gradients,
            positive_inverse_metrics,
        )
        / reference_mass_scale
    )
    maximum_dimensionless_opposite_current_residual = (
        _maximum_positive_covector_norm(
            opposite_current_residuals,
            positive_inverse_metrics,
        )
        / reference_mass_scale**5
    )

    current_mass_dimension = 5
    stress_mass_dimension = 4
    four_volume_mass_dimension = -4
    energy_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == energy_mass_dimension
        and stress_mass_dimension + 1 + four_volume_mass_dimension
        == energy_mass_dimension
        and energy_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    positive_probability_outcome = branch_probability > tolerance
    conditional_branch_normalized = math.isclose(
        conditional_trace,
        1.0,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    killing_flow = maximum_dimensionless_killing_equation_residual <= tolerance
    zero_lateral_flux = (
        abs(system_lateral_outward_energy_flux) / energy_scale <= tolerance
        and abs(battery_lateral_outward_energy_flux) / energy_scale <= tolerance
    )
    exchange_cancels = (
        dimensionless_exchange_cancellation_residual <= tolerance
        and maximum_dimensionless_opposite_current_residual <= tolerance
    )
    sector_balances_hold = (
        dimensionless_system_balance_residual <= tolerance
        and dimensionless_battery_balance_residual <= tolerance
    )
    total_energy_balance_holds = dimensionless_total_balance_residual <= tolerance
    total_energy_and_exchange_closure_holds = (
        total_energy_balance_holds and exchange_cancels
    )
    receipt_matches_both = (
        dimensionless_system_receipt_surface_residual <= tolerance
        and dimensionless_battery_receipt_surface_residual <= tolerance
    )
    killing_zero_flux_matching = (
        positive_probability_outcome
        and conditional_branch_normalized
        and future_timelike
        and killing_flow
        and zero_lateral_flux
        and exchange_cancels
        and sector_balances_hold
        and total_energy_and_exchange_closure_holds
        and receipt_matches_both
        and dimensionless_receipt_worldtube_residual <= tolerance
        and dimensions_pass
    )

    return ClosedBranchWorldtubeReceipt(
        source_receipt_id=source_receipt_id.strip(),
        quadrature_cell_count=cell_count,
        branch_probability=branch_probability,
        conditional_trace=conditional_trace,
        receipt_energy=receipt_energy,
        system_initial_surface_energy=system_initial_surface_energy,
        system_final_surface_energy=system_final_surface_energy,
        battery_initial_surface_energy=battery_initial_surface_energy,
        battery_final_surface_energy=battery_final_surface_energy,
        system_surface_energy_change=system_change,
        battery_surface_energy_change=battery_change,
        system_lateral_outward_energy_flux=system_lateral_outward_energy_flux,
        battery_lateral_outward_energy_flux=battery_lateral_outward_energy_flux,
        system_source_injection_energy=system_source_injection,
        battery_source_injection_energy=battery_source_injection,
        system_deformation_energy=system_deformation_energy,
        battery_deformation_energy=battery_deformation_energy,
        system_predicted_surface_energy_change=system_predicted_change,
        battery_predicted_surface_energy_change=battery_predicted_change,
        dimensionless_system_balance_residual=(
            dimensionless_system_balance_residual
        ),
        dimensionless_battery_balance_residual=(
            dimensionless_battery_balance_residual
        ),
        dimensionless_total_balance_residual=dimensionless_total_balance_residual,
        dimensionless_exchange_cancellation_residual=(
            dimensionless_exchange_cancellation_residual
        ),
        maximum_dimensionless_opposite_current_residual=(
            maximum_dimensionless_opposite_current_residual
        ),
        dimensionless_system_receipt_surface_residual=(
            dimensionless_system_receipt_surface_residual
        ),
        dimensionless_battery_receipt_surface_residual=(
            dimensionless_battery_receipt_surface_residual
        ),
        dimensionless_receipt_worldtube_residual=(
            dimensionless_receipt_worldtube_residual
        ),
        maximum_dimensionless_killing_equation_residual=(
            maximum_dimensionless_killing_equation_residual
        ),
        current_mass_dimension=current_mass_dimension,
        stress_mass_dimension=stress_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        energy_mass_dimension=energy_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        positive_probability_outcome=positive_probability_outcome,
        conditional_branch_normalized=conditional_branch_normalized,
        supplied_time_flow_future_timelike=future_timelike,
        supplied_killing_flow_on_samples=killing_flow,
        supplied_zero_lateral_flux=zero_lateral_flux,
        opposite_exchange_current_cancels=exchange_cancels,
        supplied_sector_balances_hold=sector_balances_hold,
        supplied_total_energy_balance_holds=total_energy_balance_holds,
        supplied_total_energy_and_exchange_closure_holds=(
            total_energy_and_exchange_closure_holds
        ),
        exclusive_branch_receipt_matches_both_sectors=receipt_matches_both,
        killing_zero_flux_receipt_matching_holds=killing_zero_flux_matching,
    )


def audit_e9d_outcome_closed_branch_worldtube(
    *,
    outcome: BatteryOutcomeReceipt,
    initial_system_energy: float,
    metrics_covariant: Sequence[Sequence[Sequence[float]]],
    orientation_observers_contravariant: Sequence[Sequence[float]],
    time_flows_contravariant: Sequence[Sequence[float]],
    exchange_currents_system_covariant: Sequence[Sequence[float]],
    exchange_currents_battery_covariant: Sequence[Sequence[float]],
    system_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    battery_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    symmetrized_time_flow_gradients_covariant: Sequence[
        Sequence[Sequence[float]]
    ],
    proper_four_volume_weights: Sequence[float],
    system_lateral_outward_energy_flux: float,
    battery_lateral_outward_energy_flux: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ClosedBranchWorldtubeReceipt:
    """Audit a typed positive-probability E9-D outcome against supplied fields.

    The adapter checks the internal outcome energy relation and obtains both
    surface ledgers from the typed battery receipt.  It verifies consistency,
    not historical authenticity: a caller can construct the dataclass by hand,
    and the local fields and worldtube remain independent supplied inputs.
    """

    if not isinstance(outcome, BatteryOutcomeReceipt):
        raise TypeError("outcome must be a BatteryOutcomeReceipt")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    initial_system_energy = _finite_scalar(
        initial_system_energy,
        "initial_system_energy",
    )
    probability = _finite_scalar(outcome.probability, "outcome.probability")
    if probability <= tolerance or probability > 1.0:
        raise ValueError("typed E9-D outcome must have positive probability")
    if not isinstance(outcome.basis_label, str) or not outcome.basis_label:
        raise ValueError("typed E9-D outcome must have a basis label")
    paid_energy = _finite_scalar(
        outcome.energy_paid_to_system,
        "outcome.energy_paid_to_system",
    )
    final_battery_energy = _finite_scalar(
        outcome.final_battery_energy,
        "outcome.final_battery_energy",
    )
    if paid_energy < 0.0 or final_battery_energy < 0.0:
        raise ValueError("typed E9-D battery energies must be non-negative")
    if outcome.conditional_system_energy is None:
        raise ValueError("typed E9-D outcome must carry conditional system energy")
    conditional_system_energy = _finite_scalar(
        outcome.conditional_system_energy,
        "outcome.conditional_system_energy",
    )
    if outcome.relative_branch_energy_residual is None:
        raise ValueError("typed E9-D outcome must carry its branch energy residual")
    reported_branch_residual = _finite_scalar(
        outcome.relative_branch_energy_residual,
        "outcome.relative_branch_energy_residual",
    )
    direct_branch_residual = abs(
        conditional_system_energy - initial_system_energy - paid_energy
    ) / reference_mass_scale
    if reported_branch_residual > tolerance or direct_branch_residual > tolerance:
        raise ValueError("typed E9-D outcome fails its branch energy relation")

    receipt = audit_closed_branch_worldtube(
        source_receipt_id=f"QNB-E9-D:{outcome.basis_label}",
        branch_probability=probability,
        conditional_trace=1.0,
        receipt_energy=paid_energy,
        metrics_covariant=metrics_covariant,
        orientation_observers_contravariant=(
            orientation_observers_contravariant
        ),
        time_flows_contravariant=time_flows_contravariant,
        exchange_currents_system_covariant=(
            exchange_currents_system_covariant
        ),
        exchange_currents_battery_covariant=(
            exchange_currents_battery_covariant
        ),
        system_stresses_contravariant=system_stresses_contravariant,
        battery_stresses_contravariant=battery_stresses_contravariant,
        symmetrized_time_flow_gradients_covariant=(
            symmetrized_time_flow_gradients_covariant
        ),
        proper_four_volume_weights=proper_four_volume_weights,
        system_initial_surface_energy=initial_system_energy,
        system_final_surface_energy=conditional_system_energy,
        battery_initial_surface_energy=final_battery_energy + paid_energy,
        battery_final_surface_energy=final_battery_energy,
        system_lateral_outward_energy_flux=(
            system_lateral_outward_energy_flux
        ),
        battery_lateral_outward_energy_flux=(
            battery_lateral_outward_energy_flux
        ),
        reference_mass_scale=reference_mass_scale,
        tolerance=tolerance,
    )
    return replace(
        receipt,
        source_receipt_id_is_provenance_label_only=False,
        typed_e9d_outcome_consistency_verified=True,
    )


def _linear_flat_stress_derivatives(
    energy_slope: float,
    longitudinal_stress_slope: float,
) -> np.ndarray:
    """Return ``partial_alpha T^{mu nu}`` for the linear counterexample."""

    derivatives = np.zeros(
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION, SPACETIME_DIMENSION),
        dtype=float,
    )
    derivatives[0, 0, 0] = energy_slope
    derivatives[1, 1, 1] = longitudinal_stress_slope
    return derivatives


def _flat_mixed_stress_divergence(stress_derivatives: np.ndarray) -> np.ndarray:
    """Compute ``partial_mu T^mu{}_nu`` in the declared Minkowski chart."""

    minkowski = np.diag((-1.0, 1.0, 1.0, 1.0))
    return np.einsum("mmk,kn->n", stress_derivatives, minkowski)


def construct_flat_receipt_current_counterexample(
    *,
    receipt_energy: float,
    duration: float,
    spatial_volume: float,
    momentum_source_a: float,
    momentum_source_b: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatReceiptCurrentCounterexample:
    """Construct two conserved-sector completions with the same receipt.

    On a flat rectangular worldtube, let ``p = delta / (T V_3)`` and

        T_S^{00} = p t,  T_S^{01} = 0,  T_S^{11} = r x.

    Then ``Q_0 = -p``, ``Q_1 = r``, lateral energy flux is zero, and the
    integrated system energy gain is ``delta`` for every ``r``.  Choosing two
    different values of ``r`` changes the local momentum current while leaving
    the scalar receipt fixed.  A complementary sector ``C^{mu nu}-T_S^{mu nu}``
    has divergence ``-Q_nu`` and keeps the same constant total stress.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    receipt_energy = _finite_scalar(receipt_energy, "receipt_energy")
    if receipt_energy < 0.0:
        raise ValueError("receipt_energy must be non-negative")
    duration = _positive_scalar(duration, "duration")
    spatial_volume = _positive_scalar(spatial_volume, "spatial_volume")
    momentum_source_a = _finite_scalar(momentum_source_a, "momentum_source_a")
    momentum_source_b = _finite_scalar(momentum_source_b, "momentum_source_b")

    four_volume = duration * spatial_volume
    energy_source_density = receipt_energy / four_volume
    profile_a_array = np.asarray(
        (-energy_source_density, momentum_source_a, 0.0, 0.0),
        dtype=float,
    )
    profile_b_array = np.asarray(
        (-energy_source_density, momentum_source_b, 0.0, 0.0),
        dtype=float,
    )
    derivatives_a = _linear_flat_stress_derivatives(
        energy_source_density,
        momentum_source_a,
    )
    derivatives_b = _linear_flat_stress_derivatives(
        energy_source_density,
        momentum_source_b,
    )
    battery_derivatives_a = -derivatives_a
    battery_derivatives_b = -derivatives_b
    computed_system_divergence_a = _flat_mixed_stress_divergence(derivatives_a)
    computed_system_divergence_b = _flat_mixed_stress_divergence(derivatives_b)
    computed_battery_divergence_a = _flat_mixed_stress_divergence(
        battery_derivatives_a
    )
    computed_battery_divergence_b = _flat_mixed_stress_divergence(
        battery_derivatives_b
    )
    profile_a = tuple(float(value) for value in profile_a_array)
    profile_b = tuple(float(value) for value in profile_b_array)
    battery_profile_a_array = -profile_a_array
    battery_profile_b_array = -profile_b_array
    battery_profile_a = tuple(float(value) for value in battery_profile_a_array)
    battery_profile_b = tuple(float(value) for value in battery_profile_b_array)
    integrated_a = -profile_a[0] * four_volume
    integrated_b = -profile_b[0] * four_volume
    energy_scale = reference_mass_scale
    current_scale = reference_mass_scale**5
    residual_a = abs(integrated_a - receipt_energy) / energy_scale
    residual_b = abs(integrated_b - receipt_energy) / energy_scale
    current_difference = abs(momentum_source_a - momentum_source_b) / current_scale
    divergence_identity_vectors = np.stack(
        (
            computed_system_divergence_a - profile_a_array,
            computed_system_divergence_b - profile_b_array,
            computed_battery_divergence_a - battery_profile_a_array,
            computed_battery_divergence_b - battery_profile_b_array,
        )
    )
    total_divergence_vectors = np.stack(
        (
            computed_system_divergence_a + computed_battery_divergence_a,
            computed_system_divergence_b + computed_battery_divergence_b,
        )
    )
    maximum_divergence_identity_residual = float(
        np.max(np.linalg.norm(divergence_identity_vectors, axis=1))
    ) / current_scale
    maximum_total_divergence_residual = float(
        np.max(np.linalg.norm(total_divergence_vectors, axis=1))
    ) / current_scale
    final_system_energy_density = energy_source_density * duration
    complement_constant_energy_density = 2.0 * final_system_energy_density
    minimum_complement_energy_density = (
        complement_constant_energy_density - final_system_energy_density
    )
    sample_stress_a = np.diag(
        (final_system_energy_density, 0.0, 0.0, 0.0)
    )
    sample_stress_b = np.diag(
        (final_system_energy_density, 0.0, 0.0, 0.0)
    )
    maximum_lateral_energy_flux_density = max(
        float(np.max(np.abs(sample_stress_a[1:, 0]))),
        float(np.max(np.abs(sample_stress_b[1:, 0]))),
    ) / reference_mass_scale**4

    current_mass_dimension = 5
    four_volume_mass_dimension = -4
    energy_mass_dimension = 1
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == energy_mass_dimension
    )
    same_receipt = residual_a <= tolerance and residual_b <= tolerance
    profiles_distinct = current_difference > tolerance
    divergence_identities_hold = maximum_divergence_identity_residual <= tolerance
    total_divergence_closes = maximum_total_divergence_residual <= tolerance
    lateral_flux_zero = maximum_lateral_energy_flux_density <= tolerance
    complement_energy_nonnegative = (
        minimum_complement_energy_density / reference_mass_scale**4
        >= -tolerance
    )
    witness = (
        same_receipt
        and profiles_distinct
        and divergence_identities_hold
        and total_divergence_closes
        and lateral_flux_zero
        and complement_energy_nonnegative
        and dimensions_pass
    )

    return FlatReceiptCurrentCounterexample(
        receipt_energy=receipt_energy,
        duration=duration,
        spatial_volume=spatial_volume,
        four_volume=four_volume,
        energy_source_density=energy_source_density,
        profile_a_current_covector=profile_a,
        profile_b_current_covector=profile_b,
        profile_a_battery_current_covector=battery_profile_a,
        profile_b_battery_current_covector=battery_profile_b,
        profile_a_computed_system_divergence_covector=tuple(
            float(value) for value in computed_system_divergence_a
        ),
        profile_b_computed_system_divergence_covector=tuple(
            float(value) for value in computed_system_divergence_b
        ),
        profile_a_computed_battery_divergence_covector=tuple(
            float(value) for value in computed_battery_divergence_a
        ),
        profile_b_computed_battery_divergence_covector=tuple(
            float(value) for value in computed_battery_divergence_b
        ),
        profile_a_integrated_energy=integrated_a,
        profile_b_integrated_energy=integrated_b,
        complement_constant_energy_density=complement_constant_energy_density,
        minimum_complement_energy_density=minimum_complement_energy_density,
        dimensionless_profile_a_receipt_residual=residual_a,
        dimensionless_profile_b_receipt_residual=residual_b,
        dimensionless_current_difference=current_difference,
        maximum_dimensionless_divergence_identity_residual=(
            maximum_divergence_identity_residual
        ),
        maximum_dimensionless_total_divergence_residual=(
            maximum_total_divergence_residual
        ),
        maximum_dimensionless_lateral_energy_flux_density=(
            maximum_lateral_energy_flux_density
        ),
        current_mass_dimension=current_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        energy_mass_dimension=energy_mass_dimension,
        dimensions_pass=dimensions_pass,
        same_flat_worldtube=True,
        same_scalar_receipt=same_receipt,
        current_profiles_distinct=profiles_distinct,
        lateral_energy_flux_zero=lateral_flux_zero,
        opposite_sector_closes_total_stress=total_divergence_closes,
        unique_current_from_receipt_claim_refuted=witness,
    )
