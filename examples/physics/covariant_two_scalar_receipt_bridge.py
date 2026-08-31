"""Conditional two-scalar exchange current and receipt non-identifiability.

The general diffeomorphism Ward identity is already part of the CE ledger.  This
module does not re-prove that theorem and does not turn a quantum battery
receipt into a gravity source.  It evaluates one explicit specialization in
four spacetime dimensions, with natural units ``hbar = c = 1`` and signature
``(-,+,+,+)``:

    S_m = integral sqrt(-g) [
        -(nabla phi)^2 / 2 - m_phi^2 phi^2 / 2
        -(nabla psi)^2 / 2 - m_psi^2 psi^2 / 2
        -lambda phi^2 psi^2 / 2
    ].

The interaction stress can be assigned to the two named sectors with a
supplied constant fraction ``alpha``.  On shell this gives equal and opposite
exchange covectors, while their sum is the uniquely defined total stress.  Two
different values of ``alpha`` keep the action and local interaction-potential
density fixed but can produce different sector currents.  That is a
constructive counterexample to inferring a unique covariant current from one
scalar density.  Here ``receipt`` means an audit record, never the E9-D battery
energy receipt.

All numeric inputs are components in one declared mass unit.  The reference
mass scale normalizes equation-of-motion and current residuals.  Consequently
only dimensionless residuals enter tolerance comparisons; dimensional values
never enter an exponential, logarithm, trigonometric function, or probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8
SPACETIME_DIMENSION = 4

Vector4 = tuple[float, float, float, float]
Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]


@dataclass(frozen=True)
class TwoScalarExchangeReceipt:
    """Pointwise Ward receipt for one supplied local two-scalar action."""

    allocation_fraction: float
    coupling: float
    interaction_energy_density: float
    interaction_d_phi: float
    interaction_d_psi: float
    interaction_gradient_covector: Vector4
    exchange_current_phi_covector: Vector4
    exchange_current_psi_covector: Vector4
    phi_sector_divergence_covector: Vector4
    psi_sector_divergence_covector: Vector4
    total_divergence_covector: Vector4
    phi_eom_residual: float
    psi_eom_residual: float
    dimensionless_eom_residual: float
    dimensionless_exchange_current_norm: float
    dimensionless_interaction_allocation_residual: float
    dimensionless_total_divergence: float
    dimensionless_ward_identity_residual: float
    dimensionless_complementarity_residual: float
    metric_signature: tuple[int, int, int, int]
    field_mass_dimension: int
    interaction_mass_dimension: int
    current_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    interaction_energy_counted_once: bool
    on_shell_within_tolerance: bool
    total_stress_conserved_on_shell: bool
    zero_coupling_exchange_vanishes: bool
    local_covariant_action_supplied: bool = True
    covariant_action_exchange_current_derived: bool = True
    interaction_allocation_dynamically_selected: bool = False
    domino_receipt_to_action_derived: bool = False
    covariant_matching_current_derived: bool = False
    physical_pointer_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class AllocationNonidentifiabilityCertificate:
    """Two allocations with one interaction density and distinct currents."""

    alpha_zero_receipt: TwoScalarExchangeReceipt
    alpha_one_receipt: TwoScalarExchangeReceipt
    dimensionless_interaction_density_difference: float
    dimensionless_current_difference: float
    dimensionless_total_interaction_allocation_difference: float
    same_action_and_interaction_density: bool
    currents_distinct: bool
    total_stress_alpha_invariant: bool
    unique_current_claim_refuted: bool
    supplied_allocation_required: bool = True
    domino_receipt_to_action_derived: bool = False
    physical_source_derived: bool = False
    record_to_gravity_source_derived: bool = False


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_scalar(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _covector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (SPACETIME_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite four-component covector")
    return array


def _lorentzian_geometry(
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    metric = np.asarray(metric_covariant, dtype=float)
    if metric.shape != (SPACETIME_DIMENSION, SPACETIME_DIMENSION):
        raise ValueError("metric_covariant must be a four-by-four matrix")
    if not np.all(np.isfinite(metric)) or not np.allclose(
        metric,
        metric.T,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("metric_covariant must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(metric)
    metric_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative = int(np.count_nonzero(eigenvalues < -tolerance * metric_scale))
    positive = int(np.count_nonzero(eigenvalues > tolerance * metric_scale))
    if negative != 1 or positive != 3:
        raise ValueError("metric_covariant must have Lorentzian signature (-,+,+,+)")
    inverse = np.linalg.inv(metric)
    observer = _covector(observer_contravariant, "observer_contravariant")
    observer_norm = float(observer @ metric @ observer)
    if not math.isclose(observer_norm, -1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("observer_contravariant must be unit timelike")
    positive_inverse = inverse + 2.0 * np.outer(observer, observer)
    positive_eigenvalues = np.linalg.eigvalsh(positive_inverse)
    if float(np.min(positive_eigenvalues)) <= tolerance:
        raise ArithmeticError("observer-induced covector norm is not positive definite")
    signature = tuple(-1 if value < 0.0 else 1 for value in eigenvalues)
    return metric, inverse, positive_inverse, signature  # type: ignore[return-value]


def _positive_covector_norm(
    covector: np.ndarray,
    positive_inverse: np.ndarray,
    tolerance: float,
) -> float:
    squared = float(covector @ positive_inverse @ covector)
    scale = max(float(np.linalg.norm(covector)) ** 2, 1.0)
    if squared < -tolerance * scale:
        raise ArithmeticError("observer-induced covector norm became negative")
    return math.sqrt(max(0.0, squared))


def _vector4(array: np.ndarray) -> Vector4:
    return tuple(float(value) for value in array)  # type: ignore[return-value]


def two_scalar_exchange_receipt(
    *,
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    phi: float,
    psi: float,
    gradient_phi_covector: Sequence[float],
    gradient_psi_covector: Sequence[float],
    box_phi: float,
    box_psi: float,
    mass_phi: float,
    mass_psi: float,
    coupling: float,
    allocation_fraction: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TwoScalarExchangeReceipt:
    """Evaluate the sector split and on-shell Ward identity at one point.

    ``phi`` and ``psi`` have mass dimension one, their covariant gradients have
    dimension two, ``box_phi`` and ``box_psi`` have dimension three, and both
    ``coupling`` and ``allocation_fraction`` are dimensionless.  The allocation
    fraction is a global supplied constant; a spacetime-dependent value would
    add derivative terms not represented by this contract.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    phi = _finite_scalar(phi, "phi")
    psi = _finite_scalar(psi, "psi")
    box_phi = _finite_scalar(box_phi, "box_phi")
    box_psi = _finite_scalar(box_psi, "box_psi")
    mass_phi = _finite_scalar(mass_phi, "mass_phi")
    mass_psi = _finite_scalar(mass_psi, "mass_psi")
    coupling = _finite_scalar(coupling, "coupling")
    allocation_fraction = _finite_scalar(
        allocation_fraction,
        "allocation_fraction",
    )
    if mass_phi < 0.0 or mass_psi < 0.0:
        raise ValueError("scalar masses must be non-negative")
    if coupling < 0.0:
        raise ValueError("coupling must be non-negative in the stable quartic branch")
    if not 0.0 <= allocation_fraction <= 1.0:
        raise ValueError("allocation_fraction must lie in [0, 1]")

    _, _, positive_inverse, signature = _lorentzian_geometry(
        metric_covariant,
        observer_contravariant,
        tolerance,
    )
    gradient_phi = _covector(gradient_phi_covector, "gradient_phi_covector")
    gradient_psi = _covector(gradient_psi_covector, "gradient_psi_covector")

    interaction = 0.5 * coupling * phi * phi * psi * psi
    interaction_d_phi = coupling * phi * psi * psi
    interaction_d_psi = coupling * phi * phi * psi
    interaction_gradient = (
        interaction_d_phi * gradient_phi
        + interaction_d_psi * gradient_psi
    )
    phi_eom_residual = (
        box_phi - mass_phi * mass_phi * phi - interaction_d_phi
    )
    psi_eom_residual = (
        box_psi - mass_psi * mass_psi * psi - interaction_d_psi
    )

    exchange_phi = (
        (1.0 - allocation_fraction) * interaction_d_phi * gradient_phi
        - allocation_fraction * interaction_d_psi * gradient_psi
    )
    exchange_psi = -exchange_phi
    phi_sector_divergence = exchange_phi + phi_eom_residual * gradient_phi
    psi_sector_divergence = exchange_psi + psi_eom_residual * gradient_psi
    total_divergence = phi_sector_divergence + psi_sector_divergence
    expected_off_shell_divergence = (
        phi_eom_residual * gradient_phi + psi_eom_residual * gradient_psi
    )
    ward_identity_difference = total_divergence - expected_off_shell_divergence
    complementarity_difference = (
        phi_sector_divergence
        + psi_sector_divergence
        - expected_off_shell_divergence
    )

    mass_cubed = reference_mass_scale**3
    mass_fourth = reference_mass_scale**4
    mass_fifth = reference_mass_scale**5
    dimensionless_eom_residual = max(
        abs(phi_eom_residual),
        abs(psi_eom_residual),
    ) / mass_cubed
    dimensionless_exchange_current_norm = (
        _positive_covector_norm(exchange_phi, positive_inverse, tolerance)
        / mass_fifth
    )
    allocated_interaction = (
        allocation_fraction * interaction
        + (1.0 - allocation_fraction) * interaction
    )
    dimensionless_interaction_allocation_residual = abs(
        allocated_interaction - interaction
    ) / mass_fourth
    dimensionless_total_divergence = (
        _positive_covector_norm(total_divergence, positive_inverse, tolerance)
        / mass_fifth
    )
    dimensionless_ward_identity_residual = (
        _positive_covector_norm(
            ward_identity_difference,
            positive_inverse,
            tolerance,
        )
        / mass_fifth
    )
    dimensionless_complementarity_residual = (
        _positive_covector_norm(
            complementarity_difference,
            positive_inverse,
            tolerance,
        )
        / mass_fifth
    )
    on_shell = dimensionless_eom_residual <= tolerance

    field_mass_dimension = 1
    interaction_mass_dimension = 4
    current_mass_dimension = 5
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        4 * field_mass_dimension == interaction_mass_dimension
        and (interaction_mass_dimension - field_mass_dimension)
        + (field_mass_dimension + 1)
        == current_mass_dimension
        and current_mass_dimension - 5 == normalized_residual_mass_dimension
    )

    return TwoScalarExchangeReceipt(
        allocation_fraction=allocation_fraction,
        coupling=coupling,
        interaction_energy_density=interaction,
        interaction_d_phi=interaction_d_phi,
        interaction_d_psi=interaction_d_psi,
        interaction_gradient_covector=_vector4(interaction_gradient),
        exchange_current_phi_covector=_vector4(exchange_phi),
        exchange_current_psi_covector=_vector4(exchange_psi),
        phi_sector_divergence_covector=_vector4(phi_sector_divergence),
        psi_sector_divergence_covector=_vector4(psi_sector_divergence),
        total_divergence_covector=_vector4(total_divergence),
        phi_eom_residual=phi_eom_residual,
        psi_eom_residual=psi_eom_residual,
        dimensionless_eom_residual=dimensionless_eom_residual,
        dimensionless_exchange_current_norm=dimensionless_exchange_current_norm,
        dimensionless_interaction_allocation_residual=(
            dimensionless_interaction_allocation_residual
        ),
        dimensionless_total_divergence=dimensionless_total_divergence,
        dimensionless_ward_identity_residual=dimensionless_ward_identity_residual,
        dimensionless_complementarity_residual=(
            dimensionless_complementarity_residual
        ),
        metric_signature=signature,
        field_mass_dimension=field_mass_dimension,
        interaction_mass_dimension=interaction_mass_dimension,
        current_mass_dimension=current_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        interaction_energy_counted_once=(
            dimensionless_interaction_allocation_residual <= tolerance
        ),
        on_shell_within_tolerance=on_shell,
        total_stress_conserved_on_shell=(
            on_shell
            and dimensionless_total_divergence <= tolerance
            and dimensionless_ward_identity_residual <= tolerance
            and dimensionless_complementarity_residual <= tolerance
        ),
        zero_coupling_exchange_vanishes=(
            coupling != 0.0 or dimensionless_exchange_current_norm <= tolerance
        ),
    )


def certify_allocation_nonidentifiability(
    *,
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    phi: float,
    psi: float,
    gradient_phi_covector: Sequence[float],
    gradient_psi_covector: Sequence[float],
    box_phi: float,
    box_psi: float,
    mass_phi: float,
    mass_psi: float,
    coupling: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AllocationNonidentifiabilityCertificate:
    """Hold the interaction density fixed and compare ``alpha=0`` with one."""

    common = dict(
        metric_covariant=metric_covariant,
        observer_contravariant=observer_contravariant,
        phi=phi,
        psi=psi,
        gradient_phi_covector=gradient_phi_covector,
        gradient_psi_covector=gradient_psi_covector,
        box_phi=box_phi,
        box_psi=box_psi,
        mass_phi=mass_phi,
        mass_psi=mass_psi,
        coupling=coupling,
        reference_mass_scale=reference_mass_scale,
        tolerance=tolerance,
    )
    alpha_zero = two_scalar_exchange_receipt(
        **common,
        allocation_fraction=0.0,
    )
    alpha_one = two_scalar_exchange_receipt(
        **common,
        allocation_fraction=1.0,
    )
    if not (
        alpha_zero.on_shell_within_tolerance
        and alpha_one.on_shell_within_tolerance
        and alpha_zero.total_stress_conserved_on_shell
        and alpha_one.total_stress_conserved_on_shell
    ):
        raise ValueError("allocation non-identifiability requires an on-shell witness")

    tolerance = _positive_scalar(tolerance, "tolerance")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    _, _, positive_inverse, _ = _lorentzian_geometry(
        metric_covariant,
        observer_contravariant,
        tolerance,
    )
    interaction_scale = reference_mass_scale**4
    current_scale = reference_mass_scale**5
    receipt_difference = abs(
        alpha_zero.interaction_energy_density
        - alpha_one.interaction_energy_density
    ) / interaction_scale
    current_difference_covector = (
        np.asarray(alpha_zero.exchange_current_phi_covector)
        - np.asarray(alpha_one.exchange_current_phi_covector)
    )
    current_difference = (
        _positive_covector_norm(
            current_difference_covector,
            positive_inverse,
            tolerance,
        )
        / current_scale
    )
    same_receipt = receipt_difference <= tolerance
    currents_distinct = current_difference > tolerance
    alpha_zero_total_interaction = (
        alpha_zero.allocation_fraction * alpha_zero.interaction_energy_density
        + (1.0 - alpha_zero.allocation_fraction)
        * alpha_zero.interaction_energy_density
    )
    alpha_one_total_interaction = (
        alpha_one.allocation_fraction * alpha_one.interaction_energy_density
        + (1.0 - alpha_one.allocation_fraction)
        * alpha_one.interaction_energy_density
    )
    total_interaction_difference = abs(
        alpha_zero_total_interaction - alpha_one_total_interaction
    ) / interaction_scale
    total_stress_alpha_invariant = (
        total_interaction_difference <= tolerance
        and alpha_zero.interaction_energy_counted_once
        and alpha_one.interaction_energy_counted_once
    )
    witness = same_receipt and currents_distinct and total_stress_alpha_invariant

    return AllocationNonidentifiabilityCertificate(
        alpha_zero_receipt=alpha_zero,
        alpha_one_receipt=alpha_one,
        dimensionless_interaction_density_difference=receipt_difference,
        dimensionless_current_difference=current_difference,
        dimensionless_total_interaction_allocation_difference=(
            total_interaction_difference
        ),
        same_action_and_interaction_density=same_receipt,
        currents_distinct=currents_distinct,
        total_stress_alpha_invariant=total_stress_alpha_invariant,
        unique_current_claim_refuted=witness,
    )
