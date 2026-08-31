"""E32 scalar-receipt admission gate for a covariant stress candidate.

This module starts *after* the E31 finite ledger has been made rank-complete.
It does not identify a Walsh coordinate with a physical field.  Instead it
certifies two narrower boundaries.

First, in four Lorentzian dimensions, a pointwise algebraic symmetric
covariant rank-two tensor made only from a supplied metric and dimensionless
scalar receipt values is proportional to the metric.  The statement excludes
derivatives, curvature, vector or foliation data, and nonlocal kernels.  The
finite certificate constructs the infinitesimal Lorentz constraints on the
ten-dimensional space of symmetric tensors.  Spatial rotations leave a
two-dimensional isotropic subspace; adding boosts leaves only ``span(eta)``.

Second, even a supplied local scalar embedding does not select the additive
normalization of an action.  For a dimensionless constant receipt ``r0`` and
``phi0 = M_star * r0``, the actions

    S_total^(a) = S_EH + S_visible + epsilon S_h^(a),

    S_h^(a) = -integral sqrt(-g) [
        (grad phi)^2 / 2 + m^2 (phi - phi0)^2 / 2 + a
    ]

with ``a = 0`` and ``a = M_star**4`` have, for every fixed ``epsilon > 0``,
the same scalar equation of motion, the same principal symbol, and the same
constant on-shell field.  Their on-shell stresses differ by
``-epsilon * M_star**4 * g``.  Thus receipt completeness plus conservation
does not choose a gravitational source.  In the limit ``epsilon -> 0`` the
hidden stress disappears from a separately supplied GR+visible equation; no
claim about convergence of metric solutions is made.

All quantities are in natural units.  The result is a finite algebraic
admission/no-go witness, not a metric derivation, a CPTP construction, a
microcausality proof, or an observational prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable

import numpy as np

from examples.physics.contextual_global_section_obstruction import (
    exact_rational_rank,
)
from examples.physics.hidden_source_factorization_receipt_rank import (
    combined_readout_rank,
    walsh_receipt_matrix,
)


SPACETIME_DIMENSION = 4
SYMMETRIC_COMPONENT_COUNT = 10
DEFAULT_TOLERANCE = 1.0e-12
COMPONENT_ORDER = (
    "00",
    "01",
    "02",
    "03",
    "11",
    "12",
    "13",
    "22",
    "23",
    "33",
)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def minkowski_metric() -> np.ndarray:
    """Return the supplied ``(-,+,+,+)`` metric in one orthonormal frame."""

    return np.diag((-1, 1, 1, 1)).astype(np.int64)


def symmetric_tensor_basis() -> tuple[np.ndarray, ...]:
    """Return the integer basis in ``COMPONENT_ORDER``."""

    basis: list[np.ndarray] = []
    for first in range(SPACETIME_DIMENSION):
        for second in range(first, SPACETIME_DIMENSION):
            tensor = np.zeros(
                (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
                dtype=np.int64,
            )
            tensor[first, second] = 1
            tensor[second, first] = 1
            basis.append(tensor)
    return tuple(basis)


def lorentz_generators() -> dict[str, np.ndarray]:
    """Return three rotations and three boosts as exact integer matrices."""

    generators: dict[str, np.ndarray] = {}
    for first, second in ((1, 2), (1, 3), (2, 3)):
        generator = np.zeros((4, 4), dtype=np.int64)
        generator[first, second] = 1
        generator[second, first] = -1
        generators[f"J{first}{second}"] = generator
    for spatial in (1, 2, 3):
        generator = np.zeros((4, 4), dtype=np.int64)
        generator[0, spatial] = 1
        generator[spatial, 0] = 1
        generators[f"K0{spatial}"] = generator
    return generators


def infinitesimal_invariance_constraint(
    generator_names: Iterable[str],
) -> np.ndarray:
    """Build exact coefficients of ``X.T @ T + T @ X = 0``.

    Every independent symmetric component contributes one row per generator.
    Zero rows are retained because the exact rank, rather than the raw row
    count, is the invariant used by the certificate.
    """

    generators = lorentz_generators()
    basis = symmetric_tensor_basis()
    rows: list[list[int]] = []
    for name in generator_names:
        if name not in generators:
            raise ValueError(f"unknown Lorentz generator: {name}")
        generator = generators[name]
        variations = tuple(
            generator.T @ tensor + tensor @ generator for tensor in basis
        )
        for first in range(SPACETIME_DIMENSION):
            for second in range(first, SPACETIME_DIMENSION):
                rows.append(
                    [
                        int(variation[first, second])
                        for variation in variations
                    ]
                )
    if not rows:
        raise ValueError("at least one Lorentz generator is required")
    return np.asarray(rows, dtype=np.int64)


def tensor_from_components(components: Iterable[float]) -> np.ndarray:
    """Construct a symmetric tensor from the canonical ten components."""

    values = np.asarray(tuple(components), dtype=np.float64)
    if values.shape != (SYMMETRIC_COMPONENT_COUNT,) or not np.isfinite(values).all():
        raise ValueError("components must contain ten finite values")
    tensor = sum(
        (value * basis for value, basis in zip(values, symmetric_tensor_basis())),
        start=np.zeros((4, 4), dtype=np.float64),
    )
    return tensor


@dataclass(frozen=True)
class LorentzNaturalTensorCertificate:
    symmetric_tensor_dimension: int
    rotation_constraint_shape: tuple[int, int]
    rotation_constraint_rank: int
    rotation_invariant_nullity: int
    full_lorentz_constraint_shape: tuple[int, int]
    full_lorentz_constraint_rank: int
    full_lorentz_invariant_nullity: int
    metric_generator_residual: int
    rotation_time_basis_residual: int
    rotation_spatial_basis_residual: int
    full_metric_span_unique: bool
    order_zero_scalar_source_form: str


def lorentz_natural_tensor_certificate() -> LorentzNaturalTensorCertificate:
    """Certify the order-zero scalar-only Lorentz invariant subspaces."""

    rotations = ("J12", "J13", "J23")
    boosts = ("K01", "K02", "K03")
    rotation_constraint = infinitesimal_invariance_constraint(rotations)
    full_constraint = infinitesimal_invariance_constraint(rotations + boosts)
    rotation_rank = exact_rational_rank(rotation_constraint)
    full_rank = exact_rational_rank(full_constraint)
    metric_components = np.asarray(
        (-1, 0, 0, 0, 1, 0, 0, 1, 0, 1),
        dtype=np.int64,
    )
    time_components = np.asarray(
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        dtype=np.int64,
    )
    spatial_components = np.asarray(
        (0, 0, 0, 0, 1, 0, 0, 1, 0, 1),
        dtype=np.int64,
    )
    metric_residual = int(np.max(np.abs(full_constraint @ metric_components)))
    time_residual = int(
        np.max(np.abs(rotation_constraint @ time_components))
    )
    spatial_residual = int(
        np.max(np.abs(rotation_constraint @ spatial_components))
    )
    rotation_nullity = SYMMETRIC_COMPONENT_COUNT - rotation_rank
    full_nullity = SYMMETRIC_COMPONENT_COUNT - full_rank
    return LorentzNaturalTensorCertificate(
        symmetric_tensor_dimension=SYMMETRIC_COMPONENT_COUNT,
        rotation_constraint_shape=tuple(rotation_constraint.shape),
        rotation_constraint_rank=rotation_rank,
        rotation_invariant_nullity=rotation_nullity,
        full_lorentz_constraint_shape=tuple(full_constraint.shape),
        full_lorentz_constraint_rank=full_rank,
        full_lorentz_invariant_nullity=full_nullity,
        metric_generator_residual=metric_residual,
        rotation_time_basis_residual=time_residual,
        rotation_spatial_basis_residual=spatial_residual,
        full_metric_span_unique=(
            full_nullity == 1
            and metric_residual == 0
            and rotation_nullity == 2
            and time_residual == 0
            and spatial_residual == 0
        ),
        order_zero_scalar_source_form="T_mn = C(r) g_mn",
    )


@dataclass(frozen=True)
class VacuumFormReceipt:
    coefficient: float
    stress_covariant: tuple[tuple[float, ...], ...]
    energy_density: float
    isotropic_pressure: float
    equation_of_state: float | None


def vacuum_form_receipt(coefficient: float) -> VacuumFormReceipt:
    """Evaluate ``T=C g`` for the rest observer of the supplied frame."""

    scalar = _finite(coefficient, "coefficient")
    stress = scalar * minkowski_metric().astype(np.float64)
    energy_density = float(stress[0, 0])
    pressure = float(np.trace(stress[1:, 1:]) / 3.0)
    equation_of_state = (
        None if energy_density == 0.0 else pressure / energy_density
    )
    return VacuumFormReceipt(
        coefficient=scalar,
        stress_covariant=tuple(tuple(float(item) for item in row) for row in stress),
        energy_density=energy_density,
        isotropic_pressure=pressure,
        equation_of_state=equation_of_state,
    )


@dataclass(frozen=True)
class AdditiveActionCountermodel:
    receipt_value: float
    reference_mass_scale: float
    scalar_mass: float
    hidden_action_coefficient: float
    constant_field_value: float
    zero_additive_density: float
    nonzero_additive_density: float
    zero_source_stress_covariant: tuple[tuple[float, ...], ...]
    nonzero_source_stress_covariant: tuple[tuple[float, ...], ...]
    normalized_stress_difference: float
    scalar_eom_difference: float
    principal_symbol_difference: float
    on_shell_eom_residual: float
    on_shell_divergence_residual: float
    zero_coefficient_hidden_stress_residual: float
    zero_coefficient_hidden_eom_coefficient: float
    same_operational_receipt_without_action_normalization: bool
    same_constant_on_shell_field: bool
    same_scalar_eom_for_positive_coefficient: bool
    same_principal_symbol_for_positive_coefficient: bool
    both_stresses_conserved_on_shell: bool
    finite_coefficient_metric_sources_distinct: bool
    additive_source_selected_by_receipt: bool
    zero_coefficient_hidden_metric_source_vanishes: bool
    metric_solution_convergence_derived: bool


def canonical_scalar_potential(
    field_value: float,
    *,
    field_minimum: float,
    scalar_mass: float,
    additive_density: float,
) -> float:
    """Return ``m^2 (phi-phi0)^2 / 2 + a`` in one declared mass unit."""

    field = _finite(field_value, "field_value")
    minimum = _finite(field_minimum, "field_minimum")
    mass = _positive(scalar_mass, "scalar_mass")
    additive = _finite(additive_density, "additive_density")
    return 0.5 * mass**2 * (field - minimum) ** 2 + additive


def canonical_scalar_potential_derivative(
    field_value: float,
    *,
    field_minimum: float,
    scalar_mass: float,
) -> float:
    """Return the exact first field derivative of the canonical potential."""

    field = _finite(field_value, "field_value")
    minimum = _finite(field_minimum, "field_minimum")
    mass = _positive(scalar_mass, "scalar_mass")
    return mass**2 * (field - minimum)


def canonical_scalar_eom(
    field_value: float,
    *,
    box_field: float,
    field_minimum: float,
    scalar_mass: float,
    hidden_action_coefficient: float,
) -> float:
    """Return ``epsilon [box(phi)-V'(phi)]`` for the supplied action."""

    epsilon = _finite(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    if epsilon < 0.0:
        raise ValueError("hidden_action_coefficient must be nonnegative")
    return epsilon * (
        _finite(box_field, "box_field")
        - canonical_scalar_potential_derivative(
            field_value,
            field_minimum=field_minimum,
            scalar_mass=scalar_mass,
        )
    )


def canonical_scalar_principal_coefficient(
    hidden_action_coefficient: float,
) -> float:
    """Return the scalar coefficient multiplying ``g^mn d_m d_n``."""

    epsilon = _finite(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    if epsilon < 0.0:
        raise ValueError("hidden_action_coefficient must be nonnegative")
    return epsilon


def canonical_scalar_stress_at_flat_point(
    field_value: float,
    *,
    gradient_covector: Iterable[float],
    field_minimum: float,
    scalar_mass: float,
    additive_density: float,
    hidden_action_coefficient: float,
) -> np.ndarray:
    """Evaluate the canonical Hilbert-stress formula in the supplied frame."""

    epsilon = canonical_scalar_principal_coefficient(
        hidden_action_coefficient
    )
    gradient = np.asarray(tuple(gradient_covector), dtype=np.float64)
    if gradient.shape != (4,) or not np.isfinite(gradient).all():
        raise ValueError("gradient_covector must contain four finite values")
    metric = minkowski_metric().astype(np.float64)
    inverse_metric = metric
    gradient_square = float(gradient @ inverse_metric @ gradient)
    potential = canonical_scalar_potential(
        field_value,
        field_minimum=field_minimum,
        scalar_mass=scalar_mass,
        additive_density=additive_density,
    )
    return epsilon * (
        np.outer(gradient, gradient)
        - metric * (0.5 * gradient_square + potential)
    )


def canonical_scalar_ward_divergence(
    field_eom: float,
    *,
    gradient_covector: Iterable[float],
) -> np.ndarray:
    """Return the on-shell Ward factor ``E_phi * d_n phi``."""

    eom = _finite(field_eom, "field_eom")
    gradient = np.asarray(tuple(gradient_covector), dtype=np.float64)
    if gradient.shape != (4,) or not np.isfinite(gradient).all():
        raise ValueError("gradient_covector must contain four finite values")
    return eom * gradient


def additive_action_countermodel(
    *,
    receipt_value: float = 0.375,
    reference_mass_scale: float = 2.0,
    scalar_mass: float = 1.5,
    hidden_action_coefficient: float = 0.25,
) -> AdditiveActionCountermodel:
    """Return two covariant actions with one receipt but distinct stresses.

    ``hidden_action_coefficient`` is the overall dimensionless ``epsilon``.
    It is not a non-minimal curvature coupling.  The comparison is made only
    for fixed positive epsilon.  At epsilon zero the hidden equation itself
    disappears; only decoupling from the supplied metric equation is claimed.
    """

    receipt = _finite(receipt_value, "receipt_value")
    mass_scale = _positive(reference_mass_scale, "reference_mass_scale")
    mass = _positive(scalar_mass, "scalar_mass")
    epsilon = _positive(
        hidden_action_coefficient,
        "hidden_action_coefficient",
    )
    field_value = mass_scale * receipt
    additive_zero = 0.0
    additive_nonzero = mass_scale**4
    constant_gradient = np.zeros(4, dtype=np.float64)
    stress_zero = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_zero,
        hidden_action_coefficient=epsilon,
    )
    stress_nonzero = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_nonzero,
        hidden_action_coefficient=epsilon,
    )
    stress_scale = epsilon * mass_scale**4
    stress_difference = float(
        np.max(np.abs(stress_nonzero - stress_zero)) / stress_scale
    )

    # The additive constant has zero field derivative.  Evaluate both branch
    # formulas at a common off-shell probe before checking the shared solution.
    probe_field = field_value + 0.25 * mass_scale
    probe_box = 0.125 * mass_scale**3
    eom_zero_probe = canonical_scalar_eom(
        probe_field,
        box_field=probe_box,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    eom_nonzero_probe = canonical_scalar_eom(
        probe_field,
        box_field=probe_box,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    scalar_eom_difference = abs(eom_zero_probe - eom_nonzero_probe) / max(
        epsilon * mass_scale**3,
        1.0,
    )
    principal_zero = canonical_scalar_principal_coefficient(epsilon)
    principal_nonzero = canonical_scalar_principal_coefficient(epsilon)
    principal_symbol_difference = abs(principal_zero - principal_nonzero)
    eom_zero_on_shell = canonical_scalar_eom(
        field_value,
        box_field=0.0,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    eom_nonzero_on_shell = canonical_scalar_eom(
        field_value,
        box_field=0.0,
        field_minimum=field_value,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    )
    on_shell_eom_residual = max(
        abs(eom_zero_on_shell),
        abs(eom_nonzero_on_shell),
    ) / max(epsilon * mass_scale**3, 1.0)
    divergence_zero = canonical_scalar_ward_divergence(
        eom_zero_on_shell,
        gradient_covector=constant_gradient,
    )
    divergence_nonzero = canonical_scalar_ward_divergence(
        eom_nonzero_on_shell,
        gradient_covector=constant_gradient,
    )
    on_shell_divergence_residual = max(
        float(np.max(np.abs(divergence_zero))),
        float(np.max(np.abs(divergence_nonzero))),
    ) / max(epsilon * mass_scale**5, 1.0)
    zero_epsilon_stress = canonical_scalar_stress_at_flat_point(
        field_value,
        gradient_covector=constant_gradient,
        field_minimum=field_value,
        scalar_mass=mass,
        additive_density=additive_nonzero,
        hidden_action_coefficient=0.0,
    )
    zero_coefficient_hidden_stress_residual = float(
        np.max(np.abs(zero_epsilon_stress))
    )
    zero_coefficient_hidden_eom_coefficient = (
        canonical_scalar_principal_coefficient(0.0)
    )
    receipt_zero_branch = (
        receipt,
        mass_scale,
        field_value,
        tuple(float(item) for item in constant_gradient),
        0.0,
    )
    receipt_nonzero_branch = (
        receipt,
        mass_scale,
        field_value,
        tuple(float(item) for item in constant_gradient),
        0.0,
    )
    same_receipt = receipt_zero_branch == receipt_nonzero_branch
    same_field = on_shell_eom_residual <= DEFAULT_TOLERANCE
    same_eom = scalar_eom_difference <= DEFAULT_TOLERANCE
    same_principal = principal_symbol_difference <= DEFAULT_TOLERANCE
    conserved = on_shell_divergence_residual <= DEFAULT_TOLERANCE
    distinct = stress_difference > DEFAULT_TOLERANCE
    return AdditiveActionCountermodel(
        receipt_value=receipt,
        reference_mass_scale=mass_scale,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
        constant_field_value=field_value,
        zero_additive_density=additive_zero,
        nonzero_additive_density=additive_nonzero,
        zero_source_stress_covariant=tuple(
            tuple(float(item) for item in row) for row in stress_zero
        ),
        nonzero_source_stress_covariant=tuple(
            tuple(float(item) for item in row) for row in stress_nonzero
        ),
        normalized_stress_difference=stress_difference,
        scalar_eom_difference=scalar_eom_difference,
        principal_symbol_difference=principal_symbol_difference,
        on_shell_eom_residual=on_shell_eom_residual,
        on_shell_divergence_residual=on_shell_divergence_residual,
        zero_coefficient_hidden_stress_residual=(
            zero_coefficient_hidden_stress_residual
        ),
        zero_coefficient_hidden_eom_coefficient=(
            zero_coefficient_hidden_eom_coefficient
        ),
        same_operational_receipt_without_action_normalization=same_receipt,
        same_constant_on_shell_field=same_field,
        same_scalar_eom_for_positive_coefficient=same_eom,
        same_principal_symbol_for_positive_coefficient=same_principal,
        both_stresses_conserved_on_shell=conserved,
        finite_coefficient_metric_sources_distinct=distinct,
        additive_source_selected_by_receipt=not (
            same_receipt
            and same_field
            and same_eom
            and same_principal
            and conserved
            and distinct
        ),
        zero_coefficient_hidden_metric_source_vanishes=(
            zero_coefficient_hidden_stress_residual <= DEFAULT_TOLERANCE
        ),
        metric_solution_convergence_derived=False,
    )


@dataclass(frozen=True)
class SourceAccountingReceipt:
    mode: str
    retained_hidden_stress_added: bool
    integrated_out_influence_response_added: bool
    rn_probability_reweighting_added_as_energy: bool
    rank_or_volume_added_as_energy: bool
    mutually_exclusive_source_accounting: bool
    declared_no_probability_energy_rebooking: bool


def source_accounting_receipt(mode: str) -> SourceAccountingReceipt:
    """Return one exclusive source-accounting mode or reject the request."""

    if mode not in {
        "retained_hidden_field",
        "integrated_out_influence",
        "receipt_only_no_source",
    }:
        raise ValueError("unknown source accounting mode")
    retained = mode == "retained_hidden_field"
    influence = mode == "integrated_out_influence"
    neither = mode == "receipt_only_no_source"
    exclusive = int(retained) + int(influence) <= 1
    no_rebooking_declared = exclusive and (retained or influence or neither)
    return SourceAccountingReceipt(
        mode=mode,
        retained_hidden_stress_added=retained,
        integrated_out_influence_response_added=influence,
        rn_probability_reweighting_added_as_energy=False,
        rank_or_volume_added_as_energy=False,
        mutually_exclusive_source_accounting=exclusive,
        declared_no_probability_energy_rebooking=no_rebooking_declared,
    )


@dataclass(frozen=True)
class ScalarReceiptSourceAdmissionCertificate:
    lorentz_tensor: LorentzNaturalTensorCertificate
    positive_vacuum_form: VacuumFormReceipt
    action_countermodel: AdditiveActionCountermodel
    source_accounting: SourceAccountingReceipt
    e31_full_receipt_combined_rank: int
    e31_receipt_kernel_rank: int
    e31_rank_complete_receipt: bool
    receipt_mass_dimension: int
    metric_mass_dimension: int
    reference_scale_mass_dimension: int
    scalar_field_mass_dimension: int
    scalar_mass_dimension: int
    derivative_mass_dimension: int
    potential_mass_dimension: int
    stress_mass_dimension: int
    action_density_mass_dimension: int
    volume_element_mass_dimension: int
    action_mass_dimension: int
    hidden_action_coefficient_mass_dimension: int
    dimensions_pass: bool
    rank_complete_receipt_selects_physical_source: bool
    scalar_only_order_zero_source_is_vacuum_form: bool
    dust_source_derived: bool
    current_gradient_or_kinetic_data_required_for_dust: bool
    local_receipt_to_field_map_derived: bool
    supplied_metric_derived_from_receipt: bool
    metric_variation_machine_verified: bool
    conditional_ward_theorem_replaced_by_numerics: bool
    cptp_quantum_dynamics_derived: bool
    qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    finite_coefficient_gr_phenomenology_derived: bool
    independent_holdout_prediction_derived: bool
    two_residual_classes_reduced: bool
    complexity_penalty_success: bool


def certificate() -> ScalarReceiptSourceAdmissionCertificate:
    """Build the canonical E32 finite admission certificate."""

    lorentz = lorentz_natural_tensor_certificate()
    # Positive vacuum density V=M_*^4 corresponds to C=-V in T=Cg.
    vacuum = vacuum_form_receipt(-16.0)
    countermodel = additive_action_countermodel()
    accounting = source_accounting_receipt("retained_hidden_field")
    walsh = walsh_receipt_matrix()
    combined_rank = combined_readout_rank(walsh)
    hidden_rank = exact_rational_rank(walsh @ walsh.T)
    rank_complete = combined_rank == 16 and hidden_rank == 7
    ambiguity = not countermodel.additive_source_selected_by_receipt

    receipt_dim = 0
    metric_dim = 0
    scale_dim = 1
    field_dim = 1
    mass_dim = 1
    derivative_dim = 1
    potential_dim = 4
    stress_dim = 4
    action_density_dim = 4
    volume_dim = -4
    action_dim = 0
    epsilon_dim = 0
    dimensions_pass = (
        receipt_dim == metric_dim == epsilon_dim == action_dim == 0
        and scale_dim == field_dim == mass_dim == derivative_dim == 1
        and potential_dim == stress_dim == action_density_dim == 4
        and action_density_dim + volume_dim == action_dim
        and 4 * scale_dim == potential_dim
        and 2 * (derivative_dim + field_dim) == action_density_dim
        and 2 * mass_dim + 2 * field_dim == action_density_dim
    )
    return ScalarReceiptSourceAdmissionCertificate(
        lorentz_tensor=lorentz,
        positive_vacuum_form=vacuum,
        action_countermodel=countermodel,
        source_accounting=accounting,
        e31_full_receipt_combined_rank=combined_rank,
        e31_receipt_kernel_rank=hidden_rank,
        e31_rank_complete_receipt=rank_complete,
        receipt_mass_dimension=receipt_dim,
        metric_mass_dimension=metric_dim,
        reference_scale_mass_dimension=scale_dim,
        scalar_field_mass_dimension=field_dim,
        scalar_mass_dimension=mass_dim,
        derivative_mass_dimension=derivative_dim,
        potential_mass_dimension=potential_dim,
        stress_mass_dimension=stress_dim,
        action_density_mass_dimension=action_density_dim,
        volume_element_mass_dimension=volume_dim,
        action_mass_dimension=action_dim,
        hidden_action_coefficient_mass_dimension=epsilon_dim,
        dimensions_pass=dimensions_pass,
        rank_complete_receipt_selects_physical_source=(
            rank_complete and not ambiguity
        ),
        scalar_only_order_zero_source_is_vacuum_form=(
            lorentz.full_metric_span_unique
            and vacuum.equation_of_state == -1.0
        ),
        dust_source_derived=False,
        current_gradient_or_kinetic_data_required_for_dust=True,
        local_receipt_to_field_map_derived=False,
        supplied_metric_derived_from_receipt=False,
        metric_variation_machine_verified=False,
        conditional_ward_theorem_replaced_by_numerics=False,
        cptp_quantum_dynamics_derived=False,
        qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        finite_coefficient_gr_phenomenology_derived=False,
        independent_holdout_prediction_derived=False,
        two_residual_classes_reduced=False,
        complexity_penalty_success=False,
    )


def run() -> dict[str, object]:
    """Return a JSON-serializable E32 receipt."""

    return asdict(certificate())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(run(), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
