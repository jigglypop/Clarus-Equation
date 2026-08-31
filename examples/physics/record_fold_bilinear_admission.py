"""E36 admission certificate for a record--fold bilinear candidate.

The module tests one deliberately narrow bridge proposed after E35.  A real
record *candidate* ``R_rec`` and a real fold field ``phi`` are assigned the
local scalar action

    S = integral sqrt(-g) [
        -(grad R_rec)^2 / 2 -(grad phi)^2 / 2
        -m_R^2 R_rec^2 / 2 -m_phi^2 phi^2 / 2
        +kappa R_rec phi
    ].

The action coefficient ``J_ns := kappa R_rec`` has mass dimension three.  It
is not a branch probability.  With the displayed sign convention the fold
equation is ``(box-m_phi^2) phi = -J_ns``.  All interaction energy is varied
once from this action.

The finite witnesses certify dimensional consistency, the exact quadratic
stability boundary, the pointwise Ward exchange identity, and the static
Schur complement.  They also exhibit the decisive limitation: an orthogonal
field rotation diagonalizes the bilinear mass matrix, so the names "record"
and "fold" have no physical privilege until an independent pointer,
preparation, or observable coupling fixes a basis.

This is not a derivation of ``Q_nonselected -> R_rec``, a measurement model,
a CPTP channel, a quantum microcausality proof, a GR solution, or a holdout
prediction.  Those claim ceilings remain explicit in :func:`certificate`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable

import numpy as np


DEFAULT_TOLERANCE = 1.0e-12
CANONICAL_STABLE_PARAMETERS = (9.0, 4.0, 2.0)
CANONICAL_TACHYON_PARAMETERS = (1.0, 1.0, 2.0)
CANONICAL_BOUNDARY_PARAMETERS = (1.0, 1.0, 1.0)


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


def _finite_covector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain four finite components")
    return result


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class DimensionAudit:
    record_field_mass_dimension: int
    fold_field_mass_dimension: int
    derivative_mass_dimension: int
    record_mass_squared_dimension: int
    fold_mass_squared_dimension: int
    mixing_kappa_mass_dimension: int
    source_coefficient_mass_dimension: int
    lagrangian_density_mass_dimension: int
    stress_mass_dimension: int
    ward_current_mass_dimension: int
    spacetime_volume_mass_dimension: int
    action_mass_dimension: int
    dimensions_pass: bool
    probability_used_as_source_coefficient: bool


def dimension_audit() -> DimensionAudit:
    """Return the natural-unit dimension ledger for the declared action."""

    record_dimension = 1
    fold_dimension = 1
    derivative_dimension = 1
    mass_squared_dimension = 2
    kappa_dimension = 2
    source_dimension = kappa_dimension + record_dimension
    lagrangian_dimension = 4
    stress_dimension = 4
    ward_dimension = 5
    volume_dimension = -4
    action_dimension = lagrangian_dimension + volume_dimension
    dimensions_pass = all(
        (
            2 * (derivative_dimension + record_dimension)
            == lagrangian_dimension,
            2 * (derivative_dimension + fold_dimension)
            == lagrangian_dimension,
            mass_squared_dimension + 2 * record_dimension
            == lagrangian_dimension,
            mass_squared_dimension + 2 * fold_dimension
            == lagrangian_dimension,
            kappa_dimension + record_dimension + fold_dimension
            == lagrangian_dimension,
            source_dimension + fold_dimension == lagrangian_dimension,
            stress_dimension == lagrangian_dimension,
            source_dimension + derivative_dimension + fold_dimension
            == ward_dimension,
            action_dimension == 0,
        )
    )
    return DimensionAudit(
        record_field_mass_dimension=record_dimension,
        fold_field_mass_dimension=fold_dimension,
        derivative_mass_dimension=derivative_dimension,
        record_mass_squared_dimension=mass_squared_dimension,
        fold_mass_squared_dimension=mass_squared_dimension,
        mixing_kappa_mass_dimension=kappa_dimension,
        source_coefficient_mass_dimension=source_dimension,
        lagrangian_density_mass_dimension=lagrangian_dimension,
        stress_mass_dimension=stress_dimension,
        ward_current_mass_dimension=ward_dimension,
        spacetime_volume_mass_dimension=volume_dimension,
        action_mass_dimension=action_dimension,
        dimensions_pass=dimensions_pass,
        probability_used_as_source_coefficient=False,
    )


@dataclass(frozen=True)
class BilinearSpectrumAudit:
    record_mass_squared: float
    fold_mass_squared: float
    mixing_kappa: float
    mass_squared_matrix: tuple[tuple[float, ...], ...]
    trace_mass_squared: float
    determinant_mass_four: float
    eigenmass_squared_high: float
    eigenmass_squared_low: float
    rotation_angle_radians: float
    rotation_angle_degrees: float
    rotation_matrix: tuple[tuple[float, ...], ...]
    rotated_mass_squared_matrix: tuple[tuple[float, ...], ...]
    rotated_off_diagonal_residual: float
    kinetic_rotation_residual: float
    positive_by_principal_minors: bool
    strictly_stable: bool
    tachyonic_mode_present: bool
    boundary_zero_mode_present: bool
    canonical_kinetic_ghost_free: bool


def bilinear_spectrum_audit(
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
) -> BilinearSpectrumAudit:
    """Diagonalize ``[[m_R^2,-kappa],[-kappa,m_phi^2]]``.

    The inputs all carry mass dimension two.  The fixed tolerance only
    classifies floating-point witnesses; it is not a fitted physical
    parameter.
    """

    record_mass = _finite(record_mass_squared, "record_mass_squared")
    fold_mass = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    matrix = np.asarray(
        ((record_mass, -kappa), (-kappa, fold_mass)),
        dtype=np.float64,
    )
    trace = record_mass + fold_mass
    determinant = record_mass * fold_mass - kappa**2
    discriminant = math.sqrt((record_mass - fold_mass) ** 2 + 4.0 * kappa**2)
    eigen_high = 0.5 * (trace + discriminant)
    eigen_low = 0.5 * (trace - discriminant)

    if kappa == 0.0:
        angle = 0.0
    else:
        angle = 0.5 * math.atan2(-2.0 * kappa, record_mass - fold_mass)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)),
        dtype=np.float64,
    )
    rotated = rotation.T @ matrix @ rotation
    kinetic_residual = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(2, dtype=np.float64)))
    )
    off_diagonal_residual = float(abs(rotated[0, 1]))
    scale_mass_squared = max(
        abs(record_mass),
        abs(fold_mass),
        abs(kappa),
        1.0,
    )
    eigen_tolerance = DEFAULT_TOLERANCE * scale_mass_squared
    positive_by_minors = record_mass > 0.0 and determinant > 0.0
    strictly_stable = positive_by_minors and eigen_low > eigen_tolerance
    tachyonic = eigen_low < -eigen_tolerance
    boundary_zero = abs(eigen_low) <= eigen_tolerance
    return BilinearSpectrumAudit(
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
        mass_squared_matrix=_tuple_matrix(matrix),
        trace_mass_squared=trace,
        determinant_mass_four=determinant,
        eigenmass_squared_high=eigen_high,
        eigenmass_squared_low=eigen_low,
        rotation_angle_radians=angle,
        rotation_angle_degrees=math.degrees(angle),
        rotation_matrix=_tuple_matrix(rotation),
        rotated_mass_squared_matrix=_tuple_matrix(rotated),
        rotated_off_diagonal_residual=off_diagonal_residual,
        kinetic_rotation_residual=kinetic_residual,
        positive_by_principal_minors=positive_by_minors,
        strictly_stable=strictly_stable,
        tachyonic_mode_present=tachyonic,
        boundary_zero_mode_present=boundary_zero,
        canonical_kinetic_ghost_free=True,
    )


def require_stable_spectrum(
    record_mass_squared: float,
    fold_mass_squared: float,
    mixing_kappa: float,
) -> BilinearSpectrumAudit:
    """Return a stable receipt or fail closed at/beyond the boundary."""

    receipt = bilinear_spectrum_audit(
        record_mass_squared,
        fold_mass_squared,
        mixing_kappa,
    )
    if not receipt.strictly_stable:
        raise ValueError(
            "bilinear spectrum is not strictly stable: "
            f"det={receipt.determinant_mass_four}, "
            f"lowest eigenmass squared={receipt.eigenmass_squared_low}"
        )
    return receipt


@dataclass(frozen=True)
class WardExchangeAudit:
    record_value: float
    fold_value: float
    source_coefficient: float
    record_eom_residual: float
    fold_eom_residual: float
    record_gradient_covector: tuple[float, ...]
    fold_gradient_covector: tuple[float, ...]
    free_fold_stress_divergence: tuple[float, ...]
    record_plus_interaction_divergence: tuple[float, ...]
    total_stress_divergence: tuple[float, ...]
    expected_total_ward_covector: tuple[float, ...]
    fold_exchange_covector: tuple[float, ...]
    opposite_exchange_covector: tuple[float, ...]
    dimensionless_ward_identity_residual: float
    dimensionless_exchange_balance_residual: float
    dimensionless_total_divergence: float
    both_field_equations_on_shell: bool
    total_stress_conserved_on_shell: bool
    interaction_counted_once: bool


def ward_exchange_audit(
    *,
    record_value: float,
    fold_value: float,
    box_record: float,
    box_fold: float,
    record_gradient_covector: Iterable[float],
    fold_gradient_covector: Iterable[float],
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
    reference_mass_scale: float = 1.0,
) -> WardExchangeAudit:
    """Evaluate the off-shell one-action Ward identity at one chart point.

    For ``E_R=(box-m_R^2)R+kappa phi`` and
    ``E_phi=(box-m_phi^2)phi+kappa R``, direct differentiation gives

    ``div(T_total) = E_R grad(R) + E_phi grad(phi)``.

    Every dimensionful residual is divided by the appropriate power of the
    supplied reference mass before numerical comparison.
    """

    record = _finite(record_value, "record_value")
    fold = _finite(fold_value, "fold_value")
    box_r = _finite(box_record, "box_record")
    box_phi = _finite(box_fold, "box_fold")
    mass_r = _finite(record_mass_squared, "record_mass_squared")
    mass_phi = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    mass_scale = _positive(reference_mass_scale, "reference_mass_scale")
    gradient_r = _finite_covector(
        record_gradient_covector,
        "record_gradient_covector",
    )
    gradient_phi = _finite_covector(
        fold_gradient_covector,
        "fold_gradient_covector",
    )

    source = kappa * record
    record_eom = box_r - mass_r * record + kappa * fold
    fold_eom = box_phi - mass_phi * fold + source
    free_fold = (box_phi - mass_phi * fold) * gradient_phi
    record_plus_interaction = (
        (box_r - mass_r * record) * gradient_r
        + kappa * (fold * gradient_r + record * gradient_phi)
    )
    total = free_fold + record_plus_interaction
    expected_total = record_eom * gradient_r + fold_eom * gradient_phi
    fold_exchange = -source * gradient_phi
    opposite_exchange = source * gradient_phi
    ward_identity_residual = float(np.max(np.abs(total - expected_total)))
    exchange_balance_residual = float(
        np.max(np.abs(fold_exchange + opposite_exchange))
    )
    current_scale = mass_scale**5
    eom_scale = mass_scale**3
    dimensionless_eom_residual = max(abs(record_eom), abs(fold_eom)) / eom_scale
    dimensionless_total = float(np.max(np.abs(total))) / current_scale
    on_shell = dimensionless_eom_residual <= DEFAULT_TOLERANCE
    return WardExchangeAudit(
        record_value=record,
        fold_value=fold,
        source_coefficient=source,
        record_eom_residual=record_eom,
        fold_eom_residual=fold_eom,
        record_gradient_covector=tuple(float(item) for item in gradient_r),
        fold_gradient_covector=tuple(float(item) for item in gradient_phi),
        free_fold_stress_divergence=tuple(float(item) for item in free_fold),
        record_plus_interaction_divergence=tuple(
            float(item) for item in record_plus_interaction
        ),
        total_stress_divergence=tuple(float(item) for item in total),
        expected_total_ward_covector=tuple(
            float(item) for item in expected_total
        ),
        fold_exchange_covector=tuple(float(item) for item in fold_exchange),
        opposite_exchange_covector=tuple(
            float(item) for item in opposite_exchange
        ),
        dimensionless_ward_identity_residual=(
            ward_identity_residual / current_scale
        ),
        dimensionless_exchange_balance_residual=(
            exchange_balance_residual / current_scale
        ),
        dimensionless_total_divergence=dimensionless_total,
        both_field_equations_on_shell=on_shell,
        total_stress_conserved_on_shell=(
            on_shell and dimensionless_total <= DEFAULT_TOLERANCE
        ),
        interaction_counted_once=True,
    )


def canonical_on_shell_ward_audit() -> WardExchangeAudit:
    """Return a nontrivial exact on-shell exchange witness."""

    record = 0.5
    fold = -0.25
    record_mass, fold_mass, kappa = CANONICAL_STABLE_PARAMETERS
    return ward_exchange_audit(
        record_value=record,
        fold_value=fold,
        box_record=record_mass * record - kappa * fold,
        box_fold=fold_mass * fold - kappa * record,
        record_gradient_covector=(0.3, -0.2, 0.1, 0.0),
        fold_gradient_covector=(-0.4, 0.05, 0.0, 0.2),
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
    )


@dataclass(frozen=True)
class SchurComplementAudit:
    record_mass_squared: float
    fold_mass_squared: float
    mixing_kappa: float
    determinant_mass_four: float
    static_effective_fold_mass_squared: float
    determinant_over_record_mass_squared: float
    positive_static_effective_mass: bool
    operator_kernel: str
    zero_momentum_local_formula_only: bool
    inverse_boundary_or_state_prescription_required: bool
    retarded_inverse_automatically_selected: bool
    closed_time_path_noise_derived: bool
    local_effective_stress_automatically_derived: bool


def schur_complement_audit(
    record_mass_squared: float = 9.0,
    fold_mass_squared: float = 4.0,
    mixing_kappa: float = 2.0,
) -> SchurComplementAudit:
    """Return the static mass Schur complement and exact operator warning."""

    record_mass = _positive(record_mass_squared, "record_mass_squared")
    fold_mass = _finite(fold_mass_squared, "fold_mass_squared")
    kappa = _finite(mixing_kappa, "mixing_kappa")
    determinant = record_mass * fold_mass - kappa**2
    effective_mass = fold_mass - kappa**2 / record_mass
    determinant_ratio = determinant / record_mass
    return SchurComplementAudit(
        record_mass_squared=record_mass,
        fold_mass_squared=fold_mass,
        mixing_kappa=kappa,
        determinant_mass_four=determinant,
        static_effective_fold_mass_squared=effective_mass,
        determinant_over_record_mass_squared=determinant_ratio,
        positive_static_effective_mass=effective_mass > 0.0,
        operator_kernel="D_phi - kappa^2 D_R^{-1}",
        zero_momentum_local_formula_only=True,
        inverse_boundary_or_state_prescription_required=True,
        retarded_inverse_automatically_selected=False,
        closed_time_path_noise_derived=False,
        local_effective_stress_automatically_derived=False,
    )


@dataclass(frozen=True)
class SourceAccountingAudit:
    mode: str
    retained_record_and_fold_fields: bool
    integrated_out_influence_kernel: bool
    original_bilinear_interaction_retained: bool
    mutually_exclusive_representations: bool
    probability_rebooked_as_energy: bool
    source_stress_counted_twice: bool


def source_accounting_audit(mode: str) -> SourceAccountingAudit:
    """Admit exactly one of retained-field and integrated-out ledgers."""

    if mode == "retained_fields":
        retained = True
        influence = False
        interaction = True
    elif mode == "integrated_out_influence":
        retained = False
        influence = True
        interaction = False
    else:
        raise ValueError(f"unknown source accounting mode: {mode}")
    return SourceAccountingAudit(
        mode=mode,
        retained_record_and_fold_fields=retained,
        integrated_out_influence_kernel=influence,
        original_bilinear_interaction_retained=interaction,
        mutually_exclusive_representations=(retained != influence),
        probability_rebooked_as_energy=False,
        source_stress_counted_twice=False,
    )


@dataclass(frozen=True)
class BasisObstructionAudit:
    witness_mass_squared_matrix: tuple[tuple[float, ...], ...]
    eigenmass_squared_set: tuple[float, float]
    absolute_rotation_angle_degrees: float
    rotated_off_diagonal_residual: float
    kinetic_rotation_residual: float
    hypothetical_pointer_vector_original_basis: tuple[float, float]
    hypothetical_pointer_vector_eigenbasis: tuple[float, float]
    hypothetical_pointer_is_extra_input: bool
    eigenmass_squared_set_basis_invariant: bool
    record_and_fold_labels_basis_invariant: bool
    bilinear_mixing_selects_pointer_basis: bool
    bilinear_mixing_derives_observed_outcome: bool
    bilinear_mixing_derives_dark_source: bool


def basis_obstruction_audit() -> BasisObstructionAudit:
    """Diagonalize a 45-degree witness without supplying pointer physics."""

    receipt = bilinear_spectrum_audit(5.0, 5.0, 1.0)
    rotation = np.asarray(receipt.rotation_matrix, dtype=np.float64)
    pointer_original = np.asarray((1.0, 0.0), dtype=np.float64)
    pointer_eigenbasis = rotation.T @ pointer_original
    return BasisObstructionAudit(
        witness_mass_squared_matrix=receipt.mass_squared_matrix,
        eigenmass_squared_set=tuple(
            sorted(
                (
                    receipt.eigenmass_squared_low,
                    receipt.eigenmass_squared_high,
                )
            )
        ),
        absolute_rotation_angle_degrees=abs(receipt.rotation_angle_degrees),
        rotated_off_diagonal_residual=receipt.rotated_off_diagonal_residual,
        kinetic_rotation_residual=receipt.kinetic_rotation_residual,
        hypothetical_pointer_vector_original_basis=(1.0, 0.0),
        hypothetical_pointer_vector_eigenbasis=tuple(
            float(item) for item in pointer_eigenbasis
        ),
        hypothetical_pointer_is_extra_input=True,
        eigenmass_squared_set_basis_invariant=True,
        record_and_fold_labels_basis_invariant=False,
        bilinear_mixing_selects_pointer_basis=False,
        bilinear_mixing_derives_observed_outcome=False,
        bilinear_mixing_derives_dark_source=False,
    )


@dataclass(frozen=True)
class RecordFoldBilinearCertificate:
    status: str
    dimensions: DimensionAudit
    stable_witness: BilinearSpectrumAudit
    tachyon_counterexample: BilinearSpectrumAudit
    boundary_counterexample: BilinearSpectrumAudit
    ward_witness: WardExchangeAudit
    static_schur_witness: SchurComplementAudit
    retained_accounting: SourceAccountingAudit
    integrated_out_accounting: SourceAccountingAudit
    basis_obstruction: BasisObstructionAudit
    source_sign_convention: str
    one_total_action_accounting_admitted: bool
    nonselected_quantum_to_record_map_derived: bool
    pointer_selection_and_durable_record_derived: bool
    probability_deformation_defined: bool
    cptp_and_normalization_derived: bool
    classical_principal_symbol_uses_metric_cone: bool
    qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    zero_stress_qm_gr_limit_derived: bool
    gravitational_solution_derived: bool
    fixed_parameter_manifest_established: bool
    independent_holdout_prediction_derived: bool
    two_residual_classes_reduced: bool
    complexity_penalized_improvement_established: bool


def certificate() -> RecordFoldBilinearCertificate:
    """Build the complete E36 finite admission/no-go certificate."""

    stable = require_stable_spectrum(*CANONICAL_STABLE_PARAMETERS)
    tachyon = bilinear_spectrum_audit(*CANONICAL_TACHYON_PARAMETERS)
    boundary = bilinear_spectrum_audit(*CANONICAL_BOUNDARY_PARAMETERS)
    return RecordFoldBilinearCertificate(
        status="CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION",
        dimensions=dimension_audit(),
        stable_witness=stable,
        tachyon_counterexample=tachyon,
        boundary_counterexample=boundary,
        ward_witness=canonical_on_shell_ward_audit(),
        static_schur_witness=schur_complement_audit(
            *CANONICAL_STABLE_PARAMETERS
        ),
        retained_accounting=source_accounting_audit("retained_fields"),
        integrated_out_accounting=source_accounting_audit(
            "integrated_out_influence"
        ),
        basis_obstruction=basis_obstruction_audit(),
        source_sign_convention=(
            "+kappa R_rec phi in L gives "
            "(box-m_phi^2)phi=-J_ns, J_ns=kappa R_rec"
        ),
        one_total_action_accounting_admitted=True,
        nonselected_quantum_to_record_map_derived=False,
        pointer_selection_and_durable_record_derived=False,
        probability_deformation_defined=False,
        cptp_and_normalization_derived=False,
        classical_principal_symbol_uses_metric_cone=True,
        qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        zero_stress_qm_gr_limit_derived=False,
        gravitational_solution_derived=False,
        fixed_parameter_manifest_established=False,
        independent_holdout_prediction_derived=False,
        two_residual_classes_reduced=False,
        complexity_penalized_improvement_established=False,
    )


def run() -> dict[str, object]:
    """Return the JSON-serializable certificate payload."""

    return asdict(certificate())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(run(), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
