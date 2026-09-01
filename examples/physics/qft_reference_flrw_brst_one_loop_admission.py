'''Finite ST-cohomology admission for the QFT redesign programme.

The order-hbar breaking in this module is supplied coefficient-space data.  It
is not produced by a Feynman integral, a UV regulator, or a BV measure
Laplacian.  The gate only checks whether a finite BRST/ST complex distinguishes
closed exact breakings from closed non-exact and non-closed controls.

All numerical coordinates are dimensionless coefficients normalized to their
declared (but not continuum-matched) operator basis scales.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ABS_TOL = 1.0e-12
REL_RANK_TOL = 1.0e-10

MINIMUM_OPERATOR_CATALOGUE = (
    'sqrt(-g)',
    'sqrt(-g) R',
    'sqrt(-g) R^2',
    'sqrt(-g) R_mn R^mn',
    'sqrt(-g) R_mnrs R^mnrs',
    'sqrt(-g) box R',
    'sqrt(-g) (nabla phi)^2',
    'sqrt(-g) m^2 phi^2',
    'sqrt(-g) xi R phi^2',
    'sqrt(-g) phi^3',
    'sqrt(-g) phi^4',
)


@dataclass(frozen=True)
class FiniteOneLoopStContract:
    '''Scope lock for a supplied order-hbar finite coefficient vector.'''

    loop_order_label: int
    breaking_provenance: str
    coefficient_normalization: str
    operator_catalogue: tuple[str, ...]
    operator_coefficient_status: tuple[str, ...]
    breaking_derived_from_loop_integral: bool
    uv_regulator_supplied: bool
    continuum_counterterm_basis_complete: bool
    local_counterterm_coefficients_computed: bool
    regulator_independence_computed: bool
    continuum_local_brst_cohomology_computed: bool
    bv_measure_laplacian_computed: bool
    ctp_doubling_computed: bool
    positive_physical_hilbert_computed: bool
    nonperturbative_m2_passed: bool


def finite_one_loop_st_contract() -> FiniteOneLoopStContract:
    return FiniteOneLoopStContract(
        loop_order_label=1,
        breaking_provenance=(
            'supplied dimensionless coefficient vector; not derived from a loop integral'
        ),
        coefficient_normalization=(
            'per-operator reference scales; no continuum matching claimed'
        ),
        operator_catalogue=MINIMUM_OPERATOR_CATALOGUE,
        operator_coefficient_status=tuple(
            'uncomputed' for _ in MINIMUM_OPERATOR_CATALOGUE
        ),
        breaking_derived_from_loop_integral=False,
        uv_regulator_supplied=False,
        continuum_counterterm_basis_complete=False,
        local_counterterm_coefficients_computed=False,
        regulator_independence_computed=False,
        continuum_local_brst_cohomology_computed=False,
        bv_measure_laplacian_computed=False,
        ctp_doubling_computed=False,
        positive_physical_hilbert_computed=False,
        nonperturbative_m2_passed=False,
    )


def validate_contract(contract: FiniteOneLoopStContract) -> None:
    '''Reject missing provenance and every unsupported continuum promotion.'''

    if contract.loop_order_label != 1:
        raise ValueError('this admission contract is labelled only at order hbar')
    if not contract.breaking_provenance.strip():
        raise ValueError('breaking provenance must be explicit')
    if 'not derived' not in contract.breaking_provenance:
        raise ValueError('the supplied breaking must not be presented as a loop result')
    if not contract.coefficient_normalization.strip():
        raise ValueError('coefficient normalization must be explicit')
    if contract.operator_catalogue != MINIMUM_OPERATOR_CATALOGUE:
        raise ValueError('the frozen diagnostic operator catalogue changed')
    if len(contract.operator_coefficient_status) != len(
        contract.operator_catalogue
    ):
        raise ValueError('every operator requires a coefficient status')
    if any(status != 'uncomputed' for status in contract.operator_coefficient_status):
        raise ValueError('this finite admission cannot claim computed loop coefficients')
    unsupported_promotions = (
        contract.breaking_derived_from_loop_integral,
        contract.uv_regulator_supplied,
        contract.continuum_counterterm_basis_complete,
        contract.local_counterterm_coefficients_computed,
        contract.regulator_independence_computed,
        contract.continuum_local_brst_cohomology_computed,
        contract.bv_measure_laplacian_computed,
        contract.ctp_doubling_computed,
        contract.positive_physical_hilbert_computed,
        contract.nonperturbative_m2_passed,
    )
    if any(unsupported_promotions):
        raise ValueError('unsupported continuum, QME, CTP, Hilbert, or M2 promotion')


@dataclass(frozen=True)
class FiniteStComplex:
    '''Coefficient complex C^-1 -> C^0 -> C^1 -> C^2.'''

    b_minus_one: np.ndarray
    b_zero: np.ndarray
    b_one: np.ndarray
    names_minus_one: tuple[str, ...]
    names_zero: tuple[str, ...]
    names_one: tuple[str, ...]
    names_two: tuple[str, ...]


def finite_st_complex() -> FiniteStComplex:
    '''Extend the E61 quartet by a nontrivial and a non-closed direction.

    Nonzero maps are b(cbar)=B, b(q)=c, and b(u)=v.  The ghost-one
    coordinate a is closed with no ghost-zero preimage.
    '''

    return FiniteStComplex(
        b_minus_one=np.array([[0.0], [0.0], [1.0]]),
        b_zero=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        b_one=np.array([[0.0, 0.0, 1.0]]),
        names_minus_one=('cbar',),
        names_zero=('x', 'q', 'B'),
        names_one=('c', 'a', 'u'),
        names_two=('v',),
    )


def validate_complex(complex_: FiniteStComplex) -> None:
    dimensions = (
        len(complex_.names_minus_one),
        len(complex_.names_zero),
        len(complex_.names_one),
        len(complex_.names_two),
    )
    expected_shapes = (
        (dimensions[1], dimensions[0]),
        (dimensions[2], dimensions[1]),
        (dimensions[3], dimensions[2]),
    )
    matrices = (
        np.asarray(complex_.b_minus_one, dtype=float),
        np.asarray(complex_.b_zero, dtype=float),
        np.asarray(complex_.b_one, dtype=float),
    )
    if tuple(matrix.shape for matrix in matrices) != expected_shapes:
        raise ValueError('BRST/ST maps do not match their ghost-number spaces')
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError('BRST/ST maps must be finite')
    for names in (
        complex_.names_minus_one,
        complex_.names_zero,
        complex_.names_one,
        complex_.names_two,
    ):
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError('each ghost-number basis requires unique nonempty names')


@dataclass(frozen=True)
class SvdAudit:
    rank: int
    threshold: float
    singular_values: tuple[float, ...]
    condition_number_on_image: float


def _svd_audit(
    matrix: np.ndarray,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_RANK_TOL,
) -> SvdAudit:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(float(abs_tol), float(rel_tol) * largest)
    retained = singular[singular > threshold]
    rank = int(retained.size)
    if rank:
        condition = float(retained[0] / retained[-1])
    else:
        condition = 1.0
    return SvdAudit(
        rank=rank,
        threshold=threshold,
        singular_values=tuple(float(value) for value in singular),
        condition_number_on_image=condition,
    )


def _pseudoinverse(
    matrix: np.ndarray,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_RANK_TOL,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(float(abs_tol), float(rel_tol) * largest)
    inverse = np.zeros_like(singular)
    inverse[singular > threshold] = 1.0 / singular[singular > threshold]
    return (vh.T * inverse) @ u.T


@dataclass(frozen=True)
class ComplexAudit:
    nilpotency_minus_one_residual: float
    nilpotency_zero_residual: float
    ranks: tuple[int, int, int]
    cohomology_dimensions: tuple[int, int, int, int]
    singular_values_minus_one: tuple[float, ...]
    singular_values_zero: tuple[float, ...]
    singular_values_one: tuple[float, ...]
    rank_thresholds: tuple[float, float, float]
    maximum_image_condition_number: float


def audit_complex(
    complex_: FiniteStComplex,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_RANK_TOL,
) -> ComplexAudit:
    validate_complex(complex_)
    maps = (
        np.asarray(complex_.b_minus_one, dtype=float),
        np.asarray(complex_.b_zero, dtype=float),
        np.asarray(complex_.b_one, dtype=float),
    )
    audits = tuple(
        _svd_audit(matrix, abs_tol=abs_tol, rel_tol=rel_tol)
        for matrix in maps
    )
    ranks = tuple(item.rank for item in audits)
    dimensions = (
        len(complex_.names_minus_one),
        len(complex_.names_zero),
        len(complex_.names_one),
        len(complex_.names_two),
    )
    cohomology = (
        dimensions[0] - ranks[0],
        dimensions[1] - ranks[1] - ranks[0],
        dimensions[2] - ranks[2] - ranks[1],
        dimensions[3] - ranks[2],
    )
    return ComplexAudit(
        nilpotency_minus_one_residual=float(np.linalg.norm(maps[1] @ maps[0])),
        nilpotency_zero_residual=float(np.linalg.norm(maps[2] @ maps[1])),
        ranks=ranks,
        cohomology_dimensions=cohomology,
        singular_values_minus_one=audits[0].singular_values,
        singular_values_zero=audits[1].singular_values,
        singular_values_one=audits[2].singular_values,
        rank_thresholds=tuple(item.threshold for item in audits),
        maximum_image_condition_number=max(
            item.condition_number_on_image for item in audits
        ),
    )


@dataclass(frozen=True)
class BreakingAudit:
    vector: tuple[float, ...]
    closed: bool
    removable: bool
    closure_residual: float
    image_distance: float
    counterterm: tuple[float, ...]
    renormalized_breaking: tuple[float, ...]
    renormalized_breaking_norm: float


def audit_breaking(
    complex_: FiniteStComplex,
    breaking: np.ndarray,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_RANK_TOL,
) -> BreakingAudit:
    validate_complex(complex_)
    vector = np.asarray(breaking, dtype=float)
    if vector.shape != (len(complex_.names_one),):
        raise ValueError('breaking must be a ghost-number-one coefficient vector')
    if not np.all(np.isfinite(vector)):
        raise ValueError('breaking coefficients must be finite')
    b_zero = np.asarray(complex_.b_zero, dtype=float)
    b_one = np.asarray(complex_.b_one, dtype=float)
    counterterm = _pseudoinverse(
        b_zero, abs_tol=abs_tol, rel_tol=rel_tol
    ) @ vector
    remainder = vector - b_zero @ counterterm
    closure = float(np.linalg.norm(b_one @ vector))
    distance = float(np.linalg.norm(remainder))
    closed = closure < abs_tol
    removable = closed and distance < abs_tol
    return BreakingAudit(
        vector=tuple(float(value) for value in vector),
        closed=closed,
        removable=removable,
        closure_residual=closure,
        image_distance=distance,
        counterterm=tuple(float(value) for value in counterterm),
        renormalized_breaking=tuple(float(value) for value in remainder),
        renormalized_breaking_norm=distance,
    )


def _basis_transformed_complex(
    complex_: FiniteStComplex,
    transforms: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> FiniteStComplex:
    s_minus_one, s_zero, s_one, s_two = transforms
    return FiniteStComplex(
        b_minus_one=s_zero @ complex_.b_minus_one @ np.linalg.inv(s_minus_one),
        b_zero=s_one @ complex_.b_zero @ np.linalg.inv(s_zero),
        b_one=s_two @ complex_.b_one @ np.linalg.inv(s_one),
        names_minus_one=complex_.names_minus_one,
        names_zero=complex_.names_zero,
        names_one=complex_.names_one,
        names_two=complex_.names_two,
    )


def _declared_quotient_coordinate(
    exact_direction: np.ndarray,
    anomaly_direction: np.ndarray,
    nonclosed_direction: np.ndarray,
    vector: np.ndarray,
) -> float:
    '''Return the anomaly coordinate in a declared algebraic decomposition.

    Unlike an Euclidean pseudoinverse remainder, this coordinate transforms
    covariantly under a simultaneous invertible change of all four declared
    directions.  It is evaluated only after closure/non-closure is checked.
    '''

    frame = np.column_stack(
        (exact_direction, anomaly_direction, nonclosed_direction)
    )
    if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(vector)):
        raise ValueError('quotient-coordinate data must be finite')
    if np.linalg.cond(frame) > 10.0:
        raise ValueError('declared quotient frame is ill-conditioned')
    coordinates = np.linalg.solve(frame, np.asarray(vector, dtype=float))
    return float(coordinates[1])


@dataclass(frozen=True)
class FiniteOneLoopStReceipt:
    loop_order_label: int
    breaking_provenance: str
    coefficient_normalization: str
    operator_catalogue: tuple[str, ...]
    operator_coefficients_all_uncomputed: bool
    breaking_derived_from_loop_integral: bool
    uv_regulator_supplied: bool
    continuum_counterterm_basis_complete: bool
    local_counterterm_coefficients_computed: bool
    regulator_independence_computed: bool
    continuum_local_brst_cohomology_computed: bool
    bv_measure_laplacian_computed: bool
    ctp_doubling_computed: bool
    positive_physical_hilbert_computed: bool
    nonperturbative_m2_passed: bool
    nilpotency_minus_one_residual: float
    nilpotency_zero_residual: float
    ranks: tuple[int, int, int]
    cohomology_dimensions: tuple[int, int, int, int]
    singular_values_minus_one: tuple[float, ...]
    singular_values_zero: tuple[float, ...]
    singular_values_one: tuple[float, ...]
    rank_thresholds: tuple[float, float, float]
    maximum_image_condition_number: float
    exact_breaking: BreakingAudit
    anomaly_control: BreakingAudit
    nonclosed_control: BreakingAudit
    zero_breaking_control: BreakingAudit
    wrong_counterterm_sign_residual: float
    nonnilpotent_control_residual: float
    nonnilpotent_control_to_tolerance_ratio: float
    basis_change_maximum_condition_number: float
    basis_nonzero_quotient_coordinate: float
    basis_covariance_residual: float
    basis_change_classification_invariant: bool
    minimum_retained_singular_to_threshold_ratio: float
    rank_tolerance_sweep_invariant: bool
    rank_ambiguity_control_detected: bool
    declared_finite_one_loop_st_admission_gate_passed: bool


def evaluate_finite_one_loop_st_admission_gate(
    contract: FiniteOneLoopStContract | None = None,
) -> FiniteOneLoopStReceipt:
    '''Evaluate the frozen finite ST admission and its negative controls.'''

    if contract is None:
        contract = finite_one_loop_st_contract()
    validate_contract(contract)
    complex_ = finite_st_complex()
    audit = audit_complex(complex_)

    exact_vector = np.array([3.0 / 8.0, 0.0, 0.0])
    anomaly_vector = np.array([0.0, 1.0, 0.0])
    nonclosed_vector = np.array([0.0, 0.0, 1.0])
    zero_vector = np.zeros(3)
    exact = audit_breaking(complex_, exact_vector)
    anomaly = audit_breaking(complex_, anomaly_vector)
    nonclosed = audit_breaking(complex_, nonclosed_vector)
    zero = audit_breaking(complex_, zero_vector)

    counterterm = np.asarray(exact.counterterm)
    wrong_sign = float(
        np.linalg.norm(exact_vector + complex_.b_zero @ counterterm)
    )
    malformed_b_one = complex_.b_one.copy()
    malformed_b_one[0, 0] = 1.0e-3
    nonnilpotent = float(np.linalg.norm(malformed_b_one @ complex_.b_zero))

    transforms = (
        np.array([[0.8]]),
        np.array(
            [
                [1.0, 0.2, 0.0],
                [0.0, 1.0, 0.1],
                [0.0, 0.0, 1.1],
            ]
        ),
        np.array(
            [
                [1.0, 0.1, 0.05],
                [0.0, 1.2, 0.2],
                [0.0, 0.0, 0.9],
            ]
        ),
        np.array([[1.3]]),
    )
    transformed = _basis_transformed_complex(complex_, transforms)
    transformed_audit = audit_complex(transformed)
    transformed_exact = audit_breaking(transformed, transforms[2] @ exact_vector)
    transformed_anomaly = audit_breaking(
        transformed, transforms[2] @ anomaly_vector
    )
    transformed_nonclosed = audit_breaking(
        transformed, transforms[2] @ nonclosed_vector
    )
    mixed_closed_vector = np.array([0.25, 0.60, 0.0])
    mixed_closed = audit_breaking(complex_, mixed_closed_vector)
    transformed_mixed_closed = audit_breaking(
        transformed, transforms[2] @ mixed_closed_vector
    )
    quotient_coordinate = _declared_quotient_coordinate(
        np.array([1.0, 0.0, 0.0]),
        anomaly_vector,
        nonclosed_vector,
        mixed_closed_vector,
    )
    transformed_quotient_coordinate = _declared_quotient_coordinate(
        transforms[2] @ np.array([1.0, 0.0, 0.0]),
        transforms[2] @ anomaly_vector,
        transforms[2] @ nonclosed_vector,
        transforms[2] @ mixed_closed_vector,
    )
    basis_covariance = abs(transformed_quotient_coordinate - quotient_coordinate)
    basis_conditions = tuple(float(np.linalg.cond(item)) for item in transforms)
    basis_classification = bool(
        transformed_audit.ranks == audit.ranks
        and transformed_audit.cohomology_dimensions == audit.cohomology_dimensions
        and transformed_exact.closed
        and transformed_exact.removable
        and transformed_anomaly.closed
        and not transformed_anomaly.removable
        and not transformed_nonclosed.closed
        and not transformed_nonclosed.removable
        and mixed_closed.closed
        and not mixed_closed.removable
        and transformed_mixed_closed.closed
        and not transformed_mixed_closed.removable
    )

    tolerance_audits = tuple(
        audit_complex(complex_, rel_tol=value)
        for value in (1.0e-8, 1.0e-10, 1.0e-12)
    )
    tolerance_invariant = all(
        item.ranks == audit.ranks
        and item.cohomology_dimensions == audit.cohomology_dimensions
        for item in tolerance_audits
    )
    retained_margins = []
    for singular_values, threshold in zip(
        (
            audit.singular_values_minus_one,
            audit.singular_values_zero,
            audit.singular_values_one,
        ),
        audit.rank_thresholds,
        strict=True,
    ):
        retained_margins.extend(
            value / threshold for value in singular_values if value > threshold
        )
    minimum_rank_margin = min(retained_margins)
    ambiguous_rank_matrix = np.diag([1.0, 5.0e-10, 0.0])
    ambiguous_ranks = tuple(
        _svd_audit(ambiguous_rank_matrix, rel_tol=value).rank
        for value in (1.0e-8, 1.0e-10, 1.0e-12)
    )
    rank_ambiguity_detected = len(set(ambiguous_ranks)) > 1
    nonnilpotent_ratio = nonnilpotent / ABS_TOL

    unsupported_flags = (
        contract.breaking_derived_from_loop_integral,
        contract.uv_regulator_supplied,
        contract.continuum_counterterm_basis_complete,
        contract.local_counterterm_coefficients_computed,
        contract.regulator_independence_computed,
        contract.continuum_local_brst_cohomology_computed,
        contract.bv_measure_laplacian_computed,
        contract.ctp_doubling_computed,
        contract.positive_physical_hilbert_computed,
        contract.nonperturbative_m2_passed,
    )
    passed = bool(
        not any(unsupported_flags)
        and all(
            status == 'uncomputed'
            for status in contract.operator_coefficient_status
        )
        and audit.nilpotency_minus_one_residual < ABS_TOL
        and audit.nilpotency_zero_residual < ABS_TOL
        and audit.ranks == (1, 1, 1)
        and audit.cohomology_dimensions == (0, 1, 1, 0)
        and exact.closed
        and exact.removable
        and exact.closure_residual < ABS_TOL
        and exact.renormalized_breaking_norm < ABS_TOL
        and anomaly.closed
        and not anomaly.removable
        and anomaly.closure_residual < ABS_TOL
        and anomaly.image_distance > 0.9
        and not nonclosed.closed
        and not nonclosed.removable
        and nonclosed.closure_residual > 0.9
        and zero.closed
        and zero.removable
        and zero.renormalized_breaking_norm < ABS_TOL
        and wrong_sign > 1.0e-2
        and nonnilpotent_ratio > 1.0e6
        and max(basis_conditions) < 10.0
        and abs(quotient_coordinate - 0.60) < ABS_TOL
        and basis_covariance < 1.0e-10
        and basis_classification
        and minimum_rank_margin > 1.0e6
        and tolerance_invariant
        and rank_ambiguity_detected
    )
    return FiniteOneLoopStReceipt(
        loop_order_label=contract.loop_order_label,
        breaking_provenance=contract.breaking_provenance,
        coefficient_normalization=contract.coefficient_normalization,
        operator_catalogue=contract.operator_catalogue,
        operator_coefficients_all_uncomputed=all(
            status == 'uncomputed'
            for status in contract.operator_coefficient_status
        ),
        breaking_derived_from_loop_integral=(
            contract.breaking_derived_from_loop_integral
        ),
        uv_regulator_supplied=contract.uv_regulator_supplied,
        continuum_counterterm_basis_complete=(
            contract.continuum_counterterm_basis_complete
        ),
        local_counterterm_coefficients_computed=(
            contract.local_counterterm_coefficients_computed
        ),
        regulator_independence_computed=contract.regulator_independence_computed,
        continuum_local_brst_cohomology_computed=(
            contract.continuum_local_brst_cohomology_computed
        ),
        bv_measure_laplacian_computed=contract.bv_measure_laplacian_computed,
        ctp_doubling_computed=contract.ctp_doubling_computed,
        positive_physical_hilbert_computed=(
            contract.positive_physical_hilbert_computed
        ),
        nonperturbative_m2_passed=contract.nonperturbative_m2_passed,
        nilpotency_minus_one_residual=audit.nilpotency_minus_one_residual,
        nilpotency_zero_residual=audit.nilpotency_zero_residual,
        ranks=audit.ranks,
        cohomology_dimensions=audit.cohomology_dimensions,
        singular_values_minus_one=audit.singular_values_minus_one,
        singular_values_zero=audit.singular_values_zero,
        singular_values_one=audit.singular_values_one,
        rank_thresholds=audit.rank_thresholds,
        maximum_image_condition_number=audit.maximum_image_condition_number,
        exact_breaking=exact,
        anomaly_control=anomaly,
        nonclosed_control=nonclosed,
        zero_breaking_control=zero,
        wrong_counterterm_sign_residual=wrong_sign,
        nonnilpotent_control_residual=nonnilpotent,
        nonnilpotent_control_to_tolerance_ratio=nonnilpotent_ratio,
        basis_change_maximum_condition_number=max(basis_conditions),
        basis_nonzero_quotient_coordinate=quotient_coordinate,
        basis_covariance_residual=basis_covariance,
        basis_change_classification_invariant=basis_classification,
        minimum_retained_singular_to_threshold_ratio=minimum_rank_margin,
        rank_tolerance_sweep_invariant=tolerance_invariant,
        rank_ambiguity_control_detected=rank_ambiguity_detected,
        declared_finite_one_loop_st_admission_gate_passed=passed,
    )
