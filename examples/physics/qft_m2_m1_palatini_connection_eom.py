'''Exact local torsion-free Palatini connection-EOM witness.

This module differentiates the E70-H Palatini density with respect to the
forty lower-symmetric affine-connection components, retains the first-
variation boundary current, traces the bulk equation to density
compatibility, and solves exact rational metric-jet fixtures.  It does not
cover unrestricted metric-affine/projective families, a global boundary
principle, full M1 BV, QME, or quantum M2.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
from math import isqrt

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
    polynomial_sum,
)
from examples.physics.qft_m2_m1_palatini_curvature_brst import (
    CONTRACT_SHA256 as E70_H_HASH,
    DIMENSION,
    JET_LOOKUP,
    MAXIMUM_TOTAL_JET_ORDER,
    MULTIINDICES,
    PalatiniJetOrderExceeded,
    ZERO_MULTIINDEX,
    _gamma_name,
    _h_name,
    evaluate_m1_palatini_curvature_brst_gate,
    generator,
    horizontal_derivative,
    m1_palatini_curvature_brst_contract,
    multi_total_derivative,
    palatini_curvature_brst_model,
    unit_multiindex,
    validate_contract as validate_e70_h_contract,
)


PRIMARY_SOURCE = 'arXiv:1306.4210'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/1306.4210'
PROJECTIVE_SOURCE = 'arXiv:1606.08756'
PROJECTIVE_SOURCE_URL = 'https://arxiv.org/abs/1606.08756'
PROJECTIVE_INVARIANCE_SOURCE = 'arXiv:1907.04137'
PROJECTIVE_INVARIANCE_SOURCE_URL = 'https://arxiv.org/abs/1907.04137'
FIRST_ORDER_SOURCE = 'hep-th/0609219'
FIRST_ORDER_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0609219'
SOURCE_ITEMS = (
    'arXiv:1306.4210: torsion-constrained Palatini and post-variation metric-affine procedures can differ',
    'arXiv:1606.08756: unrestricted Einstein--Hilbert--Palatini connections can retain an arbitrary-vector family',
    'arXiv:1907.04137: projective invariance belongs to the unrestricted metric-affine problem',
    'hep-th/0609219: d>2 first-order Einstein--Hilbert canonical precedent',
)
SOURCE_BOUNDARY = (
    'the certificate imposes a torsion-free lower-symmetric connection before '
    'variation and fixes the connection variation at the boundary; it does '
    'not transfer uniqueness to the unrestricted metric-affine/projective '
    'problem or prove a global first/second-order equivalence'
)
NORMALIZATION = (
    'all pointwise metric, density, connection, derivative, and coefficient '
    'coordinates are dimensionless exact rationals; the omitted M_P^2/2 '
    'factor does not affect the Euler or rank identities'
)
PALATINI_VARIATION_CONVENTION = (
    'torsion-free Gamma^lambda_mn=Gamma^lambda_nm is imposed before '
    'variation; delta Gamma vanishes on the boundary for this bulk gate'
)
CLAIM_CEILING = (
    'local classical torsion-free Palatini connection Euler equation, '
    'retained first-variation current, trace reduction to nabla h=0, and '
    'unique Levi--Civita reconstruction on exact positive-rho nondegenerate '
    'four-dimensional metric-density jets; no unrestricted projective '
    'classification, Palatini/GHY boundary completion, global equivalence, '
    'full M1 BV/CME, measure, QME, ST, Hilbert, HDA M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-H', E70_H_HASH),)
CONTRACT_SHA256 = (
    '729f5cb49d911013923af65e201d7ec838db12d06883eb57337f03c23a1196df'
)


ConnectionKey = tuple[int, int, int]
TensorKey = tuple[int, int, int]
NumericMatrix = tuple[tuple[Fraction, ...], ...]
NumericMetricJet = tuple[NumericMatrix, ...]


CONNECTION_KEYS: tuple[ConnectionKey, ...] = tuple(
    (upper, lower_left, lower_right)
    for upper in range(DIMENSION)
    for lower_left in range(DIMENSION)
    for lower_right in range(lower_left, DIMENSION)
)
SYMMETRIC_TENSOR_KEYS: tuple[TensorKey, ...] = tuple(
    (derivative, mu, nu)
    for derivative in range(DIMENSION)
    for mu in range(DIMENSION)
    for nu in range(mu, DIMENSION)
)


def even_jet_partial_derivative(
    polynomial: SparseSuperPolynomial,
    base_name: str,
    multiindex: tuple[int, int, int, int],
) -> SparseSuperPolynomial:
    variable = generator(base_name, multiindex)
    ((variable_even, _),) = variable.terms
    target = variable_even[0]
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        multiplicity = even_names.count(target)
        if not multiplicity:
            continue
        position = even_names.index(target)
        remaining = even_names[:position] + even_names[position + 1 :]
        result += SparseSuperPolynomial.monomial(
            even=remaining,
            odd=odd_names,
            coefficient=coefficient * multiplicity,
        )
    return result


def connection_euler_derivative(
    density: SparseSuperPolynomial,
    key: ConnectionKey,
) -> SparseSuperPolynomial:
    base_name = _gamma_name(*key)
    contributions: list[SparseSuperPolynomial] = []
    for multiindex in MULTIINDICES:
        partial = even_jet_partial_derivative(
            density,
            base_name,
            multiindex,
        )
        if partial.is_zero:
            continue
        integrated = multi_total_derivative(partial, multiindex)
        sign = -1 if sum(multiindex) % 2 else 1
        contributions.append(sign * integrated)
    return polynomial_sum(contributions)


def covariant_derivative_h(
    derivative: int,
    mu: int,
    nu: int,
    *,
    density_trace_coefficient: int = -1,
) -> SparseSuperPolynomial:
    if density_trace_coefficient not in (-1, 0, 1):
        raise ValueError('density trace coefficient must be -1, 0, or 1')
    result = horizontal_derivative(generator(_h_name(mu, nu)), derivative)
    result += polynomial_sum(
        generator(_gamma_name(mu, derivative, rho))
        * generator(_h_name(rho, nu))
        + generator(_gamma_name(nu, derivative, rho))
        * generator(_h_name(mu, rho))
        + density_trace_coefficient
        * generator(_gamma_name(rho, rho, derivative))
        * generator(_h_name(mu, nu))
        for rho in range(DIMENSION)
    )
    return result


def density_divergence_h(nu: int) -> SparseSuperPolynomial:
    return polynomial_sum(
        covariant_derivative_h(rho, nu, rho)
        for rho in range(DIMENSION)
    )


def analytic_connection_euler_component(
    upper: int,
    lower_left: int,
    lower_right: int,
) -> SparseSuperPolynomial:
    result = -covariant_derivative_h(upper, lower_left, lower_right)
    if upper == lower_left:
        result += Fraction(1, 2) * density_divergence_h(lower_right)
    if upper == lower_right:
        result += Fraction(1, 2) * density_divergence_h(lower_left)
    return result


def canonical_connection_euler_component(
    key: ConnectionKey,
) -> SparseSuperPolynomial:
    upper, lower_left, lower_right = key
    multiplicity = 1 if lower_left == lower_right else 2
    return multiplicity * analytic_connection_euler_component(*key)


def connection_euler_mismatches(
    density: SparseSuperPolynomial,
) -> tuple[SparseSuperPolynomial, ...]:
    return tuple(
        connection_euler_derivative(density, key)
        - canonical_connection_euler_component(key)
        for key in CONNECTION_KEYS
    )


def trace_reduction_residuals() -> tuple[SparseSuperPolynomial, ...]:
    factor = Fraction(DIMENSION - 1, 2)
    return tuple(
        polynomial_sum(
            analytic_connection_euler_component(mu, mu, nu)
            for mu in range(DIMENSION)
        )
        - factor * density_divergence_h(nu)
        for nu in range(DIMENSION)
    )


def _variation_generator(key: ConnectionKey, direction: int | None) -> SparseSuperPolynomial:
    suffix = 'base' if direction is None else f'd{direction}'
    return SparseSuperPolynomial.generator(
        f'delta_{_gamma_name(*key)}_{suffix}',
        odd=False,
    )


@dataclass(frozen=True)
class PalatiniFirstVariation:
    raw_variation: SparseSuperPolynomial
    bulk_variation: SparseSuperPolynomial
    boundary_current: tuple[SparseSuperPolynomial, ...]
    boundary_divergence: SparseSuperPolynomial
    decomposition_residual: SparseSuperPolynomial
    analytic_boundary_mismatch: SparseSuperPolynomial


def palatini_first_variation() -> PalatiniFirstVariation:
    density = palatini_curvature_brst_model().palatini_density
    raw_terms: list[SparseSuperPolynomial] = []
    bulk_terms: list[SparseSuperPolynomial] = []
    currents = [SparseSuperPolynomial.zero() for _ in range(DIMENSION)]
    divergence_terms: list[SparseSuperPolynomial] = []
    for key in CONNECTION_KEYS:
        base_name = _gamma_name(*key)
        eta = _variation_generator(key, None)
        partial_zero = even_jet_partial_derivative(
            density,
            base_name,
            ZERO_MULTIINDEX,
        )
        raw_terms.append(partial_zero * eta)
        euler = connection_euler_derivative(density, key)
        bulk_terms.append(euler * eta)
        for direction in range(DIMENSION):
            momentum = even_jet_partial_derivative(
                density,
                base_name,
                unit_multiindex(direction),
            )
            if momentum.is_zero:
                continue
            eta_derivative = _variation_generator(key, direction)
            raw_terms.append(momentum * eta_derivative)
            currents[direction] += momentum * eta
            divergence_terms.append(
                horizontal_derivative(momentum, direction) * eta
                + momentum * eta_derivative
            )

    analytic_currents = [SparseSuperPolynomial.zero() for _ in range(DIMENSION)]
    for direction in range(DIMENSION):
        analytic_currents[direction] += polynomial_sum(
            generator(_h_name(mu, nu))
            * _variation_generator((direction, min(mu, nu), max(mu, nu)), None)
            for mu in range(DIMENSION)
            for nu in range(DIMENSION)
        )
        analytic_currents[direction] -= polynomial_sum(
            generator(_h_name(mu, direction))
            * _variation_generator((rho, min(rho, mu), max(rho, mu)), None)
            for mu in range(DIMENSION)
            for rho in range(DIMENSION)
        )

    raw = polynomial_sum(raw_terms)
    bulk = polynomial_sum(bulk_terms)
    boundary_divergence = polynomial_sum(divergence_terms)
    analytic_mismatch = polynomial_sum(
        computed - expected
        for computed, expected in zip(currents, analytic_currents, strict=True)
    )
    return PalatiniFirstVariation(
        raw_variation=raw,
        bulk_variation=bulk,
        boundary_current=tuple(currents),
        boundary_divergence=boundary_divergence,
        decomposition_residual=raw - bulk - boundary_divergence,
        analytic_boundary_mismatch=analytic_mismatch,
    )


def _fraction_sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError('rational square root requires a nonnegative value')
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        raise ValueError('value is not an exact rational square')
    return Fraction(numerator, denominator)


def _matrix_determinant(matrix: NumericMatrix) -> Fraction:
    work = [list(row) for row in matrix]
    determinant = Fraction(1)
    for column in range(DIMENSION):
        pivot = next(
            (row for row in range(column, DIMENSION) if work[row][column]),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, DIMENSION):
            factor = work[row][column] / pivot_value
            for index in range(column, DIMENSION):
                work[row][index] -= factor * work[column][index]
    return determinant


def _matrix_inverse(matrix: NumericMatrix) -> NumericMatrix:
    size = len(matrix)
    work = [
        list(row) + [Fraction(int(i == j)) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError('matrix is singular')
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(work[row], work[column], strict=True)
                ]
    return tuple(tuple(row[size:]) for row in work)


def _matrix_rank(matrix: tuple[tuple[Fraction, ...], ...]) -> int:
    work = [list(row) for row in matrix]
    if not work:
        return 0
    row = 0
    for column in range(len(work[0])):
        pivot = next((index for index in range(row, len(work)) if work[index][column]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        pivot_value = work[row][column]
        work[row] = [value / pivot_value for value in work[row]]
        for other in range(len(work)):
            if other == row:
                continue
            factor = work[other][column]
            if factor:
                work[other] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(work[other], work[row], strict=True)
                ]
        row += 1
        if row == len(work):
            break
    return row


def _solve_square_system(
    matrix: tuple[tuple[Fraction, ...], ...],
    right_hand_side: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], int]:
    rank = _matrix_rank(matrix)
    size = len(matrix)
    if rank != size:
        raise ValueError('connection compatibility system is not full rank')
    inverse = _matrix_inverse(matrix)
    solution = tuple(
        sum(inverse[row][column] * right_hand_side[column] for column in range(size))
        for row in range(size)
    )
    return solution, rank


def _validate_metric_jet(
    metric: NumericMatrix,
    derivatives: NumericMetricJet,
) -> None:
    if len(metric) != DIMENSION or any(len(row) != DIMENSION for row in metric):
        raise ValueError('metric must be four by four')
    if len(derivatives) != DIMENSION:
        raise ValueError('metric jet needs four derivative matrices')
    for mu in range(DIMENSION):
        for nu in range(DIMENSION):
            if metric[mu][nu] != metric[nu][mu]:
                raise ValueError('metric must be symmetric')
            for direction in range(DIMENSION):
                derivative = derivatives[direction]
                if len(derivative) != DIMENSION or any(
                    len(row) != DIMENSION for row in derivative
                ):
                    raise ValueError('metric derivative must be four by four')
                if derivative[mu][nu] != derivative[nu][mu]:
                    raise ValueError('metric derivative must be symmetric')


@dataclass(frozen=True)
class MetricDensityJet:
    metric_covariant: NumericMatrix
    metric_contravariant: NumericMatrix
    metric_derivatives: NumericMetricJet
    rho: Fraction
    rho_derivatives: tuple[Fraction, ...]
    h_contravariant: NumericMatrix
    h_derivatives: NumericMetricJet


def metric_density_jet(
    metric: NumericMatrix,
    derivatives: NumericMetricJet,
) -> MetricDensityJet:
    _validate_metric_jet(metric, derivatives)
    determinant = _matrix_determinant(metric)
    if determinant >= 0:
        raise ValueError('fixture must use a nondegenerate Lorentz determinant')
    rho = _fraction_sqrt(-determinant)
    if rho <= 0:
        raise ValueError('positive-rho branch required')
    inverse = _matrix_inverse(metric)
    rho_derivatives = tuple(
        Fraction(1, 2)
        * rho
        * sum(
            inverse[mu][nu] * derivatives[direction][nu][mu]
            for mu in range(DIMENSION)
            for nu in range(DIMENSION)
        )
        for direction in range(DIMENSION)
    )
    h = tuple(
        tuple(rho * inverse[mu][nu] for nu in range(DIMENSION))
        for mu in range(DIMENSION)
    )
    h_derivatives = tuple(
        tuple(
            tuple(
                rho_derivatives[direction] * inverse[mu][nu]
                - rho
                * sum(
                    inverse[mu][alpha]
                    * derivatives[direction][alpha][beta]
                    * inverse[beta][nu]
                    for alpha in range(DIMENSION)
                    for beta in range(DIMENSION)
                )
                for nu in range(DIMENSION)
            )
            for mu in range(DIMENSION)
        )
        for direction in range(DIMENSION)
    )
    return MetricDensityJet(
        metric_covariant=metric,
        metric_contravariant=inverse,
        metric_derivatives=derivatives,
        rho=rho,
        rho_derivatives=rho_derivatives,
        h_contravariant=h,
        h_derivatives=h_derivatives,
    )


def compatibility_linear_system(
    h: NumericMatrix,
    h_derivatives: NumericMetricJet,
    *,
    density_trace_coefficient: int = -1,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    if density_trace_coefficient not in (-1, 0, 1):
        raise ValueError('density trace coefficient must be -1, 0, or 1')
    index_by_key = {key: index for index, key in enumerate(CONNECTION_KEYS)}
    rows: list[tuple[Fraction, ...]] = []
    right_hand_side: list[Fraction] = []
    for derivative, mu, nu in SYMMETRIC_TENSOR_KEYS:
        row = [Fraction(0) for _ in CONNECTION_KEYS]
        for rho in range(DIMENSION):
            first = (mu, min(derivative, rho), max(derivative, rho))
            second = (nu, min(derivative, rho), max(derivative, rho))
            trace = (rho, min(rho, derivative), max(rho, derivative))
            row[index_by_key[first]] += h[rho][nu]
            row[index_by_key[second]] += h[mu][rho]
            row[index_by_key[trace]] += (
                density_trace_coefficient * h[mu][nu]
            )
        rows.append(tuple(row))
        right_hand_side.append(-h_derivatives[derivative][mu][nu])
    return tuple(rows), tuple(right_hand_side)


def levi_civita_connection(jet: MetricDensityJet) -> tuple[Fraction, ...]:
    values: list[Fraction] = []
    inverse = jet.metric_contravariant
    derivatives = jet.metric_derivatives
    for upper, lower_left, lower_right in CONNECTION_KEYS:
        values.append(
            Fraction(1, 2)
            * sum(
                inverse[upper][rho]
                * (
                    derivatives[lower_left][rho][lower_right]
                    + derivatives[lower_right][rho][lower_left]
                    - derivatives[rho][lower_left][lower_right]
                )
                for rho in range(DIMENSION)
            )
        )
    return tuple(values)


def _system_residuals(
    matrix: tuple[tuple[Fraction, ...], ...],
    right_hand_side: tuple[Fraction, ...],
    solution: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, solution, strict=True))
        - target
        for row, target in zip(matrix, right_hand_side, strict=True)
    )


def _generic_symmetric_metric_derivatives(seed: int) -> NumericMetricJet:
    return tuple(
        tuple(
            tuple(
                Fraction(
                    (seed + direction + 1) * (min(mu, nu) + 1)
                    + (max(mu, nu) + 1) * (direction + 2),
                    seed + 7,
                )
                for nu in range(DIMENSION)
            )
            for mu in range(DIMENSION)
        )
        for direction in range(DIMENSION)
    )


def _zero_metric_derivatives() -> NumericMetricJet:
    return tuple(
        tuple(
            tuple(Fraction(0) for _ in range(DIMENSION))
            for _ in range(DIMENSION)
        )
        for _ in range(DIMENSION)
    )


def _congruence_metric(change: NumericMatrix) -> NumericMatrix:
    eta = (
        (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    return tuple(
        tuple(
            sum(
                change[alpha][mu]
                * eta[alpha][beta]
                * change[beta][nu]
                for alpha in range(DIMENSION)
                for beta in range(DIMENSION)
            )
            for nu in range(DIMENSION)
        )
        for mu in range(DIMENSION)
    )


@dataclass(frozen=True)
class MetricJetFixture:
    name: str
    metric_covariant: NumericMatrix
    metric_derivatives: NumericMetricJet


def metric_jet_fixtures() -> tuple[MetricJetFixture, ...]:
    minkowski = (
        (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    shear = (
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    e70_g_patch = (
        (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(4), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(4)),
    )
    return (
        MetricJetFixture('flat_zero_jet', minkowski, _zero_metric_derivatives()),
        MetricJetFixture('minkowski_generic_jet', minkowski, _generic_symmetric_metric_derivatives(1)),
        MetricJetFixture('unimodular_shear_generic_jet', _congruence_metric(shear), _generic_symmetric_metric_derivatives(3)),
        MetricJetFixture('e70_g_rho_four_generic_jet', e70_g_patch, _generic_symmetric_metric_derivatives(5)),
    )


@dataclass(frozen=True)
class MetricJetFixtureReceipt:
    name: str
    metric_determinant: Fraction
    rho: Fraction
    h_determinant: Fraction
    determinant_compatibility_residual: Fraction
    connection_system_rank: int
    solve_nonzero_residual_count: int
    levi_civita_nonzero_mismatch_count: int
    levi_civita_maximum_absolute_mismatch: Fraction


def evaluate_metric_jet_fixture(
    fixture: MetricJetFixture,
) -> MetricJetFixtureReceipt:
    jet = metric_density_jet(
        fixture.metric_covariant,
        fixture.metric_derivatives,
    )
    matrix, right_hand_side = compatibility_linear_system(
        jet.h_contravariant,
        jet.h_derivatives,
    )
    solution, rank = _solve_square_system(matrix, right_hand_side)
    solve_residuals = _system_residuals(matrix, right_hand_side, solution)
    levi_civita = levi_civita_connection(jet)
    mismatches = tuple(
        left - right for left, right in zip(solution, levi_civita, strict=True)
    )
    h_determinant = _matrix_determinant(jet.h_contravariant)
    return MetricJetFixtureReceipt(
        name=fixture.name,
        metric_determinant=_matrix_determinant(jet.metric_covariant),
        rho=jet.rho,
        h_determinant=h_determinant,
        determinant_compatibility_residual=h_determinant + jet.rho * jet.rho,
        connection_system_rank=rank,
        solve_nonzero_residual_count=sum(bool(value) for value in solve_residuals),
        levi_civita_nonzero_mismatch_count=sum(bool(value) for value in mismatches),
        levi_civita_maximum_absolute_mismatch=max(
            (abs(value) for value in mismatches),
            default=Fraction(0),
        ),
    )


def projective_shift_lower_symmetry_violation_count() -> int:
    vector = tuple(Fraction(index + 1) for index in range(DIMENSION))
    return sum(
        (Fraction(int(upper == nu)) * vector[mu])
        != (Fraction(int(upper == mu)) * vector[nu])
        for upper, mu, nu in product(range(DIMENSION), repeat=3)
    )


@dataclass(frozen=True)
class M1PalatiniConnectionEOMContract:
    primary_source: str
    primary_source_url: str
    projective_source: str
    projective_source_url: str
    projective_invariance_source: str
    projective_invariance_source_url: str
    first_order_source: str
    first_order_source_url: str
    source_items: tuple[str, ...]
    source_boundary: str
    normalization: str
    variation_convention: str
    dimension: int
    maximum_total_jet_order: int
    connection_keys: tuple[ConnectionKey, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    torsion_free_variation_preregistered: bool
    direct_connection_euler_computed: bool
    retained_boundary_current_computed: bool
    analytic_euler_match_computed: bool
    traced_equation_reduction_computed: bool
    positive_rho_metric_jet_fixtures_constructed: bool
    compatibility_linear_system_full_rank: bool
    unique_levi_civita_reconstruction_computed: bool
    projective_scope_controlled: bool
    live_negative_controls_computed: bool
    unrestricted_connection_variation_used: bool
    unrestricted_projective_family_classified: bool
    palatini_boundary_term_constructed: bool
    ghy_boundary_term_used: bool
    global_first_second_order_equivalence_proved: bool
    full_m1_action_assembled: bool
    full_m1_bv_functional_constructed: bool
    classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


_CONTRACT_FLAG_NAMES = (
    'torsion_free_variation_preregistered',
    'direct_connection_euler_computed',
    'retained_boundary_current_computed',
    'analytic_euler_match_computed',
    'traced_equation_reduction_computed',
    'positive_rho_metric_jet_fixtures_constructed',
    'compatibility_linear_system_full_rank',
    'unique_levi_civita_reconstruction_computed',
    'projective_scope_controlled',
    'live_negative_controls_computed',
    'unrestricted_connection_variation_used',
    'unrestricted_projective_family_classified',
    'palatini_boundary_term_constructed',
    'ghy_boundary_term_used',
    'global_first_second_order_equivalence_proved',
    'full_m1_action_assembled',
    'full_m1_bv_functional_constructed',
    'classical_master_equation_computed',
    'functional_measure_computed',
    'quantum_master_equation_computed',
    'continuum_loop_st_computed',
    'positive_physical_hilbert_proved',
    'quantum_hda_m2_proved',
    'm3_relational_observables_unlocked',
)


def m1_palatini_connection_eom_contract() -> M1PalatiniConnectionEOMContract:
    return M1PalatiniConnectionEOMContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        projective_source=PROJECTIVE_SOURCE,
        projective_source_url=PROJECTIVE_SOURCE_URL,
        projective_invariance_source=PROJECTIVE_INVARIANCE_SOURCE,
        projective_invariance_source_url=PROJECTIVE_INVARIANCE_SOURCE_URL,
        first_order_source=FIRST_ORDER_SOURCE,
        first_order_source_url=FIRST_ORDER_SOURCE_URL,
        source_items=SOURCE_ITEMS,
        source_boundary=SOURCE_BOUNDARY,
        normalization=NORMALIZATION,
        variation_convention=PALATINI_VARIATION_CONVENTION,
        dimension=DIMENSION,
        maximum_total_jet_order=MAXIMUM_TOTAL_JET_ORDER,
        connection_keys=CONNECTION_KEYS,
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        torsion_free_variation_preregistered=True,
        direct_connection_euler_computed=True,
        retained_boundary_current_computed=True,
        analytic_euler_match_computed=True,
        traced_equation_reduction_computed=True,
        positive_rho_metric_jet_fixtures_constructed=True,
        compatibility_linear_system_full_rank=True,
        unique_levi_civita_reconstruction_computed=True,
        projective_scope_controlled=True,
        live_negative_controls_computed=True,
        unrestricted_connection_variation_used=False,
        unrestricted_projective_family_classified=False,
        palatini_boundary_term_constructed=False,
        ghy_boundary_term_used=False,
        global_first_second_order_equivalence_proved=False,
        full_m1_action_assembled=False,
        full_m1_bv_functional_constructed=False,
        classical_master_equation_computed=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_local_torsion_free_palatini_connection_eom_and_'
            'levi_civita_reconstruction_not_global_bv_or_quantum_m2'
        ),
    )


def canonical_contract_payload(contract: M1PalatiniConnectionEOMContract) -> str:
    comma = chr(44)
    flags = comma.join(
        f'{name}:{getattr(contract, name)}' for name in _CONTRACT_FLAG_NAMES
    )
    return '|'.join(
        (
            f'primary={contract.primary_source}',
            f'primary_url={contract.primary_source_url}',
            f'projective={contract.projective_source}',
            f'projective_url={contract.projective_source_url}',
            f'projective_invariance={contract.projective_invariance_source}',
            f'projective_invariance_url={contract.projective_invariance_source_url}',
            f'first_order={contract.first_order_source}',
            f'first_order_url={contract.first_order_source_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_boundary={contract.source_boundary}',
            f'normalization={contract.normalization}',
            f'variation={contract.variation_convention}',
            f'dimension={contract.dimension}',
            f'max_total_jet={contract.maximum_total_jet_order}',
            f'connection_keys={comma.join(str(key) for key in contract.connection_keys)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(contract: M1PalatiniConnectionEOMContract) -> str:
    return hashlib.sha256(canonical_contract_payload(contract).encode('utf-8')).hexdigest()


def validate_contract(contract: M1PalatiniConnectionEOMContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.projective_source == PROJECTIVE_SOURCE,
        contract.projective_source_url == PROJECTIVE_SOURCE_URL,
        contract.projective_invariance_source == PROJECTIVE_INVARIANCE_SOURCE,
        contract.projective_invariance_source_url == PROJECTIVE_INVARIANCE_SOURCE_URL,
        contract.first_order_source == FIRST_ORDER_SOURCE,
        contract.first_order_source_url == FIRST_ORDER_SOURCE_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.source_boundary == SOURCE_BOUNDARY,
        contract.normalization == NORMALIZATION,
        contract.variation_convention == PALATINI_VARIATION_CONVENTION,
        contract.dimension == DIMENSION,
        contract.maximum_total_jet_order == MAXIMUM_TOTAL_JET_ORDER,
        contract.connection_keys == CONNECTION_KEYS,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == (
            'exact_local_torsion_free_palatini_connection_eom_and_'
            'levi_civita_reconstruction_not_global_bv_or_quantum_m2'
        ),
    )
    if not all(frozen):
        raise ValueError('Palatini connection-EOM source, basis, or status lock changed')
    if len(contract.connection_keys) != 40:
        raise ValueError('torsion-free connection variation requires 40 components')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('Palatini connection-EOM contract hash mismatch')
    if not all(getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[:10]):
        raise ValueError('required Palatini connection-EOM claim flag disabled')
    if any(getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]):
        raise ValueError('unsupported Palatini connection-EOM claim promoted')


@dataclass(frozen=True)
class M1PalatiniConnectionEOMReceipt:
    contract_sha256: str
    source_boundary: str
    normalization: str
    variation_convention: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_h_verified: bool
    torsion_free_connection_component_count: int
    palatini_density_term_count: int
    direct_connection_euler_component_count: int
    direct_analytic_euler_nonzero_mismatch_count: int
    direct_analytic_euler_maximum_mismatch_term_count: int
    first_variation_raw_term_count: int
    first_variation_bulk_term_count: int
    boundary_current_term_count: int
    boundary_current_divergence_term_count: int
    first_variation_decomposition_mismatch_term_count: int
    analytic_boundary_current_mismatch_term_count: int
    omitted_boundary_current_residual_term_count: int
    trace_equation_component_count: int
    trace_factor_numerator: int
    trace_factor_denominator: int
    trace_reduction_nonzero_residual_count: int
    metric_jet_fixture_count: int
    metric_jet_fixture_receipts: tuple[MetricJetFixtureReceipt, ...]
    minimum_connection_system_rank: int
    total_solve_nonzero_residual_count: int
    total_levi_civita_nonzero_mismatch_count: int
    maximum_levi_civita_absolute_mismatch: Fraction
    wrong_symmetrized_trace_nonzero_euler_component_count: int
    wrong_symmetrized_trace_total_mismatch_term_count: int
    wrong_symmetrized_trace_maximum_mismatch_term_count: int
    wrong_density_trace_connection_system_rank: int
    wrong_density_trace_levi_civita_nonzero_mismatch_count: int
    wrong_density_trace_correct_compatibility_nonzero_residual_count: int
    wrong_density_trace_maximum_absolute_mismatch: Fraction
    singular_h_connection_system_rank: int
    unrestricted_projective_shift_lower_symmetry_violation_count: int
    degenerate_metric_jet_rejected: bool
    nonsymmetric_metric_jet_rejected: bool
    terminal_jet_derivative_rejected: bool
    torsion_free_variation_preregistered: bool
    direct_connection_euler_computed: bool
    retained_boundary_current_computed: bool
    analytic_euler_match_computed: bool
    traced_equation_reduction_computed: bool
    positive_rho_metric_jet_fixtures_constructed: bool
    compatibility_linear_system_full_rank: bool
    unique_levi_civita_reconstruction_computed: bool
    projective_scope_controlled: bool
    live_negative_controls_computed: bool
    unrestricted_connection_variation_used: bool
    unrestricted_projective_family_classified: bool
    palatini_boundary_term_constructed: bool
    ghy_boundary_term_used: bool
    global_first_second_order_equivalence_proved: bool
    full_m1_action_assembled: bool
    full_m1_bv_functional_constructed: bool
    classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_palatini_connection_eom_gate_passed: bool


@lru_cache(maxsize=1)
def evaluate_m1_palatini_connection_eom_gate(
) -> M1PalatiniConnectionEOMReceipt:
    contract = m1_palatini_connection_eom_contract()
    validate_contract(contract)
    upstream_contract = m1_palatini_curvature_brst_contract()
    validate_e70_h_contract(upstream_contract)
    upstream_receipt = evaluate_m1_palatini_curvature_brst_gate()
    upstream_verified = (
        upstream_receipt.declared_m1_palatini_curvature_brst_gate_passed
    )

    density = palatini_curvature_brst_model().palatini_density
    euler_mismatches = connection_euler_mismatches(density)
    first_variation = palatini_first_variation()
    trace_residuals = trace_reduction_residuals()
    fixture_receipts = tuple(
        evaluate_metric_jet_fixture(fixture)
        for fixture in metric_jet_fixtures()
    )

    wrong_symmetrized_trace = tuple(
        connection_euler_derivative(density, key)
        - (1 if key[1] == key[2] else 2)
        * (-covariant_derivative_h(*key))
        for key in CONNECTION_KEYS
    )

    control_fixture = metric_jet_fixtures()[1]
    control_jet = metric_density_jet(
        control_fixture.metric_covariant,
        control_fixture.metric_derivatives,
    )
    wrong_matrix, wrong_right_hand_side = compatibility_linear_system(
        control_jet.h_contravariant,
        control_jet.h_derivatives,
        density_trace_coefficient=1,
    )
    wrong_solution, wrong_rank = _solve_square_system(
        wrong_matrix,
        wrong_right_hand_side,
    )
    control_levi_civita = levi_civita_connection(control_jet)
    wrong_levi_civita_mismatches = tuple(
        left - right
        for left, right in zip(
            wrong_solution,
            control_levi_civita,
            strict=True,
        )
    )
    correct_matrix, correct_right_hand_side = compatibility_linear_system(
        control_jet.h_contravariant,
        control_jet.h_derivatives,
    )
    wrong_solution_correct_residuals = _system_residuals(
        correct_matrix,
        correct_right_hand_side,
        wrong_solution,
    )

    zero = Fraction(0)
    singular_h = (
        (zero, zero, zero, zero),
        (zero, Fraction(1), zero, zero),
        (zero, zero, Fraction(1), zero),
        (zero, zero, zero, Fraction(1)),
    )
    singular_matrix, _ = compatibility_linear_system(
        singular_h,
        _zero_metric_derivatives(),
    )
    singular_rank = _matrix_rank(singular_matrix)

    degenerate_rejected = False
    try:
        metric_density_jet(singular_h, _zero_metric_derivatives())
    except ValueError:
        degenerate_rejected = True
    nonsymmetric = (
        (Fraction(-1), Fraction(1), zero, zero),
        (zero, Fraction(1), zero, zero),
        (zero, zero, Fraction(1), zero),
        (zero, zero, zero, Fraction(1)),
    )
    nonsymmetric_rejected = False
    try:
        metric_density_jet(nonsymmetric, _zero_metric_derivatives())
    except ValueError:
        nonsymmetric_rejected = True
    terminal_rejected = False
    try:
        horizontal_derivative(
            generator(
                'Gamma0_00',
                (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0),
            ),
            0,
        )
    except PalatiniJetOrderExceeded:
        terminal_rejected = True

    projective_violation = projective_shift_lower_symmetry_violation_count()
    omitted_boundary_residual = (
        first_variation.raw_variation - first_variation.bulk_variation
    )
    maximum_fixture_mismatch = max(
        receipt.levi_civita_maximum_absolute_mismatch
        for receipt in fixture_receipts
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]
    )
    passed = all(
        (
            upstream_verified,
            len(CONNECTION_KEYS) == 40,
            density.term_count == 276,
            len(euler_mismatches) == 40,
            max(value.term_count for value in euler_mismatches) == 0,
            first_variation.raw_variation.term_count == 456,
            first_variation.bulk_variation.term_count == 456,
            sum(
                value.term_count for value in first_variation.boundary_current
            )
            == 84,
            first_variation.boundary_divergence.term_count == 168,
            first_variation.decomposition_residual.is_zero,
            first_variation.analytic_boundary_mismatch.is_zero,
            omitted_boundary_residual.term_count == 168,
            len(trace_residuals) == 4,
            max(value.term_count for value in trace_residuals) == 0,
            len(fixture_receipts) == 4,
            min(value.connection_system_rank for value in fixture_receipts)
            == 40,
            sum(
                value.solve_nonzero_residual_count
                for value in fixture_receipts
            )
            == 0,
            sum(
                value.levi_civita_nonzero_mismatch_count
                for value in fixture_receipts
            )
            == 0,
            maximum_fixture_mismatch == 0,
            all(
                value.determinant_compatibility_residual == 0
                and value.rho > 0
                for value in fixture_receipts
            ),
            sum(not value.is_zero for value in wrong_symmetrized_trace) == 16,
            sum(value.term_count for value in wrong_symmetrized_trace) == 224,
            max(value.term_count for value in wrong_symmetrized_trace) == 14,
            wrong_rank == 40,
            sum(bool(value) for value in wrong_levi_civita_mismatches) == 28,
            sum(bool(value) for value in wrong_solution_correct_residuals) == 16,
            max(abs(value) for value in wrong_levi_civita_mismatches)
            == Fraction(5, 3),
            singular_rank == 33,
            projective_violation == 24,
            degenerate_rejected,
            nonsymmetric_rejected,
            terminal_rejected,
            not any(unsupported),
        )
    )

    return M1PalatiniConnectionEOMReceipt(
        contract_sha256=contract.contract_sha256,
        source_boundary=contract.source_boundary,
        normalization=contract.normalization,
        variation_convention=contract.variation_convention,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_h_verified=upstream_verified,
        torsion_free_connection_component_count=len(CONNECTION_KEYS),
        palatini_density_term_count=density.term_count,
        direct_connection_euler_component_count=len(euler_mismatches),
        direct_analytic_euler_nonzero_mismatch_count=sum(
            not value.is_zero for value in euler_mismatches
        ),
        direct_analytic_euler_maximum_mismatch_term_count=max(
            value.term_count for value in euler_mismatches
        ),
        first_variation_raw_term_count=first_variation.raw_variation.term_count,
        first_variation_bulk_term_count=first_variation.bulk_variation.term_count,
        boundary_current_term_count=sum(
            value.term_count for value in first_variation.boundary_current
        ),
        boundary_current_divergence_term_count=(
            first_variation.boundary_divergence.term_count
        ),
        first_variation_decomposition_mismatch_term_count=(
            first_variation.decomposition_residual.term_count
        ),
        analytic_boundary_current_mismatch_term_count=(
            first_variation.analytic_boundary_mismatch.term_count
        ),
        omitted_boundary_current_residual_term_count=(
            omitted_boundary_residual.term_count
        ),
        trace_equation_component_count=len(trace_residuals),
        trace_factor_numerator=DIMENSION - 1,
        trace_factor_denominator=2,
        trace_reduction_nonzero_residual_count=sum(
            not value.is_zero for value in trace_residuals
        ),
        metric_jet_fixture_count=len(fixture_receipts),
        metric_jet_fixture_receipts=fixture_receipts,
        minimum_connection_system_rank=min(
            value.connection_system_rank for value in fixture_receipts
        ),
        total_solve_nonzero_residual_count=sum(
            value.solve_nonzero_residual_count for value in fixture_receipts
        ),
        total_levi_civita_nonzero_mismatch_count=sum(
            value.levi_civita_nonzero_mismatch_count
            for value in fixture_receipts
        ),
        maximum_levi_civita_absolute_mismatch=maximum_fixture_mismatch,
        wrong_symmetrized_trace_nonzero_euler_component_count=sum(
            not value.is_zero for value in wrong_symmetrized_trace
        ),
        wrong_symmetrized_trace_total_mismatch_term_count=sum(
            value.term_count for value in wrong_symmetrized_trace
        ),
        wrong_symmetrized_trace_maximum_mismatch_term_count=max(
            value.term_count for value in wrong_symmetrized_trace
        ),
        wrong_density_trace_connection_system_rank=wrong_rank,
        wrong_density_trace_levi_civita_nonzero_mismatch_count=sum(
            bool(value) for value in wrong_levi_civita_mismatches
        ),
        wrong_density_trace_correct_compatibility_nonzero_residual_count=sum(
            bool(value) for value in wrong_solution_correct_residuals
        ),
        wrong_density_trace_maximum_absolute_mismatch=max(
            abs(value) for value in wrong_levi_civita_mismatches
        ),
        singular_h_connection_system_rank=singular_rank,
        unrestricted_projective_shift_lower_symmetry_violation_count=(
            projective_violation
        ),
        degenerate_metric_jet_rejected=degenerate_rejected,
        nonsymmetric_metric_jet_rejected=nonsymmetric_rejected,
        terminal_jet_derivative_rejected=terminal_rejected,
        torsion_free_variation_preregistered=(
            contract.torsion_free_variation_preregistered
        ),
        direct_connection_euler_computed=(
            contract.direct_connection_euler_computed
        ),
        retained_boundary_current_computed=(
            contract.retained_boundary_current_computed
        ),
        analytic_euler_match_computed=contract.analytic_euler_match_computed,
        traced_equation_reduction_computed=(
            contract.traced_equation_reduction_computed
        ),
        positive_rho_metric_jet_fixtures_constructed=(
            contract.positive_rho_metric_jet_fixtures_constructed
        ),
        compatibility_linear_system_full_rank=(
            contract.compatibility_linear_system_full_rank
        ),
        unique_levi_civita_reconstruction_computed=(
            contract.unique_levi_civita_reconstruction_computed
        ),
        projective_scope_controlled=contract.projective_scope_controlled,
        live_negative_controls_computed=contract.live_negative_controls_computed,
        unrestricted_connection_variation_used=(
            contract.unrestricted_connection_variation_used
        ),
        unrestricted_projective_family_classified=(
            contract.unrestricted_projective_family_classified
        ),
        palatini_boundary_term_constructed=(
            contract.palatini_boundary_term_constructed
        ),
        ghy_boundary_term_used=contract.ghy_boundary_term_used,
        global_first_second_order_equivalence_proved=(
            contract.global_first_second_order_equivalence_proved
        ),
        full_m1_action_assembled=contract.full_m1_action_assembled,
        full_m1_bv_functional_constructed=(
            contract.full_m1_bv_functional_constructed
        ),
        classical_master_equation_computed=(
            contract.classical_master_equation_computed
        ),
        functional_measure_computed=contract.functional_measure_computed,
        quantum_master_equation_computed=contract.quantum_master_equation_computed,
        continuum_loop_st_computed=contract.continuum_loop_st_computed,
        positive_physical_hilbert_proved=contract.positive_physical_hilbert_proved,
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        m3_relational_observables_unlocked=(
            contract.m3_relational_observables_unlocked
        ),
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_m1_palatini_connection_eom_gate_passed=passed,
    )
