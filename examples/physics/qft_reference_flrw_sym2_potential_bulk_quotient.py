'''Exact finite Eq. (17)--(18) potential traces and Eq. (22) bulk quotient.

The raw pointwise trace is kept distinct from the source-supplied bulk
representative.  This module does not complete boundary terms or derive the
minimal operator, heat kernel, determinant, loop integral, or renormalization.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    HTML_INTERNAL_HEADING,
    Matrix,
    Tensor4,
    audit_curvature,
    identity_matrix,
    kulkarni_nomizu_curvature,
    ricci_from_curvature,
    scalar_curvature,
    symmetric_fixture,
    vector_fixture,
    zero_vector,
)
from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_DATE,
    SOURCE_GAUGE,
    SOURCE_ID,
    SOURCE_THEORY,
    SOURCE_TITLE,
    SOURCE_URL,
)
from examples.physics.qft_reference_flrw_sym2_curvature_trace import (
    add_curvatures,
    curvature_squared,
    raw_symmetric_basis,
    symmetric_pairs,
    weyl_fixture,
    zero_matrix,
)


SOURCE_EQUATIONS = ('Eq17', 'Eq18', 'Eq22')
FIXTURE_DIMENSIONS = (3, 4, 5)
WEYL_FIXTURE_DIMENSIONS = (4, 5)
FRAME_CONVENTION = 'finite-Euclidean-orthonormal-normal-coordinate-point'
BUNDLE_FORMULA = 'B=Sym2(V)+scalar'
BASIS_FORMULA = (
    'pairs i<=j;B_ii[a,b]=delta_ai*delta_bi;'
    'B_ij[a,b]=delta_ai*delta_bj+delta_aj*delta_bi;extract(T)=T_ij'
)
DEWITT_FORMULA = (
    'C_abcd=(delta_ac*delta_bd+delta_ad*delta_bc-delta_ab*delta_cd)/4'
)
DEWITT_INVERSE_FORMULA = (
    'Cinv_abcd=delta_ac*delta_bd+delta_ad*delta_bc'
    '-2*delta_ab*delta_cd/(n-2)'
)
POTENTIAL_FORMULA = (
    'source-v7 Eq18 Yhh,Yhphi=H-delta*trH/2,Yphiphi=v^2'
)
HESSIAN_FIXTURE_FORMULA = (
    'H_ii=i+2;H_ij=1/(i+j+5) for zero-based i!=j'
)
BULK_QUOTIENT_FORMULA = (
    'tr((G^-1Y)^2)-Eq22Bulk=4*(H2-(trH)^2+Ric(v,v))'
)
SOURCE_TRANSCRIPTION_SHA256 = (
    '993123d20fc3f95d52d013fe7bdf7951867a6d8e2b40e53c662d93f42527af40'
)

PRIMITIVE_LENGTH_DIMENSIONS = {
    'BundleMetric': 0,
    'Curvature': -2,
    'Gradient': -1,
    'Hessian': -2,
}
QUANTITY_DIMENSION_FACTORS = {
    'BundleMetric': ('BundleMetric',),
    'TracePotentialCurvature': ('Curvature',),
    'TracePotentialGradientSquared': ('Gradient', 'Gradient'),
    'RiemannSq': ('Curvature', 'Curvature'),
    'RicciSq': ('Curvature', 'Curvature'),
    'RicciScalarSq': ('Curvature', 'Curvature'),
    'RicciGradientContraction': ('Curvature', 'Gradient', 'Gradient'),
    'RicciScalarGradientSquared': (
        'Curvature',
        'Gradient',
        'Gradient',
    ),
    'HessianSq': ('Hessian', 'Hessian'),
    'BoxPhiSq': ('Hessian', 'Hessian'),
    'ScalarGradientFourth': (
        'Gradient',
        'Gradient',
        'Gradient',
        'Gradient',
    ),
    'BulkDivergenceHessian': ('Hessian', 'Hessian'),
    'BulkDivergenceRicciGradient': (
        'Curvature',
        'Gradient',
        'Gradient',
    ),
    'TracePotentialSq': ('Curvature', 'Curvature'),
}
QUANTITY_DIMENSION_BASIS = tuple(QUANTITY_DIMENSION_FACTORS)


def _delta(left: int, right: int) -> Fraction:
    return Fraction(int(left == right))


def hessian_fixture(dimension: int) -> Matrix:
    if dimension < 2:
        raise ValueError('Hessian fixture dimension must be at least two')
    return tuple(
        tuple(
            Fraction(row + 2)
            if row == column
            else Fraction(1, row + column + 5)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def flat_hessian_fixture() -> Matrix:
    diagonal = (Fraction(1), Fraction(2), Fraction(4))
    return tuple(
        tuple(
            diagonal[row] if row == column else Fraction(0)
            for column in range(3)
        )
        for row in range(3)
    )


def matrix_trace(matrix: Matrix) -> Fraction:
    return sum(
        (matrix[index][index] for index in range(len(matrix))),
        Fraction(0),
    )


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    if len(left) != len(right):
        raise ValueError('matrix dimensions differ')
    dimension = len(left)
    if any(len(row) != dimension for row in left + right):
        raise ValueError('matrices must be square')
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(dimension)
                ),
                Fraction(0),
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def matrix_squared_trace(matrix: Matrix) -> Fraction:
    return matrix_trace(matrix_multiply(matrix, matrix))


def invert_matrix(matrix: Matrix) -> Matrix:
    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ValueError('matrix must be nonempty and square')
    augmented = [
        list(matrix[row])
        + [Fraction(int(row == column)) for column in range(dimension)]
        for row in range(dimension)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            raise ValueError('matrix is singular')
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                augmented[row][entry] - factor * augmented[column][entry]
                for entry in range(2 * dimension)
            ]
    return tuple(
        tuple(augmented[row][dimension:])
        for row in range(dimension)
    )


def matrix_identity_residual_l1(left: Matrix, right: Matrix) -> Fraction:
    product = matrix_multiply(left, right)
    dimension = len(product)
    return sum(
        (
            abs(product[row][column] - _delta(row, column))
            for row in range(dimension)
            for column in range(dimension)
        ),
        Fraction(0),
    )


def dewitt_component(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    trace_term_scale: Fraction = Fraction(1),
) -> Fraction:
    return Fraction(1, 4) * (
        _delta(a, c) * _delta(b, d)
        + _delta(a, d) * _delta(b, c)
        - trace_term_scale * _delta(a, b) * _delta(c, d)
    )


def _basis_support(
    dimension: int,
    pair: tuple[int, int],
    *,
    off_diagonal_scale: Fraction = Fraction(1),
) -> tuple[tuple[int, int, Fraction], ...]:
    tensor = raw_symmetric_basis(dimension, pair)
    return tuple(
        (
            row,
            column,
            tensor[row][column]
            * (
                off_diagonal_scale
                if pair[0] < pair[1]
                else Fraction(1)
            ),
        )
        for row in range(dimension)
        for column in range(dimension)
        if tensor[row][column] != 0
    )


def raw_dewitt_metric(
    dimension: int,
    *,
    trace_term_scale: Fraction = Fraction(1),
) -> Matrix:
    if dimension == 2:
        raise ValueError('DeWitt metric inverse has an n=2 pole')
    pairs = symmetric_pairs(dimension)
    rank = len(pairs) + 1
    data = [
        [Fraction(0) for _ in range(rank)]
        for _ in range(rank)
    ]
    for row, row_pair in enumerate(pairs):
        for column, column_pair in enumerate(pairs):
            data[row][column] = sum(
                (
                    left_value
                    * dewitt_component(
                        a,
                        b,
                        c,
                        d,
                        trace_term_scale=trace_term_scale,
                    )
                    * right_value
                    for a, b, left_value in _basis_support(
                        dimension, row_pair
                    )
                    for c, d, right_value in _basis_support(
                        dimension, column_pair
                    )
                ),
                Fraction(0),
            )
    data[-1][-1] = Fraction(1)
    return tuple(tuple(row) for row in data)


def derive_quantity_length_dimensions(
    primitive_overrides: dict[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_overrides is not None:
        dimensions.update(primitive_overrides)
    return tuple(
        sum(dimensions[factor] for factor in factors)
        for factors in QUANTITY_DIMENSION_FACTORS.values()
    )


def zero_curvature(dimension: int) -> Tensor4:
    return tuple(
        tuple(
            tuple(
                tuple(Fraction(0) for _ in range(dimension))
                for _ in range(dimension)
            )
            for _ in range(dimension)
        )
        for _ in range(dimension)
    )


def _vector_squared(vector: tuple[Fraction, ...]) -> Fraction:
    return sum((value**2 for value in vector), Fraction(0))


def _y_hh_component(
    curvature: Tensor4,
    ricci: Matrix,
    vector: tuple[Fraction, ...],
    mu: int,
    nu: int,
    rho: int,
    sigma: int,
) -> Fraction:
    ricci_scalar = scalar_curvature(ricci)
    gradient_squared = _vector_squared(vector)
    value = dewitt_component(mu, nu, rho, sigma) * (
        ricci_scalar - Fraction(1, 2) * gradient_squared
    )
    value -= Fraction(1, 2) * (
        curvature[mu][rho][nu][sigma]
        + curvature[nu][rho][mu][sigma]
    )
    value += Fraction(1, 2) * (
        _delta(mu, nu) * ricci[rho][sigma]
        + _delta(rho, sigma) * ricci[mu][nu]
    )
    value -= Fraction(1, 4) * (
        _delta(mu, rho) * ricci[nu][sigma]
        + _delta(mu, sigma) * ricci[nu][rho]
        + _delta(nu, rho) * ricci[mu][sigma]
        + _delta(nu, sigma) * ricci[mu][rho]
    )
    value -= Fraction(1, 4) * (
        _delta(mu, nu) * vector[rho] * vector[sigma]
        + _delta(rho, sigma) * vector[mu] * vector[nu]
        - _delta(mu, rho) * vector[nu] * vector[sigma]
        - _delta(mu, sigma) * vector[nu] * vector[rho]
        - _delta(nu, rho) * vector[mu] * vector[sigma]
        - _delta(nu, sigma) * vector[mu] * vector[rho]
    )
    return value


def raw_potential_matrix(
    curvature: Tensor4,
    vector: tuple[Fraction, ...],
    hessian: Matrix,
    *,
    off_diagonal_basis_scale: Fraction = Fraction(1),
    mixed_hessian_scale: Fraction = Fraction(1),
    mixed_trace_coefficient: Fraction = Fraction(-1, 2),
    mixed_upper_scale: Fraction = Fraction(1),
    mixed_lower_scale: Fraction = Fraction(1),
    scalar_block_scale: Fraction = Fraction(1),
    hh_component_delta: Fraction = Fraction(0),
) -> Matrix:
    dimension = len(curvature)
    if len(vector) != dimension or len(hessian) != dimension:
        raise ValueError('curvature, vector, and Hessian dimensions differ')
    if any(len(row) != dimension for row in hessian):
        raise ValueError('Hessian must be square')
    if any(
        hessian[row][column] != hessian[column][row]
        for row in range(dimension)
        for column in range(dimension)
    ):
        raise ValueError('Hessian must be symmetric')
    ricci = ricci_from_curvature(curvature)
    pairs = symmetric_pairs(dimension)
    rank = len(pairs) + 1
    data = [
        [Fraction(0) for _ in range(rank)]
        for _ in range(rank)
    ]
    supports = tuple(
        _basis_support(
            dimension,
            pair,
            off_diagonal_scale=off_diagonal_basis_scale,
        )
        for pair in pairs
    )
    for row, left_support in enumerate(supports):
        for column, right_support in enumerate(supports):
            data[row][column] = sum(
                (
                    left_value
                    * _y_hh_component(
                        curvature,
                        ricci,
                        vector,
                        mu,
                        nu,
                        rho,
                        sigma,
                    )
                    * right_value
                    for mu, nu, left_value in left_support
                    for rho, sigma, right_value in right_support
                ),
                Fraction(0),
            )
    hessian_trace = matrix_trace(hessian)
    for index, support in enumerate(supports):
        mixed = sum(
            (
                value
                * (
                    mixed_hessian_scale * hessian[a][b]
                    + mixed_trace_coefficient
                    * _delta(a, b)
                    * hessian_trace
                )
                for a, b, value in support
            ),
            Fraction(0),
        )
        data[index][-1] = mixed_upper_scale * mixed
        data[-1][index] = mixed_lower_scale * mixed
    data[-1][-1] = scalar_block_scale * _vector_squared(vector)
    data[0][0] += hh_component_delta
    return tuple(tuple(row) for row in data)


@dataclass(frozen=True)
class PotentialInvariants:
    riemann_squared: Fraction
    ricci_squared: Fraction
    ricci_scalar: Fraction
    scalar_gradient_squared: Fraction
    ricci_gradient_contraction: Fraction
    hessian_squared: Fraction
    box_phi: Fraction


def potential_invariants(
    curvature: Tensor4,
    vector: tuple[Fraction, ...],
    hessian: Matrix,
) -> PotentialInvariants:
    dimension = len(curvature)
    ricci = ricci_from_curvature(curvature)
    return PotentialInvariants(
        riemann_squared=curvature_squared(curvature),
        ricci_squared=sum(
            (
                ricci[row][column] ** 2
                for row in range(dimension)
                for column in range(dimension)
            ),
            Fraction(0),
        ),
        ricci_scalar=scalar_curvature(ricci),
        scalar_gradient_squared=_vector_squared(vector),
        ricci_gradient_contraction=sum(
            (
                ricci[row][column] * vector[row] * vector[column]
                for row in range(dimension)
                for column in range(dimension)
            ),
            Fraction(0),
        ),
        hessian_squared=sum(
            (
                hessian[row][column] ** 2
                for row in range(dimension)
                for column in range(dimension)
            ),
            Fraction(0),
        ),
        box_phi=matrix_trace(hessian),
    )


def trace_potential_target(
    dimension: int, invariants: PotentialInvariants
) -> Fraction:
    return (
        Fraction(dimension * (dimension - 1), 2)
        * invariants.ricci_scalar
        + Fraction(8 + 3 * dimension - dimension**2, 4)
        * invariants.scalar_gradient_squared
    )


def source_eq22_bulk_potential_squared(
    dimension: int, invariants: PotentialInvariants
) -> Fraction:
    if dimension == 2:
        raise ValueError('source Eq. (22) has an n=2 pole')
    n = dimension
    e = invariants.riemann_squared
    q = invariants.ricci_squared
    r = invariants.ricci_scalar
    x = invariants.scalar_gradient_squared
    p = invariants.ricci_gradient_contraction
    z = invariants.box_phi
    return (
        3 * e
        + Fraction(n**2 - 8 * n + 4, n - 2) * q
        + Fraction(n**3 - 5 * n**2 + 8 * n + 4, 2 * (n - 2))
        * r**2
        - (Fraction(2 * n * (n - 4), n - 2) + 4) * p
        + Fraction(n**3 - 7 * n**2 + 10 * n + 8, 2 * (2 - n))
        * r
        * x
        + 2 * z**2
        + Fraction(n**3 - n**2 + 14 * n - 40, 8 * (n - 2))
        * x**2
    )


def bulk_divergence(invariants: PotentialInvariants) -> Fraction:
    return (
        invariants.hessian_squared
        - invariants.box_phi**2
        + invariants.ricci_gradient_contraction
    )


@dataclass(frozen=True)
class PotentialTraceValues:
    trace_potential: Fraction
    trace_potential_squared_raw: Fraction


def potential_trace_values(
    curvature: Tensor4,
    vector: tuple[Fraction, ...],
    hessian: Matrix,
    *,
    inverse_metric: Matrix | None = None,
    **potential_options: Fraction,
) -> PotentialTraceValues:
    dimension = len(curvature)
    metric_inverse = (
        invert_matrix(raw_dewitt_metric(dimension))
        if inverse_metric is None
        else inverse_metric
    )
    potential = raw_potential_matrix(
        curvature,
        vector,
        hessian,
        **potential_options,
    )
    mixed = matrix_multiply(metric_inverse, potential)
    return PotentialTraceValues(
        trace_potential=matrix_trace(mixed),
        trace_potential_squared_raw=matrix_squared_trace(mixed),
    )


@dataclass(frozen=True)
class Sym2PotentialBulkContract:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_equations: tuple[str, ...]
    frame_convention: str
    fixture_dimensions: tuple[int, ...]
    weyl_fixture_dimensions: tuple[int, ...]
    bundle_formula: str
    basis_formula: str
    dewitt_formula: str
    dewitt_inverse_formula: str
    potential_formula: str
    hessian_fixture_formula: str
    bulk_quotient_formula: str
    source_transcription_sha256: str
    raw_potential_traces_computed: bool
    bulk_quotient_applied: bool
    mixed_linear_sign_determined: bool
    source_eq22_pointwise_identity_proved: bool
    integration_by_parts_source_explicit: bool
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    derivation_status: str
    finite_boundary_completed: bool
    endpoint_terms_computed: bool
    eq18_operator_derived: bool
    gauge_fixing_derived: bool
    functional_determinant_derived: bool
    heat_kernel_trace_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    evanescent_terms_controlled: bool
    independent_source_artifact_authenticated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool


def sym2_potential_bulk_contract() -> Sym2PotentialBulkContract:
    return Sym2PotentialBulkContract(
        source_id=SOURCE_ID,
        source_date=SOURCE_DATE,
        source_metadata_title=SOURCE_TITLE,
        html_internal_heading=HTML_INTERNAL_HEADING,
        source_theory=SOURCE_THEORY,
        source_gauge=SOURCE_GAUGE,
        source_url=SOURCE_URL,
        source_equations=SOURCE_EQUATIONS,
        frame_convention=FRAME_CONVENTION,
        fixture_dimensions=FIXTURE_DIMENSIONS,
        weyl_fixture_dimensions=WEYL_FIXTURE_DIMENSIONS,
        bundle_formula=BUNDLE_FORMULA,
        basis_formula=BASIS_FORMULA,
        dewitt_formula=DEWITT_FORMULA,
        dewitt_inverse_formula=DEWITT_INVERSE_FORMULA,
        potential_formula=POTENTIAL_FORMULA,
        hessian_fixture_formula=HESSIAN_FIXTURE_FORMULA,
        bulk_quotient_formula=BULK_QUOTIENT_FORMULA,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        raw_potential_traces_computed=True,
        bulk_quotient_applied=True,
        mixed_linear_sign_determined=False,
        source_eq22_pointwise_identity_proved=False,
        integration_by_parts_source_explicit=False,
        source_lorentzian_sign_extended=False,
        background_eom_used=False,
        derivation_status='finite_sym2_potential_bulk_quotient_only',
        finite_boundary_completed=False,
        endpoint_terms_computed=False,
        eq18_operator_derived=False,
        gauge_fixing_derived=False,
        functional_determinant_derived=False,
        heat_kernel_trace_derived=False,
        fp_determinant_derived=False,
        ghost_weight_derived=False,
        loop_integral_evaluated=False,
        regularization_scheme_implemented=False,
        evanescent_terms_controlled=False,
        independent_source_artifact_authenticated=False,
        renormalization_proof=False,
        continuum_st_qme_proved=False,
        local_covariance_proved=False,
        in_in_ctp_completed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
    )


def canonical_source_payload(contract: Sym2PotentialBulkContract) -> str:
    separator = chr(44)
    return '|'.join(
        (
            contract.source_id,
            contract.source_date,
            f'metadata_title={contract.source_metadata_title}',
            f'html_heading={contract.html_internal_heading}',
            f'theory={contract.source_theory}',
            f'gauge={contract.source_gauge}',
            f'equations={separator.join(contract.source_equations)}',
            f'frame={contract.frame_convention}',
            'dimensions='
            + separator.join(str(value) for value in contract.fixture_dimensions),
            'weyl_dimensions='
            + separator.join(
                str(value) for value in contract.weyl_fixture_dimensions
            ),
            f'bundle={contract.bundle_formula}',
            f'basis={contract.basis_formula}',
            f'dewitt={contract.dewitt_formula}',
            f'dewitt_inverse={contract.dewitt_inverse_formula}',
            f'potential={contract.potential_formula}',
            f'H_fixture={contract.hessian_fixture_formula}',
            f'bulk_quotient={contract.bulk_quotient_formula}',
        )
    )


def source_payload_sha256(contract: Sym2PotentialBulkContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: Sym2PotentialBulkContract) -> None:
    frozen = (
        contract.source_id == SOURCE_ID,
        contract.source_date == SOURCE_DATE,
        contract.source_metadata_title == SOURCE_TITLE,
        contract.html_internal_heading == HTML_INTERNAL_HEADING,
        contract.source_theory == SOURCE_THEORY,
        contract.source_gauge == SOURCE_GAUGE,
        contract.source_url == SOURCE_URL,
        contract.source_equations == SOURCE_EQUATIONS,
        contract.frame_convention == FRAME_CONVENTION,
        contract.fixture_dimensions == FIXTURE_DIMENSIONS,
        contract.weyl_fixture_dimensions == WEYL_FIXTURE_DIMENSIONS,
        contract.bundle_formula == BUNDLE_FORMULA,
        contract.basis_formula == BASIS_FORMULA,
        contract.dewitt_formula == DEWITT_FORMULA,
        contract.dewitt_inverse_formula == DEWITT_INVERSE_FORMULA,
        contract.potential_formula == POTENTIAL_FORMULA,
        contract.hessian_fixture_formula == HESSIAN_FIXTURE_FORMULA,
        contract.bulk_quotient_formula == BULK_QUOTIENT_FORMULA,
    )
    if not all(frozen):
        raise ValueError('source, frame, metric, potential, or fixture changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local Sym2 potential transcription hash mismatch')
    if not (
        contract.raw_potential_traces_computed
        and contract.bulk_quotient_applied
    ):
        raise ValueError('raw traces and declared bulk quotient must be computed')
    if contract.derivation_status != 'finite_sym2_potential_bulk_quotient_only':
        raise ValueError('this gate is finite potential bulk quotient only')
    required_false = (
        contract.mixed_linear_sign_determined,
        contract.source_eq22_pointwise_identity_proved,
        contract.integration_by_parts_source_explicit,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.finite_boundary_completed,
        contract.endpoint_terms_computed,
        contract.eq18_operator_derived,
        contract.gauge_fixing_derived,
        contract.functional_determinant_derived,
        contract.heat_kernel_trace_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.evanescent_terms_controlled,
        contract.independent_source_artifact_authenticated,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    if any(required_false):
        raise ValueError('a claim beyond the finite bulk quotient was enabled')


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


@dataclass(frozen=True)
class Sym2PotentialBulkReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_equations: tuple[str, ...]
    source_transcription_sha256: str
    local_transcription_lock_passed: bool
    frame_convention: str
    fixture_dimensions: tuple[int, ...]
    weyl_fixture_dimensions: tuple[int, ...]
    bundle_ranks: tuple[int, ...]
    metric_inverse_residuals_l1: tuple[str, ...]
    metric_inverse_all_passed: bool
    n2_pole_rejected: bool
    fixture_count: int
    generic_vector_fixture_count: int
    zero_vector_fixture_count: int
    weyl_added_fixture_count: int
    flat_hessian_fixture_count: int
    curvature_audit_count: int
    curvature_audits_all_passed: bool
    weyl_fixtures_nonzero_and_ricci_flat: bool
    potential_matrices_symmetric: bool
    generic_invariants_live: bool
    zero_vector_limits_all_passed: bool
    flat_divergence: str
    flat_raw_minus_bulk: str
    exact_trace_residuals: tuple[str, ...]
    exact_trace_component_count: int
    exact_trace_relations_all_passed: bool
    wrong_dewitt_metric_mismatch_l1: str
    euclidean_raw_metric_mismatch_l1: str
    off_diagonal_basis_mismatch_l1: str
    corrupted_yhh_component_mismatch_l1: str
    omitted_mixed_blocks_mismatch_l1: str
    wrong_relative_mixed_sign_mismatch_l1: str
    wrong_hessian_trace_sign_mismatch_l1: str
    omitted_scalar_block_mismatch_l1: str
    trace_square_confusion_mismatch_l1: str
    wrong_ricci_divergence_sign_mismatch_l1: str
    wrong_quotient_coefficient_mismatch_l1: str
    forced_pointwise_identity_mismatch_l1: str
    dropped_weyl_mismatch_l1: str
    n4_coefficient_copy_mismatch_l1: str
    simultaneous_mixed_sign_flip_squared_trace_residual: str
    mixed_linear_sign_determined: bool
    primitive_length_dimensions: tuple[int, ...]
    quantity_dimension_basis: tuple[str, ...]
    quantity_length_dimensions: tuple[int, ...]
    corrupted_gradient_length_dimensions: tuple[int, ...]
    corrupted_hessian_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    raw_potential_traces_computed: bool
    bulk_quotient_applied: bool
    derivation_status: str
    source_eq22_pointwise_identity_proved: bool
    integration_by_parts_source_explicit: bool
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    finite_boundary_completed: bool
    endpoint_terms_computed: bool
    eq18_operator_derived: bool
    gauge_fixing_derived: bool
    functional_determinant_derived: bool
    heat_kernel_trace_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    evanescent_terms_controlled: bool
    independent_source_artifact_authenticated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    declared_finite_sym2_potential_bulk_gate_passed: bool


@dataclass(frozen=True)
class _PotentialFixture:
    dimension: int
    kind: str
    curvature: Tensor4
    generic_curvature: Tensor4
    vector: tuple[Fraction, ...]
    hessian: Matrix


def _trace_relation_mismatch(
    values: PotentialTraceValues,
    dimension: int,
    invariants: PotentialInvariants,
) -> Fraction:
    return abs(
        values.trace_potential
        - trace_potential_target(dimension, invariants)
    ) + abs(
        values.trace_potential_squared_raw
        - source_eq22_bulk_potential_squared(dimension, invariants)
        - 4 * bulk_divergence(invariants)
    )


def evaluate_sym2_potential_bulk_gate() -> Sym2PotentialBulkReceipt:
    contract = sym2_potential_bulk_contract()
    validate_contract(contract)

    metrics: dict[int, Matrix] = {}
    inverse_metrics: dict[int, Matrix] = {}
    wrong_inverse_metrics: dict[int, Matrix] = {}
    inverse_residuals: list[Fraction] = []
    for dimension in contract.fixture_dimensions:
        metric = raw_dewitt_metric(dimension)
        inverse = invert_matrix(metric)
        metrics[dimension] = metric
        inverse_metrics[dimension] = inverse
        inverse_residuals.append(
            matrix_identity_residual_l1(inverse, metric)
        )
        wrong_inverse_metrics[dimension] = invert_matrix(
            raw_dewitt_metric(dimension, trace_term_scale=Fraction(2))
        )

    zero_invariants = PotentialInvariants(
        riemann_squared=Fraction(0),
        ricci_squared=Fraction(0),
        ricci_scalar=Fraction(0),
        scalar_gradient_squared=Fraction(0),
        ricci_gradient_contraction=Fraction(0),
        hessian_squared=Fraction(0),
        box_phi=Fraction(0),
    )
    n2_metric_rejected = False
    n2_source_rejected = False
    try:
        raw_dewitt_metric(2)
    except ValueError:
        n2_metric_rejected = True
    try:
        source_eq22_bulk_potential_squared(2, zero_invariants)
    except ValueError:
        n2_source_rejected = True
    n2_pole_rejected = n2_metric_rejected and n2_source_rejected

    fixtures: list[_PotentialFixture] = []
    curvature_audits = []
    weyl_checks: list[bool] = []
    for dimension in contract.fixture_dimensions:
        symmetric = symmetric_fixture(dimension)
        generic_curvature = kulkarni_nomizu_curvature(symmetric)
        generic_ricci = ricci_from_curvature(generic_curvature)
        curvature_audits.append(
            audit_curvature(generic_curvature, generic_ricci, symmetric)
        )
        vector = vector_fixture(dimension)
        hessian = hessian_fixture(dimension)
        fixtures.append(
            _PotentialFixture(
                dimension,
                'generic-vector',
                generic_curvature,
                generic_curvature,
                vector,
                hessian,
            )
        )
        fixtures.append(
            _PotentialFixture(
                dimension,
                'generic-zero-vector',
                generic_curvature,
                generic_curvature,
                zero_vector(dimension),
                hessian,
            )
        )
        if dimension in contract.weyl_fixture_dimensions:
            weyl = weyl_fixture(dimension)
            zero_symmetric = zero_matrix(dimension)
            weyl_ricci = ricci_from_curvature(weyl)
            curvature_audits.append(
                audit_curvature(weyl, weyl_ricci, zero_symmetric)
            )
            weyl_checks.append(
                curvature_squared(weyl) > 0
                and weyl_ricci == zero_symmetric
            )
            combined = add_curvatures(generic_curvature, weyl)
            combined_ricci = ricci_from_curvature(combined)
            curvature_audits.append(
                audit_curvature(combined, combined_ricci, symmetric)
            )
            fixtures.append(
                _PotentialFixture(
                    dimension,
                    'weyl-added',
                    combined,
                    generic_curvature,
                    vector,
                    hessian,
                )
            )

    fixtures.append(
        _PotentialFixture(
            3,
            'flat-hessian',
            zero_curvature(3),
            zero_curvature(3),
            zero_vector(3),
            flat_hessian_fixture(),
        )
    )

    exact_residuals: list[Fraction] = []
    potential_symmetry_checks: list[bool] = []
    generic_liveness: list[bool] = []
    zero_vector_checks: list[bool] = []
    flat_divergence_value: Fraction | None = None
    flat_raw_minus_bulk_value: Fraction | None = None

    wrong_dewitt_control = Fraction(0)
    euclidean_metric_control = Fraction(0)
    off_diagonal_control = Fraction(0)
    corrupted_yhh_control = Fraction(0)
    omitted_mixed_control = Fraction(0)
    wrong_relative_mixed_control = Fraction(0)
    wrong_hessian_trace_control = Fraction(0)
    omitted_scalar_control = Fraction(0)
    trace_square_control = Fraction(0)
    wrong_ricci_divergence_control = Fraction(0)
    wrong_quotient_control = Fraction(0)
    forced_pointwise_control = Fraction(0)
    dropped_weyl_control = Fraction(0)
    n4_copy_control = Fraction(0)
    simultaneous_mixed_sign_residual = Fraction(0)

    for fixture in fixtures:
        dimension = fixture.dimension
        invariants = potential_invariants(
            fixture.curvature,
            fixture.vector,
            fixture.hessian,
        )
        inverse_metric = inverse_metrics[dimension]
        correct = potential_trace_values(
            fixture.curvature,
            fixture.vector,
            fixture.hessian,
            inverse_metric=inverse_metric,
        )
        trace_target = trace_potential_target(dimension, invariants)
        bulk_target = source_eq22_bulk_potential_squared(
            dimension, invariants
        )
        divergence = bulk_divergence(invariants)
        exact_residuals.extend(
            (
                correct.trace_potential - trace_target,
                correct.trace_potential_squared_raw
                - bulk_target
                - 4 * divergence,
            )
        )

        potential = raw_potential_matrix(
            fixture.curvature,
            fixture.vector,
            fixture.hessian,
        )
        potential_symmetry_checks.append(
            all(
                potential[row][column] == potential[column][row]
                for row in range(len(potential))
                for column in range(len(potential))
            )
        )
        if fixture.kind == 'generic-vector':
            generic_liveness.append(
                all(
                    value != 0
                    for value in (
                        invariants.riemann_squared,
                        invariants.ricci_squared,
                        invariants.ricci_scalar,
                        invariants.scalar_gradient_squared,
                        invariants.ricci_gradient_contraction,
                        invariants.hessian_squared,
                        invariants.box_phi,
                        divergence,
                    )
                )
            )
        if fixture.kind == 'generic-zero-vector':
            zero_vector_checks.append(
                invariants.scalar_gradient_squared == 0
                and invariants.ricci_gradient_contraction == 0
            )
        if fixture.kind == 'flat-hessian':
            flat_divergence_value = divergence
            flat_raw_minus_bulk_value = (
                correct.trace_potential_squared_raw - bulk_target
            )

        wrong_dewitt_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=wrong_inverse_metrics[dimension],
            ),
            dimension,
            invariants,
        )
        euclidean_metric_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=identity_matrix(len(inverse_metric)),
            ),
            dimension,
            invariants,
        )
        off_diagonal_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                off_diagonal_basis_scale=Fraction(2),
            ),
            dimension,
            invariants,
        )
        corrupted_yhh_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                hh_component_delta=Fraction(1),
            ),
            dimension,
            invariants,
        )
        omitted_mixed_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                mixed_upper_scale=Fraction(0),
                mixed_lower_scale=Fraction(0),
            ),
            dimension,
            invariants,
        )
        wrong_relative_mixed_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                mixed_lower_scale=Fraction(-1),
            ),
            dimension,
            invariants,
        )
        wrong_hessian_trace_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                mixed_trace_coefficient=Fraction(1, 2),
            ),
            dimension,
            invariants,
        )
        omitted_scalar_control += _trace_relation_mismatch(
            potential_trace_values(
                fixture.curvature,
                fixture.vector,
                fixture.hessian,
                inverse_metric=inverse_metric,
                scalar_block_scale=Fraction(0),
            ),
            dimension,
            invariants,
        )
        trace_square_control += abs(
            correct.trace_potential**2
            - bulk_target
            - 4 * divergence
        )
        wrong_divergence = (
            invariants.hessian_squared
            - invariants.box_phi**2
            - invariants.ricci_gradient_contraction
        )
        wrong_ricci_divergence_control += abs(
            correct.trace_potential_squared_raw
            - bulk_target
            - 4 * wrong_divergence
        )
        wrong_quotient_control += abs(
            correct.trace_potential_squared_raw
            - bulk_target
            - 3 * divergence
        )
        forced_pointwise_control += abs(
            correct.trace_potential_squared_raw - bulk_target
        )
        simultaneous_sign = potential_trace_values(
            fixture.curvature,
            fixture.vector,
            fixture.hessian,
            inverse_metric=inverse_metric,
            mixed_upper_scale=Fraction(-1),
            mixed_lower_scale=Fraction(-1),
        )
        simultaneous_mixed_sign_residual += abs(
            simultaneous_sign.trace_potential - correct.trace_potential
        ) + abs(
            simultaneous_sign.trace_potential_squared_raw
            - correct.trace_potential_squared_raw
        )
        if fixture.kind == 'weyl-added':
            dropped_weyl_control += _trace_relation_mismatch(
                potential_trace_values(
                    fixture.generic_curvature,
                    fixture.vector,
                    fixture.hessian,
                    inverse_metric=inverse_metric,
                ),
                dimension,
                invariants,
            )
        if dimension != 4:
            n4_copy_control += abs(
                correct.trace_potential
                - trace_potential_target(4, invariants)
            ) + abs(
                correct.trace_potential_squared_raw
                - source_eq22_bulk_potential_squared(4, invariants)
                - 4 * divergence
            )

    controls = (
        wrong_dewitt_control,
        euclidean_metric_control,
        off_diagonal_control,
        corrupted_yhh_control,
        omitted_mixed_control,
        wrong_relative_mixed_control,
        wrong_hessian_trace_control,
        omitted_scalar_control,
        trace_square_control,
        wrong_ricci_divergence_control,
        wrong_quotient_control,
        forced_pointwise_control,
        dropped_weyl_control,
        n4_copy_control,
    )
    primitive_dimensions = tuple(
        PRIMITIVE_LENGTH_DIMENSIONS[name]
        for name in ('BundleMetric', 'Curvature', 'Gradient', 'Hessian')
    )
    quantity_dimensions = derive_quantity_length_dimensions()
    corrupted_gradient_dimensions = derive_quantity_length_dimensions(
        {'Gradient': 0}
    )
    corrupted_hessian_dimensions = derive_quantity_length_dimensions(
        {'Hessian': -1}
    )
    expected_dimensions = (0, -2, -2) + (-4,) * 11
    expected_corrupted_gradient = (
        0,
        -2,
        0,
        -4,
        -4,
        -4,
        -2,
        -2,
        -4,
        -4,
        0,
        -4,
        -2,
        -4,
    )
    expected_corrupted_hessian = (
        0,
        -2,
        -2,
        -4,
        -4,
        -4,
        -4,
        -4,
        -2,
        -2,
        -4,
        -2,
        -4,
        -4,
    )
    dimension_gate = (
        primitive_dimensions == (0, -2, -1, -2)
        and quantity_dimensions == expected_dimensions
        and corrupted_gradient_dimensions == expected_corrupted_gradient
        and corrupted_hessian_dimensions == expected_corrupted_hessian
        and corrupted_gradient_dimensions != quantity_dimensions
        and corrupted_hessian_dimensions != quantity_dimensions
    )
    required_false = (
        contract.mixed_linear_sign_determined,
        contract.source_eq22_pointwise_identity_proved,
        contract.integration_by_parts_source_explicit,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.finite_boundary_completed,
        contract.endpoint_terms_computed,
        contract.eq18_operator_derived,
        contract.gauge_fixing_derived,
        contract.functional_determinant_derived,
        contract.heat_kernel_trace_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.evanescent_terms_controlled,
        contract.independent_source_artifact_authenticated,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    source_lock = source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    exact_passed = all(value == 0 for value in exact_residuals)
    geometry_passed = all(audit.passed for audit in curvature_audits)
    weyl_passed = len(weyl_checks) == 2 and all(weyl_checks)
    flat_passed = (
        flat_divergence_value == -28
        and flat_raw_minus_bulk_value == -112
    )
    gate_passed = (
        source_lock
        and tuple(len(metric) for metric in metrics.values()) == (7, 11, 16)
        and all(value == 0 for value in inverse_residuals)
        and n2_pole_rejected
        and len(fixtures) == 9
        and len(curvature_audits) == 7
        and geometry_passed
        and weyl_passed
        and all(potential_symmetry_checks)
        and len(generic_liveness) == 3
        and all(generic_liveness)
        and len(zero_vector_checks) == 3
        and all(zero_vector_checks)
        and flat_passed
        and len(exact_residuals) == 18
        and exact_passed
        and all(value > 0 for value in controls)
        and simultaneous_mixed_sign_residual == 0
        and dimension_gate
        and contract.raw_potential_traces_computed
        and contract.bulk_quotient_applied
        and not any(required_false)
    )

    return Sym2PotentialBulkReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        html_internal_heading=contract.html_internal_heading,
        source_equations=contract.source_equations,
        source_transcription_sha256=contract.source_transcription_sha256,
        local_transcription_lock_passed=source_lock,
        frame_convention=contract.frame_convention,
        fixture_dimensions=contract.fixture_dimensions,
        weyl_fixture_dimensions=contract.weyl_fixture_dimensions,
        bundle_ranks=tuple(len(metric) for metric in metrics.values()),
        metric_inverse_residuals_l1=tuple(
            _fraction_text(value) for value in inverse_residuals
        ),
        metric_inverse_all_passed=all(
            value == 0 for value in inverse_residuals
        ),
        n2_pole_rejected=n2_pole_rejected,
        fixture_count=len(fixtures),
        generic_vector_fixture_count=sum(
            fixture.kind == 'generic-vector' for fixture in fixtures
        ),
        zero_vector_fixture_count=sum(
            fixture.kind == 'generic-zero-vector' for fixture in fixtures
        ),
        weyl_added_fixture_count=sum(
            fixture.kind == 'weyl-added' for fixture in fixtures
        ),
        flat_hessian_fixture_count=sum(
            fixture.kind == 'flat-hessian' for fixture in fixtures
        ),
        curvature_audit_count=len(curvature_audits),
        curvature_audits_all_passed=geometry_passed,
        weyl_fixtures_nonzero_and_ricci_flat=weyl_passed,
        potential_matrices_symmetric=all(potential_symmetry_checks),
        generic_invariants_live=all(generic_liveness),
        zero_vector_limits_all_passed=all(zero_vector_checks),
        flat_divergence=_fraction_text(
            Fraction(0)
            if flat_divergence_value is None
            else flat_divergence_value
        ),
        flat_raw_minus_bulk=_fraction_text(
            Fraction(0)
            if flat_raw_minus_bulk_value is None
            else flat_raw_minus_bulk_value
        ),
        exact_trace_residuals=tuple(
            _fraction_text(value) for value in exact_residuals
        ),
        exact_trace_component_count=len(exact_residuals),
        exact_trace_relations_all_passed=exact_passed,
        wrong_dewitt_metric_mismatch_l1=_fraction_text(
            wrong_dewitt_control
        ),
        euclidean_raw_metric_mismatch_l1=_fraction_text(
            euclidean_metric_control
        ),
        off_diagonal_basis_mismatch_l1=_fraction_text(
            off_diagonal_control
        ),
        corrupted_yhh_component_mismatch_l1=_fraction_text(
            corrupted_yhh_control
        ),
        omitted_mixed_blocks_mismatch_l1=_fraction_text(
            omitted_mixed_control
        ),
        wrong_relative_mixed_sign_mismatch_l1=_fraction_text(
            wrong_relative_mixed_control
        ),
        wrong_hessian_trace_sign_mismatch_l1=_fraction_text(
            wrong_hessian_trace_control
        ),
        omitted_scalar_block_mismatch_l1=_fraction_text(
            omitted_scalar_control
        ),
        trace_square_confusion_mismatch_l1=_fraction_text(
            trace_square_control
        ),
        wrong_ricci_divergence_sign_mismatch_l1=_fraction_text(
            wrong_ricci_divergence_control
        ),
        wrong_quotient_coefficient_mismatch_l1=_fraction_text(
            wrong_quotient_control
        ),
        forced_pointwise_identity_mismatch_l1=_fraction_text(
            forced_pointwise_control
        ),
        dropped_weyl_mismatch_l1=_fraction_text(dropped_weyl_control),
        n4_coefficient_copy_mismatch_l1=_fraction_text(n4_copy_control),
        simultaneous_mixed_sign_flip_squared_trace_residual=_fraction_text(
            simultaneous_mixed_sign_residual
        ),
        mixed_linear_sign_determined=contract.mixed_linear_sign_determined,
        primitive_length_dimensions=primitive_dimensions,
        quantity_dimension_basis=QUANTITY_DIMENSION_BASIS,
        quantity_length_dimensions=quantity_dimensions,
        corrupted_gradient_length_dimensions=(
            corrupted_gradient_dimensions
        ),
        corrupted_hessian_length_dimensions=(
            corrupted_hessian_dimensions
        ),
        dimension_gate_passed=dimension_gate,
        raw_potential_traces_computed=contract.raw_potential_traces_computed,
        bulk_quotient_applied=contract.bulk_quotient_applied,
        derivation_status=contract.derivation_status,
        source_eq22_pointwise_identity_proved=(
            contract.source_eq22_pointwise_identity_proved
        ),
        integration_by_parts_source_explicit=(
            contract.integration_by_parts_source_explicit
        ),
        source_lorentzian_sign_extended=(
            contract.source_lorentzian_sign_extended
        ),
        background_eom_used=contract.background_eom_used,
        finite_boundary_completed=contract.finite_boundary_completed,
        endpoint_terms_computed=contract.endpoint_terms_computed,
        eq18_operator_derived=contract.eq18_operator_derived,
        gauge_fixing_derived=contract.gauge_fixing_derived,
        functional_determinant_derived=contract.functional_determinant_derived,
        heat_kernel_trace_derived=contract.heat_kernel_trace_derived,
        fp_determinant_derived=contract.fp_determinant_derived,
        ghost_weight_derived=contract.ghost_weight_derived,
        loop_integral_evaluated=contract.loop_integral_evaluated,
        regularization_scheme_implemented=(
            contract.regularization_scheme_implemented
        ),
        evanescent_terms_controlled=contract.evanescent_terms_controlled,
        independent_source_artifact_authenticated=(
            contract.independent_source_artifact_authenticated
        ),
        renormalization_proof=contract.renormalization_proof,
        continuum_st_qme_proved=contract.continuum_st_qme_proved,
        local_covariance_proved=contract.local_covariance_proved,
        in_in_ctp_completed=contract.in_in_ctp_completed,
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        declared_finite_sym2_potential_bulk_gate_passed=gate_passed,
    )
