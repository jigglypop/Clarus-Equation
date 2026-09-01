'''Finite exact Sym^2(V)+scalar curvature traces for source-v7 Eq. (20)--(22).

This module constructs local Euclidean representation matrices on declared
rational algebraic-curvature fixtures.  It does not derive the minimal
operator, potential traces, determinant, heat kernel, loop integral, or
renormalization.
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
    symmetric_fixture,
)
from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_DATE,
    SOURCE_GAUGE,
    SOURCE_ID,
    SOURCE_THEORY,
    SOURCE_TITLE,
    SOURCE_URL,
)


SOURCE_EQUATIONS = ('Eq20', 'Eq21', 'Eq22')
FIXTURE_DIMENSIONS = (3, 4, 5)
WEYL_FIXTURE_DIMENSIONS = (4, 5)
FRAME_CONVENTION = 'finite-Euclidean-orthonormal'
BUNDLE_FORMULA = 'B=Sym2(V)+scalar'
BASIS_FORMULA = (
    'pairs i<=j;B_ii[a,b]=delta_ai*delta_bi;'
    'B_ij[a,b]=delta_ai*delta_bj+delta_aj*delta_bi;extract(T)=T_ij'
)
CURVATURE_ACTION_FORMULA = (
    '(W_mn h)[a,b]=R[a,c,m,n]h[c,b]+R[b,c,m,n]h[a,c];scalar=0'
)
GENERIC_CURVATURE_FORMULA = 'R=delta wedge S with locked rational S fixture'
WEYL_FIXTURE_FORMULA = (
    'C_1212=1;C_1313=-1;C_2424=-1;C_3434=1;Riemann symmetries;'
    'embed first four axes for n=5'
)
SOURCE_TRANSCRIPTION_SHA256 = (
    '23826e568c1fd9e995437e9fb088f23372e7ca97167003b19a4016989e70e1a7'
)

PRIMITIVE_LENGTH_DIMENSIONS = {
    'Identity': 0,
    'Curvature': -2,
}
TRACE_DIMENSION_FACTORS = {
    'IdentityTrace': ('Identity',),
    'RiemannSq': ('Curvature', 'Curvature'),
    'BundleCurvatureSqTrace': ('Curvature', 'Curvature'),
}
TRACE_DIMENSION_BASIS = tuple(TRACE_DIMENSION_FACTORS)


def _delta(left: int, right: int) -> Fraction:
    return Fraction(int(left == right))


def zero_matrix(dimension: int) -> Matrix:
    return tuple(
        tuple(Fraction(0) for _ in range(dimension))
        for _ in range(dimension)
    )


def symmetric_pairs(dimension: int) -> tuple[tuple[int, int], ...]:
    if dimension < 2:
        raise ValueError('representation dimension must be at least two')
    return tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left, dimension)
    )


def raw_symmetric_basis(
    dimension: int, pair: tuple[int, int]
) -> Matrix:
    left, right = pair
    if not (0 <= left <= right < dimension):
        raise ValueError('raw symmetric basis pair is out of range')
    return tuple(
        tuple(
            _delta(row, left) * _delta(column, right)
            + (
                Fraction(0)
                if left == right
                else _delta(row, right) * _delta(column, left)
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def bundle_rank(dimension: int) -> int:
    return dimension * (dimension + 1) // 2 + 1


def bundle_identity_matrix(
    dimension: int, *, include_scalar: bool = True
) -> Matrix:
    rank = bundle_rank(dimension) if include_scalar else bundle_rank(dimension) - 1
    return identity_matrix(rank)


def _zero_tensor4(dimension: int) -> list[list[list[list[Fraction]]]]:
    return [
        [
            [
                [Fraction(0) for _ in range(dimension)]
                for _ in range(dimension)
            ]
            for _ in range(dimension)
        ]
        for _ in range(dimension)
    ]


def _freeze_tensor4(
    data: list[list[list[list[Fraction]]]],
) -> Tensor4:
    return tuple(
        tuple(
            tuple(tuple(data[a][b][c]) for c in range(len(data)))
            for b in range(len(data))
        )
        for a in range(len(data))
    )


def weyl_fixture(dimension: int) -> Tensor4:
    if dimension not in WEYL_FIXTURE_DIMENSIONS:
        raise ValueError('the locked Weyl fixture is defined only for n=4,5')
    data = _zero_tensor4(dimension)
    sections = (
        (0, 1, Fraction(1)),
        (0, 2, Fraction(-1)),
        (1, 3, Fraction(-1)),
        (2, 3, Fraction(1)),
    )
    for left, right, value in sections:
        data[left][right][left][right] = value
        data[left][right][right][left] = -value
        data[right][left][left][right] = -value
        data[right][left][right][left] = value
    return _freeze_tensor4(data)


def add_curvatures(left: Tensor4, right: Tensor4) -> Tensor4:
    if len(left) != len(right):
        raise ValueError('curvature dimensions differ')
    dimension = len(left)
    return tuple(
        tuple(
            tuple(
                tuple(
                    left[a][b][c][d] + right[a][b][c][d]
                    for d in range(dimension)
                )
                for c in range(dimension)
            )
            for b in range(dimension)
        )
        for a in range(dimension)
    )


def curvature_squared(curvature: Tensor4) -> Fraction:
    dimension = len(curvature)
    return sum(
        (
            curvature[a][b][c][d] ** 2
            for a in range(dimension)
            for b in range(dimension)
            for c in range(dimension)
            for d in range(dimension)
        ),
        Fraction(0),
    )


def _curvature_action_component(
    curvature: Tensor4,
    a: int,
    c: int,
    mu: int,
    nu: int,
    *,
    wrong_index_placement: bool,
) -> Fraction:
    if wrong_index_placement:
        return curvature[a][mu][c][nu]
    return curvature[a][c][mu][nu]


def symmetric_curvature_action(
    curvature: Tensor4,
    mu: int,
    nu: int,
    tensor: Matrix,
    *,
    action_scale: Fraction = Fraction(1),
    first_slot_scale: Fraction = Fraction(1),
    second_slot_scale: Fraction = Fraction(1),
    wrong_index_placement: bool = False,
) -> Matrix:
    dimension = len(curvature)
    if len(tensor) != dimension or any(
        len(row) != dimension for row in tensor
    ):
        raise ValueError('tensor and curvature dimensions differ')
    return tuple(
        tuple(
            action_scale
            * sum(
                (
                    first_slot_scale
                    * _curvature_action_component(
                        curvature,
                        a,
                        c,
                        mu,
                        nu,
                        wrong_index_placement=wrong_index_placement,
                    )
                    * tensor[c][b]
                    + second_slot_scale
                    * _curvature_action_component(
                        curvature,
                        b,
                        c,
                        mu,
                        nu,
                        wrong_index_placement=wrong_index_placement,
                    )
                    * tensor[a][c]
                    for c in range(dimension)
                ),
                Fraction(0),
            )
            for b in range(dimension)
        )
        for a in range(dimension)
    )


def bundle_curvature_matrix(
    curvature: Tensor4,
    mu: int,
    nu: int,
    *,
    action_scale: Fraction = Fraction(1),
    first_slot_scale: Fraction = Fraction(1),
    second_slot_scale: Fraction = Fraction(1),
    off_diagonal_extraction_scale: Fraction = Fraction(1),
    wrong_index_placement: bool = False,
) -> Matrix:
    dimension = len(curvature)
    pairs = symmetric_pairs(dimension)
    rank = len(pairs) + 1
    data = [
        [Fraction(0) for _ in range(rank)]
        for _ in range(rank)
    ]
    for column, input_pair in enumerate(pairs):
        output = symmetric_curvature_action(
            curvature,
            mu,
            nu,
            raw_symmetric_basis(dimension, input_pair),
            action_scale=action_scale,
            first_slot_scale=first_slot_scale,
            second_slot_scale=second_slot_scale,
            wrong_index_placement=wrong_index_placement,
        )
        for row, (left, right) in enumerate(pairs):
            scale = (
                off_diagonal_extraction_scale
                if left < right
                else Fraction(1)
            )
            data[row][column] = scale * output[left][right]
    return tuple(tuple(row) for row in data)


def matrix_product_trace(left: Matrix, right: Matrix) -> Fraction:
    if len(left) != len(right):
        raise ValueError('matrix dimensions differ')
    dimension = len(left)
    return sum(
        (
            left[row][column] * right[column][row]
            for row in range(dimension)
            for column in range(dimension)
        ),
        Fraction(0),
    )


def matrix_frobenius_squared(matrix: Matrix) -> Fraction:
    return sum(
        (value**2 for row in matrix for value in row),
        Fraction(0),
    )


def bundle_curvature_squared_trace(
    curvature: Tensor4,
    *,
    action_scale: Fraction = Fraction(1),
    first_slot_scale: Fraction = Fraction(1),
    second_slot_scale: Fraction = Fraction(1),
    off_diagonal_extraction_scale: Fraction = Fraction(1),
    wrong_index_placement: bool = False,
    linear_sign: Fraction = Fraction(1),
    frobenius: bool = False,
) -> Fraction:
    dimension = len(curvature)
    total = Fraction(0)
    for mu in range(dimension):
        for nu in range(dimension):
            matrix = bundle_curvature_matrix(
                curvature,
                mu,
                nu,
                action_scale=linear_sign * action_scale,
                first_slot_scale=first_slot_scale,
                second_slot_scale=second_slot_scale,
                off_diagonal_extraction_scale=(
                    off_diagonal_extraction_scale
                ),
                wrong_index_placement=wrong_index_placement,
            )
            total += (
                matrix_frobenius_squared(matrix)
                if frobenius
                else matrix_product_trace(matrix, matrix)
            )
    return total


def scalar_curvature_blocks_are_zero(curvature: Tensor4) -> bool:
    dimension = len(curvature)
    scalar_index = bundle_rank(dimension) - 1
    return all(
        matrix[scalar_index][index] == 0
        and matrix[index][scalar_index] == 0
        for mu in range(dimension)
        for nu in range(dimension)
        for matrix in (bundle_curvature_matrix(curvature, mu, nu),)
        for index in range(bundle_rank(dimension))
    )


def derive_trace_length_dimensions(
    primitive_overrides: dict[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_overrides is not None:
        dimensions.update(primitive_overrides)
    return tuple(
        sum(dimensions[factor] for factor in factors)
        for factors in TRACE_DIMENSION_FACTORS.values()
    )


def raw_symmetric_coordinates(tensor: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        tensor[left][right]
        for left, right in symmetric_pairs(len(tensor))
    )


def raw_basis_roundtrip_passed(dimension: int) -> bool:
    pairs = symmetric_pairs(dimension)
    for column, pair in enumerate(pairs):
        coordinates = raw_symmetric_coordinates(
            raw_symmetric_basis(dimension, pair)
        )
        if coordinates != tuple(
            Fraction(int(row == column)) for row in range(len(pairs))
        ):
            return False
    return True


def _replace_curvature_component(
    curvature: Tensor4,
    indices: tuple[int, int, int, int],
    delta: Fraction,
) -> Tensor4:
    dimension = len(curvature)
    data = [
        [
            [
                [curvature[a][b][c][d] for d in range(dimension)]
                for c in range(dimension)
            ]
            for b in range(dimension)
        ]
        for a in range(dimension)
    ]
    a, b, c, d = indices
    data[a][b][c][d] += delta
    return _freeze_tensor4(data)


@dataclass(frozen=True)
class Sym2CurvatureTraceContract:
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
    curvature_action_formula: str
    generic_curvature_formula: str
    weyl_fixture_formula: str
    source_transcription_sha256: str
    finite_sym2_bundle_curvature_traces_computed: bool
    w_linear_sign_determined: bool
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    derivation_status: str
    eq22_trY_derived: bool
    eq22_trY2_derived: bool
    eq18_operator_derived: bool
    gauge_fixing_derived: bool
    functional_determinant_derived: bool
    heat_kernel_trace_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    finite_boundary_completed: bool
    evanescent_terms_controlled: bool
    independent_source_artifact_authenticated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool


def sym2_curvature_trace_contract() -> Sym2CurvatureTraceContract:
    return Sym2CurvatureTraceContract(
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
        curvature_action_formula=CURVATURE_ACTION_FORMULA,
        generic_curvature_formula=GENERIC_CURVATURE_FORMULA,
        weyl_fixture_formula=WEYL_FIXTURE_FORMULA,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        finite_sym2_bundle_curvature_traces_computed=True,
        w_linear_sign_determined=False,
        source_lorentzian_sign_extended=False,
        background_eom_used=False,
        derivation_status='finite_sym2_bundle_curvature_trace_only',
        eq22_trY_derived=False,
        eq22_trY2_derived=False,
        eq18_operator_derived=False,
        gauge_fixing_derived=False,
        functional_determinant_derived=False,
        heat_kernel_trace_derived=False,
        fp_determinant_derived=False,
        ghost_weight_derived=False,
        loop_integral_evaluated=False,
        regularization_scheme_implemented=False,
        finite_boundary_completed=False,
        evanescent_terms_controlled=False,
        independent_source_artifact_authenticated=False,
        renormalization_proof=False,
        continuum_st_qme_proved=False,
        local_covariance_proved=False,
        in_in_ctp_completed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
    )


def canonical_source_payload(contract: Sym2CurvatureTraceContract) -> str:
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
            f'action={contract.curvature_action_formula}',
            f'generic_curvature={contract.generic_curvature_formula}',
            f'weyl_fixture={contract.weyl_fixture_formula}',
        )
    )


def source_payload_sha256(contract: Sym2CurvatureTraceContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: Sym2CurvatureTraceContract) -> None:
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
        contract.curvature_action_formula == CURVATURE_ACTION_FORMULA,
        contract.generic_curvature_formula == GENERIC_CURVATURE_FORMULA,
        contract.weyl_fixture_formula == WEYL_FIXTURE_FORMULA,
    )
    if not all(frozen):
        raise ValueError('source, frame, representation, or fixture contract changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local Sym2 curvature-trace transcription hash mismatch')
    if not contract.finite_sym2_bundle_curvature_traces_computed:
        raise ValueError('the declared finite representation traces must be computed')
    if contract.derivation_status != 'finite_sym2_bundle_curvature_trace_only':
        raise ValueError('this gate is finite Sym2 curvature trace only')
    required_false = (
        contract.w_linear_sign_determined,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.eq22_trY_derived,
        contract.eq22_trY2_derived,
        contract.eq18_operator_derived,
        contract.gauge_fixing_derived,
        contract.functional_determinant_derived,
        contract.heat_kernel_trace_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.finite_boundary_completed,
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
        raise ValueError('a claim beyond the finite representation trace was enabled')


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


@dataclass(frozen=True)
class Sym2CurvatureTraceReceipt:
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
    raw_basis_roundtrip_all_passed: bool
    generic_fixture_count: int
    weyl_added_fixture_count: int
    curvature_audit_count: int
    curvature_audits_all_passed: bool
    weyl_fixtures_nonzero_and_ricci_flat: bool
    corrupted_curvature_rejected: bool
    scalar_curvature_blocks_all_zero: bool
    exact_trace_residuals: tuple[str, ...]
    exact_trace_component_count: int
    exact_trace_contractions_all_passed: bool
    missing_scalar_identity_mismatch_l1: str
    off_diagonal_normalization_mismatch_l1: str
    half_action_mismatch_l1: str
    omitted_second_slot_mismatch_l1: str
    wrong_relative_slot_sign_mismatch_l1: str
    wrong_curvature_index_mismatch_l1: str
    frobenius_vs_matrix_trace_mismatch_l1: str
    dropped_weyl_mismatch_l1: str
    omitted_generic_fixture_magnitude: str
    w_linear_sign_flip_squared_trace_residual: str
    w_linear_sign_determined: bool
    primitive_length_dimensions: tuple[int, ...]
    trace_dimension_basis: tuple[str, ...]
    trace_length_dimensions: tuple[int, ...]
    corrupted_trace_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    finite_sym2_bundle_curvature_traces_computed: bool
    derivation_status: str
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    eq22_trY_derived: bool
    eq22_trY2_derived: bool
    eq18_operator_derived: bool
    gauge_fixing_derived: bool
    functional_determinant_derived: bool
    heat_kernel_trace_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    finite_boundary_completed: bool
    evanescent_terms_controlled: bool
    independent_source_artifact_authenticated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    declared_finite_sym2_curvature_trace_gate_passed: bool


@dataclass(frozen=True)
class _CurvatureFixture:
    dimension: int
    kind: str
    curvature: Tensor4
    generic_curvature: Tensor4


def evaluate_sym2_curvature_trace_gate() -> Sym2CurvatureTraceReceipt:
    contract = sym2_curvature_trace_contract()
    validate_contract(contract)

    identity_residuals: list[Fraction] = []
    curvature_residuals: list[Fraction] = []
    curvature_audits = []
    weyl_checks: list[bool] = []
    fixtures: list[_CurvatureFixture] = []
    scalar_block_checks: list[bool] = []
    basis_checks: list[bool] = []

    missing_scalar_identity = Fraction(0)
    off_diagonal_control = Fraction(0)
    half_action_control = Fraction(0)
    omitted_second_slot_control = Fraction(0)
    wrong_relative_sign_control = Fraction(0)
    wrong_index_control = Fraction(0)
    frobenius_control = Fraction(0)
    dropped_weyl_control = Fraction(0)
    generic_fixture_magnitude = Fraction(0)
    w_sign_flip_residual = Fraction(0)

    for dimension in contract.fixture_dimensions:
        basis_checks.append(raw_basis_roundtrip_passed(dimension))
        identity = bundle_identity_matrix(dimension)
        identity_trace = sum(
            (identity[index][index] for index in range(len(identity))),
            Fraction(0),
        )
        identity_residuals.append(identity_trace - bundle_rank(dimension))
        identity_without_scalar = bundle_identity_matrix(
            dimension, include_scalar=False
        )
        missing_scalar_identity += abs(
            sum(
                (
                    identity_without_scalar[index][index]
                    for index in range(len(identity_without_scalar))
                ),
                Fraction(0),
            )
            - bundle_rank(dimension)
        )

        symmetric = symmetric_fixture(dimension)
        generic_curvature = kulkarni_nomizu_curvature(symmetric)
        generic_ricci = ricci_from_curvature(generic_curvature)
        curvature_audits.append(
            audit_curvature(generic_curvature, generic_ricci, symmetric)
        )
        generic_fixture_magnitude += curvature_squared(generic_curvature)
        fixtures.append(
            _CurvatureFixture(
                dimension=dimension,
                kind='generic',
                curvature=generic_curvature,
                generic_curvature=generic_curvature,
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
                _CurvatureFixture(
                    dimension=dimension,
                    kind='weyl-added',
                    curvature=combined,
                    generic_curvature=generic_curvature,
                )
            )

    for fixture in fixtures:
        dimension = fixture.dimension
        curvature = fixture.curvature
        expected = -(dimension + 2) * curvature_squared(curvature)
        correct = bundle_curvature_squared_trace(curvature)
        curvature_residuals.append(correct - expected)
        scalar_block_checks.append(
            scalar_curvature_blocks_are_zero(curvature)
        )
        off_diagonal_control += abs(
            bundle_curvature_squared_trace(
                curvature,
                off_diagonal_extraction_scale=Fraction(2),
            )
            - expected
        )
        half_action_control += abs(
            bundle_curvature_squared_trace(
                curvature, action_scale=Fraction(1, 2)
            )
            - expected
        )
        omitted_second_slot_control += abs(
            bundle_curvature_squared_trace(
                curvature, second_slot_scale=Fraction(0)
            )
            - expected
        )
        wrong_relative_sign_control += abs(
            bundle_curvature_squared_trace(
                curvature, second_slot_scale=Fraction(-1)
            )
            - expected
        )
        wrong_index_control += abs(
            bundle_curvature_squared_trace(
                curvature, wrong_index_placement=True
            )
            - expected
        )
        frobenius_control += abs(
            bundle_curvature_squared_trace(curvature, frobenius=True)
            - expected
        )
        w_sign_flip_residual += abs(
            bundle_curvature_squared_trace(
                curvature, linear_sign=Fraction(-1)
            )
            - correct
        )
        if fixture.kind == 'weyl-added':
            dropped_weyl_control += abs(
                bundle_curvature_squared_trace(fixture.generic_curvature)
                - expected
            )

    base_dimension = contract.fixture_dimensions[0]
    base_symmetric = symmetric_fixture(base_dimension)
    base_curvature = kulkarni_nomizu_curvature(base_symmetric)
    corrupted_curvature = _replace_curvature_component(
        base_curvature, (0, 1, 0, 1), Fraction(1)
    )
    corrupted_ricci = ricci_from_curvature(corrupted_curvature)
    corrupted_rejected = not audit_curvature(
        corrupted_curvature, corrupted_ricci, base_symmetric
    ).passed

    controls = (
        missing_scalar_identity,
        off_diagonal_control,
        half_action_control,
        omitted_second_slot_control,
        wrong_relative_sign_control,
        wrong_index_control,
        frobenius_control,
        dropped_weyl_control,
        generic_fixture_magnitude,
    )
    exact_residuals = identity_residuals + curvature_residuals
    primitive_dimensions = tuple(
        PRIMITIVE_LENGTH_DIMENSIONS[name]
        for name in ('Identity', 'Curvature')
    )
    trace_dimensions = derive_trace_length_dimensions()
    corrupted_trace_dimensions = derive_trace_length_dimensions(
        {'Curvature': -1}
    )
    dimension_gate = (
        primitive_dimensions == (0, -2)
        and TRACE_DIMENSION_BASIS
        == ('IdentityTrace', 'RiemannSq', 'BundleCurvatureSqTrace')
        and trace_dimensions == (0, -4, -4)
        and corrupted_trace_dimensions == (0, -2, -2)
    )
    required_false = (
        contract.w_linear_sign_determined,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.eq22_trY_derived,
        contract.eq22_trY2_derived,
        contract.eq18_operator_derived,
        contract.gauge_fixing_derived,
        contract.functional_determinant_derived,
        contract.heat_kernel_trace_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.finite_boundary_completed,
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
    gate_passed = (
        source_lock
        and tuple(bundle_rank(n) for n in contract.fixture_dimensions)
        == (7, 11, 16)
        and all(basis_checks)
        and len(curvature_audits) == 7
        and geometry_passed
        and weyl_passed
        and corrupted_rejected
        and all(scalar_block_checks)
        and len(identity_residuals) == 3
        and len(curvature_residuals) == 5
        and len(exact_residuals) == 8
        and exact_passed
        and all(value > 0 for value in controls)
        and w_sign_flip_residual == 0
        and dimension_gate
        and contract.finite_sym2_bundle_curvature_traces_computed
        and not any(required_false)
    )

    return Sym2CurvatureTraceReceipt(
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
        bundle_ranks=tuple(
            bundle_rank(value) for value in contract.fixture_dimensions
        ),
        raw_basis_roundtrip_all_passed=all(basis_checks),
        generic_fixture_count=len(contract.fixture_dimensions),
        weyl_added_fixture_count=len(contract.weyl_fixture_dimensions),
        curvature_audit_count=len(curvature_audits),
        curvature_audits_all_passed=geometry_passed,
        weyl_fixtures_nonzero_and_ricci_flat=weyl_passed,
        corrupted_curvature_rejected=corrupted_rejected,
        scalar_curvature_blocks_all_zero=all(scalar_block_checks),
        exact_trace_residuals=tuple(
            _fraction_text(value) for value in exact_residuals
        ),
        exact_trace_component_count=len(exact_residuals),
        exact_trace_contractions_all_passed=exact_passed,
        missing_scalar_identity_mismatch_l1=_fraction_text(
            missing_scalar_identity
        ),
        off_diagonal_normalization_mismatch_l1=_fraction_text(
            off_diagonal_control
        ),
        half_action_mismatch_l1=_fraction_text(half_action_control),
        omitted_second_slot_mismatch_l1=_fraction_text(
            omitted_second_slot_control
        ),
        wrong_relative_slot_sign_mismatch_l1=_fraction_text(
            wrong_relative_sign_control
        ),
        wrong_curvature_index_mismatch_l1=_fraction_text(
            wrong_index_control
        ),
        frobenius_vs_matrix_trace_mismatch_l1=_fraction_text(
            frobenius_control
        ),
        dropped_weyl_mismatch_l1=_fraction_text(dropped_weyl_control),
        omitted_generic_fixture_magnitude=_fraction_text(
            generic_fixture_magnitude
        ),
        w_linear_sign_flip_squared_trace_residual=_fraction_text(
            w_sign_flip_residual
        ),
        w_linear_sign_determined=contract.w_linear_sign_determined,
        primitive_length_dimensions=primitive_dimensions,
        trace_dimension_basis=TRACE_DIMENSION_BASIS,
        trace_length_dimensions=trace_dimensions,
        corrupted_trace_length_dimensions=corrupted_trace_dimensions,
        dimension_gate_passed=dimension_gate,
        finite_sym2_bundle_curvature_traces_computed=(
            contract.finite_sym2_bundle_curvature_traces_computed
        ),
        derivation_status=contract.derivation_status,
        source_lorentzian_sign_extended=(
            contract.source_lorentzian_sign_extended
        ),
        background_eom_used=contract.background_eom_used,
        eq22_trY_derived=contract.eq22_trY_derived,
        eq22_trY2_derived=contract.eq22_trY2_derived,
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
        finite_boundary_completed=contract.finite_boundary_completed,
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
        declared_finite_sym2_curvature_trace_gate_passed=gate_passed,
    )
