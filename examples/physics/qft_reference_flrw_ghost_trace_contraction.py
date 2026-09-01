'''Finite exact ghost-vector trace contractions for the v7 reference source.

The calculation is local Euclidean tensor algebra on declared rational
fixtures. It does not derive the Faddeev--Popov operator, determinant, ghost
weight, heat-kernel formula, loop integral, or renormalization.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_heat_kernel_ghost_reconstruction import (
    HTML_INTERNAL_HEADING,
)
from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_DATE,
    SOURCE_GAUGE,
    SOURCE_ID,
    SOURCE_THEORY,
    SOURCE_TITLE,
    SOURCE_URL,
)


SOURCE_EQUATIONS = ('Eq24', 'Eq25', 'Eq26', 'Eq27')
FIXTURE_DIMENSIONS = (3, 4, 5)
FRAME_CONVENTION = 'finite-Euclidean-orthonormal'
CURVATURE_FORMULA = (
    'R_abcd=delta_ac*S_bd-delta_ad*S_bc-delta_bc*S_ad+delta_bd*S_ac'
)
RICCI_FORMULA = 'Ric_bd=sum_a R_ab ad'
POTENTIAL_FORMULA = 'Potential_ab=-Ric_ab+v_a*v_b'
FIELD_STRENGTH_FORMULA = 'W_mn[a,b]=R_abmn'
SYMMETRIC_FIXTURE_FORMULA = (
    'S_ii=i+1;S_ij=1/(i+j+3) for zero-based i!=j'
)
VECTOR_FIXTURE_FORMULA = 'v_i=1/(i+2) for zero-based i'
SOURCE_TRANSCRIPTION_SHA256 = (
    '38657a0defe69d3391f1affede36221d65f241a6cd413d4263a9ad735aa45488'
)
PRIMITIVE_LENGTH_DIMENSIONS = {
    'RicciTensor': -2,
    'RicciScalar': -2,
    'Potential': -2,
    'FieldStrength': -2,
    'ScalarGradientSquared': -2,
}
INVARIANT_PRIMITIVE_FACTORS = {
    'RiemannSq': ('FieldStrength', 'FieldStrength'),
    'RicciSq': ('RicciTensor', 'RicciTensor'),
    'RicciScalar': ('RicciScalar',),
    'ScalarGradientSquared': ('ScalarGradientSquared',),
    'RicciGradientContraction': (
        'RicciTensor',
        'ScalarGradientSquared',
    ),
    'ScalarGradientFourth': (
        'ScalarGradientSquared',
        'ScalarGradientSquared',
    ),
}
INVARIANT_DIMENSION_BASIS = tuple(INVARIANT_PRIMITIVE_FACTORS)

Matrix = tuple[tuple[Fraction, ...], ...]
Tensor4 = tuple[
    tuple[tuple[tuple[Fraction, ...], ...], ...],
    ...,
]


def _delta(left: int, right: int) -> Fraction:
    return Fraction(int(left == right))


def symmetric_fixture(dimension: int) -> Matrix:
    if dimension < 2:
        raise ValueError('fixture dimension must be at least two')
    return tuple(
        tuple(
            Fraction(row + 1)
            if row == column
            else Fraction(1, row + column + 3)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def vector_fixture(dimension: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(1, index + 2) for index in range(dimension))


def zero_vector(dimension: int) -> tuple[Fraction, ...]:
    return (Fraction(0),) * dimension


def kulkarni_nomizu_curvature(symmetric: Matrix) -> Tensor4:
    dimension = len(symmetric)
    if any(len(row) != dimension for row in symmetric):
        raise ValueError('S must be square')
    if any(
        symmetric[row][column] != symmetric[column][row]
        for row in range(dimension)
        for column in range(dimension)
    ):
        raise ValueError('S must be symmetric')
    return tuple(
        tuple(
            tuple(
                tuple(
                    _delta(a, c) * symmetric[b][d]
                    - _delta(a, d) * symmetric[b][c]
                    - _delta(b, c) * symmetric[a][d]
                    + _delta(b, d) * symmetric[a][c]
                    for d in range(dimension)
                )
                for c in range(dimension)
            )
            for b in range(dimension)
        )
        for a in range(dimension)
    )


def ricci_from_curvature(curvature: Tensor4) -> Matrix:
    dimension = len(curvature)
    return tuple(
        tuple(
            sum(
                (curvature[a][b][a][d] for a in range(dimension)),
                Fraction(0),
            )
            for d in range(dimension)
        )
        for b in range(dimension)
    )


def wrong_ricci_contraction(curvature: Tensor4) -> Matrix:
    dimension = len(curvature)
    return tuple(
        tuple(
            sum(
                (curvature[b][a][a][d] for a in range(dimension)),
                Fraction(0),
            )
            for d in range(dimension)
        )
        for b in range(dimension)
    )


def analytic_ricci_from_s(symmetric: Matrix) -> Matrix:
    dimension = len(symmetric)
    trace_s = sum(
        (symmetric[index][index] for index in range(dimension)), Fraction(0)
    )
    return tuple(
        tuple(
            (dimension - 2) * symmetric[row][column]
            + trace_s * _delta(row, column)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def scalar_curvature(ricci: Matrix) -> Fraction:
    return sum(
        (ricci[index][index] for index in range(len(ricci))), Fraction(0)
    )


def matrix_squared_trace(matrix: Matrix) -> Fraction:
    dimension = len(matrix)
    return sum(
        (
            matrix[row][column] * matrix[column][row]
            for row in range(dimension)
            for column in range(dimension)
        ),
        Fraction(0),
    )


def matrix_trace(matrix: Matrix) -> Fraction:
    return sum(
        (matrix[index][index] for index in range(len(matrix))), Fraction(0)
    )


def identity_matrix(dimension: int) -> Matrix:
    if dimension < 1:
        raise ValueError('identity dimension must be positive')
    return tuple(
        tuple(_delta(row, column) for column in range(dimension))
        for row in range(dimension)
    )


def rank_deficient_identity_control(dimension: int) -> Matrix:
    identity = identity_matrix(dimension)
    return tuple(
        tuple(
            Fraction(0)
            if row == dimension - 1 and column == dimension - 1
            else identity[row][column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def ghost_potential(
    ricci: Matrix,
    vector: tuple[Fraction, ...],
    *,
    outer_sign: int = 1,
) -> Matrix:
    dimension = len(ricci)
    if len(vector) != dimension:
        raise ValueError('vector and Ricci dimensions differ')
    return tuple(
        tuple(
            -ricci[row][column]
            + outer_sign * vector[row] * vector[column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )


@dataclass(frozen=True)
class GhostInvariants:
    riemann_squared: Fraction
    ricci_squared: Fraction
    ricci_scalar: Fraction
    scalar_gradient_squared: Fraction
    ricci_gradient_contraction: Fraction


def ghost_invariants(
    curvature: Tensor4,
    ricci: Matrix,
    vector: tuple[Fraction, ...],
) -> GhostInvariants:
    dimension = len(ricci)
    riemann_squared = sum(
        (
            curvature[a][b][c][d] ** 2
            for a in range(dimension)
            for b in range(dimension)
            for c in range(dimension)
            for d in range(dimension)
        ),
        Fraction(0),
    )
    ricci_squared = sum(
        (
            ricci[row][column] ** 2
            for row in range(dimension)
            for column in range(dimension)
        ),
        Fraction(0),
    )
    scalar_gradient_squared = sum(
        (value**2 for value in vector), Fraction(0)
    )
    ricci_gradient_contraction = sum(
        (
            ricci[row][column] * vector[row] * vector[column]
            for row in range(dimension)
            for column in range(dimension)
        ),
        Fraction(0),
    )
    return GhostInvariants(
        riemann_squared=riemann_squared,
        ricci_squared=ricci_squared,
        ricci_scalar=scalar_curvature(ricci),
        scalar_gradient_squared=scalar_gradient_squared,
        ricci_gradient_contraction=ricci_gradient_contraction,
    )


def field_strength_matrix_trace(
    curvature: Tensor4,
    *,
    linear_sign: int = 1,
) -> Fraction:
    dimension = len(curvature)
    return sum(
        (
            linear_sign
            * curvature[a][b][mu][nu]
            * linear_sign
            * curvature[b][a][mu][nu]
            for mu in range(dimension)
            for nu in range(dimension)
            for a in range(dimension)
            for b in range(dimension)
        ),
        Fraction(0),
    )


def field_strength_frobenius(curvature: Tensor4) -> Fraction:
    dimension = len(curvature)
    return sum(
        (
            curvature[a][b][mu][nu] ** 2
            for mu in range(dimension)
            for nu in range(dimension)
            for a in range(dimension)
            for b in range(dimension)
        ),
        Fraction(0),
    )


def wrong_field_strength_index_trace(curvature: Tensor4) -> Fraction:
    dimension = len(curvature)
    return sum(
        (
            curvature[a][mu][b][nu] * curvature[b][mu][a][nu]
            for mu in range(dimension)
            for nu in range(dimension)
            for a in range(dimension)
            for b in range(dimension)
        ),
        Fraction(0),
    )


@dataclass(frozen=True)
class CurvatureAudit:
    first_pair_antisymmetric: bool
    second_pair_antisymmetric: bool
    pair_exchange_symmetric: bool
    first_bianchi_identity: bool
    ricci_symmetric: bool
    ricci_matches_analytic_s_contraction: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.first_pair_antisymmetric,
                self.second_pair_antisymmetric,
                self.pair_exchange_symmetric,
                self.first_bianchi_identity,
                self.ricci_symmetric,
                self.ricci_matches_analytic_s_contraction,
            )
        )


def audit_curvature(
    curvature: Tensor4,
    ricci: Matrix,
    symmetric: Matrix,
) -> CurvatureAudit:
    dimension = len(curvature)
    first_pair = all(
        curvature[a][b][c][d] == -curvature[b][a][c][d]
        for a in range(dimension)
        for b in range(dimension)
        for c in range(dimension)
        for d in range(dimension)
    )
    second_pair = all(
        curvature[a][b][c][d] == -curvature[a][b][d][c]
        for a in range(dimension)
        for b in range(dimension)
        for c in range(dimension)
        for d in range(dimension)
    )
    pair_exchange = all(
        curvature[a][b][c][d] == curvature[c][d][a][b]
        for a in range(dimension)
        for b in range(dimension)
        for c in range(dimension)
        for d in range(dimension)
    )
    bianchi = all(
        curvature[a][b][c][d]
        + curvature[a][c][d][b]
        + curvature[a][d][b][c]
        == 0
        for a in range(dimension)
        for b in range(dimension)
        for c in range(dimension)
        for d in range(dimension)
    )
    ricci_symmetric = all(
        ricci[row][column] == ricci[column][row]
        for row in range(dimension)
        for column in range(dimension)
    )
    return CurvatureAudit(
        first_pair_antisymmetric=first_pair,
        second_pair_antisymmetric=second_pair,
        pair_exchange_symmetric=pair_exchange,
        first_bianchi_identity=bianchi,
        ricci_symmetric=ricci_symmetric,
        ricci_matches_analytic_s_contraction=(
            ricci == analytic_ricci_from_s(symmetric)
        ),
    )


def _replace_curvature_component(
    curvature: Tensor4,
    indices: tuple[int, int, int, int],
    delta: Fraction,
) -> Tensor4:
    data = [
        [
            [list(curvature[a][b][c]) for c in range(len(curvature))]
            for b in range(len(curvature))
        ]
        for a in range(len(curvature))
    ]
    a, b, c, d = indices
    data[a][b][c][d] += delta
    return tuple(
        tuple(
            tuple(tuple(data[a][b][c]) for c in range(len(curvature)))
            for b in range(len(curvature))
        )
        for a in range(len(curvature))
    )


@dataclass(frozen=True)
class GhostTraceContract:
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
    curvature_formula: str
    ricci_formula: str
    potential_formula: str
    field_strength_formula: str
    symmetric_fixture_formula: str
    vector_fixture_formula: str
    source_transcription_sha256: str
    source_eq22_reused_for_ghost: bool
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    ghost_weight_applied: bool
    finite_trace_contractions_computed: bool
    w_linear_sign_determined: bool
    derivation_status: str
    fp_operator_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    eq19_heat_kernel_derived: bool
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


def ghost_trace_contract() -> GhostTraceContract:
    return GhostTraceContract(
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
        curvature_formula=CURVATURE_FORMULA,
        ricci_formula=RICCI_FORMULA,
        potential_formula=POTENTIAL_FORMULA,
        field_strength_formula=FIELD_STRENGTH_FORMULA,
        symmetric_fixture_formula=SYMMETRIC_FIXTURE_FORMULA,
        vector_fixture_formula=VECTOR_FIXTURE_FORMULA,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        source_eq22_reused_for_ghost=False,
        source_lorentzian_sign_extended=False,
        background_eom_used=False,
        ghost_weight_applied=False,
        finite_trace_contractions_computed=True,
        w_linear_sign_determined=False,
        derivation_status='finite_ghost_trace_contraction_only',
        fp_operator_derived=False,
        fp_determinant_derived=False,
        ghost_weight_derived=False,
        eq19_heat_kernel_derived=False,
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


def canonical_source_payload(contract: GhostTraceContract) -> str:
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
            f'curvature={contract.curvature_formula}',
            f'ricci={contract.ricci_formula}',
            f'potential={contract.potential_formula}',
            f'field_strength={contract.field_strength_formula}',
            f'S_fixture={contract.symmetric_fixture_formula}',
            f'v_fixture={contract.vector_fixture_formula}',
        )
    )


def source_payload_sha256(contract: GhostTraceContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: GhostTraceContract) -> None:
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
        contract.curvature_formula == CURVATURE_FORMULA,
        contract.ricci_formula == RICCI_FORMULA,
        contract.potential_formula == POTENTIAL_FORMULA,
        contract.field_strength_formula == FIELD_STRENGTH_FORMULA,
        contract.symmetric_fixture_formula == SYMMETRIC_FIXTURE_FORMULA,
        contract.vector_fixture_formula == VECTOR_FIXTURE_FORMULA,
    )
    if not all(frozen):
        raise ValueError('source, frame, formula, or fixture contract changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local ghost-trace transcription hash mismatch')
    required_false = (
        contract.source_eq22_reused_for_ghost,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.ghost_weight_applied,
        contract.w_linear_sign_determined,
        contract.fp_operator_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.eq19_heat_kernel_derived,
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
        raise ValueError('unsupported ghost, loop, boundary, continuum, or M2 claim')
    if not contract.finite_trace_contractions_computed:
        raise ValueError('the declared finite trace contractions must be computed')
    if contract.derivation_status != 'finite_ghost_trace_contraction_only':
        raise ValueError('this gate is finite ghost trace contraction only')


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def derive_invariant_length_dimensions(
    primitive_overrides: dict[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_overrides is not None:
        dimensions.update(primitive_overrides)
    return tuple(
        sum(dimensions[factor] for factor in factors)
        for factors in INVARIANT_PRIMITIVE_FACTORS.values()
    )


@dataclass(frozen=True)
class GhostTraceReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_equations: tuple[str, ...]
    source_transcription_sha256: str
    local_transcription_lock_passed: bool
    frame_convention: str
    fixture_dimensions: tuple[int, ...]
    generic_fixture_count: int
    zero_vector_fixture_count: int
    curvature_audit_count: int
    curvature_audits_all_passed: bool
    corrupted_curvature_rejected: bool
    generic_invariants_all_nonzero: bool
    exact_trace_residuals: tuple[str, ...]
    exact_trace_component_count: int
    exact_trace_contractions_all_passed: bool
    zero_vector_limits_all_passed: bool
    frobenius_vs_matrix_trace_mismatch_l1: str
    wrong_ricci_contraction_mismatch_l1: str
    wrong_outer_sign_mismatch_l1: str
    omitted_outer_product_mismatch_l1: str
    omitted_cross_term_mismatch_l1: str
    wrong_w_index_placement_mismatch_l1: str
    omitted_generic_fixture_magnitude: str
    rank_deficient_identity_mismatch_l1: str
    w_linear_sign_flip_squared_trace_residual: str
    w_linear_sign_determined: bool
    primitive_operator_length_dimensions: tuple[int, ...]
    invariant_dimension_basis: tuple[str, ...]
    invariant_length_dimensions: tuple[int, ...]
    corrupted_invariant_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    source_eq22_reused_for_ghost: bool
    source_lorentzian_sign_extended: bool
    background_eom_used: bool
    ghost_weight_applied: bool
    finite_trace_contractions_computed: bool
    derivation_status: str
    fp_operator_derived: bool
    fp_determinant_derived: bool
    ghost_weight_derived: bool
    eq19_heat_kernel_derived: bool
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
    declared_finite_ghost_trace_contraction_gate_passed: bool


def evaluate_ghost_trace_contraction_gate() -> GhostTraceReceipt:
    contract = ghost_trace_contract()
    validate_contract(contract)

    trace_residuals: list[Fraction] = []
    curvature_audits: list[CurvatureAudit] = []
    generic_invariants_nonzero: list[bool] = []
    zero_vector_limits: list[bool] = []
    frobenius_control = Fraction(0)
    wrong_ricci_control = Fraction(0)
    wrong_outer_sign_control = Fraction(0)
    omitted_outer_control = Fraction(0)
    omitted_cross_control = Fraction(0)
    wrong_w_index_control = Fraction(0)
    generic_fixture_magnitude = Fraction(0)
    rank_deficient_identity_control_mismatch = Fraction(0)
    w_sign_flip_residual = Fraction(0)

    for dimension in contract.fixture_dimensions:
        symmetric = symmetric_fixture(dimension)
        curvature = kulkarni_nomizu_curvature(symmetric)
        ricci = ricci_from_curvature(curvature)
        curvature_audits.append(audit_curvature(curvature, ricci, symmetric))

        generic_vector = vector_fixture(dimension)
        generic = ghost_invariants(curvature, ricci, generic_vector)
        generic_potential = ghost_potential(ricci, generic_vector)
        generic_residuals = (
            matrix_trace(identity_matrix(dimension)) - dimension,
            matrix_trace(generic_potential)
            - (-generic.ricci_scalar + generic.scalar_gradient_squared),
            matrix_squared_trace(generic_potential)
            - (
                generic.ricci_squared
                - 2 * generic.ricci_gradient_contraction
                + generic.scalar_gradient_squared**2
            ),
            field_strength_matrix_trace(curvature)
            - (-generic.riemann_squared),
        )
        trace_residuals.extend(generic_residuals)
        generic_invariants_nonzero.append(
            all(
                value != 0
                for value in (
                    generic.riemann_squared,
                    generic.ricci_squared,
                    generic.ricci_scalar,
                    generic.scalar_gradient_squared,
                    generic.ricci_gradient_contraction,
                )
            )
        )

        zero = zero_vector(dimension)
        zero_invariants = ghost_invariants(curvature, ricci, zero)
        zero_potential = ghost_potential(ricci, zero)
        zero_residuals = (
            matrix_trace(identity_matrix(dimension)) - dimension,
            matrix_trace(zero_potential) - (-zero_invariants.ricci_scalar),
            matrix_squared_trace(zero_potential)
            - zero_invariants.ricci_squared,
            field_strength_matrix_trace(curvature)
            - (-zero_invariants.riemann_squared),
        )
        trace_residuals.extend(zero_residuals)
        zero_vector_limits.append(
            zero_invariants.scalar_gradient_squared == 0
            and zero_invariants.ricci_gradient_contraction == 0
            and all(value == 0 for value in zero_residuals)
        )

        frobenius_control += abs(
            field_strength_frobenius(curvature)
            - (-generic.riemann_squared)
        )
        wrong_ricci = wrong_ricci_contraction(curvature)
        wrong_ricci_potential = ghost_potential(wrong_ricci, generic_vector)
        wrong_ricci_control += abs(
            matrix_trace(wrong_ricci_potential)
            - (-generic.ricci_scalar + generic.scalar_gradient_squared)
        )
        wrong_outer = ghost_potential(
            ricci, generic_vector, outer_sign=-1
        )
        wrong_outer_sign_control += abs(
            matrix_trace(wrong_outer)
            - (-generic.ricci_scalar + generic.scalar_gradient_squared)
        )
        wrong_outer_sign_control += abs(
            matrix_squared_trace(wrong_outer)
            - (
                generic.ricci_squared
                - 2 * generic.ricci_gradient_contraction
                + generic.scalar_gradient_squared**2
            )
        )
        omitted_outer = ghost_potential(
            ricci, generic_vector, outer_sign=0
        )
        omitted_outer_control += abs(
            matrix_trace(omitted_outer)
            - (-generic.ricci_scalar + generic.scalar_gradient_squared)
        )
        omitted_outer_control += abs(
            matrix_squared_trace(omitted_outer)
            - (
                generic.ricci_squared
                - 2 * generic.ricci_gradient_contraction
                + generic.scalar_gradient_squared**2
            )
        )
        omitted_cross_control += abs(
            generic.ricci_squared
            + generic.scalar_gradient_squared**2
            - (
                generic.ricci_squared
                - 2 * generic.ricci_gradient_contraction
                + generic.scalar_gradient_squared**2
            )
        )
        wrong_w_index_control += abs(
            wrong_field_strength_index_trace(curvature)
            - (-generic.riemann_squared)
        )
        generic_fixture_magnitude += (
            abs(generic.scalar_gradient_squared)
            + abs(generic.ricci_gradient_contraction)
        )
        rank_deficient_identity_control_mismatch += abs(
            matrix_trace(rank_deficient_identity_control(dimension))
            - dimension
        )
        w_sign_flip_residual += abs(
            field_strength_matrix_trace(curvature, linear_sign=-1)
            - field_strength_matrix_trace(curvature, linear_sign=1)
        )

    base_symmetric = symmetric_fixture(contract.fixture_dimensions[0])
    base_curvature = kulkarni_nomizu_curvature(base_symmetric)
    corrupted_curvature = _replace_curvature_component(
        base_curvature, (0, 1, 0, 1), Fraction(1)
    )
    corrupted_ricci = ricci_from_curvature(corrupted_curvature)
    corrupted_rejected = not audit_curvature(
        corrupted_curvature, corrupted_ricci, base_symmetric
    ).passed

    controls = (
        frobenius_control,
        wrong_ricci_control,
        wrong_outer_sign_control,
        omitted_outer_control,
        omitted_cross_control,
        wrong_w_index_control,
        generic_fixture_magnitude,
        rank_deficient_identity_control_mismatch,
    )
    primitive_dimensions = tuple(
        PRIMITIVE_LENGTH_DIMENSIONS[name]
        for name in ('RicciTensor', 'Potential', 'FieldStrength')
    )
    invariant_dimensions = derive_invariant_length_dimensions()
    corrupted_invariant_dimensions = derive_invariant_length_dimensions(
        {'ScalarGradientSquared': -1}
    )
    dimension_gate = (
        primitive_dimensions == (-2, -2, -2)
        and INVARIANT_DIMENSION_BASIS
        == (
            'RiemannSq',
            'RicciSq',
            'RicciScalar',
            'ScalarGradientSquared',
            'RicciGradientContraction',
            'ScalarGradientFourth',
        )
        and invariant_dimensions == (-4, -4, -2, -2, -4, -4)
        and corrupted_invariant_dimensions == (-4, -4, -2, -1, -3, -2)
        and corrupted_invariant_dimensions != invariant_dimensions
    )
    required_false = (
        contract.source_eq22_reused_for_ghost,
        contract.source_lorentzian_sign_extended,
        contract.background_eom_used,
        contract.ghost_weight_applied,
        contract.w_linear_sign_determined,
        contract.fp_operator_derived,
        contract.fp_determinant_derived,
        contract.ghost_weight_derived,
        contract.eq19_heat_kernel_derived,
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
    trace_passed = all(value == 0 for value in trace_residuals)
    geometry_passed = all(audit.passed for audit in curvature_audits)
    gate_passed = (
        source_lock
        and geometry_passed
        and corrupted_rejected
        and all(generic_invariants_nonzero)
        and all(zero_vector_limits)
        and len(trace_residuals) == 24
        and trace_passed
        and all(value > 0 for value in controls)
        and w_sign_flip_residual == 0
        and dimension_gate
        and contract.finite_trace_contractions_computed
        and not any(required_false)
    )

    return GhostTraceReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        html_internal_heading=contract.html_internal_heading,
        source_equations=contract.source_equations,
        source_transcription_sha256=contract.source_transcription_sha256,
        local_transcription_lock_passed=source_lock,
        frame_convention=contract.frame_convention,
        fixture_dimensions=contract.fixture_dimensions,
        generic_fixture_count=len(contract.fixture_dimensions),
        zero_vector_fixture_count=len(contract.fixture_dimensions),
        curvature_audit_count=len(curvature_audits),
        curvature_audits_all_passed=geometry_passed,
        corrupted_curvature_rejected=corrupted_rejected,
        generic_invariants_all_nonzero=all(generic_invariants_nonzero),
        exact_trace_residuals=tuple(
            _fraction_text(value) for value in trace_residuals
        ),
        exact_trace_component_count=len(trace_residuals),
        exact_trace_contractions_all_passed=trace_passed,
        zero_vector_limits_all_passed=all(zero_vector_limits),
        frobenius_vs_matrix_trace_mismatch_l1=_fraction_text(
            frobenius_control
        ),
        wrong_ricci_contraction_mismatch_l1=_fraction_text(
            wrong_ricci_control
        ),
        wrong_outer_sign_mismatch_l1=_fraction_text(
            wrong_outer_sign_control
        ),
        omitted_outer_product_mismatch_l1=_fraction_text(
            omitted_outer_control
        ),
        omitted_cross_term_mismatch_l1=_fraction_text(
            omitted_cross_control
        ),
        wrong_w_index_placement_mismatch_l1=_fraction_text(
            wrong_w_index_control
        ),
        omitted_generic_fixture_magnitude=_fraction_text(
            generic_fixture_magnitude
        ),
        rank_deficient_identity_mismatch_l1=_fraction_text(
            rank_deficient_identity_control_mismatch
        ),
        w_linear_sign_flip_squared_trace_residual=_fraction_text(
            w_sign_flip_residual
        ),
        w_linear_sign_determined=contract.w_linear_sign_determined,
        primitive_operator_length_dimensions=primitive_dimensions,
        invariant_dimension_basis=INVARIANT_DIMENSION_BASIS,
        invariant_length_dimensions=invariant_dimensions,
        corrupted_invariant_length_dimensions=(
            corrupted_invariant_dimensions
        ),
        dimension_gate_passed=dimension_gate,
        source_eq22_reused_for_ghost=contract.source_eq22_reused_for_ghost,
        source_lorentzian_sign_extended=(
            contract.source_lorentzian_sign_extended
        ),
        background_eom_used=contract.background_eom_used,
        ghost_weight_applied=contract.ghost_weight_applied,
        finite_trace_contractions_computed=(
            contract.finite_trace_contractions_computed
        ),
        derivation_status=contract.derivation_status,
        fp_operator_derived=contract.fp_operator_derived,
        fp_determinant_derived=contract.fp_determinant_derived,
        ghost_weight_derived=contract.ghost_weight_derived,
        eq19_heat_kernel_derived=contract.eq19_heat_kernel_derived,
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
        declared_finite_ghost_trace_contraction_gate_passed=gate_passed,
    )
