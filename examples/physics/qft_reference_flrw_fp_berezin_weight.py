'''Finite linear FP variation and Berezin determinant/weight gate.

The differential calculation is a local Euclidean jet identity.  The
Grassmann calculation is a separate finite exterior-algebra identity.  No
global functional determinant, measure, zero-mode resolution, or
renormalization is derived here.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import math

from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    HTML_INTERNAL_HEADING,
    Matrix,
    Tensor4,
    audit_curvature,
    kulkarni_nomizu_curvature,
    ricci_from_curvature,
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
    curvature_squared,
    weyl_fixture,
    zero_matrix,
)
from examples.physics.qft_reference_flrw_sym2_potential_bulk_quotient import (
    invert_matrix,
    matrix_multiply,
    zero_curvature,
)


SOURCE_EQUATIONS = ('Eq11', 'Eq13', 'Eq24', 'Eq25', 'Eq26', 'Eq28')
FIXTURE_DIMENSIONS = (3, 4, 5)
FRAME_CONVENTION = 'finite-Euclidean-orthonormal-normal-coordinate-point'
LINEARIZED_SPLIT_FORMULA = (
    'delta_h=grad_xi+transpose;delta_varphi=xi.dot(v);'
    'chi=div(h)-grad(trace(h))/2-varphi*v'
)
JET_FORMULA = (
    'K_mnr=S_mnr+R_mnrs*xi_s/2;K_mnr-K_nmr=R_mnrs*xi_s'
)
FP_FORMULA = (
    'delta_chi=box(xi)+Ric*xi-v*(v.dot(xi));Delta_FP=-delta_chi'
)
BEREZIN_FORMULA = (
    'oriented integral exp(-barc_i*M_ij*c_j)=det(M);N=1,2,3'
)
WEIGHT_FORMULA = (
    'Wghost=-log(abs(detM/detM0));Wboson=log(detA/detA0)/2;ratio=-2'
)
BEREZIN_VARIABLE_ORDER = 'bar_1,...,bar_N,c_1,...,c_N'
BEREZIN_MATRIX_PAYLOAD = (
    'N1=[[2]];N2=[[2,1],[3,5]];'
    'N3=[[2,1,0],[1,3,1],[0,2,4]];singular=[[1,2],[2,4]]'
)
REFERENCE_SCALE = Fraction(2)
SOURCE_TRANSCRIPTION_SHA256 = (
    '0f98583d2bc462d2f4252499a0e3f59c6573169b717f66f4c682849c0298ff49'
)

PRIMITIVE_LENGTH_DIMENSIONS = {
    'GaugeParameter': 1,
    'Derivative': -1,
    'Curvature': -2,
    'ScalarGradient': -1,
}
FP_DIMENSION_FACTORS = {
    'GaugeParameter': ('GaugeParameter',),
    'LaplacianGaugeParameter': (
        'Derivative',
        'Derivative',
        'GaugeParameter',
    ),
    'RicciGaugeParameter': ('Curvature', 'GaugeParameter'),
    'GradientOuterGaugeParameter': (
        'ScalarGradient',
        'ScalarGradient',
        'GaugeParameter',
    ),
    'FPOperator': ('Derivative', 'Derivative'),
    'DeterminantRatio': (),
    'LogDetRatio': (),
    'RelativeWeight': (),
}
FP_DIMENSION_BASIS = tuple(FP_DIMENSION_FACTORS)

Tensor3 = tuple[tuple[tuple[Fraction, ...], ...], ...]
GrassmannPolynomial = dict[int, Fraction]


def xi_fixture(dimension: int) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(((-1) ** index) * (index + 1), index + 2)
        for index in range(dimension)
    )


def symmetric_second_jet_fixture(dimension: int) -> Tensor3:
    return tuple(
        tuple(
            tuple(
                Fraction(mu + nu + rho + 3, (mu + 1) * (nu + 1) * (rho + 2))
                for rho in range(dimension)
            )
            for nu in range(dimension)
        )
        for mu in range(dimension)
    )


def covariant_second_jet(
    curvature: Tensor4,
    xi: tuple[Fraction, ...],
    *,
    commutator_sign: Fraction = Fraction(1),
) -> Tensor3:
    dimension = len(curvature)
    if len(xi) != dimension:
        raise ValueError('curvature and gauge-parameter dimensions differ')
    symmetric = symmetric_second_jet_fixture(dimension)
    return tuple(
        tuple(
            tuple(
                symmetric[mu][nu][rho]
                + commutator_sign
                * Fraction(1, 2)
                * sum(
                    (
                        curvature[mu][nu][rho][sigma] * xi[sigma]
                        for sigma in range(dimension)
                    ),
                    Fraction(0),
                )
                for rho in range(dimension)
            )
            for nu in range(dimension)
        )
        for mu in range(dimension)
    )


def commutator_residual_l1(
    curvature: Tensor4,
    xi: tuple[Fraction, ...],
    second_jet: Tensor3,
) -> Fraction:
    dimension = len(curvature)
    return sum(
        (
            abs(
                second_jet[mu][nu][rho]
                - second_jet[nu][mu][rho]
                - sum(
                    (
                        curvature[mu][nu][rho][sigma] * xi[sigma]
                        for sigma in range(dimension)
                    ),
                    Fraction(0),
                )
            )
            for mu in range(dimension)
            for nu in range(dimension)
            for rho in range(dimension)
        ),
        Fraction(0),
    )


def expanded_gauge_variation(
    second_jet: Tensor3,
    xi: tuple[Fraction, ...],
    scalar_gradient: tuple[Fraction, ...],
    *,
    gauge_trace_coefficient: Fraction = Fraction(1, 2),
    scalar_gauge_term_scale: Fraction = Fraction(1),
) -> tuple[Fraction, ...]:
    dimension = len(second_jet)
    gradient_pairing = sum(
        (
            scalar_gradient[index] * xi[index]
            for index in range(dimension)
        ),
        Fraction(0),
    )
    return tuple(
        sum(
            (second_jet[mu][mu][nu] for mu in range(dimension)),
            Fraction(0),
        )
        + sum(
            (second_jet[mu][nu][mu] for mu in range(dimension)),
            Fraction(0),
        )
        - 2
        * gauge_trace_coefficient
        * sum(
            (second_jet[nu][mu][mu] for mu in range(dimension)),
            Fraction(0),
        )
        - scalar_gauge_term_scale * scalar_gradient[nu] * gradient_pairing
        for nu in range(dimension)
    )


def target_gauge_variation(
    curvature: Tensor4,
    second_jet: Tensor3,
    xi: tuple[Fraction, ...],
    scalar_gradient: tuple[Fraction, ...],
    *,
    ricci_sign: Fraction = Fraction(1),
) -> tuple[Fraction, ...]:
    dimension = len(curvature)
    ricci = ricci_from_curvature(curvature)
    gradient_pairing = sum(
        (
            scalar_gradient[index] * xi[index]
            for index in range(dimension)
        ),
        Fraction(0),
    )
    return tuple(
        sum(
            (second_jet[mu][mu][nu] for mu in range(dimension)),
            Fraction(0),
        )
        + ricci_sign
        * sum(
            (ricci[nu][rho] * xi[rho] for rho in range(dimension)),
            Fraction(0),
        )
        - scalar_gradient[nu] * gradient_pairing
        for nu in range(dimension)
    )


def fp_potential_action(
    curvature: Tensor4,
    xi: tuple[Fraction, ...],
    scalar_gradient: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    dimension = len(curvature)
    ricci = ricci_from_curvature(curvature)
    gradient_pairing = sum(
        (
            scalar_gradient[index] * xi[index]
            for index in range(dimension)
        ),
        Fraction(0),
    )
    return tuple(
        -sum(
            (ricci[nu][rho] * xi[rho] for rho in range(dimension)),
            Fraction(0),
        )
        + scalar_gradient[nu] * gradient_pairing
        for nu in range(dimension)
    )


def derive_fp_length_dimensions(
    primitive_overrides: dict[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_overrides is not None:
        dimensions.update(primitive_overrides)
    return tuple(
        sum(dimensions[factor] for factor in factors)
        for factors in FP_DIMENSION_FACTORS.values()
    )


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def determinant_leibniz(matrix: Matrix) -> Fraction:
    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ValueError('determinant matrix must be nonempty and square')
    return sum(
        (
            permutation_sign(permutation)
            * math.prod(
                matrix[row][permutation[row]]
                for row in range(dimension)
            )
            for permutation in itertools.permutations(range(dimension))
        ),
        Fraction(0),
    )


def determinant_reference_ratio(
    matrix: Matrix,
    reference: Matrix,
) -> Fraction:
    determinant = determinant_leibniz(matrix)
    reference_determinant = determinant_leibniz(reference)
    if determinant == 0 or reference_determinant == 0:
        raise ValueError('determinant ratio requires nonsingular matrices')
    return determinant / reference_determinant


def _grassmann_monomial_sign(left_mask: int, right_mask: int) -> int:
    swaps = 0
    bit = 0
    remaining = left_mask
    while remaining:
        if remaining & 1:
            swaps += (right_mask & ((1 << bit) - 1)).bit_count()
        remaining >>= 1
        bit += 1
    return -1 if swaps % 2 else 1


def grassmann_multiply(
    left: GrassmannPolynomial,
    right: GrassmannPolynomial,
) -> GrassmannPolynomial:
    result: GrassmannPolynomial = {}
    for left_mask, left_value in left.items():
        for right_mask, right_value in right.items():
            if left_mask & right_mask:
                continue
            mask = left_mask | right_mask
            value = (
                left_value
                * right_value
                * _grassmann_monomial_sign(left_mask, right_mask)
            )
            result[mask] = result.get(mask, Fraction(0)) + value
    return {mask: value for mask, value in result.items() if value != 0}


def grassmann_add(
    left: GrassmannPolynomial,
    right: GrassmannPolynomial,
) -> GrassmannPolynomial:
    result = dict(left)
    for mask, value in right.items():
        result[mask] = result.get(mask, Fraction(0)) + value
    return {mask: value for mask, value in result.items() if value != 0}


def grassmann_scale(
    polynomial: GrassmannPolynomial,
    scale: Fraction,
) -> GrassmannPolynomial:
    return {
        mask: scale * value
        for mask, value in polynomial.items()
        if scale * value != 0
    }


def grassmann_bilinear(
    matrix: Matrix,
    *,
    exponent_sign: Fraction = Fraction(-1),
) -> GrassmannPolynomial:
    dimension = len(matrix)
    result: GrassmannPolynomial = {}
    for row in range(dimension):
        for column in range(dimension):
            coefficient = exponent_sign * matrix[row][column]
            if coefficient == 0:
                continue
            bar_mask = 1 << row
            ghost_mask = 1 << (dimension + column)
            monomial = grassmann_multiply(
                {bar_mask: Fraction(1)},
                {ghost_mask: Fraction(1)},
            )
            result = grassmann_add(
                result,
                grassmann_scale(monomial, coefficient),
            )
    return result


def grassmann_exponential(
    polynomial: GrassmannPolynomial,
    maximum_order: int,
) -> GrassmannPolynomial:
    result: GrassmannPolynomial = {0: Fraction(1)}
    power: GrassmannPolynomial = {0: Fraction(1)}
    for order in range(1, maximum_order + 1):
        power = grassmann_multiply(power, polynomial)
        result = grassmann_add(
            result,
            grassmann_scale(power, Fraction(1, math.factorial(order))),
        )
    return result


def berezin_orientation_sign(dimension: int) -> int:
    return -1 if (dimension * (dimension + 1) // 2) % 2 else 1


def berezin_gaussian_integral(
    matrix: Matrix,
    *,
    exponent_sign: Fraction = Fraction(-1),
    orientation_sign: int | None = None,
) -> Fraction:
    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ValueError('Berezin matrix must be nonempty and square')
    exponential = grassmann_exponential(
        grassmann_bilinear(matrix, exponent_sign=exponent_sign),
        dimension,
    )
    top_mask = (1 << (2 * dimension)) - 1
    orientation = (
        berezin_orientation_sign(dimension)
        if orientation_sign is None
        else orientation_sign
    )
    return orientation * exponential.get(top_mask, Fraction(0))


def berezin_matrix_fixtures() -> tuple[Matrix, ...]:
    return (
        ((Fraction(2),),),
        (
            (Fraction(2), Fraction(1)),
            (Fraction(3), Fraction(5)),
        ),
        (
            (Fraction(2), Fraction(1), Fraction(0)),
            (Fraction(1), Fraction(3), Fraction(1)),
            (Fraction(0), Fraction(2), Fraction(4)),
        ),
    )


def singular_berezin_fixture() -> Matrix:
    return (
        (Fraction(1), Fraction(2)),
        (Fraction(2), Fraction(4)),
    )


def matrix_scale(matrix: Matrix, scale: Fraction) -> Matrix:
    return tuple(
        tuple(scale * value for value in row)
        for row in matrix
    )


def matrix_transpose(matrix: Matrix) -> Matrix:
    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ValueError('transpose matrix must be nonempty and square')
    return tuple(
        tuple(matrix[column][row] for column in range(dimension))
        for row in range(dimension)
    )


def similarity_fixture(dimension: int) -> Matrix:
    return tuple(
        tuple(
            Fraction(index + 1) if row == index else Fraction(0)
            for index in range(dimension)
        )
        for row in range(dimension)
    )


def similarity_transform(matrix: Matrix, basis: Matrix) -> Matrix:
    return matrix_multiply(
        matrix_multiply(invert_matrix(basis), matrix),
        basis,
    )


def permutation_basis(dimension: int) -> Matrix:
    if dimension < 1:
        raise ValueError('permutation basis dimension must be positive')
    return tuple(
        tuple(
            Fraction(1)
            if column == (row + 1) % dimension
            else Fraction(0)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


@dataclass(frozen=True)
class FPBerezinContract:
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
    linearized_split_formula: str
    jet_formula: str
    fp_formula: str
    berezin_formula: str
    weight_formula: str
    berezin_variable_order: str
    berezin_matrix_payload: str
    reference_scale: Fraction
    source_transcription_sha256: str
    linearized_background_split_assumed: bool
    source_fp_sign_convention_adopted: bool
    finite_fp_variation_computed: bool
    finite_berezin_identity_computed: bool
    dimensionless_reference_ratio_used: bool
    euclidean_real_boson_gaussian_assumed: bool
    relative_ghost_weight_computed: bool
    derivation_status: str
    fp_derivation_source_explicit: bool
    grassmann_measure_source_explicit: bool
    ghost_minus_two_derivation_source_explicit: bool
    action_prefactor_derived: bool
    overall_operator_sign_phase_resolved: bool
    global_fp_operator_completed: bool
    boundary_conditions_completed: bool
    zero_mode_sector_resolved: bool
    functional_measure_derived: bool
    functional_determinant_computed: bool
    log_branch_resolved: bool
    brst_bv_measure_proved: bool
    heat_kernel_derived: bool
    loop_integral_evaluated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool


def fp_berezin_contract() -> FPBerezinContract:
    return FPBerezinContract(
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
        linearized_split_formula=LINEARIZED_SPLIT_FORMULA,
        jet_formula=JET_FORMULA,
        fp_formula=FP_FORMULA,
        berezin_formula=BEREZIN_FORMULA,
        weight_formula=WEIGHT_FORMULA,
        berezin_variable_order=BEREZIN_VARIABLE_ORDER,
        berezin_matrix_payload=BEREZIN_MATRIX_PAYLOAD,
        reference_scale=REFERENCE_SCALE,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        linearized_background_split_assumed=True,
        source_fp_sign_convention_adopted=True,
        finite_fp_variation_computed=True,
        finite_berezin_identity_computed=True,
        dimensionless_reference_ratio_used=True,
        euclidean_real_boson_gaussian_assumed=True,
        relative_ghost_weight_computed=True,
        derivation_status='finite_linear_fp_variation_and_berezin_weight_only',
        fp_derivation_source_explicit=False,
        grassmann_measure_source_explicit=False,
        ghost_minus_two_derivation_source_explicit=False,
        action_prefactor_derived=False,
        overall_operator_sign_phase_resolved=False,
        global_fp_operator_completed=False,
        boundary_conditions_completed=False,
        zero_mode_sector_resolved=False,
        functional_measure_derived=False,
        functional_determinant_computed=False,
        log_branch_resolved=False,
        brst_bv_measure_proved=False,
        heat_kernel_derived=False,
        loop_integral_evaluated=False,
        renormalization_proof=False,
        continuum_st_qme_proved=False,
        local_covariance_proved=False,
        in_in_ctp_completed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
    )


def canonical_source_payload(contract: FPBerezinContract) -> str:
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
            f'linearized_split={contract.linearized_split_formula}',
            f'jet={contract.jet_formula}',
            f'fp={contract.fp_formula}',
            f'berezin={contract.berezin_formula}',
            f'weight={contract.weight_formula}',
            f'variable_order={contract.berezin_variable_order}',
            f'matrices={contract.berezin_matrix_payload}',
            f'reference_scale={contract.reference_scale}',
        )
    )


def source_payload_sha256(contract: FPBerezinContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: FPBerezinContract) -> None:
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
        contract.linearized_split_formula == LINEARIZED_SPLIT_FORMULA,
        contract.jet_formula == JET_FORMULA,
        contract.fp_formula == FP_FORMULA,
        contract.berezin_formula == BEREZIN_FORMULA,
        contract.weight_formula == WEIGHT_FORMULA,
        contract.berezin_variable_order == BEREZIN_VARIABLE_ORDER,
        contract.berezin_matrix_payload == BEREZIN_MATRIX_PAYLOAD,
        contract.reference_scale == REFERENCE_SCALE,
    )
    if not all(frozen):
        raise ValueError('source, frame, FP, Berezin, or fixture contract changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local FP/Berezin transcription hash mismatch')
    required_true = (
        contract.linearized_background_split_assumed,
        contract.source_fp_sign_convention_adopted,
        contract.finite_fp_variation_computed,
        contract.finite_berezin_identity_computed,
        contract.dimensionless_reference_ratio_used,
        contract.euclidean_real_boson_gaussian_assumed,
        contract.relative_ghost_weight_computed,
    )
    if not all(required_true):
        raise ValueError('a required finite FP/Berezin assumption was disabled')
    if contract.derivation_status != (
        'finite_linear_fp_variation_and_berezin_weight_only'
    ):
        raise ValueError('this gate is finite FP/Berezin weight only')
    required_false = (
        contract.fp_derivation_source_explicit,
        contract.grassmann_measure_source_explicit,
        contract.ghost_minus_two_derivation_source_explicit,
        contract.action_prefactor_derived,
        contract.overall_operator_sign_phase_resolved,
        contract.global_fp_operator_completed,
        contract.boundary_conditions_completed,
        contract.zero_mode_sector_resolved,
        contract.functional_measure_derived,
        contract.functional_determinant_computed,
        contract.log_branch_resolved,
        contract.brst_bv_measure_proved,
        contract.heat_kernel_derived,
        contract.loop_integral_evaluated,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    if any(required_false):
        raise ValueError('a claim beyond finite FP/Berezin algebra was enabled')


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def _vector_mismatch_l1(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> Fraction:
    if len(left) != len(right):
        raise ValueError('vector dimensions differ')
    return sum(
        (abs(left[index] - right[index]) for index in range(len(left))),
        Fraction(0),
    )


def _tensor3_scale(tensor: Tensor3, scale: Fraction) -> Tensor3:
    return tuple(
        tuple(
            tuple(scale * value for value in row)
            for row in plane
        )
        for plane in tensor
    )


@dataclass(frozen=True)
class FPBerezinReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_equations: tuple[str, ...]
    source_transcription_sha256: str
    local_transcription_lock_passed: bool
    frame_convention: str
    fixture_dimensions: tuple[int, ...]
    fixture_count: int
    generic_vector_fixture_count: int
    zero_vector_fixture_count: int
    pure_weyl_fixture_count: int
    flat_fixture_count: int
    curvature_audit_count: int
    curvature_audits_all_passed: bool
    weyl_fixtures_nonzero_and_ricci_flat: bool
    commutator_residuals_l1: tuple[str, ...]
    commutator_identity_all_passed: bool
    gauge_variation_residuals: tuple[str, ...]
    fp_operator_residuals: tuple[str, ...]
    exact_component_count: int
    exact_fp_variation_all_passed: bool
    gauge_parameter_rescaling_residual_l1: str
    gauge_parameter_rescaling_covariant: bool
    generic_laplacian_ricci_scalar_terms_live: bool
    zero_vector_limits_all_passed: bool
    flat_curvature_limit_passed: bool
    wrong_commutator_sign_mismatch_l1: str
    wrong_gauge_trace_coefficient_mismatch_l1: str
    omitted_scalar_gauge_term_mismatch_l1: str
    flipped_scalar_gauge_term_mismatch_l1: str
    wrong_ricci_sign_mismatch_l1: str
    wrong_fp_operator_sign_mismatch_l1: str
    berezin_fixture_dimensions: tuple[int, ...]
    determinant_values: tuple[str, ...]
    berezin_integral_values: tuple[str, ...]
    berezin_residuals: tuple[str, ...]
    finite_berezin_identity_all_passed: bool
    transpose_determinant_covariant: bool
    diagonal_similarity_determinant_covariant: bool
    permutation_basis_determinant_covariant: bool
    positive_exponent_sign_mismatch_l1: str
    wrong_orientation_mismatch_l1: str
    inverse_determinant_confusion_mismatch_l1: str
    singular_determinant: str
    singular_berezin_integral: str
    singular_reference_ratio_rejected: bool
    zero_mode_rejected: bool
    reference_scale: str
    dimensionless_reference_ratios: tuple[str, ...]
    expected_reference_ratios: tuple[str, ...]
    reference_scale_law_passed: bool
    operator_sign_abs_reference_ratios_preserved: bool
    odd_dimension_operator_sign_changed: bool
    ghost_effective_action_exponent: str
    real_boson_effective_action_exponent: str
    relative_ghost_weight: str
    wrong_inverse_ghost_weight_mismatch: str
    half_ghost_weight_mismatch: str
    doubled_ghost_multiplicity_mismatch: str
    primitive_dimension_basis: tuple[str, ...]
    primitive_length_dimensions: tuple[int, ...]
    quantity_dimension_basis: tuple[str, ...]
    quantity_length_dimensions: tuple[int, ...]
    corrupted_gradient_length_dimensions: tuple[int, ...]
    corrupted_derivative_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    linearized_background_split_assumed: bool
    source_fp_sign_convention_adopted: bool
    finite_fp_variation_computed: bool
    finite_berezin_identity_computed: bool
    dimensionless_reference_ratio_used: bool
    euclidean_real_boson_gaussian_assumed: bool
    relative_ghost_weight_computed: bool
    derivation_status: str
    fp_derivation_source_explicit: bool
    grassmann_measure_source_explicit: bool
    ghost_minus_two_derivation_source_explicit: bool
    action_prefactor_derived: bool
    overall_operator_sign_phase_resolved: bool
    global_fp_operator_completed: bool
    boundary_conditions_completed: bool
    zero_mode_sector_resolved: bool
    functional_measure_derived: bool
    functional_determinant_computed: bool
    log_branch_resolved: bool
    brst_bv_measure_proved: bool
    heat_kernel_derived: bool
    loop_integral_evaluated: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    declared_finite_fp_berezin_gate_passed: bool


@dataclass(frozen=True)
class _FPFixture:
    dimension: int
    kind: str
    curvature: Tensor4
    scalar_gradient: tuple[Fraction, ...]


def _laplacian_action(second_jet: Tensor3) -> tuple[Fraction, ...]:
    dimension = len(second_jet)
    return tuple(
        sum(
            (second_jet[mu][mu][nu] for mu in range(dimension)),
            Fraction(0),
        )
        for nu in range(dimension)
    )


def evaluate_fp_berezin_gate() -> FPBerezinReceipt:
    contract = fp_berezin_contract()
    validate_contract(contract)

    fixtures: list[_FPFixture] = []
    curvature_audits = []
    weyl_checks: list[bool] = []
    for dimension in contract.fixture_dimensions:
        symmetric = symmetric_fixture(dimension)
        curvature = kulkarni_nomizu_curvature(symmetric)
        ricci = ricci_from_curvature(curvature)
        curvature_audits.append(
            audit_curvature(curvature, ricci, symmetric)
        )
        fixtures.extend(
            (
                _FPFixture(
                    dimension,
                    'generic-vector',
                    curvature,
                    vector_fixture(dimension),
                ),
                _FPFixture(
                    dimension,
                    'generic-zero-vector',
                    curvature,
                    zero_vector(dimension),
                ),
            )
        )
        if dimension in (4, 5):
            weyl = weyl_fixture(dimension)
            weyl_ricci = ricci_from_curvature(weyl)
            zero = zero_matrix(dimension)
            curvature_audits.append(
                audit_curvature(weyl, weyl_ricci, zero)
            )
            weyl_checks.append(
                curvature_squared(weyl) > 0 and weyl_ricci == zero
            )
            fixtures.append(
                _FPFixture(
                    dimension,
                    'pure-weyl',
                    weyl,
                    vector_fixture(dimension),
                )
            )

    flat_curvature = zero_curvature(3)
    flat_zero = zero_matrix(3)
    curvature_audits.append(
        audit_curvature(flat_curvature, flat_zero, flat_zero)
    )
    fixtures.append(
        _FPFixture(
            3,
            'flat-vector',
            flat_curvature,
            vector_fixture(3),
        )
    )

    commutator_residuals: list[Fraction] = []
    gauge_residuals: list[Fraction] = []
    fp_residuals: list[Fraction] = []
    rescaling_residual = Fraction(0)
    wrong_commutator_control = Fraction(0)
    wrong_trace_control = Fraction(0)
    omitted_scalar_control = Fraction(0)
    flipped_scalar_control = Fraction(0)
    wrong_ricci_control = Fraction(0)
    wrong_fp_sign_control = Fraction(0)
    generic_liveness: list[bool] = []
    zero_vector_checks: list[bool] = []
    flat_checks: list[bool] = []

    for fixture in fixtures:
        dimension = fixture.dimension
        xi = xi_fixture(dimension)
        second_jet = covariant_second_jet(fixture.curvature, xi)
        commutator_residuals.append(
            commutator_residual_l1(
                fixture.curvature,
                xi,
                second_jet,
            )
        )
        expanded = expanded_gauge_variation(
            second_jet,
            xi,
            fixture.scalar_gradient,
        )
        target = target_gauge_variation(
            fixture.curvature,
            second_jet,
            xi,
            fixture.scalar_gradient,
        )
        gauge_residuals.extend(
            expanded[index] - target[index]
            for index in range(dimension)
        )
        laplacian = _laplacian_action(second_jet)
        potential = fp_potential_action(
            fixture.curvature,
            xi,
            fixture.scalar_gradient,
        )
        fp_target = tuple(
            -laplacian[index] + potential[index]
            for index in range(dimension)
        )
        fp_residuals.extend(
            -expanded[index] - fp_target[index]
            for index in range(dimension)
        )

        scale = Fraction(3, 2)
        scaled_xi = tuple(scale * value for value in xi)
        scaled_jet = _tensor3_scale(second_jet, scale)
        scaled_expanded = expanded_gauge_variation(
            scaled_jet,
            scaled_xi,
            fixture.scalar_gradient,
        )
        rescaling_residual += _vector_mismatch_l1(
            scaled_expanded,
            tuple(scale * value for value in expanded),
        )

        wrong_second_jet = covariant_second_jet(
            fixture.curvature,
            xi,
            commutator_sign=Fraction(-1),
        )
        wrong_commutator_control += commutator_residual_l1(
            fixture.curvature,
            xi,
            wrong_second_jet,
        )
        wrong_trace_control += _vector_mismatch_l1(
            expanded_gauge_variation(
                second_jet,
                xi,
                fixture.scalar_gradient,
                gauge_trace_coefficient=Fraction(0),
            ),
            target,
        )
        omitted_scalar_control += _vector_mismatch_l1(
            expanded_gauge_variation(
                second_jet,
                xi,
                fixture.scalar_gradient,
                scalar_gauge_term_scale=Fraction(0),
            ),
            target,
        )
        flipped_scalar_control += _vector_mismatch_l1(
            expanded_gauge_variation(
                second_jet,
                xi,
                fixture.scalar_gradient,
                scalar_gauge_term_scale=Fraction(-1),
            ),
            target,
        )
        wrong_ricci_control += _vector_mismatch_l1(
            target_gauge_variation(
                fixture.curvature,
                second_jet,
                xi,
                fixture.scalar_gradient,
                ricci_sign=Fraction(-1),
            ),
            expanded,
        )
        wrong_fp_sign_control += _vector_mismatch_l1(
            expanded,
            fp_target,
        )

        ricci = ricci_from_curvature(fixture.curvature)
        ricci_action = tuple(
            sum(
                (ricci[nu][rho] * xi[rho] for rho in range(dimension)),
                Fraction(0),
            )
            for nu in range(dimension)
        )
        scalar_pairing = sum(
            (
                fixture.scalar_gradient[index] * xi[index]
                for index in range(dimension)
            ),
            Fraction(0),
        )
        scalar_action = tuple(
            value * scalar_pairing
            for value in fixture.scalar_gradient
        )
        if fixture.kind == 'generic-vector':
            generic_liveness.append(
                any(value != 0 for value in laplacian)
                and any(value != 0 for value in ricci_action)
                and any(value != 0 for value in scalar_action)
            )
        if fixture.kind == 'generic-zero-vector':
            zero_vector_checks.append(
                scalar_pairing == 0
                and all(value == 0 for value in scalar_action)
            )
        if fixture.kind == 'flat-vector':
            flat_checks.append(
                all(value == 0 for value in ricci_action)
                and any(value != 0 for value in scalar_action)
            )

    matrices = berezin_matrix_fixtures()
    determinant_values = tuple(
        determinant_leibniz(matrix) for matrix in matrices
    )
    berezin_values = tuple(
        berezin_gaussian_integral(matrix) for matrix in matrices
    )
    berezin_residuals = tuple(
        berezin_values[index] - determinant_values[index]
        for index in range(len(matrices))
    )
    transpose_covariant = all(
        determinant_leibniz(matrix_transpose(matrix)) == determinant
        and berezin_gaussian_integral(matrix_transpose(matrix)) == determinant
        for matrix, determinant in zip(
            matrices,
            determinant_values,
            strict=True,
        )
    )
    diagonal_similarity_covariant = all(
        determinant_leibniz(
            similarity_transform(matrix, similarity_fixture(len(matrix)))
        )
        == determinant
        and berezin_gaussian_integral(
            similarity_transform(matrix, similarity_fixture(len(matrix)))
        )
        == determinant
        for matrix, determinant in zip(
            matrices,
            determinant_values,
            strict=True,
        )
    )
    permutation_covariant = all(
        determinant_leibniz(
            similarity_transform(matrix, permutation_basis(len(matrix)))
        )
        == determinant
        and berezin_gaussian_integral(
            similarity_transform(matrix, permutation_basis(len(matrix)))
        )
        == determinant
        for matrix, determinant in zip(
            matrices,
            determinant_values,
            strict=True,
        )
    )
    positive_exponent_control = sum(
        (
            abs(
                berezin_gaussian_integral(
                    matrix,
                    exponent_sign=Fraction(1),
                )
                - determinant
            )
            for matrix, determinant in zip(
                matrices,
                determinant_values,
                strict=True,
            )
        ),
        Fraction(0),
    )
    wrong_orientation_control = sum(
        (
            abs(
                berezin_gaussian_integral(
                    matrix,
                    orientation_sign=-berezin_orientation_sign(len(matrix)),
                )
                - determinant
            )
            for matrix, determinant in zip(
                matrices,
                determinant_values,
                strict=True,
            )
        ),
        Fraction(0),
    )
    inverse_determinant_control = sum(
        (
            abs(berezin_values[index] - 1 / determinant_values[index])
            for index in range(len(matrices))
        ),
        Fraction(0),
    )

    singular = singular_berezin_fixture()
    singular_determinant = determinant_leibniz(singular)
    singular_integral = berezin_gaussian_integral(singular)
    singular_ratio_rejected = False
    try:
        determinant_reference_ratio(singular, singular)
    except ValueError:
        singular_ratio_rejected = True

    reference_ratios = tuple(
        determinant_reference_ratio(
            matrix_scale(matrix, contract.reference_scale),
            matrix,
        )
        for matrix in matrices
    )
    expected_ratios = tuple(
        contract.reference_scale ** len(matrix)
        for matrix in matrices
    )
    sign_abs_ratios_preserved = all(
        abs(
            determinant_reference_ratio(
                matrix_scale(matrix, -contract.reference_scale),
                matrix_scale(matrix, Fraction(-1)),
            )
        )
        == abs(reference_ratios[index])
        for index, matrix in enumerate(matrices)
    )
    odd_sign_changed = all(
        (
            determinant_leibniz(matrix_scale(matrix, Fraction(-1)))
            == (
                (-1) ** len(matrix)
            )
            * determinant_values[index]
        )
        for index, matrix in enumerate(matrices)
    ) and any(
        determinant_leibniz(matrix_scale(matrix, Fraction(-1)))
        != determinant_values[index]
        for index, matrix in enumerate(matrices)
        if len(matrix) % 2 == 1
    )

    ghost_exponent = Fraction(-1)
    boson_exponent = Fraction(1, 2)
    relative_weight = ghost_exponent / boson_exponent
    wrong_inverse_weight_control = abs(
        Fraction(1) / boson_exponent - relative_weight
    )
    half_ghost_weight_control = abs(
        Fraction(-1, 2) / boson_exponent - relative_weight
    )
    doubled_ghost_control = abs(
        Fraction(-2) / boson_exponent - relative_weight
    )

    primitive_dimensions = tuple(PRIMITIVE_LENGTH_DIMENSIONS.values())
    quantity_dimensions = derive_fp_length_dimensions()
    corrupted_gradient_dimensions = derive_fp_length_dimensions(
        {'ScalarGradient': 0}
    )
    corrupted_derivative_dimensions = derive_fp_length_dimensions(
        {'Derivative': 0}
    )
    expected_dimensions = (1, -1, -1, -1, -2, 0, 0, 0)
    dimension_gate_passed = (
        FP_DIMENSION_BASIS
        == (
            'GaugeParameter',
            'LaplacianGaugeParameter',
            'RicciGaugeParameter',
            'GradientOuterGaugeParameter',
            'FPOperator',
            'DeterminantRatio',
            'LogDetRatio',
            'RelativeWeight',
        )
        and quantity_dimensions == expected_dimensions
        and corrupted_gradient_dimensions != quantity_dimensions
        and corrupted_derivative_dimensions != quantity_dimensions
    )

    exact_fp_passed = (
        all(value == 0 for value in gauge_residuals)
        and all(value == 0 for value in fp_residuals)
    )
    berezin_passed = all(value == 0 for value in berezin_residuals)
    controls_nonzero = all(
        value > 0
        for value in (
            wrong_commutator_control,
            wrong_trace_control,
            omitted_scalar_control,
            flipped_scalar_control,
            wrong_ricci_control,
            wrong_fp_sign_control,
            positive_exponent_control,
            wrong_orientation_control,
            inverse_determinant_control,
            wrong_inverse_weight_control,
            half_ghost_weight_control,
            doubled_ghost_control,
        )
    )
    bounded_false = (
        contract.fp_derivation_source_explicit,
        contract.grassmann_measure_source_explicit,
        contract.ghost_minus_two_derivation_source_explicit,
        contract.action_prefactor_derived,
        contract.overall_operator_sign_phase_resolved,
        contract.global_fp_operator_completed,
        contract.boundary_conditions_completed,
        contract.zero_mode_sector_resolved,
        contract.functional_measure_derived,
        contract.functional_determinant_computed,
        contract.log_branch_resolved,
        contract.brst_bv_measure_proved,
        contract.heat_kernel_derived,
        contract.loop_integral_evaluated,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    declared_passed = all(
        (
            len(fixtures) == 9,
            all(audit.passed for audit in curvature_audits),
            all(weyl_checks),
            all(value == 0 for value in commutator_residuals),
            exact_fp_passed,
            rescaling_residual == 0,
            all(generic_liveness),
            all(zero_vector_checks),
            all(flat_checks),
            berezin_passed,
            transpose_covariant,
            diagonal_similarity_covariant,
            permutation_covariant,
            singular_determinant == 0,
            singular_integral == 0,
            singular_ratio_rejected,
            reference_ratios == expected_ratios,
            sign_abs_ratios_preserved,
            odd_sign_changed,
            relative_weight == -2,
            dimension_gate_passed,
            controls_nonzero,
            not any(bounded_false),
        )
    )

    return FPBerezinReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        html_internal_heading=contract.html_internal_heading,
        source_equations=contract.source_equations,
        source_transcription_sha256=contract.source_transcription_sha256,
        local_transcription_lock_passed=True,
        frame_convention=contract.frame_convention,
        fixture_dimensions=contract.fixture_dimensions,
        fixture_count=len(fixtures),
        generic_vector_fixture_count=sum(
            fixture.kind == 'generic-vector' for fixture in fixtures
        ),
        zero_vector_fixture_count=sum(
            fixture.kind == 'generic-zero-vector' for fixture in fixtures
        ),
        pure_weyl_fixture_count=sum(
            fixture.kind == 'pure-weyl' for fixture in fixtures
        ),
        flat_fixture_count=sum(
            fixture.kind == 'flat-vector' for fixture in fixtures
        ),
        curvature_audit_count=len(curvature_audits),
        curvature_audits_all_passed=all(
            audit.passed for audit in curvature_audits
        ),
        weyl_fixtures_nonzero_and_ricci_flat=all(weyl_checks),
        commutator_residuals_l1=tuple(
            _fraction_text(value) for value in commutator_residuals
        ),
        commutator_identity_all_passed=all(
            value == 0 for value in commutator_residuals
        ),
        gauge_variation_residuals=tuple(
            _fraction_text(value) for value in gauge_residuals
        ),
        fp_operator_residuals=tuple(
            _fraction_text(value) for value in fp_residuals
        ),
        exact_component_count=len(gauge_residuals) + len(fp_residuals),
        exact_fp_variation_all_passed=exact_fp_passed,
        gauge_parameter_rescaling_residual_l1=_fraction_text(
            rescaling_residual
        ),
        gauge_parameter_rescaling_covariant=rescaling_residual == 0,
        generic_laplacian_ricci_scalar_terms_live=all(generic_liveness),
        zero_vector_limits_all_passed=all(zero_vector_checks),
        flat_curvature_limit_passed=all(flat_checks),
        wrong_commutator_sign_mismatch_l1=_fraction_text(
            wrong_commutator_control
        ),
        wrong_gauge_trace_coefficient_mismatch_l1=_fraction_text(
            wrong_trace_control
        ),
        omitted_scalar_gauge_term_mismatch_l1=_fraction_text(
            omitted_scalar_control
        ),
        flipped_scalar_gauge_term_mismatch_l1=_fraction_text(
            flipped_scalar_control
        ),
        wrong_ricci_sign_mismatch_l1=_fraction_text(wrong_ricci_control),
        wrong_fp_operator_sign_mismatch_l1=_fraction_text(
            wrong_fp_sign_control
        ),
        berezin_fixture_dimensions=tuple(len(matrix) for matrix in matrices),
        determinant_values=tuple(
            _fraction_text(value) for value in determinant_values
        ),
        berezin_integral_values=tuple(
            _fraction_text(value) for value in berezin_values
        ),
        berezin_residuals=tuple(
            _fraction_text(value) for value in berezin_residuals
        ),
        finite_berezin_identity_all_passed=berezin_passed,
        transpose_determinant_covariant=transpose_covariant,
        diagonal_similarity_determinant_covariant=(
            diagonal_similarity_covariant
        ),
        permutation_basis_determinant_covariant=permutation_covariant,
        positive_exponent_sign_mismatch_l1=_fraction_text(
            positive_exponent_control
        ),
        wrong_orientation_mismatch_l1=_fraction_text(
            wrong_orientation_control
        ),
        inverse_determinant_confusion_mismatch_l1=_fraction_text(
            inverse_determinant_control
        ),
        singular_determinant=_fraction_text(singular_determinant),
        singular_berezin_integral=_fraction_text(singular_integral),
        singular_reference_ratio_rejected=singular_ratio_rejected,
        zero_mode_rejected=(
            singular_determinant == 0
            and singular_integral == 0
            and singular_ratio_rejected
        ),
        reference_scale=_fraction_text(contract.reference_scale),
        dimensionless_reference_ratios=tuple(
            _fraction_text(value) for value in reference_ratios
        ),
        expected_reference_ratios=tuple(
            _fraction_text(value) for value in expected_ratios
        ),
        reference_scale_law_passed=reference_ratios == expected_ratios,
        operator_sign_abs_reference_ratios_preserved=(
            sign_abs_ratios_preserved
        ),
        odd_dimension_operator_sign_changed=odd_sign_changed,
        ghost_effective_action_exponent=_fraction_text(ghost_exponent),
        real_boson_effective_action_exponent=_fraction_text(
            boson_exponent
        ),
        relative_ghost_weight=_fraction_text(relative_weight),
        wrong_inverse_ghost_weight_mismatch=_fraction_text(
            wrong_inverse_weight_control
        ),
        half_ghost_weight_mismatch=_fraction_text(
            half_ghost_weight_control
        ),
        doubled_ghost_multiplicity_mismatch=_fraction_text(
            doubled_ghost_control
        ),
        primitive_dimension_basis=tuple(PRIMITIVE_LENGTH_DIMENSIONS),
        primitive_length_dimensions=primitive_dimensions,
        quantity_dimension_basis=FP_DIMENSION_BASIS,
        quantity_length_dimensions=quantity_dimensions,
        corrupted_gradient_length_dimensions=(
            corrupted_gradient_dimensions
        ),
        corrupted_derivative_length_dimensions=(
            corrupted_derivative_dimensions
        ),
        dimension_gate_passed=dimension_gate_passed,
        linearized_background_split_assumed=(
            contract.linearized_background_split_assumed
        ),
        source_fp_sign_convention_adopted=(
            contract.source_fp_sign_convention_adopted
        ),
        finite_fp_variation_computed=contract.finite_fp_variation_computed,
        finite_berezin_identity_computed=(
            contract.finite_berezin_identity_computed
        ),
        dimensionless_reference_ratio_used=(
            contract.dimensionless_reference_ratio_used
        ),
        euclidean_real_boson_gaussian_assumed=(
            contract.euclidean_real_boson_gaussian_assumed
        ),
        relative_ghost_weight_computed=(
            contract.relative_ghost_weight_computed
        ),
        derivation_status=contract.derivation_status,
        fp_derivation_source_explicit=(
            contract.fp_derivation_source_explicit
        ),
        grassmann_measure_source_explicit=(
            contract.grassmann_measure_source_explicit
        ),
        ghost_minus_two_derivation_source_explicit=(
            contract.ghost_minus_two_derivation_source_explicit
        ),
        action_prefactor_derived=contract.action_prefactor_derived,
        overall_operator_sign_phase_resolved=(
            contract.overall_operator_sign_phase_resolved
        ),
        global_fp_operator_completed=contract.global_fp_operator_completed,
        boundary_conditions_completed=contract.boundary_conditions_completed,
        zero_mode_sector_resolved=contract.zero_mode_sector_resolved,
        functional_measure_derived=contract.functional_measure_derived,
        functional_determinant_computed=(
            contract.functional_determinant_computed
        ),
        log_branch_resolved=contract.log_branch_resolved,
        brst_bv_measure_proved=contract.brst_bv_measure_proved,
        heat_kernel_derived=contract.heat_kernel_derived,
        loop_integral_evaluated=contract.loop_integral_evaluated,
        renormalization_proof=contract.renormalization_proof,
        continuum_st_qme_proved=contract.continuum_st_qme_proved,
        local_covariance_proved=contract.local_covariance_proved,
        in_in_ctp_completed=contract.in_in_ctp_completed,
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        declared_finite_fp_berezin_gate_passed=declared_passed,
    )
