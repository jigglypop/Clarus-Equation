'''Finite raw-operator trace to heat-kernel coefficient synthesis gate.

The raw bundle and ghost traces are computed from finite Euclidean matrices.
The universal Eq. (19) heat-kernel formula is a source-supplied theorem input.
Published Eq. (23)/(27)/(28) coefficients are consulted only after the exact
fixture fit.  This module does not derive the heat-kernel theorem, a global
functional determinant, a loop integral, boundary completion, or
renormalization.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_fp_berezin_weight import (
    SOURCE_TRANSCRIPTION_SHA256 as FP_BEREZIN_HASH,
    fp_berezin_contract,
    validate_contract as validate_fp_berezin_contract,
)
from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    SOURCE_TRANSCRIPTION_SHA256 as GHOST_TRACE_HASH,
    Matrix,
    Tensor4,
    audit_curvature,
    field_strength_matrix_trace,
    ghost_potential,
    ghost_trace_contract,
    identity_matrix,
    kulkarni_nomizu_curvature,
    matrix_squared_trace as ghost_matrix_squared_trace,
    matrix_trace as ghost_matrix_trace,
    ricci_from_curvature,
    validate_contract as validate_ghost_trace_contract,
    zero_vector,
)
from examples.physics.qft_reference_flrw_heat_kernel_ghost_reconstruction import (
    GHOST_WEIGHT,
    RAW_BASIS,
    combine_ghost_weight,
    equation_23_coefficients,
    equation_27_ghost_coefficients,
    four_dimensional_bulk_gb_quotient,
    source_equation_28_with_p_slot,
)
from examples.physics.qft_reference_flrw_heat_kernel_trace_identity_assembly import (
    EQ19_FORMULA,
    SOURCE_TRANSCRIPTION_SHA256 as TRACE_IDENTITY_HASH,
    trace_identity_assembly_contract,
    validate_contract as validate_trace_identity_contract,
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
    SOURCE_TRANSCRIPTION_SHA256 as SYM2_CURVATURE_HASH,
    add_curvatures,
    bundle_curvature_squared_trace,
    bundle_identity_matrix,
    curvature_squared,
    sym2_curvature_trace_contract,
    validate_contract as validate_sym2_curvature_contract,
    weyl_fixture,
    zero_matrix,
)
from examples.physics.qft_reference_flrw_sym2_potential_bulk_quotient import (
    SOURCE_TRANSCRIPTION_SHA256 as SYM2_POTENTIAL_HASH,
    PotentialInvariants,
    bulk_divergence,
    invert_matrix,
    matrix_trace,
    potential_invariants,
    potential_trace_values,
    raw_dewitt_metric,
    sym2_potential_bulk_contract,
    validate_contract as validate_sym2_potential_contract,
    zero_curvature,
)


SOURCE_EQUATIONS = (
    'Eq17',
    'Eq18',
    'Eq19',
    'Eq20',
    'Eq21',
    'Eq22',
    'Eq23',
    'Eq24',
    'Eq25',
    'Eq26',
    'Eq27',
    'Eq28',
    'Eq29',
)
FRAME_CONVENTION = 'finite-Euclidean-orthonormal-normal-coordinate-point'
IDENTIFICATION_DIMENSIONS = (4, 5, 6, 7)
HOLDOUT_DIMENSIONS = (8,)
FIXTURE_TYPES = (
    'pure-weyl',
    'trace-curvature',
    'traceless-curvature',
    'flat-vector',
    'flat-hessian',
    'traceless-vector',
    'trace-vector',
    'generic-a',
    'generic-b',
    'generic-c',
    'generic-hessian',
    'generic-vector',
)
FIXTURES_PER_DIMENSION = len(FIXTURE_TYPES)
UPSTREAM_HASHES = (
    ('trace-identity', TRACE_IDENTITY_HASH),
    ('ghost-trace', GHOST_TRACE_HASH),
    ('sym2-curvature', SYM2_CURVATURE_HASH),
    ('sym2-potential', SYM2_POTENTIAL_HASH),
    ('fp-berezin', FP_BEREZIN_HASH),
)
BULK_QUOTIENT_FORMULA = (
    'trY2_bulk=trY2_raw-4*(H2-BoxPhi2+RicciGradPhiGradPhi)'
)
FIT_FORMULA = (
    'independent raw Eq19 densities fit exact B7='
    '(RiemannSq,RicciSq,R2,P,R_X,X2,BoxPhi2);'
    'source Eq23/Eq27 targets read only after fit'
)
LIFT_FORMULA = 'p_i(n)=360*(n-2)*c_i(n);degree<=3 admitted'
GB_FORMULA = (
    'n=4 integrated-bulk only:RiemannSq->4*RicciSq-R2'
)
FIXTURE_FORMULA = (
    'per-n 12 exact fixtures:pure Weyl,identity-S,traceless-S,'
    'flat-v,flat-H,traceless-S+v,identity-S+v,'
    'five seeded S+tC/v/H variants;seeds=1..5;'
    'C=locked 4D Ricci-flat Weyl zero-embedded for n>4'
)
COMMON_LIFT_FACTOR = 360
ADMITTED_LIFT_DEGREE = 3
SOURCE_TRANSCRIPTION_SHA256 = (
    'b7f30c9948b1853335a519d65b5c6796b2fd8ec58fd7532ecdad714b4bc0d83c'
)

PRIMITIVE_LENGTH_DIMENSIONS = {
    'Curvature': -2,
    'ScalarGradient': -1,
    'ScalarHessian': -2,
    'Coefficient': 0,
}
SYNTHESIS_DIMENSION_FACTORS = {
    'RiemannSq': ('Curvature', 'Curvature'),
    'RicciSq': ('Curvature', 'Curvature'),
    'R2': ('Curvature', 'Curvature'),
    'RicciGradPhiGradPhi': (
        'Curvature',
        'ScalarGradient',
        'ScalarGradient',
    ),
    'R_X': ('Curvature', 'ScalarGradient', 'ScalarGradient'),
    'X2': (
        'ScalarGradient',
        'ScalarGradient',
        'ScalarGradient',
        'ScalarGradient',
    ),
    'BoxPhi2': ('ScalarHessian', 'ScalarHessian'),
    'BulkDivergence': ('ScalarHessian', 'ScalarHessian'),
    'Eq19Density': ('Curvature', 'Curvature'),
    'Coefficient': ('Coefficient',),
}
SYNTHESIS_DIMENSION_BASIS = tuple(SYNTHESIS_DIMENSION_FACTORS)


@dataclass(frozen=True)
class OperatorTraceSynthesisContract:
    source_id: str
    source_date: str
    source_metadata_title: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_equations: tuple[str, ...]
    frame_convention: str
    raw_basis: tuple[str, ...]
    identification_dimensions: tuple[int, ...]
    holdout_dimensions: tuple[int, ...]
    fixture_types: tuple[str, ...]
    fixtures_per_dimension: int
    eq19_formula: str
    bulk_quotient_formula: str
    fit_formula: str
    lift_formula: str
    gb_formula: str
    fixture_formula: str
    common_lift_factor: int
    admitted_lift_degree: int
    ghost_weight: int
    upstream_hashes: tuple[tuple[str, str], ...]
    source_transcription_sha256: str
    upstream_contracts_verified: bool
    eq19_source_supplied_theorem: bool
    raw_bundle_traces_independently_computed: bool
    raw_ghost_traces_independently_computed: bool
    source_targets_used_only_after_fit: bool
    bulk_divergence_quotient_adopted: bool
    ghost_weight_from_finite_berezin_adopted: bool
    four_dimensional_gb_bulk_quotient_adopted: bool
    polynomial_lift_degree_bound_admitted: bool
    derivation_status: str
    source_eq22_coefficients_used_as_fit_input: bool
    source_eq23_eq27_used_as_fit_input: bool
    eq19_theorem_independently_derived: bool
    all_n_symbolic_identity_proved: bool
    bulk_divergence_pointwise_zero_claimed: bool
    gauss_bonnet_pointwise_zero_claimed: bool
    finite_boundary_completed: bool
    global_minimal_operator_derived: bool
    functional_measure_derived: bool
    functional_determinant_computed: bool
    heat_kernel_proper_time_integral_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    evanescent_terms_controlled: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool


def operator_trace_synthesis_contract() -> OperatorTraceSynthesisContract:
    return OperatorTraceSynthesisContract(
        source_id=SOURCE_ID,
        source_date=SOURCE_DATE,
        source_metadata_title=SOURCE_TITLE,
        source_theory=SOURCE_THEORY,
        source_gauge=SOURCE_GAUGE,
        source_url=SOURCE_URL,
        source_equations=SOURCE_EQUATIONS,
        frame_convention=FRAME_CONVENTION,
        raw_basis=RAW_BASIS,
        identification_dimensions=IDENTIFICATION_DIMENSIONS,
        holdout_dimensions=HOLDOUT_DIMENSIONS,
        fixture_types=FIXTURE_TYPES,
        fixtures_per_dimension=FIXTURES_PER_DIMENSION,
        eq19_formula=EQ19_FORMULA,
        bulk_quotient_formula=BULK_QUOTIENT_FORMULA,
        fit_formula=FIT_FORMULA,
        lift_formula=LIFT_FORMULA,
        gb_formula=GB_FORMULA,
        fixture_formula=FIXTURE_FORMULA,
        common_lift_factor=COMMON_LIFT_FACTOR,
        admitted_lift_degree=ADMITTED_LIFT_DEGREE,
        ghost_weight=GHOST_WEIGHT,
        upstream_hashes=UPSTREAM_HASHES,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        upstream_contracts_verified=True,
        eq19_source_supplied_theorem=True,
        raw_bundle_traces_independently_computed=True,
        raw_ghost_traces_independently_computed=True,
        source_targets_used_only_after_fit=True,
        bulk_divergence_quotient_adopted=True,
        ghost_weight_from_finite_berezin_adopted=True,
        four_dimensional_gb_bulk_quotient_adopted=True,
        polynomial_lift_degree_bound_admitted=True,
        derivation_status='finite_raw_trace_to_source_coefficient_synthesis_only',
        source_eq22_coefficients_used_as_fit_input=False,
        source_eq23_eq27_used_as_fit_input=False,
        eq19_theorem_independently_derived=False,
        all_n_symbolic_identity_proved=False,
        bulk_divergence_pointwise_zero_claimed=False,
        gauss_bonnet_pointwise_zero_claimed=False,
        finite_boundary_completed=False,
        global_minimal_operator_derived=False,
        functional_measure_derived=False,
        functional_determinant_computed=False,
        heat_kernel_proper_time_integral_derived=False,
        loop_integral_evaluated=False,
        regularization_scheme_implemented=False,
        evanescent_terms_controlled=False,
        renormalization_proof=False,
        continuum_st_qme_proved=False,
        local_covariance_proved=False,
        in_in_ctp_completed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
    )


def canonical_source_payload(contract: OperatorTraceSynthesisContract) -> str:
    separator = chr(44)
    upstream = separator.join(
        f'{name}:{value}' for name, value in contract.upstream_hashes
    )
    return '|'.join(
        (
            contract.source_id,
            contract.source_date,
            f'metadata_title={contract.source_metadata_title}',
            f'theory={contract.source_theory}',
            f'gauge={contract.source_gauge}',
            f'equations={separator.join(contract.source_equations)}',
            f'frame={contract.frame_convention}',
            f'basis={separator.join(contract.raw_basis)}',
            'identify='
            + separator.join(
                str(value) for value in contract.identification_dimensions
            ),
            'holdout='
            + separator.join(str(value) for value in contract.holdout_dimensions),
            f'fixture_types={separator.join(contract.fixture_types)}',
            f'fixtures_per_dimension={contract.fixtures_per_dimension}',
            f'eq19={contract.eq19_formula}',
            f'bulk={contract.bulk_quotient_formula}',
            f'fit={contract.fit_formula}',
            f'lift={contract.lift_formula}',
            f'gb={contract.gb_formula}',
            f'fixtures={contract.fixture_formula}',
            f'lift_factor={contract.common_lift_factor}',
            f'degree={contract.admitted_lift_degree}',
            f'ghost_weight={contract.ghost_weight}',
            f'upstream={upstream}',
        )
    )


def source_payload_sha256(contract: OperatorTraceSynthesisContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: OperatorTraceSynthesisContract) -> None:
    frozen = (
        contract.source_id == SOURCE_ID,
        contract.source_date == SOURCE_DATE,
        contract.source_metadata_title == SOURCE_TITLE,
        contract.source_theory == SOURCE_THEORY,
        contract.source_gauge == SOURCE_GAUGE,
        contract.source_url == SOURCE_URL,
        contract.source_equations == SOURCE_EQUATIONS,
        contract.frame_convention == FRAME_CONVENTION,
        contract.raw_basis == RAW_BASIS,
        contract.identification_dimensions == IDENTIFICATION_DIMENSIONS,
        contract.holdout_dimensions == HOLDOUT_DIMENSIONS,
        contract.fixture_types == FIXTURE_TYPES,
        contract.fixtures_per_dimension == FIXTURES_PER_DIMENSION,
        contract.eq19_formula == EQ19_FORMULA,
        contract.bulk_quotient_formula == BULK_QUOTIENT_FORMULA,
        contract.fit_formula == FIT_FORMULA,
        contract.lift_formula == LIFT_FORMULA,
        contract.gb_formula == GB_FORMULA,
        contract.fixture_formula == FIXTURE_FORMULA,
        contract.common_lift_factor == COMMON_LIFT_FACTOR,
        contract.admitted_lift_degree == ADMITTED_LIFT_DEGREE,
        contract.ghost_weight == GHOST_WEIGHT,
        contract.upstream_hashes == UPSTREAM_HASHES,
    )
    if not all(frozen):
        raise ValueError('source, upstream, basis, fixture, or fit contract changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('operator-trace synthesis transcription hash mismatch')
    required_true = (
        contract.upstream_contracts_verified,
        contract.eq19_source_supplied_theorem,
        contract.raw_bundle_traces_independently_computed,
        contract.raw_ghost_traces_independently_computed,
        contract.source_targets_used_only_after_fit,
        contract.bulk_divergence_quotient_adopted,
        contract.ghost_weight_from_finite_berezin_adopted,
        contract.four_dimensional_gb_bulk_quotient_adopted,
        contract.polynomial_lift_degree_bound_admitted,
    )
    if not all(required_true):
        raise ValueError('a required synthesis assumption was disabled')
    if contract.derivation_status != (
        'finite_raw_trace_to_source_coefficient_synthesis_only'
    ):
        raise ValueError('this gate is finite coefficient synthesis only')
    required_false = (
        contract.source_eq22_coefficients_used_as_fit_input,
        contract.source_eq23_eq27_used_as_fit_input,
        contract.eq19_theorem_independently_derived,
        contract.all_n_symbolic_identity_proved,
        contract.bulk_divergence_pointwise_zero_claimed,
        contract.gauss_bonnet_pointwise_zero_claimed,
        contract.finite_boundary_completed,
        contract.global_minimal_operator_derived,
        contract.functional_measure_derived,
        contract.functional_determinant_computed,
        contract.heat_kernel_proper_time_integral_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.evanescent_terms_controlled,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    if any(required_false):
        raise ValueError('a claim beyond finite trace synthesis was enabled')


def validate_upstream_contracts() -> None:
    validate_trace_identity_contract(trace_identity_assembly_contract())
    validate_ghost_trace_contract(ghost_trace_contract())
    validate_sym2_curvature_contract(sym2_curvature_trace_contract())
    validate_sym2_potential_contract(sym2_potential_bulk_contract())
    validate_fp_berezin_contract(fp_berezin_contract())


def derive_synthesis_length_dimensions(
    primitive_overrides: dict[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_overrides is not None:
        dimensions.update(primitive_overrides)
    return tuple(
        sum(dimensions[factor] for factor in factors)
        for factors in SYNTHESIS_DIMENSION_FACTORS.values()
    )


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def _scale_curvature(curvature: Tensor4, scale: Fraction) -> Tensor4:
    dimension = len(curvature)
    return tuple(
        tuple(
            tuple(
                tuple(
                    scale * curvature[a][b][c][d]
                    for d in range(dimension)
                )
                for c in range(dimension)
            )
            for b in range(dimension)
        )
        for a in range(dimension)
    )


def _embedded_weyl_fixture(dimension: int) -> Tensor4:
    if dimension < 4:
        raise ValueError('the embedded Weyl fixture requires n>=4')
    base = weyl_fixture(4)
    return tuple(
        tuple(
            tuple(
                tuple(
                    base[a][b][c][d]
                    if max(a, b, c, d) < 4
                    else Fraction(0)
                    for d in range(dimension)
                )
                for c in range(dimension)
            )
            for b in range(dimension)
        )
        for a in range(dimension)
    )


def _diagonal_matrix(
    dimension: int,
    diagonal: tuple[Fraction, ...],
) -> Matrix:
    if len(diagonal) > dimension:
        raise ValueError('diagonal payload exceeds matrix dimension')
    padded = diagonal + (Fraction(0),) * (dimension - len(diagonal))
    return tuple(
        tuple(
            padded[row] if row == column else Fraction(0)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _identity_symmetric(dimension: int) -> Matrix:
    return _diagonal_matrix(
        dimension,
        tuple(Fraction(1) for _ in range(dimension)),
    )


def _traceless_symmetric(dimension: int) -> Matrix:
    if dimension < 2:
        raise ValueError('traceless fixture requires at least two dimensions')
    return _diagonal_matrix(
        dimension,
        (Fraction(1), Fraction(-1)),
    )


def _seeded_symmetric(dimension: int, seed: int) -> Matrix:
    if seed < 1:
        raise ValueError('fixture seed must be positive')
    return tuple(
        tuple(
            Fraction(
                (
                    -1
                    if (min(row, column) + max(row, column) + seed) % 3 == 0
                    else 1
                )
                * (
                    (min(row, column) + 1) * (max(row, column) + 2)
                    + seed * (min(row, column) + max(row, column) + 1)
                ),
                seed + min(row, column) + max(row, column) + 2,
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _seeded_vector(dimension: int, seed: int) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(
            (-1 if (index + seed) % 2 else 1) * (seed + index + 1),
            seed * (index + 1) + 2,
        )
        for index in range(dimension)
    )


def _seeded_hessian(dimension: int, seed: int) -> Matrix:
    return _seeded_symmetric(dimension, seed + 7)


@dataclass(frozen=True)
class SynthesisFixture:
    dimension: int
    kind: str
    symmetric: Matrix
    weyl_scale: Fraction
    curvature: Tensor4
    vector: tuple[Fraction, ...]
    hessian: Matrix


def _make_fixture(
    dimension: int,
    kind: str,
    symmetric: Matrix,
    weyl_scale: Fraction,
    vector: tuple[Fraction, ...],
    hessian: Matrix,
) -> SynthesisFixture:
    curvature = kulkarni_nomizu_curvature(symmetric)
    if weyl_scale != 0:
        curvature = add_curvatures(
            curvature,
            _scale_curvature(
                _embedded_weyl_fixture(dimension),
                weyl_scale,
            ),
        )
    return SynthesisFixture(
        dimension=dimension,
        kind=kind,
        symmetric=symmetric,
        weyl_scale=weyl_scale,
        curvature=curvature,
        vector=vector,
        hessian=hessian,
    )


def synthesis_fixtures(dimension: int) -> tuple[SynthesisFixture, ...]:
    if dimension < 4:
        raise ValueError('full B7 identification requires n>=4')
    zero_s = zero_matrix(dimension)
    zero_v = zero_vector(dimension)
    zero_h = zero_matrix(dimension)
    identity_s = _identity_symmetric(dimension)
    traceless_s = _traceless_symmetric(dimension)
    fixtures = (
        _make_fixture(
            dimension,
            'pure-weyl',
            zero_s,
            Fraction(1),
            zero_v,
            zero_h,
        ),
        _make_fixture(
            dimension,
            'trace-curvature',
            identity_s,
            Fraction(0),
            zero_v,
            zero_h,
        ),
        _make_fixture(
            dimension,
            'traceless-curvature',
            traceless_s,
            Fraction(0),
            zero_v,
            zero_h,
        ),
        _make_fixture(
            dimension,
            'flat-vector',
            zero_s,
            Fraction(0),
            _seeded_vector(dimension, 1),
            zero_h,
        ),
        _make_fixture(
            dimension,
            'flat-hessian',
            zero_s,
            Fraction(0),
            zero_v,
            _diagonal_matrix(
                dimension,
                (Fraction(1), Fraction(2), Fraction(4)),
            ),
        ),
        _make_fixture(
            dimension,
            'traceless-vector',
            traceless_s,
            Fraction(0),
            _seeded_vector(dimension, 2),
            zero_h,
        ),
        _make_fixture(
            dimension,
            'trace-vector',
            identity_s,
            Fraction(0),
            _seeded_vector(dimension, 3),
            zero_h,
        ),
        _make_fixture(
            dimension,
            'generic-a',
            _seeded_symmetric(dimension, 1),
            Fraction(1, 2),
            _seeded_vector(dimension, 1),
            _seeded_hessian(dimension, 1),
        ),
        _make_fixture(
            dimension,
            'generic-b',
            _seeded_symmetric(dimension, 2),
            Fraction(-1),
            _seeded_vector(dimension, 2),
            _seeded_hessian(dimension, 2),
        ),
        _make_fixture(
            dimension,
            'generic-c',
            _seeded_symmetric(dimension, 3),
            Fraction(2),
            _seeded_vector(dimension, 3),
            _seeded_hessian(dimension, 3),
        ),
        _make_fixture(
            dimension,
            'generic-hessian',
            _seeded_symmetric(dimension, 4),
            Fraction(1),
            zero_v,
            _seeded_hessian(dimension, 4),
        ),
        _make_fixture(
            dimension,
            'generic-vector',
            _seeded_symmetric(dimension, 5),
            Fraction(-1, 2),
            _seeded_vector(dimension, 4),
            zero_h,
        ),
    )
    if tuple(fixture.kind for fixture in fixtures) != FIXTURE_TYPES:
        raise ValueError('fixture ordering changed')
    return fixtures


def invariant_row(invariants: PotentialInvariants) -> tuple[Fraction, ...]:
    return (
        invariants.riemann_squared,
        invariants.ricci_squared,
        invariants.ricci_scalar**2,
        invariants.ricci_gradient_contraction,
        invariants.ricci_scalar * invariants.scalar_gradient_squared,
        invariants.scalar_gradient_squared**2,
        invariants.box_phi**2,
    )


def eq19_density(
    row: tuple[Fraction, ...],
    ricci_scalar: Fraction,
    *,
    trace_identity: Fraction,
    trace_potential: Fraction,
    trace_potential_squared: Fraction,
    trace_field_strength_squared: Fraction,
) -> Fraction:
    if len(row) != len(RAW_BASIS):
        raise ValueError('Eq. (19) requires the fixed seven-invariant basis')
    e, q, r_squared, _, _, _, _ = row
    if ricci_scalar**2 != r_squared:
        raise ValueError('signed Ricci scalar does not match the invariant row')
    return (
        Fraction(2 * e - 2 * q + 5 * r_squared, 360)
        * trace_identity
        + Fraction(1, 2) * trace_potential_squared
        - Fraction(1, 6) * ricci_scalar * trace_potential
        + Fraction(1, 12) * trace_field_strength_squared
    )


def _matrix_rank(rows: tuple[tuple[Fraction, ...], ...]) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError('rank matrix must be rectangular and nonempty')
    work = [list(row) for row in rows]
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [
            value / pivot_value for value in work[pivot_row]
        ]
        for row in range(len(work)):
            if row == pivot_row or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                work[row][index] - factor * work[pivot_row][index]
                for index in range(width)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


@dataclass(frozen=True)
class ExactFit:
    coefficients: tuple[Fraction, ...]
    residuals: tuple[Fraction, ...]
    rank: int
    selected_indices: tuple[int, ...]


def exact_fit(
    rows: tuple[tuple[Fraction, ...], ...],
    values: tuple[Fraction, ...],
) -> ExactFit:
    if len(rows) != len(values) or not rows:
        raise ValueError('fit rows and values must be nonempty and aligned')
    width = len(rows[0])
    rank = _matrix_rank(rows)
    if rank < width:
        raise ValueError(f'design rank {rank} is smaller than basis rank {width}')
    selected: list[int] = []
    selected_rows: tuple[tuple[Fraction, ...], ...] = ()
    for index, row in enumerate(rows):
        candidate = selected_rows + (row,)
        if _matrix_rank(candidate) > len(selected_rows):
            selected.append(index)
            selected_rows = candidate
        if len(selected) == width:
            break
    if len(selected) != width:
        raise ValueError('could not select a square full-rank design')
    inverse = invert_matrix(selected_rows)
    selected_values = tuple(values[index] for index in selected)
    coefficients = tuple(
        sum(
            (
                inverse[row][column] * selected_values[column]
                for column in range(width)
            ),
            Fraction(0),
        )
        for row in range(width)
    )
    residuals = tuple(
        sum(
            (
                row[index] * coefficients[index]
                for index in range(width)
            ),
            Fraction(0),
        )
        - value
        for row, value in zip(rows, values, strict=True)
    )
    return ExactFit(
        coefficients=coefficients,
        residuals=residuals,
        rank=rank,
        selected_indices=tuple(selected),
    )


def interpolate_polynomial(
    points: tuple[int, ...],
    values: tuple[Fraction, ...],
    degree: int,
) -> tuple[Fraction, ...]:
    if len(points) != degree + 1 or len(values) != len(points):
        raise ValueError('polynomial interpolation needs degree+1 points')
    rows = tuple(
        tuple(Fraction(point) ** power for power in range(degree + 1))
        for point in points
    )
    fit = exact_fit(rows, values)
    if any(fit.residuals):
        raise ValueError('polynomial interpolation residual is nonzero')
    return fit.coefficients


def evaluate_polynomial(
    coefficients: tuple[Fraction, ...],
    point: int,
) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


@dataclass(frozen=True)
class RawSynthesisSample:
    dimension: int
    kind: str
    invariant_row: tuple[Fraction, ...]
    ricci_scalar: Fraction
    bulk_divergence: Fraction
    euler_density: Fraction
    bosonic_identity_trace: Fraction
    bosonic_potential_trace: Fraction
    bosonic_potential_squared_raw: Fraction
    bosonic_potential_squared_bulk: Fraction
    bosonic_field_strength_trace: Fraction
    bosonic_density: Fraction
    ghost_identity_trace: Fraction
    ghost_potential_trace: Fraction
    ghost_potential_squared_trace: Fraction
    ghost_field_strength_trace: Fraction
    ghost_density: Fraction
    curvature_audit_passed: bool


def raw_synthesis_sample(
    fixture: SynthesisFixture,
    *,
    inverse_metric: Matrix,
    bosonic_field_strength_override: Fraction | None = None,
    ghost_field_strength_override: Fraction | None = None,
) -> RawSynthesisSample:
    invariants = potential_invariants(
        fixture.curvature,
        fixture.vector,
        fixture.hessian,
    )
    row = invariant_row(invariants)
    potential_values = potential_trace_values(
        fixture.curvature,
        fixture.vector,
        fixture.hessian,
        inverse_metric=inverse_metric,
    )
    divergence = bulk_divergence(invariants)
    bosonic_bulk_squared = (
        potential_values.trace_potential_squared_raw - 4 * divergence
    )
    bosonic_identity = matrix_trace(
        bundle_identity_matrix(fixture.dimension)
    )
    bosonic_field_strength = (
        bundle_curvature_squared_trace(fixture.curvature)
        if bosonic_field_strength_override is None
        else bosonic_field_strength_override
    )
    bosonic_density = eq19_density(
        row,
        invariants.ricci_scalar,
        trace_identity=bosonic_identity,
        trace_potential=potential_values.trace_potential,
        trace_potential_squared=bosonic_bulk_squared,
        trace_field_strength_squared=bosonic_field_strength,
    )

    ricci = ricci_from_curvature(fixture.curvature)
    ghost_y = ghost_potential(ricci, fixture.vector)
    ghost_identity = ghost_matrix_trace(identity_matrix(fixture.dimension))
    ghost_potential_trace = ghost_matrix_trace(ghost_y)
    ghost_potential_squared = ghost_matrix_squared_trace(ghost_y)
    ghost_field_strength = (
        field_strength_matrix_trace(fixture.curvature)
        if ghost_field_strength_override is None
        else ghost_field_strength_override
    )
    ghost_density = eq19_density(
        row,
        invariants.ricci_scalar,
        trace_identity=ghost_identity,
        trace_potential=ghost_potential_trace,
        trace_potential_squared=ghost_potential_squared,
        trace_field_strength_squared=ghost_field_strength,
    )
    ricci_for_audit = ricci_from_curvature(fixture.curvature)
    audit = audit_curvature(
        fixture.curvature,
        ricci_for_audit,
        fixture.symmetric,
    )
    return RawSynthesisSample(
        dimension=fixture.dimension,
        kind=fixture.kind,
        invariant_row=row,
        ricci_scalar=invariants.ricci_scalar,
        bulk_divergence=divergence,
        euler_density=(
            row[0] - 4 * row[1] + row[2]
        ),
        bosonic_identity_trace=bosonic_identity,
        bosonic_potential_trace=potential_values.trace_potential,
        bosonic_potential_squared_raw=(
            potential_values.trace_potential_squared_raw
        ),
        bosonic_potential_squared_bulk=bosonic_bulk_squared,
        bosonic_field_strength_trace=bosonic_field_strength,
        bosonic_density=bosonic_density,
        ghost_identity_trace=ghost_identity,
        ghost_potential_trace=ghost_potential_trace,
        ghost_potential_squared_trace=ghost_potential_squared,
        ghost_field_strength_trace=ghost_field_strength,
        ghost_density=ghost_density,
        curvature_audit_passed=audit.passed,
    )


def _coefficient_mismatch_l1(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> Fraction:
    if len(left) != len(right):
        raise ValueError('coefficient vectors use different bases')
    return sum(
        (abs(left[index] - right[index]) for index in range(len(left))),
        Fraction(0),
    )


def _residual_l1(values: tuple[Fraction, ...]) -> Fraction:
    return sum((abs(value) for value in values), Fraction(0))


def _no_weyl_design_rank(
    fixtures: tuple[SynthesisFixture, ...],
) -> int:
    rows = tuple(
        invariant_row(
            potential_invariants(
                kulkarni_nomizu_curvature(fixture.symmetric),
                fixture.vector,
                fixture.hessian,
            )
        )
        for fixture in fixtures
    )
    return _matrix_rank(rows)


@dataclass(frozen=True)
class OperatorTraceSynthesisReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    source_equations: tuple[str, ...]
    source_transcription_sha256: str
    upstream_hashes: tuple[tuple[str, str], ...]
    local_transcription_lock_passed: bool
    upstream_contracts_verified: bool
    frame_convention: str
    raw_basis: tuple[str, ...]
    identification_dimensions: tuple[int, ...]
    holdout_dimensions: tuple[int, ...]
    fixture_types: tuple[str, ...]
    fixtures_per_dimension: int
    fixture_count: int
    curvature_audit_count: int
    curvature_audits_all_passed: bool
    direct_field_strength_audit_count: int
    direct_field_strength_audit_residuals: tuple[str, ...]
    direct_field_strength_audits_all_passed: bool
    pure_weyl_nonzero_and_ricci_flat: bool
    generic_invariants_live: bool
    design_ranks: tuple[int, ...]
    no_weyl_design_ranks: tuple[int, ...]
    selected_fit_indices: tuple[tuple[int, ...], ...]
    full_rank_identification_passed: bool
    no_weyl_rank_loss_detected: bool
    bosonic_coefficients: tuple[tuple[str, ...], ...]
    ghost_coefficients: tuple[tuple[str, ...], ...]
    source_eq23_coefficients: tuple[tuple[str, ...], ...]
    source_eq27_coefficients: tuple[tuple[str, ...], ...]
    bosonic_fit_residuals: tuple[str, ...]
    ghost_fit_residuals: tuple[str, ...]
    exact_fit_residual_count: int
    exact_fits_all_passed: bool
    source_coefficient_residuals: tuple[str, ...]
    source_coefficient_component_count: int
    source_coefficients_all_matched: bool
    bosonic_lift_polynomials: tuple[tuple[str, ...], ...]
    ghost_lift_polynomials: tuple[tuple[str, ...], ...]
    lift_holdout_residuals: tuple[str, ...]
    lift_holdout_component_count: int
    polynomial_lift_holdout_passed: bool
    combined_coefficients: tuple[tuple[str, ...], ...]
    n4_combined_raw_coefficients: tuple[str, ...]
    n4_combined_p_coefficient: str
    n4_gb_reduced_coefficients: tuple[str, ...]
    source_eq28_with_p_slot: tuple[str, ...]
    n4_combination_and_gb_passed: bool
    generic_euler_density_mismatch_l1: str
    pointwise_gauss_bonnet_rejected: bool
    omitted_bulk_quotient_residual_l1: str
    wrong_plus_bulk_quotient_residual_l1: str
    wrong_field_strength_sign_mismatch_l1: str
    omitted_scalar_identity_mismatch_l1: str
    omitted_r_potential_term_mismatch_l1: str
    wrong_ghost_outer_sign_mismatch_l1: str
    coefficient_permutation_mismatch_l1: str
    corrupted_raw_density_residual_l1: str
    wrong_ghost_weight_plus_two_mismatch_l1: str
    wrong_ghost_weight_minus_one_mismatch_l1: str
    wrong_ghost_weight_zero_mismatch_l1: str
    premature_bosonic_p_deletion_mismatch_l1: str
    n4_copy_to_n8_mismatch_l1: str
    n3_full_rank_identification_rejected: bool
    n2_pole_rejected: bool
    primitive_dimension_basis: tuple[str, ...]
    primitive_length_dimensions: tuple[int, ...]
    quantity_dimension_basis: tuple[str, ...]
    quantity_length_dimensions: tuple[int, ...]
    corrupted_gradient_length_dimensions: tuple[int, ...]
    corrupted_curvature_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    eq19_source_supplied_theorem: bool
    raw_bundle_traces_independently_computed: bool
    raw_ghost_traces_independently_computed: bool
    source_targets_used_only_after_fit: bool
    bulk_divergence_quotient_adopted: bool
    ghost_weight_from_finite_berezin_adopted: bool
    four_dimensional_gb_bulk_quotient_adopted: bool
    polynomial_lift_degree_bound_admitted: bool
    admitted_lift_degree: int
    derivation_status: str
    source_eq22_coefficients_used_as_fit_input: bool
    source_eq23_eq27_used_as_fit_input: bool
    eq19_theorem_independently_derived: bool
    all_n_symbolic_identity_proved: bool
    bulk_divergence_pointwise_zero_claimed: bool
    gauss_bonnet_pointwise_zero_claimed: bool
    finite_boundary_completed: bool
    global_minimal_operator_derived: bool
    functional_measure_derived: bool
    functional_determinant_computed: bool
    heat_kernel_proper_time_integral_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    evanescent_terms_controlled: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    declared_operator_trace_synthesis_gate_passed: bool


def evaluate_operator_trace_synthesis_gate() -> OperatorTraceSynthesisReceipt:
    contract = operator_trace_synthesis_contract()
    validate_contract(contract)
    validate_upstream_contracts()

    dimensions = (
        contract.identification_dimensions + contract.holdout_dimensions
    )
    fixtures_by_dimension: dict[int, tuple[SynthesisFixture, ...]] = {}
    samples_by_dimension: dict[int, tuple[RawSynthesisSample, ...]] = {}
    bosonic_fits: dict[int, ExactFit] = {}
    ghost_fits: dict[int, ExactFit] = {}
    source_bosonic: dict[int, tuple[Fraction, ...]] = {}
    source_ghost: dict[int, tuple[Fraction, ...]] = {}
    design_ranks: list[int] = []
    no_weyl_ranks: list[int] = []
    direct_field_strength_residuals: list[Fraction] = []

    for dimension in dimensions:
        fixtures = synthesis_fixtures(dimension)
        inverse_metric = invert_matrix(raw_dewitt_metric(dimension))
        for audit_index in (0, 7):
            audit_fixture = fixtures[audit_index]
            e_value = curvature_squared(audit_fixture.curvature)
            direct_field_strength_residuals.extend(
                (
                    bundle_curvature_squared_trace(
                        audit_fixture.curvature
                    )
                    + (dimension + 2) * e_value,
                    field_strength_matrix_trace(
                        audit_fixture.curvature
                    )
                    + e_value,
                )
            )
        samples = tuple(
            raw_synthesis_sample(
                fixture,
                inverse_metric=inverse_metric,
                bosonic_field_strength_override=(
                    -(dimension + 2)
                    * curvature_squared(fixture.curvature)
                ),
                ghost_field_strength_override=(
                    -curvature_squared(fixture.curvature)
                ),
            )
            for fixture in fixtures
        )
        rows = tuple(sample.invariant_row for sample in samples)
        bosonic_fit = exact_fit(
            rows,
            tuple(sample.bosonic_density for sample in samples),
        )
        ghost_fit = exact_fit(
            rows,
            tuple(sample.ghost_density for sample in samples),
        )
        fixtures_by_dimension[dimension] = fixtures
        samples_by_dimension[dimension] = samples
        bosonic_fits[dimension] = bosonic_fit
        ghost_fits[dimension] = ghost_fit
        source_bosonic[dimension] = equation_23_coefficients(dimension)
        source_ghost[dimension] = equation_27_ghost_coefficients(dimension)
        design_ranks.append(bosonic_fit.rank)
        no_weyl_ranks.append(_no_weyl_design_rank(fixtures))

    bosonic_fit_residuals = tuple(
        residual
        for dimension in dimensions
        for residual in bosonic_fits[dimension].residuals
    )
    ghost_fit_residuals = tuple(
        residual
        for dimension in dimensions
        for residual in ghost_fits[dimension].residuals
    )
    source_coefficient_residuals = tuple(
        residual
        for dimension in dimensions
        for residual in (
            tuple(
                bosonic_fits[dimension].coefficients[index]
                - source_bosonic[dimension][index]
                for index in range(len(RAW_BASIS))
            )
            + tuple(
                ghost_fits[dimension].coefficients[index]
                - source_ghost[dimension][index]
                for index in range(len(RAW_BASIS))
            )
        )
    )

    bosonic_polynomials: list[tuple[Fraction, ...]] = []
    ghost_polynomials: list[tuple[Fraction, ...]] = []
    holdout_dimension = contract.holdout_dimensions[0]
    lift_holdout_residuals: list[Fraction] = []
    for index in range(len(RAW_BASIS)):
        bosonic_lift_values = tuple(
            Fraction(
                contract.common_lift_factor * (dimension - 2)
            )
            * bosonic_fits[dimension].coefficients[index]
            for dimension in contract.identification_dimensions
        )
        ghost_lift_values = tuple(
            Fraction(
                contract.common_lift_factor * (dimension - 2)
            )
            * ghost_fits[dimension].coefficients[index]
            for dimension in contract.identification_dimensions
        )
        bosonic_polynomial = interpolate_polynomial(
            contract.identification_dimensions,
            bosonic_lift_values,
            contract.admitted_lift_degree,
        )
        ghost_polynomial = interpolate_polynomial(
            contract.identification_dimensions,
            ghost_lift_values,
            contract.admitted_lift_degree,
        )
        bosonic_polynomials.append(bosonic_polynomial)
        ghost_polynomials.append(ghost_polynomial)
        lift_holdout_residuals.extend(
            (
                evaluate_polynomial(
                    bosonic_polynomial,
                    holdout_dimension,
                )
                - Fraction(
                    contract.common_lift_factor
                    * (holdout_dimension - 2)
                )
                * bosonic_fits[holdout_dimension].coefficients[index],
                evaluate_polynomial(
                    ghost_polynomial,
                    holdout_dimension,
                )
                - Fraction(
                    contract.common_lift_factor
                    * (holdout_dimension - 2)
                )
                * ghost_fits[holdout_dimension].coefficients[index],
            )
        )

    combined_coefficients = {
        dimension: combine_ghost_weight(
            bosonic_fits[dimension].coefficients,
            ghost_fits[dimension].coefficients,
            contract.ghost_weight,
        )
        for dimension in dimensions
    }
    n4_combined = combined_coefficients[4]
    n4_reduced = four_dimensional_bulk_gb_quotient(
        n4_combined,
        spacetime_dimension=4,
    )
    source_eq28 = source_equation_28_with_p_slot()

    n4_samples = samples_by_dimension[4]
    n4_rows = tuple(sample.invariant_row for sample in n4_samples)
    omitted_bulk_values = tuple(
        sample.bosonic_density + 2 * sample.bulk_divergence
        for sample in n4_samples
    )
    wrong_plus_bulk_values = tuple(
        sample.bosonic_density + 4 * sample.bulk_divergence
        for sample in n4_samples
    )
    omitted_bulk_fit = exact_fit(n4_rows, omitted_bulk_values)
    wrong_plus_bulk_fit = exact_fit(n4_rows, wrong_plus_bulk_values)

    wrong_field_strength_values = tuple(
        sample.bosonic_density
        - Fraction(1, 6) * sample.bosonic_field_strength_trace
        for sample in n4_samples
    )
    wrong_field_strength_fit = exact_fit(
        n4_rows,
        wrong_field_strength_values,
    )
    wrong_field_strength_control = _coefficient_mismatch_l1(
        wrong_field_strength_fit.coefficients,
        source_bosonic[4],
    )

    omitted_scalar_identity_values = tuple(
        sample.bosonic_density
        - Fraction(
            2 * sample.invariant_row[0]
            - 2 * sample.invariant_row[1]
            + 5 * sample.invariant_row[2],
            360,
        )
        for sample in n4_samples
    )
    omitted_scalar_identity_fit = exact_fit(
        n4_rows,
        omitted_scalar_identity_values,
    )
    omitted_scalar_identity_control = _coefficient_mismatch_l1(
        omitted_scalar_identity_fit.coefficients,
        source_bosonic[4],
    )

    omitted_r_potential_values = tuple(
        sample.bosonic_density
        + Fraction(1, 6)
        * sample.ricci_scalar
        * sample.bosonic_potential_trace
        for sample in n4_samples
    )
    omitted_r_potential_fit = exact_fit(
        n4_rows,
        omitted_r_potential_values,
    )
    omitted_r_potential_control = _coefficient_mismatch_l1(
        omitted_r_potential_fit.coefficients,
        source_bosonic[4],
    )

    wrong_ghost_outer_values: list[Fraction] = []
    for fixture, sample in zip(
        fixtures_by_dimension[4],
        n4_samples,
        strict=True,
    ):
        wrong_y = ghost_potential(
            ricci_from_curvature(fixture.curvature),
            fixture.vector,
            outer_sign=-1,
        )
        wrong_ghost_outer_values.append(
            eq19_density(
                sample.invariant_row,
                sample.ricci_scalar,
                trace_identity=sample.ghost_identity_trace,
                trace_potential=ghost_matrix_trace(wrong_y),
                trace_potential_squared=ghost_matrix_squared_trace(wrong_y),
                trace_field_strength_squared=(
                    sample.ghost_field_strength_trace
                ),
            )
        )
    wrong_ghost_outer_fit = exact_fit(
        n4_rows,
        tuple(wrong_ghost_outer_values),
    )
    wrong_ghost_outer_control = _coefficient_mismatch_l1(
        wrong_ghost_outer_fit.coefficients,
        source_ghost[4],
    )

    permuted_bosonic = list(bosonic_fits[4].coefficients)
    permuted_bosonic[0], permuted_bosonic[1] = (
        permuted_bosonic[1],
        permuted_bosonic[0],
    )
    coefficient_permutation_control = _coefficient_mismatch_l1(
        tuple(permuted_bosonic),
        source_bosonic[4],
    )

    corrupted_density_values = [
        sample.bosonic_density for sample in n4_samples
    ]
    corrupted_density_values[-1] += Fraction(1)
    corrupted_density_fit = exact_fit(
        n4_rows,
        tuple(corrupted_density_values),
    )
    corrupted_density_control = _residual_l1(
        corrupted_density_fit.residuals
    )

    correct_raw_n4 = combined_coefficients[4]
    wrong_weight_controls = {}
    for weight in (2, -1, 0):
        wrong_weight_controls[weight] = _coefficient_mismatch_l1(
            combine_ghost_weight(
                bosonic_fits[4].coefficients,
                ghost_fits[4].coefficients,
                weight,
            ),
            correct_raw_n4,
        )
    premature_bosonic = list(bosonic_fits[4].coefficients)
    premature_bosonic[3] = Fraction(0)
    premature_p_control = _coefficient_mismatch_l1(
        combine_ghost_weight(
            tuple(premature_bosonic),
            ghost_fits[4].coefficients,
            contract.ghost_weight,
        ),
        correct_raw_n4,
    )
    n4_copy_control = (
        _coefficient_mismatch_l1(
            bosonic_fits[4].coefficients,
            bosonic_fits[8].coefficients,
        )
        + _coefficient_mismatch_l1(
            ghost_fits[4].coefficients,
            ghost_fits[8].coefficients,
        )
    )

    generic_euler_control = sum(
        (
            abs(sample.euler_density)
            for sample in n4_samples
            if sample.kind.startswith('generic')
        ),
        Fraction(0),
    )

    n3_rejected = False
    try:
        synthesis_fixtures(3)
    except ValueError:
        n3_rejected = True
    n2_fixture_rejected = False
    n2_metric_rejected = False
    n2_source_rejected = False
    try:
        synthesis_fixtures(2)
    except ValueError:
        n2_fixture_rejected = True
    try:
        raw_dewitt_metric(2)
    except ValueError:
        n2_metric_rejected = True
    try:
        equation_23_coefficients(2)
    except ValueError:
        n2_source_rejected = True
    n2_rejected = (
        n2_fixture_rejected and n2_metric_rejected and n2_source_rejected
    )

    primitive_dimensions = tuple(PRIMITIVE_LENGTH_DIMENSIONS.values())
    quantity_dimensions = derive_synthesis_length_dimensions()
    corrupted_gradient_dimensions = derive_synthesis_length_dimensions(
        {'ScalarGradient': 0}
    )
    corrupted_curvature_dimensions = derive_synthesis_length_dimensions(
        {'Curvature': -1}
    )
    dimension_gate_passed = (
        tuple(RAW_BASIS) == SYNTHESIS_DIMENSION_BASIS[:7]
        and quantity_dimensions == (-4,) * 9 + (0,)
        and corrupted_gradient_dimensions != quantity_dimensions
        and corrupted_curvature_dimensions != quantity_dimensions
    )

    all_samples = tuple(
        sample
        for dimension in dimensions
        for sample in samples_by_dimension[dimension]
    )
    generic_samples = tuple(
        sample
        for sample in all_samples
        if sample.kind in ('generic-a', 'generic-b', 'generic-c')
    )
    generic_invariants_live = all(
        all(value != 0 for value in sample.invariant_row)
        and sample.bulk_divergence != 0
        for sample in generic_samples
    )
    pure_weyl_checks = tuple(
        sample.invariant_row[0] > 0
        and sample.invariant_row[1] == 0
        and sample.invariant_row[2] == 0
        for sample in all_samples
        if sample.kind == 'pure-weyl'
    )
    exact_fits_passed = (
        all(value == 0 for value in bosonic_fit_residuals)
        and all(value == 0 for value in ghost_fit_residuals)
    )
    source_coefficients_passed = all(
        value == 0 for value in source_coefficient_residuals
    )
    lift_holdout_passed = all(
        value == 0 for value in lift_holdout_residuals
    )
    n4_combination_passed = (
        n4_combined[3] == 0
        and n4_reduced == source_eq28
    )

    controls_nonzero = all(
        value > 0
        for value in (
            _residual_l1(omitted_bulk_fit.residuals),
            _residual_l1(wrong_plus_bulk_fit.residuals),
            wrong_field_strength_control,
            omitted_scalar_identity_control,
            omitted_r_potential_control,
            wrong_ghost_outer_control,
            coefficient_permutation_control,
            corrupted_density_control,
            wrong_weight_controls[2],
            wrong_weight_controls[-1],
            wrong_weight_controls[0],
            premature_p_control,
            n4_copy_control,
            generic_euler_control,
        )
    )
    bounded_false = (
        contract.source_eq22_coefficients_used_as_fit_input,
        contract.source_eq23_eq27_used_as_fit_input,
        contract.eq19_theorem_independently_derived,
        contract.all_n_symbolic_identity_proved,
        contract.bulk_divergence_pointwise_zero_claimed,
        contract.gauss_bonnet_pointwise_zero_claimed,
        contract.finite_boundary_completed,
        contract.global_minimal_operator_derived,
        contract.functional_measure_derived,
        contract.functional_determinant_computed,
        contract.heat_kernel_proper_time_integral_derived,
        contract.loop_integral_evaluated,
        contract.regularization_scheme_implemented,
        contract.evanescent_terms_controlled,
        contract.renormalization_proof,
        contract.continuum_st_qme_proved,
        contract.local_covariance_proved,
        contract.in_in_ctp_completed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
    )
    declared_passed = all(
        (
            len(all_samples)
            == len(dimensions) * contract.fixtures_per_dimension,
            all(sample.curvature_audit_passed for sample in all_samples),
            all(value == 0 for value in direct_field_strength_residuals),
            all(pure_weyl_checks),
            generic_invariants_live,
            tuple(design_ranks) == (7,) * len(dimensions),
            all(rank < 7 for rank in no_weyl_ranks),
            exact_fits_passed,
            source_coefficients_passed,
            lift_holdout_passed,
            n4_combination_passed,
            controls_nonzero,
            n3_rejected,
            n2_rejected,
            dimension_gate_passed,
            not any(bounded_false),
        )
    )

    return OperatorTraceSynthesisReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        source_equations=contract.source_equations,
        source_transcription_sha256=contract.source_transcription_sha256,
        upstream_hashes=contract.upstream_hashes,
        local_transcription_lock_passed=True,
        upstream_contracts_verified=contract.upstream_contracts_verified,
        frame_convention=contract.frame_convention,
        raw_basis=contract.raw_basis,
        identification_dimensions=contract.identification_dimensions,
        holdout_dimensions=contract.holdout_dimensions,
        fixture_types=contract.fixture_types,
        fixtures_per_dimension=contract.fixtures_per_dimension,
        fixture_count=len(all_samples),
        curvature_audit_count=len(all_samples),
        curvature_audits_all_passed=all(
            sample.curvature_audit_passed for sample in all_samples
        ),
        direct_field_strength_audit_count=len(
            direct_field_strength_residuals
        ),
        direct_field_strength_audit_residuals=tuple(
            _fraction_text(value)
            for value in direct_field_strength_residuals
        ),
        direct_field_strength_audits_all_passed=all(
            value == 0 for value in direct_field_strength_residuals
        ),
        pure_weyl_nonzero_and_ricci_flat=all(pure_weyl_checks),
        generic_invariants_live=generic_invariants_live,
        design_ranks=tuple(design_ranks),
        no_weyl_design_ranks=tuple(no_weyl_ranks),
        selected_fit_indices=tuple(
            bosonic_fits[dimension].selected_indices
            for dimension in dimensions
        ),
        full_rank_identification_passed=(
            tuple(design_ranks) == (7,) * len(dimensions)
        ),
        no_weyl_rank_loss_detected=all(
            rank < 7 for rank in no_weyl_ranks
        ),
        bosonic_coefficients=tuple(
            tuple(
                _fraction_text(value)
                for value in bosonic_fits[dimension].coefficients
            )
            for dimension in dimensions
        ),
        ghost_coefficients=tuple(
            tuple(
                _fraction_text(value)
                for value in ghost_fits[dimension].coefficients
            )
            for dimension in dimensions
        ),
        source_eq23_coefficients=tuple(
            tuple(
                _fraction_text(value)
                for value in source_bosonic[dimension]
            )
            for dimension in dimensions
        ),
        source_eq27_coefficients=tuple(
            tuple(
                _fraction_text(value)
                for value in source_ghost[dimension]
            )
            for dimension in dimensions
        ),
        bosonic_fit_residuals=tuple(
            _fraction_text(value) for value in bosonic_fit_residuals
        ),
        ghost_fit_residuals=tuple(
            _fraction_text(value) for value in ghost_fit_residuals
        ),
        exact_fit_residual_count=(
            len(bosonic_fit_residuals) + len(ghost_fit_residuals)
        ),
        exact_fits_all_passed=exact_fits_passed,
        source_coefficient_residuals=tuple(
            _fraction_text(value)
            for value in source_coefficient_residuals
        ),
        source_coefficient_component_count=len(
            source_coefficient_residuals
        ),
        source_coefficients_all_matched=source_coefficients_passed,
        bosonic_lift_polynomials=tuple(
            tuple(_fraction_text(value) for value in polynomial)
            for polynomial in bosonic_polynomials
        ),
        ghost_lift_polynomials=tuple(
            tuple(_fraction_text(value) for value in polynomial)
            for polynomial in ghost_polynomials
        ),
        lift_holdout_residuals=tuple(
            _fraction_text(value) for value in lift_holdout_residuals
        ),
        lift_holdout_component_count=len(lift_holdout_residuals),
        polynomial_lift_holdout_passed=lift_holdout_passed,
        combined_coefficients=tuple(
            tuple(
                _fraction_text(value)
                for value in combined_coefficients[dimension]
            )
            for dimension in dimensions
        ),
        n4_combined_raw_coefficients=tuple(
            _fraction_text(value) for value in n4_combined
        ),
        n4_combined_p_coefficient=_fraction_text(n4_combined[3]),
        n4_gb_reduced_coefficients=tuple(
            _fraction_text(value) for value in n4_reduced
        ),
        source_eq28_with_p_slot=tuple(
            _fraction_text(value) for value in source_eq28
        ),
        n4_combination_and_gb_passed=n4_combination_passed,
        generic_euler_density_mismatch_l1=_fraction_text(
            generic_euler_control
        ),
        pointwise_gauss_bonnet_rejected=generic_euler_control > 0,
        omitted_bulk_quotient_residual_l1=_fraction_text(
            _residual_l1(omitted_bulk_fit.residuals)
        ),
        wrong_plus_bulk_quotient_residual_l1=_fraction_text(
            _residual_l1(wrong_plus_bulk_fit.residuals)
        ),
        wrong_field_strength_sign_mismatch_l1=_fraction_text(
            wrong_field_strength_control
        ),
        omitted_scalar_identity_mismatch_l1=_fraction_text(
            omitted_scalar_identity_control
        ),
        omitted_r_potential_term_mismatch_l1=_fraction_text(
            omitted_r_potential_control
        ),
        wrong_ghost_outer_sign_mismatch_l1=_fraction_text(
            wrong_ghost_outer_control
        ),
        coefficient_permutation_mismatch_l1=_fraction_text(
            coefficient_permutation_control
        ),
        corrupted_raw_density_residual_l1=_fraction_text(
            corrupted_density_control
        ),
        wrong_ghost_weight_plus_two_mismatch_l1=_fraction_text(
            wrong_weight_controls[2]
        ),
        wrong_ghost_weight_minus_one_mismatch_l1=_fraction_text(
            wrong_weight_controls[-1]
        ),
        wrong_ghost_weight_zero_mismatch_l1=_fraction_text(
            wrong_weight_controls[0]
        ),
        premature_bosonic_p_deletion_mismatch_l1=_fraction_text(
            premature_p_control
        ),
        n4_copy_to_n8_mismatch_l1=_fraction_text(n4_copy_control),
        n3_full_rank_identification_rejected=n3_rejected,
        n2_pole_rejected=n2_rejected,
        primitive_dimension_basis=tuple(PRIMITIVE_LENGTH_DIMENSIONS),
        primitive_length_dimensions=primitive_dimensions,
        quantity_dimension_basis=SYNTHESIS_DIMENSION_BASIS,
        quantity_length_dimensions=quantity_dimensions,
        corrupted_gradient_length_dimensions=(
            corrupted_gradient_dimensions
        ),
        corrupted_curvature_length_dimensions=(
            corrupted_curvature_dimensions
        ),
        dimension_gate_passed=dimension_gate_passed,
        eq19_source_supplied_theorem=(
            contract.eq19_source_supplied_theorem
        ),
        raw_bundle_traces_independently_computed=(
            contract.raw_bundle_traces_independently_computed
        ),
        raw_ghost_traces_independently_computed=(
            contract.raw_ghost_traces_independently_computed
        ),
        source_targets_used_only_after_fit=(
            contract.source_targets_used_only_after_fit
        ),
        bulk_divergence_quotient_adopted=(
            contract.bulk_divergence_quotient_adopted
        ),
        ghost_weight_from_finite_berezin_adopted=(
            contract.ghost_weight_from_finite_berezin_adopted
        ),
        four_dimensional_gb_bulk_quotient_adopted=(
            contract.four_dimensional_gb_bulk_quotient_adopted
        ),
        polynomial_lift_degree_bound_admitted=(
            contract.polynomial_lift_degree_bound_admitted
        ),
        admitted_lift_degree=contract.admitted_lift_degree,
        derivation_status=contract.derivation_status,
        source_eq22_coefficients_used_as_fit_input=(
            contract.source_eq22_coefficients_used_as_fit_input
        ),
        source_eq23_eq27_used_as_fit_input=(
            contract.source_eq23_eq27_used_as_fit_input
        ),
        eq19_theorem_independently_derived=(
            contract.eq19_theorem_independently_derived
        ),
        all_n_symbolic_identity_proved=(
            contract.all_n_symbolic_identity_proved
        ),
        bulk_divergence_pointwise_zero_claimed=(
            contract.bulk_divergence_pointwise_zero_claimed
        ),
        gauss_bonnet_pointwise_zero_claimed=(
            contract.gauss_bonnet_pointwise_zero_claimed
        ),
        finite_boundary_completed=contract.finite_boundary_completed,
        global_minimal_operator_derived=(
            contract.global_minimal_operator_derived
        ),
        functional_measure_derived=contract.functional_measure_derived,
        functional_determinant_computed=(
            contract.functional_determinant_computed
        ),
        heat_kernel_proper_time_integral_derived=(
            contract.heat_kernel_proper_time_integral_derived
        ),
        loop_integral_evaluated=contract.loop_integral_evaluated,
        regularization_scheme_implemented=(
            contract.regularization_scheme_implemented
        ),
        evanescent_terms_controlled=contract.evanescent_terms_controlled,
        renormalization_proof=contract.renormalization_proof,
        continuum_st_qme_proved=contract.continuum_st_qme_proved,
        local_covariance_proved=contract.local_covariance_proved,
        in_in_ctp_completed=contract.in_in_ctp_completed,
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        declared_operator_trace_synthesis_gate_passed=declared_passed,
    )
