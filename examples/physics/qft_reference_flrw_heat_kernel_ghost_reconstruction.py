'''Exact source-coefficient assembly for arXiv:1706.02622v7, Eqs. (23)--(29).

This module evaluates the published n-dependent coefficient formulae.  It does
not derive the heat-kernel traces, a ghost determinant, or a loop integral.
The Gauss--Bonnet operation is an explicitly four-dimensional integrated-bulk
quotient; no finite-boundary completion is claimed.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_COEFFICIENTS,
    SOURCE_DATE,
    SOURCE_GAUGE,
    SOURCE_ID,
    SOURCE_PREFACTOR,
    SOURCE_THEORY,
    SOURCE_TITLE,
    SOURCE_URL,
)


HTML_INTERNAL_HEADING = 'One-Loop in first order quantum gravity'
SOURCE_EQUATIONS = ('Eq23', 'Eq27', 'Eq28', 'Eq29')
RAW_BASIS = (
    'RiemannSq',
    'RicciSq',
    'R2',
    'RicciGradPhiGradPhi',
    'R_X',
    'X2',
    'BoxPhi2',
)
REDUCED_BASIS = ('RicciSq', 'R2', 'RicciGradPhiGradPhi', 'R_X', 'X2', 'BoxPhi2')
COMMON_DENOMINATOR = 360
GHOST_WEIGHT = -2
EQ23_FORMULAE = (
    '(482-29*n+n^2)/360',
    '(724-1440*n+181*n^2-n^3)/(360*(n-2))',
    '5*(140+264*n-145*n^2+25*n^3)/(720*(n-2))',
    '-360*(-4-2*n+n^2)/(360*(n-2))',
    '-15*(32+62*n-37*n^2+5*n^3)/(360*(n-2))',
    '45*(n^3-n^2+14*n-40)/(720*(n-2))',
    '360/360',
)
EQ27_FORMULAE = (
    '(2*n-30)/360',
    '(180-2*n)/360',
    '(5*n+60)/360',
    '-360/360',
    '-60/360',
    '180/360',
    '0/360',
)
EXPECTED_EQ23_AT_FOUR = tuple(
    Fraction(value)
    for value in ('191/180', '-551/180', '119/72', '-2', '-1/6', '2', '1')
)
EXPECTED_EQ27_AT_FOUR = tuple(
    Fraction(value)
    for value in ('-11/180', '43/90', '2/9', '-1', '-1/6', '1/2', '0')
)
EXPECTED_RAW_AT_FOUR = tuple(
    Fraction(value)
    for value in ('71/60', '-241/60', '29/24', '0', '1/6', '1', '1')
)
EXPECTED_EQ23_NUMERATORS = (382, -1102, 595, -720, -60, 720, 360)
EXPECTED_EQ27_NUMERATORS = (-22, 172, 80, -360, -60, 180, 0)
EXPECTED_RAW_NUMERATORS = (426, -1446, 435, 0, 60, 360, 360)
EXPECTED_REDUCED_WITH_P = tuple(
    Fraction(value) for value in ('43/60', '1/40', '0', '1/6', '1', '1')
)
PRIMITIVE_LENGTH_DIMENSIONS = {
    'RiemannTensor': -2,
    'RicciTensor': -2,
    'RicciScalar': -2,
    'ScalarGradientSquared': -2,
    'BoxScalar': -2,
}
MONOMIAL_PRIMITIVE_FACTORS = {
    'RiemannSq': ('RiemannTensor', 'RiemannTensor'),
    'RicciSq': ('RicciTensor', 'RicciTensor'),
    'R2': ('RicciScalar', 'RicciScalar'),
    'RicciGradPhiGradPhi': ('RicciTensor', 'ScalarGradientSquared'),
    'R_X': ('RicciScalar', 'ScalarGradientSquared'),
    'X2': ('ScalarGradientSquared', 'ScalarGradientSquared'),
    'BoxPhi2': ('BoxScalar', 'BoxScalar'),
}
SOURCE_TRANSCRIPTION_SHA256 = (
    '88cb5281c058f0983281d2e20017be987de6e2ab6bb53af41fa6fcc205ae9f17'
)


@dataclass(frozen=True)
class HeatKernelGhostContract:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_equations: tuple[str, ...]
    source_prefactor: str
    raw_basis: tuple[str, ...]
    reduced_basis: tuple[str, ...]
    eq23_formulae: tuple[str, ...]
    eq27_formulae: tuple[str, ...]
    source_transcription_sha256: str
    spacetime_dimension: int
    common_denominator: int
    ghost_weight: int
    gauss_bonnet_bulk_quotient_used: bool
    gauss_bonnet_pointwise_identity_claimed: bool
    derivation_status: str
    heat_kernel_trace_derived: bool
    ghost_determinant_derived: bool
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


def heat_kernel_ghost_contract() -> HeatKernelGhostContract:
    return HeatKernelGhostContract(
        source_id=SOURCE_ID,
        source_date=SOURCE_DATE,
        source_metadata_title=SOURCE_TITLE,
        html_internal_heading=HTML_INTERNAL_HEADING,
        source_theory=SOURCE_THEORY,
        source_gauge=SOURCE_GAUGE,
        source_url=SOURCE_URL,
        source_equations=SOURCE_EQUATIONS,
        source_prefactor=SOURCE_PREFACTOR,
        raw_basis=RAW_BASIS,
        reduced_basis=REDUCED_BASIS,
        eq23_formulae=EQ23_FORMULAE,
        eq27_formulae=EQ27_FORMULAE,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        spacetime_dimension=4,
        common_denominator=COMMON_DENOMINATOR,
        ghost_weight=GHOST_WEIGHT,
        gauss_bonnet_bulk_quotient_used=True,
        gauss_bonnet_pointwise_identity_claimed=False,
        derivation_status='source_coefficient_assembly_only',
        heat_kernel_trace_derived=False,
        ghost_determinant_derived=False,
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


def canonical_source_payload(contract: HeatKernelGhostContract) -> str:
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
            f'prefactor={contract.source_prefactor}',
            f'raw_basis={separator.join(contract.raw_basis)}',
            f'reduced_basis={separator.join(contract.reduced_basis)}',
            f'eq23={separator.join(contract.eq23_formulae)}',
            f'eq27={separator.join(contract.eq27_formulae)}',
            f'n={contract.spacetime_dimension}',
            f'denominator={contract.common_denominator}',
            f'ghost_weight={contract.ghost_weight}',
        )
    )


def source_payload_sha256(contract: HeatKernelGhostContract) -> str:
    return hashlib.sha256(canonical_source_payload(contract).encode('utf-8')).hexdigest()


def validate_contract(contract: HeatKernelGhostContract) -> None:
    frozen = (
        contract.source_id == SOURCE_ID,
        contract.source_date == SOURCE_DATE,
        contract.source_metadata_title == SOURCE_TITLE,
        contract.html_internal_heading == HTML_INTERNAL_HEADING,
        contract.source_theory == SOURCE_THEORY,
        contract.source_gauge == SOURCE_GAUGE,
        contract.source_url == SOURCE_URL,
        contract.source_equations == SOURCE_EQUATIONS,
        contract.source_prefactor == SOURCE_PREFACTOR,
        contract.raw_basis == RAW_BASIS,
        contract.reduced_basis == REDUCED_BASIS,
        contract.eq23_formulae == EQ23_FORMULAE,
        contract.eq27_formulae == EQ27_FORMULAE,
        contract.spacetime_dimension == 4,
        contract.common_denominator == COMMON_DENOMINATOR,
        contract.ghost_weight == GHOST_WEIGHT,
    )
    if not all(frozen):
        raise ValueError('source metadata, equations, basis, or coefficient formula changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local source transcription hash mismatch')
    if not contract.gauss_bonnet_bulk_quotient_used:
        raise ValueError('the source assembly requires the declared 4D bulk quotient')
    if contract.gauss_bonnet_pointwise_identity_claimed:
        raise ValueError('Gauss-Bonnet is not a pointwise-zero identity in this gate')
    if contract.derivation_status != 'source_coefficient_assembly_only':
        raise ValueError('this gate is source coefficient assembly only')
    unsupported = (
        contract.heat_kernel_trace_derived,
        contract.ghost_determinant_derived,
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
    if any(unsupported):
        raise ValueError('unsupported derivation, boundary, continuum, or M2 promotion')


def equation_23_coefficients(n: int) -> tuple[Fraction, ...]:
    if n == 2:
        raise ValueError('Eq. (23) is singular at n=2')
    return (
        Fraction(482 - 29 * n + n**2, 360),
        Fraction(724 - 1440 * n + 181 * n**2 - n**3, 360 * (n - 2)),
        Fraction(5 * (140 + 264 * n - 145 * n**2 + 25 * n**3), 720 * (n - 2)),
        Fraction(-360 * (-4 - 2 * n + n**2), 360 * (n - 2)),
        Fraction(-15 * (32 + 62 * n - 37 * n**2 + 5 * n**3), 360 * (n - 2)),
        Fraction(45 * (n**3 - n**2 + 14 * n - 40), 720 * (n - 2)),
        Fraction(360, 360),
    )


def equation_27_ghost_coefficients(n: int) -> tuple[Fraction, ...]:
    return (
        Fraction(2 * n - 30, 360),
        Fraction(180 - 2 * n, 360),
        Fraction(5 * n + 60, 360),
        Fraction(-360, 360),
        Fraction(-60, 360),
        Fraction(180, 360),
        Fraction(0, 360),
    )


def combine_ghost_weight(
    bosonic: Sequence[Fraction],
    ghost: Sequence[Fraction],
    ghost_weight: int,
) -> tuple[Fraction, ...]:
    if len(bosonic) != len(ghost):
        raise ValueError('bosonic and ghost vectors must share one ordered basis')
    return tuple(left + ghost_weight * right for left, right in zip(bosonic, ghost))


def four_dimensional_bulk_gb_quotient(
    coefficients: Sequence[Fraction],
    *,
    spacetime_dimension: int = 4,
) -> tuple[Fraction, ...]:
    if spacetime_dimension != 4:
        raise ValueError('this Gauss-Bonnet quotient is admitted only after n=4 specialization')
    if len(coefficients) != len(RAW_BASIS):
        raise ValueError('Gauss-Bonnet quotient requires the seven-component raw basis')
    riemann, ricci, r_squared, p_term, r_x, x_squared, box_squared = coefficients
    return (
        4 * riemann + ricci,
        -riemann + r_squared,
        p_term,
        r_x,
        x_squared,
        box_squared,
    )


def source_equation_28_with_p_slot() -> tuple[Fraction, ...]:
    return (
        Fraction(SOURCE_COEFFICIENTS[0]),
        Fraction(SOURCE_COEFFICIENTS[1]),
        Fraction(0),
        Fraction(SOURCE_COEFFICIENTS[2]),
        Fraction(SOURCE_COEFFICIENTS[3]),
        Fraction(SOURCE_COEFFICIENTS[4]),
    )


def scaled_integer_vector(
    coefficients: Sequence[Fraction],
    denominator: int = COMMON_DENOMINATOR,
) -> tuple[int, ...]:
    scaled = tuple(value * denominator for value in coefficients)
    if any(value.denominator != 1 for value in scaled):
        raise ValueError('coefficient vector does not share the declared integer scale')
    return tuple(value.numerator for value in scaled)


def derive_monomial_length_dimensions(
    primitive_dimensions: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_dimensions is not None:
        dimensions.update(primitive_dimensions)
    return tuple(
        sum(dimensions[factor] for factor in MONOMIAL_PRIMITIVE_FACTORS[basis])
        for basis in RAW_BASIS
    )


def _l1_distance(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise ValueError('distance requires equal-length coefficient vectors')
    return sum((abs(a - b) for a, b in zip(left, right)), Fraction(0))


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


@dataclass(frozen=True)
class BulkRepresentativePoint:
    ricci_squared: Fraction
    r_squared: Fraction
    p_term: Fraction
    r_x: Fraction
    x_squared: Fraction
    box_squared: Fraction

    @property
    def riemann_squared_representative(self) -> Fraction:
        return 4 * self.ricci_squared - self.r_squared

    def raw_components(self) -> tuple[Fraction, ...]:
        return (
            self.riemann_squared_representative,
            self.ricci_squared,
            self.r_squared,
            self.p_term,
            self.r_x,
            self.x_squared,
            self.box_squared,
        )

    def reduced_components(self) -> tuple[Fraction, ...]:
        return (
            self.ricci_squared,
            self.r_squared,
            self.p_term,
            self.r_x,
            self.x_squared,
            self.box_squared,
        )


def coefficient_density(
    coefficients: Sequence[Fraction], components: Sequence[Fraction]
) -> Fraction:
    if len(coefficients) != len(components):
        raise ValueError('coefficient and component vectors must have equal length')
    return sum((coefficient * value for coefficient, value in zip(coefficients, components)), Fraction(0))


@dataclass(frozen=True)
class HeatKernelGhostReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_equations: tuple[str, ...]
    source_prefactor: str
    source_transcription_sha256: str
    local_transcription_lock_passed: bool
    independent_source_artifact_authenticated: bool
    spacetime_dimension: int
    raw_basis: tuple[str, ...]
    reduced_basis: tuple[str, ...]
    equation_23_integer_numerators: tuple[int, ...]
    equation_27_integer_numerators: tuple[int, ...]
    equation_23_exact_vector: tuple[str, ...]
    equation_27_exact_vector: tuple[str, ...]
    ghost_weight: int
    raw_integer_numerators: tuple[int, ...]
    raw_exact_vector: tuple[str, ...]
    p_term_cancels_only_after_ghost_subtraction: bool
    gauss_bonnet_bulk_quotient_used: bool
    gauss_bonnet_pointwise_identity_claimed: bool
    reduced_exact_vector_with_p_slot: tuple[str, ...]
    source_equation_28_vector_with_p_slot: tuple[str, ...]
    primitive_length_dimensions: tuple[tuple[str, int], ...]
    monomial_length_dimensions: tuple[int, ...]
    corrupted_x_dimension_vector: tuple[int, ...]
    dimension_gate_passed: bool
    bulk_representative_sample_count: int
    bulk_representative_samples_all_exact: bool
    broken_gb_representative_residual: str
    wrong_plus_two_ghost_mismatch_l1: str
    wrong_minus_one_ghost_mismatch_l1: str
    omitted_ghost_mismatch_l1: str
    premature_p_deletion_input_mismatch_l1: str
    wrong_gauss_bonnet_sign_mismatch_l1: str
    raw_equation_28_confusion_mismatch_l1: str
    permuted_eq23_basis_mismatch_l1: str
    omitted_r_squared_mismatch_l1: str
    omitted_r_x_mismatch_l1: str
    omitted_x_squared_mismatch_l1: str
    dimension_five_raw_mismatch_l1: str
    derivation_status: str
    heat_kernel_trace_derived: bool
    ghost_determinant_derived: bool
    loop_integral_evaluated: bool
    regularization_scheme_implemented: bool
    finite_boundary_completed: bool
    evanescent_terms_controlled: bool
    renormalization_proof: bool
    continuum_st_qme_proved: bool
    local_covariance_proved: bool
    in_in_ctp_completed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    declared_source_coefficient_assembly_gate_passed: bool


def evaluate_heat_kernel_ghost_reconstruction_gate() -> HeatKernelGhostReceipt:
    contract = heat_kernel_ghost_contract()
    validate_contract(contract)
    bosonic = equation_23_coefficients(contract.spacetime_dimension)
    ghost = equation_27_ghost_coefficients(contract.spacetime_dimension)
    raw = combine_ghost_weight(bosonic, ghost, contract.ghost_weight)
    reduced = four_dimensional_bulk_gb_quotient(
        raw, spacetime_dimension=contract.spacetime_dimension
    )
    source_final = source_equation_28_with_p_slot()

    bosonic_numerators = scaled_integer_vector(bosonic)
    ghost_numerators = scaled_integer_vector(ghost)
    raw_numerators = scaled_integer_vector(raw)
    p_cancellation = bosonic[3] != 0 and ghost[3] != 0 and raw[3] == 0

    monomial_dimensions = derive_monomial_length_dimensions()
    corrupted_dimensions = derive_monomial_length_dimensions(
        {'ScalarGradientSquared': -1}
    )
    expected_dimension = -contract.spacetime_dimension
    dimension_gate = (
        all(value == expected_dimension for value in monomial_dimensions)
        and corrupted_dimensions != monomial_dimensions
    )

    samples = (
        BulkRepresentativePoint(
            Fraction(3), Fraction(4), Fraction(5), Fraction(11), Fraction(13), Fraction(17)
        ),
        BulkRepresentativePoint(
            Fraction(-2, 3),
            Fraction(5, 7),
            Fraction(-11, 13),
            Fraction(17, 19),
            Fraction(-23, 29),
            Fraction(31, 37),
        ),
        BulkRepresentativePoint(
            Fraction(0), Fraction(-1, 5), Fraction(7, 9), Fraction(0), Fraction(2, 11), Fraction(-3, 8)
        ),
    )
    sample_residuals = tuple(
        coefficient_density(raw, sample.raw_components())
        - coefficient_density(reduced, sample.reduced_components())
        for sample in samples
    )
    broken_raw_components = list(samples[0].raw_components())
    broken_raw_components[0] += Fraction(1, 7)
    broken_residual = coefficient_density(raw, broken_raw_components) - coefficient_density(
        reduced, samples[0].reduced_components()
    )

    plus_two = four_dimensional_bulk_gb_quotient(
        combine_ghost_weight(bosonic, ghost, 2)
    )
    minus_one = four_dimensional_bulk_gb_quotient(
        combine_ghost_weight(bosonic, ghost, -1)
    )
    no_ghost = four_dimensional_bulk_gb_quotient(bosonic)
    wrong_gb = (
        -4 * raw[0] + raw[1],
        raw[0] + raw[2],
        raw[3],
        raw[4],
        raw[5],
        raw[6],
    )
    raw_without_riemann = raw[1:]
    permuted_bosonic = list(bosonic)
    permuted_bosonic[1], permuted_bosonic[2] = (
        permuted_bosonic[2],
        permuted_bosonic[1],
    )
    omitted_r_squared = list(source_final)
    omitted_r_squared[1] = Fraction(0)
    omitted_r_x = list(source_final)
    omitted_r_x[3] = Fraction(0)
    omitted_x_squared = list(source_final)
    omitted_x_squared[4] = Fraction(0)
    dimension_five_raw = combine_ghost_weight(
        equation_23_coefficients(5),
        equation_27_ghost_coefficients(5),
        contract.ghost_weight,
    )
    controls = (
        _l1_distance(plus_two, source_final),
        _l1_distance(minus_one, source_final),
        _l1_distance(no_ghost, source_final),
        abs(bosonic[3]) + abs(ghost[3]),
        _l1_distance(wrong_gb, source_final),
        _l1_distance(raw_without_riemann, source_final),
        _l1_distance(permuted_bosonic, EXPECTED_EQ23_AT_FOUR),
        _l1_distance(omitted_r_squared, source_final),
        _l1_distance(omitted_r_x, source_final),
        _l1_distance(omitted_x_squared, source_final),
        _l1_distance(dimension_five_raw, EXPECTED_RAW_AT_FOUR),
    )
    unsupported = (
        contract.heat_kernel_trace_derived,
        contract.ghost_determinant_derived,
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
    gate_passed = (
        source_lock
        and bosonic == EXPECTED_EQ23_AT_FOUR
        and ghost == EXPECTED_EQ27_AT_FOUR
        and bosonic_numerators == EXPECTED_EQ23_NUMERATORS
        and ghost_numerators == EXPECTED_EQ27_NUMERATORS
        and raw == EXPECTED_RAW_AT_FOUR
        and raw_numerators == EXPECTED_RAW_NUMERATORS
        and p_cancellation
        and reduced == EXPECTED_REDUCED_WITH_P
        and reduced == source_final
        and dimension_gate
        and all(residual == 0 for residual in sample_residuals)
        and broken_residual != 0
        and all(control > 0 for control in controls)
        and contract.gauss_bonnet_bulk_quotient_used
        and not contract.gauss_bonnet_pointwise_identity_claimed
        and not any(unsupported)
    )

    return HeatKernelGhostReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        html_internal_heading=contract.html_internal_heading,
        source_theory=contract.source_theory,
        source_gauge=contract.source_gauge,
        source_url=contract.source_url,
        source_equations=contract.source_equations,
        source_prefactor=contract.source_prefactor,
        source_transcription_sha256=contract.source_transcription_sha256,
        local_transcription_lock_passed=source_lock,
        independent_source_artifact_authenticated=(
            contract.independent_source_artifact_authenticated
        ),
        spacetime_dimension=contract.spacetime_dimension,
        raw_basis=contract.raw_basis,
        reduced_basis=contract.reduced_basis,
        equation_23_integer_numerators=bosonic_numerators,
        equation_27_integer_numerators=ghost_numerators,
        equation_23_exact_vector=tuple(_fraction_text(value) for value in bosonic),
        equation_27_exact_vector=tuple(_fraction_text(value) for value in ghost),
        ghost_weight=contract.ghost_weight,
        raw_integer_numerators=raw_numerators,
        raw_exact_vector=tuple(_fraction_text(value) for value in raw),
        p_term_cancels_only_after_ghost_subtraction=p_cancellation,
        gauss_bonnet_bulk_quotient_used=contract.gauss_bonnet_bulk_quotient_used,
        gauss_bonnet_pointwise_identity_claimed=(
            contract.gauss_bonnet_pointwise_identity_claimed
        ),
        reduced_exact_vector_with_p_slot=tuple(
            _fraction_text(value) for value in reduced
        ),
        source_equation_28_vector_with_p_slot=tuple(
            _fraction_text(value) for value in source_final
        ),
        primitive_length_dimensions=tuple(PRIMITIVE_LENGTH_DIMENSIONS.items()),
        monomial_length_dimensions=monomial_dimensions,
        corrupted_x_dimension_vector=corrupted_dimensions,
        dimension_gate_passed=dimension_gate,
        bulk_representative_sample_count=len(samples),
        bulk_representative_samples_all_exact=all(
            residual == 0 for residual in sample_residuals
        ),
        broken_gb_representative_residual=_fraction_text(broken_residual),
        wrong_plus_two_ghost_mismatch_l1=_fraction_text(controls[0]),
        wrong_minus_one_ghost_mismatch_l1=_fraction_text(controls[1]),
        omitted_ghost_mismatch_l1=_fraction_text(controls[2]),
        premature_p_deletion_input_mismatch_l1=_fraction_text(controls[3]),
        wrong_gauss_bonnet_sign_mismatch_l1=_fraction_text(controls[4]),
        raw_equation_28_confusion_mismatch_l1=_fraction_text(controls[5]),
        permuted_eq23_basis_mismatch_l1=_fraction_text(controls[6]),
        omitted_r_squared_mismatch_l1=_fraction_text(controls[7]),
        omitted_r_x_mismatch_l1=_fraction_text(controls[8]),
        omitted_x_squared_mismatch_l1=_fraction_text(controls[9]),
        dimension_five_raw_mismatch_l1=_fraction_text(controls[10]),
        derivation_status=contract.derivation_status,
        heat_kernel_trace_derived=contract.heat_kernel_trace_derived,
        ghost_determinant_derived=contract.ghost_determinant_derived,
        loop_integral_evaluated=contract.loop_integral_evaluated,
        regularization_scheme_implemented=contract.regularization_scheme_implemented,
        finite_boundary_completed=contract.finite_boundary_completed,
        evanescent_terms_controlled=contract.evanescent_terms_controlled,
        renormalization_proof=contract.renormalization_proof,
        continuum_st_qme_proved=contract.continuum_st_qme_proved,
        local_covariance_proved=contract.local_covariance_proved,
        in_in_ctp_completed=contract.in_in_ctp_completed,
        positive_physical_hilbert_proved=contract.positive_physical_hilbert_proved,
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        declared_source_coefficient_assembly_gate_passed=gate_passed,
    )
