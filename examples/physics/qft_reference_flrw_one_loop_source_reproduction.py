'''Exact-rational reproduction of arXiv:1706.02622v7, Eqs. (28)--(30).

This module does not evaluate a loop integral.  It locks a published
Einstein--massless-quantum-scalar counterterm to an exact rational basis and
checks its supplied background-EOM reduction in two algebraic ways.
'''

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib


SOURCE_ID = 'arXiv:1706.02622v7'
SOURCE_DATE = '2021-09-11'
SOURCE_TITLE = 'One-loop divergences in first order Einstein-Hilbert gravity'
SOURCE_THEORY = 'Einstein-Hilbert+minimally-coupled-massless-quantum-scalar'
SOURCE_GAUGE = 'de-Donder-harmonic'
SOURCE_URL = 'https://arxiv.org/html/1706.02622v7'
SOURCE_EQUATION_28 = 'Eq28'
SOURCE_EQUATION_30 = 'Eq30'
SOURCE_PREFACTOR = '(4*pi)^-2*epsilon^-1'
ORDERED_BASIS = ('RicciSq', 'R2', 'R_X', 'X2', 'BoxPhi2')
SOURCE_COEFFICIENTS = ('43/60', '1/40', '1/6', '1', '1')
SOURCE_EQ30 = '203/40*R2'
SOURCE_TRANSCRIPTION_SHA256 = (
    '37653a585f767212830cf49ba21cc9661c6509fa3f49d4f78c4a16ce5c869189'
)
PRIMITIVE_LENGTH_DIMENSIONS = {
    'RicciTensor': -2,
    'RicciScalar': -2,
    'ScalarGradientSquared': -2,
    'BoxScalar': -2,
}
MONOMIAL_PRIMITIVE_FACTORS = {
    'RicciSq': ('RicciTensor', 'RicciTensor'),
    'R2': ('RicciScalar', 'RicciScalar'),
    'R_X': ('RicciScalar', 'ScalarGradientSquared'),
    'X2': ('ScalarGradientSquared', 'ScalarGradientSquared'),
    'BoxPhi2': ('BoxScalar', 'BoxScalar'),
}


@dataclass(frozen=True)
class OneLoopSourceContract:
    source_id: str
    source_date: str
    source_title: str
    source_theory: str
    source_gauge: str
    source_url: str
    equation_28_id: str
    equation_30_id: str
    prefactor: str
    ordered_basis: tuple[str, ...]
    coefficients: tuple[str, ...]
    equation_30: str
    source_transcription_sha256: str
    spacetime_dimension: int
    quantum_scalar_multiplicity: int
    scalar_loop_retained: bool
    scalar_background_zero_removes_scalar_loop: bool
    gauss_bonnet_identity_used_by_source: bool
    boundary_counterterm_computed: bool
    derivation_status: str
    loop_integral_evaluated: bool
    heat_kernel_trace_derived: bool
    ghost_determinant_derived: bool
    regularization_scheme_implemented: bool
    independent_feynman_diagram_check: bool
    renormalization_proof: bool
    pure_einstein_coefficients_claimed: bool
    continuum_st_qme_proved: bool
    in_in_ctp_computed: bool
    positive_physical_hilbert_computed: bool
    nonperturbative_m2_passed: bool


def one_loop_source_contract() -> OneLoopSourceContract:
    return OneLoopSourceContract(
        source_id=SOURCE_ID,
        source_date=SOURCE_DATE,
        source_title=SOURCE_TITLE,
        source_theory=SOURCE_THEORY,
        source_gauge=SOURCE_GAUGE,
        source_url=SOURCE_URL,
        equation_28_id=SOURCE_EQUATION_28,
        equation_30_id=SOURCE_EQUATION_30,
        prefactor=SOURCE_PREFACTOR,
        ordered_basis=ORDERED_BASIS,
        coefficients=SOURCE_COEFFICIENTS,
        equation_30=SOURCE_EQ30,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        spacetime_dimension=4,
        quantum_scalar_multiplicity=1,
        scalar_loop_retained=True,
        scalar_background_zero_removes_scalar_loop=False,
        gauss_bonnet_identity_used_by_source=True,
        boundary_counterterm_computed=False,
        derivation_status='source_reproduction_only',
        loop_integral_evaluated=False,
        heat_kernel_trace_derived=False,
        ghost_determinant_derived=False,
        regularization_scheme_implemented=False,
        independent_feynman_diagram_check=False,
        renormalization_proof=False,
        pure_einstein_coefficients_claimed=False,
        continuum_st_qme_proved=False,
        in_in_ctp_computed=False,
        positive_physical_hilbert_computed=False,
        nonperturbative_m2_passed=False,
    )


def canonical_source_payload(contract: OneLoopSourceContract) -> str:
    separator = chr(44)
    return '|'.join(
        (
            contract.source_id,
            contract.source_date,
            f'title={contract.source_title}',
            f'theory={contract.source_theory}',
            f'gauge={contract.source_gauge}',
            contract.equation_28_id,
            f'prefactor={contract.prefactor}',
            f'basis={separator.join(contract.ordered_basis)}',
            f'coeff={separator.join(contract.coefficients)}',
            f'{contract.equation_30_id}={contract.equation_30}',
        )
    )


def source_payload_sha256(contract: OneLoopSourceContract) -> str:
    payload = canonical_source_payload(contract).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def derive_monomial_length_dimensions(
    primitive_dimensions: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    '''Derive Eq. (28) monomial dimensions from primitive length exponents.'''
    dimensions = dict(PRIMITIVE_LENGTH_DIMENSIONS)
    if primitive_dimensions is not None:
        dimensions.update(primitive_dimensions)
    return tuple(
        sum(dimensions[factor] for factor in MONOMIAL_PRIMITIVE_FACTORS[basis])
        for basis in ORDERED_BASIS
    )


def validate_contract(contract: OneLoopSourceContract) -> None:
    frozen_metadata = (
        contract.source_id == SOURCE_ID,
        contract.source_date == SOURCE_DATE,
        contract.source_title == SOURCE_TITLE,
        contract.source_theory == SOURCE_THEORY,
        contract.source_gauge == SOURCE_GAUGE,
        contract.source_url == SOURCE_URL,
        contract.equation_28_id == SOURCE_EQUATION_28,
        contract.equation_30_id == SOURCE_EQUATION_30,
        contract.prefactor == SOURCE_PREFACTOR,
        contract.ordered_basis == ORDERED_BASIS,
        contract.coefficients == SOURCE_COEFFICIENTS,
        contract.equation_30 == SOURCE_EQ30,
    )
    if not all(frozen_metadata):
        raise ValueError('source edition, equation, field content, or rational table changed')
    computed_hash = source_payload_sha256(contract)
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or computed_hash != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('source transcription hash mismatch')
    if contract.spacetime_dimension != 4:
        raise ValueError('the source reduction and Gauss-Bonnet step are four-dimensional')
    if contract.quantum_scalar_multiplicity != 1:
        raise ValueError('v7 Eq. (28) retains exactly one quantum scalar')
    if not contract.scalar_loop_retained:
        raise ValueError('the scalar determinant cannot be dropped from this source gate')
    if contract.scalar_background_zero_removes_scalar_loop:
        raise ValueError('zero scalar background does not remove its quantum determinant')
    if not contract.gauss_bonnet_identity_used_by_source:
        raise ValueError('v7 Eq. (28) records use of the four-dimensional GB identity')
    if contract.boundary_counterterm_computed:
        raise ValueError('this source gate has no finite-boundary completion')
    if contract.derivation_status != 'source_reproduction_only':
        raise ValueError('this module is source reproduction only')
    unsupported_promotions = (
        contract.loop_integral_evaluated,
        contract.heat_kernel_trace_derived,
        contract.ghost_determinant_derived,
        contract.regularization_scheme_implemented,
        contract.independent_feynman_diagram_check,
        contract.renormalization_proof,
        contract.pure_einstein_coefficients_claimed,
        contract.continuum_st_qme_proved,
        contract.in_in_ctp_computed,
        contract.positive_physical_hilbert_computed,
        contract.nonperturbative_m2_passed,
    )
    if any(unsupported_promotions):
        raise ValueError('unsupported loop, pure-Einstein, QME, CTP, Hilbert, or M2 claim')


def _parse_fraction(value: str) -> Fraction:
    parsed = Fraction(value)
    canonical = (
        str(parsed.numerator)
        if parsed.denominator == 1
        else f'{parsed.numerator}/{parsed.denominator}'
    )
    if canonical != value:
        raise ValueError('source rationals must be stored canonically')
    return parsed


def source_coefficient_vector(
    contract: OneLoopSourceContract,
) -> tuple[Fraction, ...]:
    validate_contract(contract)
    return tuple(_parse_fraction(value) for value in contract.coefficients)


@dataclass(frozen=True)
class RationalBackgroundPoint:
    ricci_scalar: Fraction
    ricci_tensor_squared: Fraction
    scalar_gradient_squared: Fraction
    box_scalar: Fraction


def counterterm_density(
    coefficients: tuple[Fraction, ...],
    point: RationalBackgroundPoint,
) -> Fraction:
    if len(coefficients) != len(ORDERED_BASIS):
        raise ValueError('counterterm vector has the wrong basis dimension')
    values = (
        point.ricci_tensor_squared,
        point.ricci_scalar**2,
        point.ricci_scalar * point.scalar_gradient_squared,
        point.scalar_gradient_squared**2,
        point.box_scalar**2,
    )
    return sum(
        (coefficient * value for coefficient, value in zip(coefficients, values, strict=True)),
        start=Fraction(0),
    )


@dataclass(frozen=True)
class BackgroundEomAudit:
    trace_r_coefficient: str
    trace_x_coefficient: str
    scalar_gradient_squared_over_r: str
    ricci_gradient_outer_product_coefficient: str
    ricci_tensor_squared_over_r_squared: str
    box_scalar_over_r: str


def derive_background_eom_reduction(dimension: int = 4) -> BackgroundEomAudit:
    '''Derive X=2R, Ricci^2=R^2 and box(phi)=0 from source Eq. (3).'''

    if dimension != 4:
        raise ValueError('the locked EOM reduction is four-dimensional')
    trace_r = Fraction(dimension, 2) - 1
    trace_x = -Fraction(dimension, 4) + Fraction(1, 2)
    x_over_r = -trace_r / trace_x
    metric_r_after_trace = Fraction(1, 2) - Fraction(1, 4) * x_over_r
    if metric_r_after_trace != 0:
        raise AssertionError('source tensor EOM did not cancel its metric term')
    ricci_gradient_coefficient = Fraction(1, 2)
    q_over_r2 = ricci_gradient_coefficient**2 * x_over_r**2
    return BackgroundEomAudit(
        trace_r_coefficient=str(trace_r),
        trace_x_coefficient=str(trace_x),
        scalar_gradient_squared_over_r=str(x_over_r),
        ricci_gradient_outer_product_coefficient=str(
            ricci_gradient_coefficient
        ),
        ricci_tensor_squared_over_r_squared=str(q_over_r2),
        box_scalar_over_r='0',
    )


def eom_ideal_identity(
    coefficients: tuple[Fraction, ...],
    point: RationalBackgroundPoint,
    equation_30_coefficient: Fraction,
) -> tuple[Fraction, Fraction]:
    '''Return the two sides of the exact Eq. (28)-to-(30) residual identity.'''

    r = point.ricci_scalar
    delta_q = point.ricci_tensor_squared - r**2
    delta_x = point.scalar_gradient_squared - 2 * r
    left = counterterm_density(coefficients, point) - equation_30_coefficient * r**2
    right = (
        Fraction(43, 60) * delta_q
        + Fraction(25, 6) * r * delta_x
        + delta_x**2
        + point.box_scalar**2
    )
    return left, right


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def _l1_fraction_distance(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> Fraction:
    return sum(
        (abs(a - b) for a, b in zip(left, right, strict=True)),
        start=Fraction(0),
    )


@dataclass(frozen=True)
class OneLoopSourceReproductionReceipt:
    source_id: str
    source_date: str
    source_title: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_equations: tuple[str, str]
    source_transcription_sha256: str
    source_lock_passed: bool
    prefactor: str
    ordered_basis: tuple[str, ...]
    exact_coefficient_vector: tuple[str, ...]
    monomial_length_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    quantum_scalar_multiplicity: int
    scalar_loop_retained: bool
    scalar_background_zero_removes_scalar_loop: bool
    background_eom: BackgroundEomAudit
    on_shell_term_contributions: tuple[str, ...]
    direct_equation_30_coefficient: str
    source_equation_30_coefficient: str
    on_shell_rational_sample_count: int
    on_shell_samples_all_exact: bool
    off_shell_rational_sample_count: int
    eom_ideal_identity_all_exact: bool
    stale_pure_gravity_vector_mismatch_l1: str
    equation_28_30_confusion_mismatch_l1: str
    scalar_background_zero_shortcut_residual: str
    omitted_x_squared_control_residual: str
    omitted_r_x_control_residual: str
    wrong_eom_substitution_residual: str
    half_ricci_coefficient_control_residual: str
    linear_box_scalar_control_residual: str
    gauss_bonnet_identity_used_by_source: bool
    boundary_counterterm_computed: bool
    derivation_status: str
    loop_integral_evaluated: bool
    heat_kernel_trace_derived: bool
    ghost_determinant_derived: bool
    regularization_scheme_implemented: bool
    independent_feynman_diagram_check: bool
    renormalization_proof: bool
    pure_einstein_coefficients_claimed: bool
    continuum_st_qme_proved: bool
    in_in_ctp_computed: bool
    positive_physical_hilbert_computed: bool
    nonperturbative_m2_passed: bool
    declared_one_loop_source_reproduction_gate_passed: bool


def evaluate_one_loop_source_reproduction_gate(
    contract: OneLoopSourceContract | None = None,
) -> OneLoopSourceReproductionReceipt:
    if contract is None:
        contract = one_loop_source_contract()
    validate_contract(contract)
    coefficients = source_coefficient_vector(contract)
    equation_30 = Fraction(203, 40)
    eom = derive_background_eom_reduction(contract.spacetime_dimension)
    x_over_r = _parse_fraction(eom.scalar_gradient_squared_over_r)
    q_over_r2 = _parse_fraction(eom.ricci_tensor_squared_over_r_squared)

    unit_on_shell = RationalBackgroundPoint(
        ricci_scalar=Fraction(1),
        ricci_tensor_squared=q_over_r2,
        scalar_gradient_squared=x_over_r,
        box_scalar=Fraction(0),
    )
    contributions = (
        coefficients[0] * unit_on_shell.ricci_tensor_squared,
        coefficients[1] * unit_on_shell.ricci_scalar**2,
        coefficients[2]
        * unit_on_shell.ricci_scalar
        * unit_on_shell.scalar_gradient_squared,
        coefficients[3] * unit_on_shell.scalar_gradient_squared**2,
        coefficients[4] * unit_on_shell.box_scalar**2,
    )
    direct_equation_30 = sum(contributions, start=Fraction(0))

    on_shell_r_values = (
        Fraction(-2),
        Fraction(-1, 3),
        Fraction(0),
        Fraction(5, 7),
    )
    on_shell_exact = True
    for r in on_shell_r_values:
        point = RationalBackgroundPoint(
            ricci_scalar=r,
            ricci_tensor_squared=q_over_r2 * r**2,
            scalar_gradient_squared=x_over_r * r,
            box_scalar=Fraction(0),
        )
        left, right = eom_ideal_identity(coefficients, point, equation_30)
        on_shell_exact = on_shell_exact and (
            counterterm_density(coefficients, point) == equation_30 * r**2
            and left == 0
            and right == 0
        )

    off_shell_points = (
        RationalBackgroundPoint(Fraction(1), Fraction(2), Fraction(3), Fraction(4)),
        RationalBackgroundPoint(
            Fraction(-2, 3), Fraction(5, 7), Fraction(-1, 5), Fraction(2, 9)
        ),
        RationalBackgroundPoint(
            Fraction(0), Fraction(-3, 2), Fraction(4, 3), Fraction(-5, 4)
        ),
    )
    identity_exact = all(
        (lambda sides: sides[0] == sides[1])(
            eom_ideal_identity(coefficients, point, equation_30)
        )
        for point in off_shell_points
    )

    stale_vector = (
        Fraction(7, 10),
        Fraction(1, 60),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    equation_confusion_vector = (
        Fraction(0),
        equation_30,
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    phi_zero_shortcut = RationalBackgroundPoint(
        Fraction(1), Fraction(1), Fraction(0), Fraction(0)
    )
    phi_zero_residual = abs(
        counterterm_density(coefficients, phi_zero_shortcut) - equation_30
    )
    without_x_squared = list(coefficients)
    without_x_squared[3] = Fraction(0)
    without_r_x = list(coefficients)
    without_r_x[2] = Fraction(0)
    wrong_eom_point = RationalBackgroundPoint(
        Fraction(1), Fraction(1, 4), Fraction(1), Fraction(0)
    )
    half_ricci = list(coefficients)
    half_ricci[0] = Fraction(43, 120)
    box_control_point = RationalBackgroundPoint(
        Fraction(0), Fraction(0), Fraction(0), Fraction(2)
    )
    correct_box_density = counterterm_density(coefficients, box_control_point)
    wrong_linear_box_density = Fraction(2)

    unsupported_flags = (
        contract.loop_integral_evaluated,
        contract.heat_kernel_trace_derived,
        contract.ghost_determinant_derived,
        contract.regularization_scheme_implemented,
        contract.independent_feynman_diagram_check,
        contract.renormalization_proof,
        contract.pure_einstein_coefficients_claimed,
        contract.continuum_st_qme_proved,
        contract.in_in_ctp_computed,
        contract.positive_physical_hilbert_computed,
        contract.nonperturbative_m2_passed,
    )
    source_lock = source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    monomial_dimensions = derive_monomial_length_dimensions()
    expected_density_dimension = -contract.spacetime_dimension
    dimension_gate = all(
        value == expected_density_dimension for value in monomial_dimensions
    )
    stale_mismatch = _l1_fraction_distance(coefficients, stale_vector)
    confusion_mismatch = _l1_fraction_distance(
        coefficients, equation_confusion_vector
    )
    omitted_x2_residual = abs(
        counterterm_density(tuple(without_x_squared), unit_on_shell)
        - equation_30
    )
    omitted_rx_residual = abs(
        counterterm_density(tuple(without_r_x), unit_on_shell) - equation_30
    )
    wrong_eom_residual = abs(
        counterterm_density(coefficients, wrong_eom_point) - equation_30
    )
    half_ricci_residual = abs(
        counterterm_density(tuple(half_ricci), unit_on_shell) - equation_30
    )
    linear_box_residual = abs(correct_box_density - wrong_linear_box_density)

    passed = bool(
        source_lock
        and coefficients
        == (
            Fraction(43, 60),
            Fraction(1, 40),
            Fraction(1, 6),
            Fraction(1),
            Fraction(1),
        )
        and dimension_gate
        and direct_equation_30 == equation_30
        and on_shell_exact
        and identity_exact
        and stale_mismatch > 0
        and confusion_mismatch > 0
        and phi_zero_residual > 0
        and omitted_x2_residual == 4
        and omitted_rx_residual == Fraction(1, 3)
        and wrong_eom_residual > 0
        and half_ricci_residual == Fraction(43, 120)
        and linear_box_residual == 2
        and contract.quantum_scalar_multiplicity == 1
        and contract.scalar_loop_retained
        and not contract.scalar_background_zero_removes_scalar_loop
        and contract.gauss_bonnet_identity_used_by_source
        and not contract.boundary_counterterm_computed
        and contract.derivation_status == 'source_reproduction_only'
        and not any(unsupported_flags)
    )
    return OneLoopSourceReproductionReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_title=contract.source_title,
        source_theory=contract.source_theory,
        source_gauge=contract.source_gauge,
        source_url=contract.source_url,
        source_equations=(contract.equation_28_id, contract.equation_30_id),
        source_transcription_sha256=contract.source_transcription_sha256,
        source_lock_passed=source_lock,
        prefactor=contract.prefactor,
        ordered_basis=contract.ordered_basis,
        exact_coefficient_vector=tuple(_fraction_text(value) for value in coefficients),
        monomial_length_dimensions=monomial_dimensions,
        dimension_gate_passed=dimension_gate,
        quantum_scalar_multiplicity=contract.quantum_scalar_multiplicity,
        scalar_loop_retained=contract.scalar_loop_retained,
        scalar_background_zero_removes_scalar_loop=(
            contract.scalar_background_zero_removes_scalar_loop
        ),
        background_eom=eom,
        on_shell_term_contributions=tuple(
            _fraction_text(value) for value in contributions
        ),
        direct_equation_30_coefficient=_fraction_text(direct_equation_30),
        source_equation_30_coefficient=_fraction_text(equation_30),
        on_shell_rational_sample_count=len(on_shell_r_values),
        on_shell_samples_all_exact=on_shell_exact,
        off_shell_rational_sample_count=len(off_shell_points),
        eom_ideal_identity_all_exact=identity_exact,
        stale_pure_gravity_vector_mismatch_l1=_fraction_text(stale_mismatch),
        equation_28_30_confusion_mismatch_l1=_fraction_text(confusion_mismatch),
        scalar_background_zero_shortcut_residual=_fraction_text(phi_zero_residual),
        omitted_x_squared_control_residual=_fraction_text(omitted_x2_residual),
        omitted_r_x_control_residual=_fraction_text(omitted_rx_residual),
        wrong_eom_substitution_residual=_fraction_text(wrong_eom_residual),
        half_ricci_coefficient_control_residual=_fraction_text(
            half_ricci_residual
        ),
        linear_box_scalar_control_residual=_fraction_text(linear_box_residual),
        gauss_bonnet_identity_used_by_source=(
            contract.gauss_bonnet_identity_used_by_source
        ),
        boundary_counterterm_computed=contract.boundary_counterterm_computed,
        derivation_status=contract.derivation_status,
        loop_integral_evaluated=contract.loop_integral_evaluated,
        heat_kernel_trace_derived=contract.heat_kernel_trace_derived,
        ghost_determinant_derived=contract.ghost_determinant_derived,
        regularization_scheme_implemented=(
            contract.regularization_scheme_implemented
        ),
        independent_feynman_diagram_check=(
            contract.independent_feynman_diagram_check
        ),
        renormalization_proof=contract.renormalization_proof,
        pure_einstein_coefficients_claimed=(
            contract.pure_einstein_coefficients_claimed
        ),
        continuum_st_qme_proved=contract.continuum_st_qme_proved,
        in_in_ctp_computed=contract.in_in_ctp_computed,
        positive_physical_hilbert_computed=(
            contract.positive_physical_hilbert_computed
        ),
        nonperturbative_m2_passed=contract.nonperturbative_m2_passed,
        declared_one_loop_source_reproduction_gate_passed=passed,
    )
