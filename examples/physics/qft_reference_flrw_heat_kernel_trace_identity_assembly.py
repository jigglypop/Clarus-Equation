'''Symbolic source-trace assembly for arXiv:1706.02622v7, Eqs. (19)--(27).

Eq. (19), the Eq. (22) trace identities, the ghost operator conventions, and
the downstream ghost weight are source-supplied inputs. This module checks an
exact rational-polynomial assembly identity. It does not derive heat-kernel
trace tensors, a determinant, a loop integral, or renormalization.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_heat_kernel_ghost_reconstruction import (
    GHOST_WEIGHT,
    HTML_INTERNAL_HEADING,
    RAW_BASIS,
    derive_monomial_length_dimensions,
    equation_23_coefficients,
    equation_27_ghost_coefficients,
)
from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_DATE,
    SOURCE_GAUGE,
    SOURCE_ID,
    SOURCE_PREFACTOR,
    SOURCE_THEORY,
    SOURCE_TITLE,
    SOURCE_URL,
)


SOURCE_EQUATIONS = ('Eq19', 'Eq22', 'Eq23', 'Eq24', 'Eq25', 'Eq26', 'Eq27')
VERIFICATION_DIMENSIONS = (3, 4, 5)
EQ19_FORMULA = (
    '(2*RiemannSq-2*RicciSq+5*R2)*trI/360'
    '+trPotentialSq/2-R*trPotential/6+trFieldStrengthSq/12'
)
EQ22_TRACE_FORMULAE = (
    'trI=n*(n+1)/2+1',
    'trPotential_R=n*(n-1)/2',
    'trPotential_X=(8+3*n-n^2)/4',
    'trPotentialSq_RiemannSq=3',
    'trPotentialSq_RicciSq=(n^2-8*n+4)/(n-2)',
    'trPotentialSq_R2=(n^3-5*n^2+8*n+4)/(2*(n-2))',
    'trPotentialSq_P=-(2*n*(n-4)/(n-2)+4)',
    'trPotentialSq_RX=(n^3-7*n^2+10*n+8)/(2*(2-n))',
    'trPotentialSq_X2=(n^3-n^2+14*n-40)/(8*(n-2))',
    'trPotentialSq_BoxPhi2=2',
    'trFieldStrengthSq_RiemannSq=-(n+2)',
)
GHOST_TRACE_FORMULAE = (
    'trI=n',
    'trPotential=-R+X',
    'trPotentialSq=RicciSq-2*P+X2',
    'trFieldStrengthSq=-RiemannSq',
)
SOURCE_TRANSCRIPTION_SHA256 = (
    '684ace59f009a4ce2a3c680b835df786ea9bab0803ce308365645b5331811ebc'
)


def _trim_polynomial(values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    trimmed = list(values)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed) if trimmed else (Fraction(0),)


def _poly_add(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    size = max(len(left), len(right))
    return _trim_polynomial(
        tuple(
            (left[index] if index < len(left) else Fraction(0))
            + (right[index] if index < len(right) else Fraction(0))
            for index in range(size)
        )
    )


def _poly_scale(
    values: tuple[Fraction, ...], factor: Fraction
) -> tuple[Fraction, ...]:
    return _trim_polynomial(tuple(factor * value for value in values))


def _poly_multiply(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return _trim_polynomial(tuple(result))


def _poly_evaluate(values: tuple[Fraction, ...], point: int) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(values):
        result = result * point + coefficient
    return result


@dataclass(frozen=True)
class RationalPolynomial:
    numerator: tuple[Fraction, ...]
    denominator: tuple[Fraction, ...] = (Fraction(1),)

    def __post_init__(self) -> None:
        numerator = _trim_polynomial(
            tuple(Fraction(value) for value in self.numerator)
        )
        denominator = _trim_polynomial(
            tuple(Fraction(value) for value in self.denominator)
        )
        if all(value == 0 for value in denominator):
            raise ValueError('rational polynomial denominator cannot be zero')
        if denominator[-1] < 0:
            numerator = _poly_scale(numerator, Fraction(-1))
            denominator = _poly_scale(denominator, Fraction(-1))
        object.__setattr__(self, 'numerator', numerator)
        object.__setattr__(self, 'denominator', denominator)

    @classmethod
    def constant(cls, value: int | Fraction) -> RationalPolynomial:
        return cls((Fraction(value),))

    @classmethod
    def variable(cls) -> RationalPolynomial:
        return cls((Fraction(0), Fraction(1)))

    @staticmethod
    def _coerce(
        value: int | Fraction | RationalPolynomial,
    ) -> RationalPolynomial:
        if isinstance(value, RationalPolynomial):
            return value
        return RationalPolynomial.constant(value)

    def __add__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        other_value = self._coerce(other)
        return RationalPolynomial(
            _poly_add(
                _poly_multiply(self.numerator, other_value.denominator),
                _poly_multiply(other_value.numerator, self.denominator),
            ),
            _poly_multiply(self.denominator, other_value.denominator),
        )

    def __radd__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        return self + other

    def __neg__(self) -> RationalPolynomial:
        return RationalPolynomial(
            _poly_scale(self.numerator, Fraction(-1)), self.denominator
        )

    def __sub__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        return self + (-self._coerce(other))

    def __rsub__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        return self._coerce(other) - self

    def __mul__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        other_value = self._coerce(other)
        return RationalPolynomial(
            _poly_multiply(self.numerator, other_value.numerator),
            _poly_multiply(self.denominator, other_value.denominator),
        )

    def __rmul__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        return self * other

    def __truediv__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        other_value = self._coerce(other)
        if all(value == 0 for value in other_value.numerator):
            raise ZeroDivisionError('cannot divide by the zero rational polynomial')
        return RationalPolynomial(
            _poly_multiply(self.numerator, other_value.denominator),
            _poly_multiply(self.denominator, other_value.numerator),
        )

    def __rtruediv__(
        self, other: int | Fraction | RationalPolynomial
    ) -> RationalPolynomial:
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> RationalPolynomial:
        if exponent < 0:
            raise ValueError('negative rational-polynomial powers are not admitted')
        result = RationalPolynomial.constant(1)
        for _ in range(exponent):
            result = result * self
        return result

    def cross_residual(
        self, other: RationalPolynomial
    ) -> tuple[Fraction, ...]:
        return _poly_add(
            _poly_multiply(self.numerator, other.denominator),
            _poly_scale(
                _poly_multiply(other.numerator, self.denominator), Fraction(-1)
            ),
        )

    def equivalent(self, other: RationalPolynomial) -> bool:
        return all(value == 0 for value in self.cross_residual(other))

    def evaluate(self, point: int) -> Fraction:
        numerator = _poly_evaluate(self.numerator, point)
        denominator = _poly_evaluate(self.denominator, point)
        if denominator == 0:
            raise ZeroDivisionError(f'rational polynomial has a pole at n={point}')
        return numerator / denominator


ZERO = RationalPolynomial.constant(0)


@dataclass(frozen=True)
class TraceInputs:
    identity: RationalPolynomial
    potential_r: RationalPolynomial
    potential_x: RationalPolynomial
    potential_squared: tuple[RationalPolynomial, ...]
    field_strength_squared: tuple[RationalPolynomial, ...]


def _zero_vector() -> tuple[RationalPolynomial, ...]:
    return tuple(ZERO for _ in RAW_BASIS)


def eq22_trace_inputs() -> TraceInputs:
    n = RationalPolynomial.variable()
    potential_squared = list(_zero_vector())
    potential_squared[0] = RationalPolynomial.constant(3)
    potential_squared[1] = (n**2 - 8 * n + 4) / (n - 2)
    potential_squared[2] = (
        n**3 - 5 * n**2 + 8 * n + 4
    ) / (2 * (n - 2))
    potential_squared[3] = -(2 * n * (n - 4) / (n - 2) + 4)
    potential_squared[4] = (
        n**3 - 7 * n**2 + 10 * n + 8
    ) / (2 * (2 - n))
    potential_squared[5] = (
        n**3 - n**2 + 14 * n - 40
    ) / (8 * (n - 2))
    potential_squared[6] = RationalPolynomial.constant(2)
    field_strength_squared = list(_zero_vector())
    field_strength_squared[0] = -(n + 2)
    return TraceInputs(
        identity=n * (n + 1) / 2 + 1,
        potential_r=n * (n - 1) / 2,
        potential_x=(8 + 3 * n - n**2) / 4,
        potential_squared=tuple(potential_squared),
        field_strength_squared=tuple(field_strength_squared),
    )


def ghost_trace_inputs() -> TraceInputs:
    n = RationalPolynomial.variable()
    potential_squared = list(_zero_vector())
    potential_squared[1] = RationalPolynomial.constant(1)
    potential_squared[3] = RationalPolynomial.constant(-2)
    potential_squared[5] = RationalPolynomial.constant(1)
    field_strength_squared = list(_zero_vector())
    field_strength_squared[0] = RationalPolynomial.constant(-1)
    return TraceInputs(
        identity=n,
        potential_r=RationalPolynomial.constant(-1),
        potential_x=RationalPolynomial.constant(1),
        potential_squared=tuple(potential_squared),
        field_strength_squared=tuple(field_strength_squared),
    )


def universal_eq19_assembly(
    traces: TraceInputs,
    *,
    include_r_potential: bool = True,
) -> tuple[RationalPolynomial, ...]:
    result = list(_zero_vector())
    for index, geometric_factor in enumerate((2, -2, 5)):
        result[index] = traces.identity * Fraction(geometric_factor, 360)
    for index, value in enumerate(traces.potential_squared):
        result[index] = result[index] + value / 2
    if include_r_potential:
        result[2] = result[2] - traces.potential_r / 6
        result[4] = result[4] - traces.potential_x / 6
    for index, value in enumerate(traces.field_strength_squared):
        result[index] = result[index] + value / 12
    return tuple(result)


def source_eq23_rational_functions() -> tuple[RationalPolynomial, ...]:
    n = RationalPolynomial.variable()
    return (
        (482 - 29 * n + n**2) / 360,
        (724 - 1440 * n + 181 * n**2 - n**3) / (360 * (n - 2)),
        5 * (140 + 264 * n - 145 * n**2 + 25 * n**3)
        / (720 * (n - 2)),
        -360 * (-4 - 2 * n + n**2) / (360 * (n - 2)),
        -15 * (32 + 62 * n - 37 * n**2 + 5 * n**3)
        / (360 * (n - 2)),
        45 * (n**3 - n**2 + 14 * n - 40) / (720 * (n - 2)),
        RationalPolynomial.constant(1),
    )


def source_eq27_rational_functions() -> tuple[RationalPolynomial, ...]:
    n = RationalPolynomial.variable()
    return (
        (2 * n - 30) / 360,
        (180 - 2 * n) / 360,
        (5 * n + 60) / 360,
        RationalPolynomial.constant(-1),
        RationalPolynomial.constant(Fraction(-1, 6)),
        RationalPolynomial.constant(Fraction(1, 2)),
        RationalPolynomial.constant(0),
    )


def _vector_cross_residuals(
    derived: tuple[RationalPolynomial, ...],
    target: tuple[RationalPolynomial, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    if len(derived) != len(target):
        raise ValueError('symbolic vectors must use one ordered basis')
    return tuple(
        left.cross_residual(right) for left, right in zip(derived, target)
    )


def _symbolic_mismatch_count(
    candidate: tuple[RationalPolynomial, ...],
    target: tuple[RationalPolynomial, ...],
) -> int:
    return sum(
        not left.equivalent(right) for left, right in zip(candidate, target)
    )


def _replace_trace_component(
    traces: TraceInputs,
    *,
    identity: RationalPolynomial | None = None,
    potential_squared_index: int | None = None,
    potential_squared_value: RationalPolynomial | None = None,
    field_strength_sign: int = 1,
) -> TraceInputs:
    potential_squared = list(traces.potential_squared)
    if potential_squared_index is not None:
        if potential_squared_value is None:
            raise ValueError('a replacement trace component value is required')
        potential_squared[potential_squared_index] = potential_squared_value
    return TraceInputs(
        identity=traces.identity if identity is None else identity,
        potential_r=traces.potential_r,
        potential_x=traces.potential_x,
        potential_squared=tuple(potential_squared),
        field_strength_squared=tuple(
            field_strength_sign * value
            for value in traces.field_strength_squared
        ),
    )


@dataclass(frozen=True)
class TraceIdentityAssemblyContract:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_theory: str
    source_gauge: str
    source_url: str
    source_prefactor: str
    source_equations: tuple[str, ...]
    ordered_basis: tuple[str, ...]
    eq19_formula: str
    eq22_trace_formulae: tuple[str, ...]
    ghost_trace_formulae: tuple[str, ...]
    verification_dimensions: tuple[int, ...]
    downstream_ghost_weight: int
    source_transcription_sha256: str
    source_bulk_total_derivatives_omitted: bool
    gauss_bonnet_applied_in_this_gate: bool
    derivation_status: str
    universal_heat_kernel_formula_derived: bool
    eq22_trace_tensors_derived: bool
    ghost_determinant_derived: bool
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


def trace_identity_assembly_contract() -> TraceIdentityAssemblyContract:
    return TraceIdentityAssemblyContract(
        source_id=SOURCE_ID,
        source_date=SOURCE_DATE,
        source_metadata_title=SOURCE_TITLE,
        html_internal_heading=HTML_INTERNAL_HEADING,
        source_theory=SOURCE_THEORY,
        source_gauge=SOURCE_GAUGE,
        source_url=SOURCE_URL,
        source_prefactor=SOURCE_PREFACTOR,
        source_equations=SOURCE_EQUATIONS,
        ordered_basis=RAW_BASIS,
        eq19_formula=EQ19_FORMULA,
        eq22_trace_formulae=EQ22_TRACE_FORMULAE,
        ghost_trace_formulae=GHOST_TRACE_FORMULAE,
        verification_dimensions=VERIFICATION_DIMENSIONS,
        downstream_ghost_weight=GHOST_WEIGHT,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        source_bulk_total_derivatives_omitted=True,
        gauss_bonnet_applied_in_this_gate=False,
        derivation_status='source_trace_identity_assembly_only',
        universal_heat_kernel_formula_derived=False,
        eq22_trace_tensors_derived=False,
        ghost_determinant_derived=False,
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


def canonical_source_payload(contract: TraceIdentityAssemblyContract) -> str:
    separator = chr(44)
    return '|'.join(
        (
            contract.source_id,
            contract.source_date,
            f'metadata_title={contract.source_metadata_title}',
            f'html_heading={contract.html_internal_heading}',
            f'theory={contract.source_theory}',
            f'gauge={contract.source_gauge}',
            f'prefactor={contract.source_prefactor}',
            f'equations={separator.join(contract.source_equations)}',
            f'basis={separator.join(contract.ordered_basis)}',
            f'eq19={contract.eq19_formula}',
            f'eq22={separator.join(contract.eq22_trace_formulae)}',
            f'ghost_traces={separator.join(contract.ghost_trace_formulae)}',
            'verification_n='
            + separator.join(str(value) for value in contract.verification_dimensions),
            f'downstream_ghost_weight={contract.downstream_ghost_weight}',
        )
    )


def source_payload_sha256(contract: TraceIdentityAssemblyContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: TraceIdentityAssemblyContract) -> None:
    frozen = (
        contract.source_id == SOURCE_ID,
        contract.source_date == SOURCE_DATE,
        contract.source_metadata_title == SOURCE_TITLE,
        contract.html_internal_heading == HTML_INTERNAL_HEADING,
        contract.source_theory == SOURCE_THEORY,
        contract.source_gauge == SOURCE_GAUGE,
        contract.source_url == SOURCE_URL,
        contract.source_prefactor == SOURCE_PREFACTOR,
        contract.source_equations == SOURCE_EQUATIONS,
        contract.ordered_basis == RAW_BASIS,
        contract.eq19_formula == EQ19_FORMULA,
        contract.eq22_trace_formulae == EQ22_TRACE_FORMULAE,
        contract.ghost_trace_formulae == GHOST_TRACE_FORMULAE,
        contract.verification_dimensions == VERIFICATION_DIMENSIONS,
        contract.downstream_ghost_weight == GHOST_WEIGHT,
    )
    if not all(frozen):
        raise ValueError(
            'source metadata, trace formula, basis, or dimension lock changed'
        )
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('local trace-transcription hash mismatch')
    if not contract.source_bulk_total_derivatives_omitted:
        raise ValueError(
            'the source bulk total-derivative convention must be explicit'
        )
    if contract.gauss_bonnet_applied_in_this_gate:
        raise ValueError('Gauss-Bonnet belongs to the downstream assembly gate')
    if contract.derivation_status != 'source_trace_identity_assembly_only':
        raise ValueError('this gate is source trace-identity assembly only')
    unsupported = (
        contract.universal_heat_kernel_formula_derived,
        contract.eq22_trace_tensors_derived,
        contract.ghost_determinant_derived,
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
    if any(unsupported):
        raise ValueError(
            'unsupported trace, determinant, boundary, continuum, or M2 claim'
        )


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def _polynomial_text(values: tuple[Fraction, ...]) -> str:
    return chr(44).join(
        _fraction_text(value) for value in _trim_polynomial(values)
    )


def _numeric_l1(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> Fraction:
    return sum((abs(a - b) for a, b in zip(left, right)), Fraction(0))


@dataclass(frozen=True)
class TraceIdentityAssemblyReceipt:
    source_id: str
    source_date: str
    source_metadata_title: str
    html_internal_heading: str
    source_equations: tuple[str, ...]
    source_transcription_sha256: str
    local_transcription_lock_passed: bool
    ordered_basis: tuple[str, ...]
    verification_dimensions: tuple[int, ...]
    eq23_symbolic_cross_residuals: tuple[str, ...]
    eq27_symbolic_cross_residuals: tuple[str, ...]
    eq23_symbolic_identity_passed: bool
    eq27_symbolic_identity_passed: bool
    exact_spot_component_count: int
    exact_spot_checks_all_passed: bool
    n_two_pole_rejected: bool
    n_four_only_eq23_impostor_mismatch_l1: str
    n_four_only_eq27_impostor_mismatch_l1: str
    missing_r_potential_nonzero_component_count: int
    wrong_eq22_field_strength_sign_nonzero_component_count: int
    missing_scalar_trace_identity_nonzero_component_count: int
    omitted_eq22_p_nonzero_component_count: int
    omitted_eq22_r_x_nonzero_component_count: int
    wrong_ghost_p_sign_nonzero_component_count: int
    wrong_ghost_field_strength_sign_nonzero_component_count: int
    permuted_curvature_basis_nonzero_component_count: int
    monomial_length_dimensions: tuple[int, ...]
    corrupted_x_dimension_vector: tuple[int, ...]
    universal_contribution_length_dimensions: tuple[int, ...]
    corrupted_potential_contribution_dimensions: tuple[int, ...]
    dimension_gate_passed: bool
    downstream_ghost_weight: int
    source_bulk_total_derivatives_omitted: bool
    gauss_bonnet_applied_in_this_gate: bool
    derivation_status: str
    universal_heat_kernel_formula_derived: bool
    eq22_trace_tensors_derived: bool
    ghost_determinant_derived: bool
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
    declared_source_trace_identity_assembly_gate_passed: bool


def evaluate_trace_identity_assembly_gate() -> TraceIdentityAssemblyReceipt:
    contract = trace_identity_assembly_contract()
    validate_contract(contract)

    eq22 = eq22_trace_inputs()
    ghost_traces = ghost_trace_inputs()
    derived_eq23 = universal_eq19_assembly(eq22)
    derived_eq27 = universal_eq19_assembly(ghost_traces)
    target_eq23 = source_eq23_rational_functions()
    target_eq27 = source_eq27_rational_functions()
    residual_eq23 = _vector_cross_residuals(derived_eq23, target_eq23)
    residual_eq27 = _vector_cross_residuals(derived_eq27, target_eq27)
    eq23_identity = all(
        all(value == 0 for value in item) for item in residual_eq23
    )
    eq27_identity = all(
        all(value == 0 for value in item) for item in residual_eq27
    )

    spot_checks: list[bool] = []
    for dimension in contract.verification_dimensions:
        derived_23_at_n = tuple(
            value.evaluate(dimension) for value in derived_eq23
        )
        derived_27_at_n = tuple(
            value.evaluate(dimension) for value in derived_eq27
        )
        spot_checks.extend(
            left == right
            for left, right in zip(
                derived_23_at_n, equation_23_coefficients(dimension)
            )
        )
        spot_checks.extend(
            left == right
            for left, right in zip(
                derived_27_at_n, equation_27_ghost_coefficients(dimension)
            )
        )
    n_two_pole_rejected = False
    try:
        tuple(value.evaluate(2) for value in derived_eq23)
    except ZeroDivisionError:
        n_two_pole_rejected = True

    eq23_at_four = equation_23_coefficients(4)
    eq27_at_four = equation_27_ghost_coefficients(4)
    impostor_23 = sum(
        (
            _numeric_l1(eq23_at_four, equation_23_coefficients(dimension))
            for dimension in (3, 5)
        ),
        Fraction(0),
    )
    impostor_27 = sum(
        (
            _numeric_l1(
                eq27_at_four, equation_27_ghost_coefficients(dimension)
            )
            for dimension in (3, 5)
        ),
        Fraction(0),
    )

    missing_r_potential = universal_eq19_assembly(
        eq22, include_r_potential=False
    )
    wrong_eq22_w = universal_eq19_assembly(
        _replace_trace_component(eq22, field_strength_sign=-1)
    )
    missing_scalar_identity = universal_eq19_assembly(
        _replace_trace_component(eq22, identity=eq22.identity - 1)
    )
    omitted_p = universal_eq19_assembly(
        _replace_trace_component(
            eq22,
            potential_squared_index=3,
            potential_squared_value=ZERO,
        )
    )
    omitted_r_x = universal_eq19_assembly(
        _replace_trace_component(
            eq22,
            potential_squared_index=4,
            potential_squared_value=ZERO,
        )
    )
    wrong_ghost_p = universal_eq19_assembly(
        _replace_trace_component(
            ghost_traces,
            potential_squared_index=3,
            potential_squared_value=RationalPolynomial.constant(2),
        )
    )
    wrong_ghost_w = universal_eq19_assembly(
        _replace_trace_component(ghost_traces, field_strength_sign=-1)
    )
    permuted_curvature = list(derived_eq23)
    permuted_curvature[1], permuted_curvature[2] = (
        permuted_curvature[2],
        permuted_curvature[1],
    )
    control_counts = (
        _symbolic_mismatch_count(missing_r_potential, target_eq23),
        _symbolic_mismatch_count(wrong_eq22_w, target_eq23),
        _symbolic_mismatch_count(missing_scalar_identity, target_eq23),
        _symbolic_mismatch_count(omitted_p, target_eq23),
        _symbolic_mismatch_count(omitted_r_x, target_eq23),
        _symbolic_mismatch_count(wrong_ghost_p, target_eq27),
        _symbolic_mismatch_count(wrong_ghost_w, target_eq27),
        _symbolic_mismatch_count(tuple(permuted_curvature), target_eq23),
    )

    monomial_dimensions = derive_monomial_length_dimensions()
    corrupted_x_dimensions = derive_monomial_length_dimensions(
        {'ScalarGradientSquared': -1}
    )
    primitive_curvature = -2
    primitive_potential = -2
    primitive_field_strength = -2
    contribution_dimensions = (
        2 * primitive_curvature,
        2 * primitive_potential,
        primitive_curvature + primitive_potential,
        2 * primitive_field_strength,
    )
    corrupted_contribution_dimensions = (
        2 * primitive_curvature,
        2 * -1,
        primitive_curvature + -1,
        2 * -1,
    )
    dimension_gate = (
        monomial_dimensions == (-4,) * len(RAW_BASIS)
        and corrupted_x_dimensions != monomial_dimensions
        and contribution_dimensions == (-4, -4, -4, -4)
        and corrupted_contribution_dimensions != contribution_dimensions
    )

    unsupported = (
        contract.universal_heat_kernel_formula_derived,
        contract.eq22_trace_tensors_derived,
        contract.ghost_determinant_derived,
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
    gate_passed = (
        source_lock
        and eq23_identity
        and eq27_identity
        and all(spot_checks)
        and len(spot_checks) == 42
        and n_two_pole_rejected
        and impostor_23 > 0
        and impostor_27 > 0
        and all(count > 0 for count in control_counts)
        and dimension_gate
        and contract.source_bulk_total_derivatives_omitted
        and not contract.gauss_bonnet_applied_in_this_gate
        and not any(unsupported)
    )

    return TraceIdentityAssemblyReceipt(
        source_id=contract.source_id,
        source_date=contract.source_date,
        source_metadata_title=contract.source_metadata_title,
        html_internal_heading=contract.html_internal_heading,
        source_equations=contract.source_equations,
        source_transcription_sha256=contract.source_transcription_sha256,
        local_transcription_lock_passed=source_lock,
        ordered_basis=contract.ordered_basis,
        verification_dimensions=contract.verification_dimensions,
        eq23_symbolic_cross_residuals=tuple(
            _polynomial_text(item) for item in residual_eq23
        ),
        eq27_symbolic_cross_residuals=tuple(
            _polynomial_text(item) for item in residual_eq27
        ),
        eq23_symbolic_identity_passed=eq23_identity,
        eq27_symbolic_identity_passed=eq27_identity,
        exact_spot_component_count=len(spot_checks),
        exact_spot_checks_all_passed=all(spot_checks),
        n_two_pole_rejected=n_two_pole_rejected,
        n_four_only_eq23_impostor_mismatch_l1=_fraction_text(impostor_23),
        n_four_only_eq27_impostor_mismatch_l1=_fraction_text(impostor_27),
        missing_r_potential_nonzero_component_count=control_counts[0],
        wrong_eq22_field_strength_sign_nonzero_component_count=control_counts[1],
        missing_scalar_trace_identity_nonzero_component_count=control_counts[2],
        omitted_eq22_p_nonzero_component_count=control_counts[3],
        omitted_eq22_r_x_nonzero_component_count=control_counts[4],
        wrong_ghost_p_sign_nonzero_component_count=control_counts[5],
        wrong_ghost_field_strength_sign_nonzero_component_count=(
            control_counts[6]
        ),
        permuted_curvature_basis_nonzero_component_count=control_counts[7],
        monomial_length_dimensions=monomial_dimensions,
        corrupted_x_dimension_vector=corrupted_x_dimensions,
        universal_contribution_length_dimensions=contribution_dimensions,
        corrupted_potential_contribution_dimensions=(
            corrupted_contribution_dimensions
        ),
        dimension_gate_passed=dimension_gate,
        downstream_ghost_weight=contract.downstream_ghost_weight,
        source_bulk_total_derivatives_omitted=(
            contract.source_bulk_total_derivatives_omitted
        ),
        gauss_bonnet_applied_in_this_gate=(
            contract.gauss_bonnet_applied_in_this_gate
        ),
        derivation_status=contract.derivation_status,
        universal_heat_kernel_formula_derived=(
            contract.universal_heat_kernel_formula_derived
        ),
        eq22_trace_tensors_derived=contract.eq22_trace_tensors_derived,
        ghost_determinant_derived=contract.ghost_determinant_derived,
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
        declared_source_trace_identity_assembly_gate_passed=gate_passed,
    )
