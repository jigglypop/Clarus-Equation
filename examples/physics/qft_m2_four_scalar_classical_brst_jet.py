'''Exact local-jet classical BRST gate for the M1 chi-plus-four-X model.

The gate implements a sparse supercommutative polynomial algebra over the
rationals.  Metric, chi, four X scalars, and diffeomorphism-ghost jets are independent
symbols through second derivative order.  Consequently, a zero residual is a
coefficient-wise polynomial identity for the 27 declared base-field squares
inside that algebra, not a floating-point sample.  Images of second-jet
generators themselves are not defined; that extension would require third
jets.

This module proves neither the BV classical master equation nor a quantum
Slavnov--Taylor identity.  It contains no antifields, measure Laplacian,
regulator, physical inner product, or Hamiltonian constraint operators.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from typing import Iterable, Mapping

from examples.physics.qft_m2_quantum_admission_closure import (
    SOURCE_TRANSCRIPTION_SHA256 as E70_A_HASH,
    m2_admission_closure_contract,
    validate_contract as validate_e70_a_contract,
)


PRIMARY_SOURCE = 'arXiv:2206.00780v2'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
PRIMARY_SOURCE_DATE = '2025-06-01'
SOURCE_ITEMS = (
    'Definition 3.1 and Eq. (32)--(33): classical diffeomorphism BRST action',
    'Proposition 3.3 and Eq. (35): nilpotency from the diffeomorphism algebra',
    'Remark 2.14: quantum lifting requires anomaly-free ST identities',
)
SOURCE_RELATION = (
    'convention-adapted unscaled covariant reconstruction; Prinz v2 uses '
    'rescaled perturbative h,C,barC,B with kappa,zeta; not literal transcription'
)
SIGN_CONVENTION = (
    's g=L_c g; s chi=c^rho partial_rho chi; '
    's X^A=c^rho partial_rho X^A; '
    's c^mu=c^rho partial_rho c^mu; s bar_c_mu=B_mu; s B_mu=0'
)
GRADED_LEIBNIZ = 's(uv)=s(u)v+(-1)^parity(u)u s(v)'
M1_FIELD_CONTENT = (
    'Einstein gravity plus matter scalar chi with V(chi) and four '
    'dimensionless positive-internal-metric Klein-Gordon reference scalars '
    'X^0,...,X^3'
)
MATTER_SCALAR_LABELS = ('chi',)
REFERENCE_SCALAR_LABELS = ('X0', 'X1', 'X2', 'X3')
SCALAR_LABELS = MATTER_SCALAR_LABELS + REFERENCE_SCALAR_LABELS
JET_SCOPE = (
    'four spacetime dimensions',
    'independent metric, chi, and four-X jets through derivative order two',
    'Grassmann-odd ghost jets through derivative order two',
    'symmetric second partial derivatives',
    'exact sparse supercommutative polynomials over Fraction',
    'differential images through first jets only; 27 base-field squares audited',
)
DIMENSION_CONVENTION = (
    '[x]=L; [partial]=L^-1; [c]=L; [chi]=L^-1; '
    '[g]=[X^A]=[s]=L^0; '
    '[bar_c]=[B] left common but otherwise gauge-fixing dependent'
)
CLAIM_CEILING = (
    'exact nilpotency of 27 base-field components in the independent symmetric '
    'second-jet polynomial algebra plus coefficient-level first-transformation '
    'checks for full M1 chi plus four-X content; no differential on second jets'
)
UPSTREAM_HASHES = (('E70-A', E70_A_HASH),)
CONTRACT_SHA256 = (
    'f994c00debc61bda8ca9ebc867ae98f13b5fa3a0ea89a46f18e193dcb61f9bec'
)


@dataclass(frozen=True)
class FourScalarClassicalBrstJetContract:
    primary_source: str
    primary_source_url: str
    primary_source_date: str
    source_items: tuple[str, ...]
    source_relation: str
    source_contains_four_reference_scalars: bool
    sign_convention: str
    graded_leibniz: str
    m1_field_content: str
    matter_scalar_labels: tuple[str, ...]
    reference_scalar_labels: tuple[str, ...]
    scalar_labels: tuple[str, ...]
    spacetime_dimension: int
    maximum_jet_order: int
    jet_scope: tuple[str, ...]
    dimension_convention: str
    claim_ceiling: str
    upstream_hashes: tuple[tuple[str, str], ...]
    contract_sha256: str
    classical_brst_nilpotency_computed: bool
    second_jet_generator_images_defined: bool
    first_transformations_checked: bool
    action_density_invariance_computed: bool
    gauge_fixing_fermion_constructed: bool
    bv_antifields_constructed: bool
    classical_master_equation_computed: bool
    quantum_master_equation_computed: bool
    loop_anomaly_cancellation_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    relational_observable_interpretation_proved: bool
    derivation_status: str


def four_scalar_classical_brst_jet_contract(
) -> FourScalarClassicalBrstJetContract:
    return FourScalarClassicalBrstJetContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        primary_source_date=PRIMARY_SOURCE_DATE,
        source_items=SOURCE_ITEMS,
        source_relation=SOURCE_RELATION,
        source_contains_four_reference_scalars=False,
        sign_convention=SIGN_CONVENTION,
        graded_leibniz=GRADED_LEIBNIZ,
        m1_field_content=M1_FIELD_CONTENT,
        matter_scalar_labels=MATTER_SCALAR_LABELS,
        reference_scalar_labels=REFERENCE_SCALAR_LABELS,
        scalar_labels=SCALAR_LABELS,
        spacetime_dimension=4,
        maximum_jet_order=2,
        jet_scope=JET_SCOPE,
        dimension_convention=DIMENSION_CONVENTION,
        claim_ceiling=CLAIM_CEILING,
        upstream_hashes=UPSTREAM_HASHES,
        contract_sha256=CONTRACT_SHA256,
        classical_brst_nilpotency_computed=True,
        second_jet_generator_images_defined=False,
        first_transformations_checked=True,
        action_density_invariance_computed=False,
        gauge_fixing_fermion_constructed=False,
        bv_antifields_constructed=False,
        classical_master_equation_computed=False,
        quantum_master_equation_computed=False,
        loop_anomaly_cancellation_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        relational_observable_interpretation_proved=False,
        derivation_status=(
            'exact_27_base_component_second_jet_classical_brst_nilpotency_only'
        ),
    )


def canonical_contract_payload(
    contract: FourScalarClassicalBrstJetContract,
) -> str:
    comma = chr(44)
    upstream = comma.join(
        f'{name}:{value}' for name, value in contract.upstream_hashes
    )
    flags = comma.join(
        f'{name}:{getattr(contract, name)}'
        for name in (
            'classical_brst_nilpotency_computed',
            'second_jet_generator_images_defined',
            'first_transformations_checked',
            'action_density_invariance_computed',
            'gauge_fixing_fermion_constructed',
            'bv_antifields_constructed',
            'classical_master_equation_computed',
            'quantum_master_equation_computed',
            'loop_anomaly_cancellation_computed',
            'positive_physical_hilbert_proved',
            'quantum_hda_m2_proved',
            'relational_observable_interpretation_proved',
        )
    )
    return '|'.join(
        (
            f'source={contract.primary_source}',
            f'url={contract.primary_source_url}',
            f'date={contract.primary_source_date}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_relation={contract.source_relation}',
            f'source_four_reference_scalars={contract.source_contains_four_reference_scalars}',
            f'sign={contract.sign_convention}',
            f'leibniz={contract.graded_leibniz}',
            f'field_content={contract.m1_field_content}',
            f'matter_scalars={comma.join(contract.matter_scalar_labels)}',
            f'reference_scalars={comma.join(contract.reference_scalar_labels)}',
            f'scalars={comma.join(contract.scalar_labels)}',
            f'dimension={contract.spacetime_dimension}',
            f'jet_order={contract.maximum_jet_order}',
            f'jet_scope={comma.join(contract.jet_scope)}',
            f'dimensions={contract.dimension_convention}',
            f'ceiling={contract.claim_ceiling}',
            f'upstream={upstream}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(
    contract: FourScalarClassicalBrstJetContract,
) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: FourScalarClassicalBrstJetContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.primary_source_date == PRIMARY_SOURCE_DATE,
        contract.source_items == SOURCE_ITEMS,
        contract.source_relation == SOURCE_RELATION,
        not contract.source_contains_four_reference_scalars,
        contract.sign_convention == SIGN_CONVENTION,
        contract.graded_leibniz == GRADED_LEIBNIZ,
        contract.m1_field_content == M1_FIELD_CONTENT,
        contract.matter_scalar_labels == MATTER_SCALAR_LABELS,
        contract.reference_scalar_labels == REFERENCE_SCALAR_LABELS,
        contract.scalar_labels == SCALAR_LABELS,
        contract.spacetime_dimension == 4,
        contract.maximum_jet_order == 2,
        contract.jet_scope == JET_SCOPE,
        contract.dimension_convention == DIMENSION_CONVENTION,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.derivation_status
        == 'exact_27_base_component_second_jet_classical_brst_nilpotency_only',
    )
    if not all(frozen):
        raise ValueError('classical BRST source, field, jet, or status lock changed')
    if (
        len(contract.matter_scalar_labels) != 1
        or len(contract.reference_scalar_labels) != 4
        or len(contract.scalar_labels) != 5
        or len(set(contract.scalar_labels)) != 5
        or contract.scalar_labels
        != contract.matter_scalar_labels + contract.reference_scalar_labels
    ):
        raise ValueError('the M1 BRST gate requires chi plus four distinct X scalars')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('classical BRST jet contract hash mismatch')
    if not (
        contract.classical_brst_nilpotency_computed
        and contract.first_transformations_checked
    ):
        raise ValueError('the declared classical gate must keep both core checks')
    if contract.second_jet_generator_images_defined:
        raise ValueError('this gate does not define the differential on second jets')
    unsupported = (
        contract.action_density_invariance_computed,
        contract.gauge_fixing_fermion_constructed,
        contract.bv_antifields_constructed,
        contract.classical_master_equation_computed,
        contract.quantum_master_equation_computed,
        contract.loop_anomaly_cancellation_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.relational_observable_interpretation_proved,
    )
    if any(unsupported):
        raise ValueError('unsupported action, BV, quantum, Hilbert, or M2 promotion')


TermKey = tuple[tuple[str, ...], tuple[str, ...]]


class SparseSuperPolynomial:
    '''Sparse polynomial with commuting even and exterior odd generators.'''

    def __init__(
        self,
        terms: Mapping[TermKey, Fraction | int] | None = None,
    ) -> None:
        cleaned: dict[TermKey, Fraction] = {}
        for key, raw in (terms or {}).items():
            coefficient = Fraction(raw)
            if coefficient:
                cleaned[key] = cleaned.get(key, Fraction(0)) + coefficient
        self.terms = {
            key: value for key, value in cleaned.items() if value
        }

    @classmethod
    def zero(cls) -> 'SparseSuperPolynomial':
        return cls()

    @classmethod
    def scalar(cls, value: Fraction | int) -> 'SparseSuperPolynomial':
        coefficient = Fraction(value)
        return cls({((), ()): coefficient}) if coefficient else cls.zero()

    @classmethod
    def generator(
        cls, name: str, *, odd: bool
    ) -> 'SparseSuperPolynomial':
        if not name:
            raise ValueError('generator names must be nonempty')
        key = ((), (name,)) if odd else ((name,), ())
        return cls({key: Fraction(1)})

    @classmethod
    def monomial(
        cls,
        *,
        even: tuple[str, ...] = (),
        odd: tuple[str, ...] = (),
        coefficient: Fraction | int = 1,
    ) -> 'SparseSuperPolynomial':
        if len(set(odd)) != len(odd):
            return cls.zero()
        return cls(
            {
                (tuple(sorted(even)), tuple(sorted(odd))): Fraction(coefficient)
            }
        )

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    def __add__(self, other: object) -> 'SparseSuperPolynomial':
        if not isinstance(other, SparseSuperPolynomial):
            return NotImplemented
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Fraction(0)) + value
        return SparseSuperPolynomial(terms)

    def __radd__(self, other: object) -> 'SparseSuperPolynomial':
        if other == 0:
            return self
        return self.__add__(other)

    def __neg__(self) -> 'SparseSuperPolynomial':
        return SparseSuperPolynomial(
            {key: -value for key, value in self.terms.items()}
        )

    def __sub__(self, other: object) -> 'SparseSuperPolynomial':
        if not isinstance(other, SparseSuperPolynomial):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: object) -> 'SparseSuperPolynomial':
        if isinstance(other, (int, Fraction)):
            return SparseSuperPolynomial(
                {key: value * Fraction(other) for key, value in self.terms.items()}
            )
        if not isinstance(other, SparseSuperPolynomial):
            return NotImplemented
        result: dict[TermKey, Fraction] = {}
        for (even_left, odd_left), coefficient_left in self.terms.items():
            for (even_right, odd_right), coefficient_right in other.terms.items():
                if set(odd_left).intersection(odd_right):
                    continue
                inversions = sum(
                    left > right for left in odd_left for right in odd_right
                )
                sign = -1 if inversions % 2 else 1
                key = (
                    tuple(sorted(even_left + even_right)),
                    tuple(sorted(odd_left + odd_right)),
                )
                result[key] = result.get(key, Fraction(0)) + (
                    sign * coefficient_left * coefficient_right
                )
        return SparseSuperPolynomial(result)

    def __rmul__(self, other: object) -> 'SparseSuperPolynomial':
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SparseSuperPolynomial)
            and self.terms == other.terms
        )


def polynomial_sum(
    values: Iterable[SparseSuperPolynomial],
) -> SparseSuperPolynomial:
    return sum(values, SparseSuperPolynomial.zero())


def apply_odd_derivation(
    polynomial: SparseSuperPolynomial,
    generator_images: Mapping[str, SparseSuperPolynomial],
    *,
    graded: bool = True,
) -> SparseSuperPolynomial:
    '''Apply an odd derivation, optionally breaking the graded sign as a control.'''

    result = SparseSuperPolynomial.zero()
    for (even, odd), coefficient in polynomial.terms.items():
        even_shell = SparseSuperPolynomial.monomial(even=even)
        odd_shell = SparseSuperPolynomial.monomial(odd=odd)
        for index, name in enumerate(even):
            if name not in generator_images:
                raise ValueError(f'missing differential image for {name}')
            remaining = even[:index] + even[index + 1 :]
            contribution = (
                SparseSuperPolynomial.monomial(even=remaining)
                * generator_images[name]
                * odd_shell
            )
            result += coefficient * contribution
        for index, name in enumerate(odd):
            if name not in generator_images:
                raise ValueError(f'missing differential image for {name}')
            remaining = odd[:index] + odd[index + 1 :]
            sign = -1 if graded and index % 2 else 1
            contribution = (
                even_shell
                * generator_images[name]
                * SparseSuperPolynomial.monomial(odd=remaining)
            )
            result += coefficient * sign * contribution
    return result


def _metric_name(mu: int, nu: int) -> str:
    left, right = sorted((mu, nu))
    return f'g{left}{right}'


def _metric_first_name(mu: int, nu: int, derivative: int) -> str:
    return f'd{_metric_name(mu, nu)}_{derivative}'


def _metric_second_name(
    mu: int,
    nu: int,
    first: int,
    second: int,
    *,
    symmetric: bool,
) -> str:
    if symmetric:
        first, second = sorted((first, second))
    return f'd2{_metric_name(mu, nu)}_{first}_{second}'


def _scalar_name(label: str) -> str:
    return label


def _scalar_first_name(label: str, derivative: int) -> str:
    return f'd{label}_{derivative}'


def _scalar_second_name(
    label: str,
    first: int,
    second: int,
    *,
    symmetric: bool,
) -> str:
    if symmetric:
        first, second = sorted((first, second))
    return f'd2{label}_{first}_{second}'


def _ghost_name(mu: int) -> str:
    return f'c{mu}'


def _ghost_first_name(mu: int, derivative: int) -> str:
    return f'dc{mu}_{derivative}'


def _ghost_second_name(
    mu: int,
    first: int,
    second: int,
    *,
    symmetric: bool,
) -> str:
    if symmetric:
        first, second = sorted((first, second))
    return f'd2c{mu}_{first}_{second}'


def _antighost_name(mu: int) -> str:
    return f'barc{mu}'


def _auxiliary_name(mu: int) -> str:
    return f'B{mu}'


@dataclass(frozen=True)
class JetDifferential:
    generator_images: Mapping[str, SparseSuperPolynomial]
    base_field_names: tuple[str, ...]
    geometric_map_names: tuple[str, ...]
    generator_names: tuple[str, ...]
    odd_generator_names: tuple[str, ...]
    mass_dimensions: Mapping[str, int]
    ghost_numbers: Mapping[str, int]
    symmetric_second_jets: bool
    ghosts_are_odd: bool


def build_jet_differential(
    *,
    ghost_transport_sign: int = 1,
    ghosts_are_odd: bool = True,
    symmetric_second_jets: bool = True,
    scalar_transport: bool = True,
    metric_lie_slots: tuple[bool, bool] = (True, True),
    broken_auxiliary: bool = False,
) -> JetDifferential:
    '''Build the prolonged second-jet differential and selected controls.'''

    if ghost_transport_sign not in (-1, 1):
        raise ValueError('the ghost transport control sign must be plus or minus one')
    if len(metric_lie_slots) != 2:
        raise ValueError('the two covariant metric Lie slots must be declared')

    dimension = 4
    zero = SparseSuperPolynomial.zero

    def even(name: str) -> SparseSuperPolynomial:
        return SparseSuperPolynomial.generator(name, odd=False)

    def ghost(name: str) -> SparseSuperPolynomial:
        return SparseSuperPolynomial.generator(name, odd=ghosts_are_odd)

    def odd(name: str) -> SparseSuperPolynomial:
        return SparseSuperPolynomial.generator(name, odd=True)

    images: dict[str, SparseSuperPolynomial] = {}
    mass_dimensions: dict[str, int] = {}
    ghost_numbers: dict[str, int] = {}
    odd_names: set[str] = set()

    derivative_pairs = tuple(
        (first, second)
        for first in range(dimension)
        for second in range(first if symmetric_second_jets else 0, dimension)
    )

    for mu in range(dimension):
        name = _ghost_name(mu)
        mass_dimensions[name] = -1
        ghost_numbers[name] = 1
        if ghosts_are_odd:
            odd_names.add(name)
        for derivative in range(dimension):
            first_name = _ghost_first_name(mu, derivative)
            mass_dimensions[first_name] = 0
            ghost_numbers[first_name] = 1
            if ghosts_are_odd:
                odd_names.add(first_name)
        for first, second in derivative_pairs:
            second_name = _ghost_second_name(
                mu,
                first,
                second,
                symmetric=symmetric_second_jets,
            )
            mass_dimensions[second_name] = 1
            ghost_numbers[second_name] = 1
            if ghosts_are_odd:
                odd_names.add(second_name)

    metric_pairs = tuple(
        (mu, nu) for mu in range(dimension) for nu in range(mu, dimension)
    )
    for mu, nu in metric_pairs:
        name = _metric_name(mu, nu)
        mass_dimensions[name] = 0
        ghost_numbers[name] = 0
        for derivative in range(dimension):
            first_name = _metric_first_name(mu, nu, derivative)
            mass_dimensions[first_name] = 1
            ghost_numbers[first_name] = 0
        for first, second in derivative_pairs:
            second_name = _metric_second_name(
                mu,
                nu,
                first,
                second,
                symmetric=symmetric_second_jets,
            )
            mass_dimensions[second_name] = 2
            ghost_numbers[second_name] = 0

    for label in SCALAR_LABELS:
        name = _scalar_name(label)
        scalar_dimension = 1 if label in MATTER_SCALAR_LABELS else 0
        mass_dimensions[name] = scalar_dimension
        ghost_numbers[name] = 0
        for derivative in range(dimension):
            first_name = _scalar_first_name(label, derivative)
            mass_dimensions[first_name] = scalar_dimension + 1
            ghost_numbers[first_name] = 0
        for first, second in derivative_pairs:
            second_name = _scalar_second_name(
                label,
                first,
                second,
                symmetric=symmetric_second_jets,
            )
            mass_dimensions[second_name] = scalar_dimension + 2
            ghost_numbers[second_name] = 0

    for mu in range(dimension):
        antighost = _antighost_name(mu)
        auxiliary = _auxiliary_name(mu)
        ghost_numbers[antighost] = -1
        ghost_numbers[auxiliary] = 0
        odd_names.add(antighost)

    for mu in range(dimension):
        images[_ghost_name(mu)] = ghost_transport_sign * polynomial_sum(
            ghost(_ghost_name(rho))
            * ghost(_ghost_first_name(mu, rho))
            for rho in range(dimension)
        )
        for derivative in range(dimension):
            images[_ghost_first_name(mu, derivative)] = (
                ghost_transport_sign
                * polynomial_sum(
                    (
                        ghost(_ghost_first_name(rho, derivative))
                        * ghost(_ghost_first_name(mu, rho))
                    )
                    + (
                        ghost(_ghost_name(rho))
                        * ghost(
                            _ghost_second_name(
                                mu,
                                rho,
                                derivative,
                                symmetric=symmetric_second_jets,
                            )
                        )
                    )
                    for rho in range(dimension)
                )
            )

    for label in SCALAR_LABELS:
        if scalar_transport:
            images[_scalar_name(label)] = polynomial_sum(
                ghost(_ghost_name(rho))
                * even(_scalar_first_name(label, rho))
                for rho in range(dimension)
            )
        else:
            images[_scalar_name(label)] = zero()
        for derivative in range(dimension):
            if scalar_transport:
                images[_scalar_first_name(label, derivative)] = polynomial_sum(
                    (
                        ghost(_ghost_first_name(rho, derivative))
                        * even(_scalar_first_name(label, rho))
                    )
                    + (
                        ghost(_ghost_name(rho))
                        * even(
                            _scalar_second_name(
                                label,
                                rho,
                                derivative,
                                symmetric=symmetric_second_jets,
                            )
                        )
                    )
                    for rho in range(dimension)
                )
            else:
                images[_scalar_first_name(label, derivative)] = zero()

    for mu, nu in metric_pairs:
        base_terms: list[SparseSuperPolynomial] = []
        for rho in range(dimension):
            base_terms.append(
                ghost(_ghost_name(rho))
                * even(_metric_first_name(mu, nu, rho))
            )
            if metric_lie_slots[0]:
                base_terms.append(
                    even(_metric_name(rho, nu))
                    * ghost(_ghost_first_name(rho, mu))
                )
            if metric_lie_slots[1]:
                base_terms.append(
                    even(_metric_name(mu, rho))
                    * ghost(_ghost_first_name(rho, nu))
                )
        images[_metric_name(mu, nu)] = polynomial_sum(base_terms)

        for derivative in range(dimension):
            first_terms: list[SparseSuperPolynomial] = []
            for rho in range(dimension):
                first_terms.extend(
                    (
                        ghost(_ghost_first_name(rho, derivative))
                        * even(_metric_first_name(mu, nu, rho)),
                        ghost(_ghost_name(rho))
                        * even(
                            _metric_second_name(
                                mu,
                                nu,
                                rho,
                                derivative,
                                symmetric=symmetric_second_jets,
                            )
                        ),
                    )
                )
                if metric_lie_slots[0]:
                    first_terms.extend(
                        (
                            even(_metric_first_name(rho, nu, derivative))
                            * ghost(_ghost_first_name(rho, mu)),
                            even(_metric_name(rho, nu))
                            * ghost(
                                _ghost_second_name(
                                    rho,
                                    mu,
                                    derivative,
                                    symmetric=symmetric_second_jets,
                                )
                            ),
                        )
                    )
                if metric_lie_slots[1]:
                    first_terms.extend(
                        (
                            even(_metric_first_name(mu, rho, derivative))
                            * ghost(_ghost_first_name(rho, nu)),
                            even(_metric_name(mu, rho))
                            * ghost(
                                _ghost_second_name(
                                    rho,
                                    nu,
                                    derivative,
                                    symmetric=symmetric_second_jets,
                                )
                            ),
                        )
                    )
            images[_metric_first_name(mu, nu, derivative)] = polynomial_sum(
                first_terms
            )

    for mu in range(dimension):
        antighost = _antighost_name(mu)
        auxiliary = _auxiliary_name(mu)
        images[antighost] = even(auxiliary)
        images[auxiliary] = (
            ghost(_ghost_name(0)) * even(auxiliary)
            if broken_auxiliary
            else zero()
        )

    base_names = (
        tuple(_ghost_name(mu) for mu in range(dimension))
        + tuple(_scalar_name(label) for label in SCALAR_LABELS)
        + tuple(_metric_name(mu, nu) for mu, nu in metric_pairs)
        + tuple(_antighost_name(mu) for mu in range(dimension))
        + tuple(_auxiliary_name(mu) for mu in range(dimension))
    )
    geometric_map_names = tuple(
        name for name in images if name in mass_dimensions
    )
    generator_names = tuple(sorted(ghost_numbers))
    return JetDifferential(
        generator_images=images,
        base_field_names=base_names,
        geometric_map_names=geometric_map_names,
        generator_names=generator_names,
        odd_generator_names=tuple(sorted(odd_names)),
        mass_dimensions=mass_dimensions,
        ghost_numbers=ghost_numbers,
        symmetric_second_jets=symmetric_second_jets,
        ghosts_are_odd=ghosts_are_odd,
    )


def locked_base_transformation_targets(
) -> Mapping[str, SparseSuperPolynomial]:
    '''Independently assemble the coefficient-level base transformation oracle.'''

    def even(name: str) -> SparseSuperPolynomial:
        return SparseSuperPolynomial.generator(name, odd=False)

    def odd(name: str) -> SparseSuperPolynomial:
        return SparseSuperPolynomial.generator(name, odd=True)

    targets: dict[str, SparseSuperPolynomial] = {}
    for mu in range(4):
        targets[_ghost_name(mu)] = polynomial_sum(
            odd(_ghost_name(rho)) * odd(_ghost_first_name(mu, rho))
            for rho in range(4)
        )
    for label in SCALAR_LABELS:
        targets[label] = polynomial_sum(
            odd(_ghost_name(rho)) * even(_scalar_first_name(label, rho))
            for rho in range(4)
        )
    for mu in range(4):
        for nu in range(mu, 4):
            targets[_metric_name(mu, nu)] = polynomial_sum(
                (
                    odd(_ghost_name(rho))
                    * even(_metric_first_name(mu, nu, rho))
                )
                + (
                    even(_metric_name(rho, nu))
                    * odd(_ghost_first_name(rho, mu))
                )
                + (
                    even(_metric_name(mu, rho))
                    * odd(_ghost_first_name(rho, nu))
                )
                for rho in range(4)
            )
    for mu in range(4):
        targets[_antighost_name(mu)] = even(_auxiliary_name(mu))
        targets[_auxiliary_name(mu)] = SparseSuperPolynomial.zero()
    return targets


def nilpotency_residuals(
    differential: JetDifferential,
    *,
    graded: bool = True,
) -> tuple[tuple[str, SparseSuperPolynomial], ...]:
    return tuple(
        (
            name,
            apply_odd_derivation(
                differential.generator_images[name],
                differential.generator_images,
                graded=graded,
            ),
        )
        for name in differential.base_field_names
    )


def _polynomial_quantum_numbers(
    polynomial: SparseSuperPolynomial,
    assignments: Mapping[str, int],
) -> set[int]:
    values: set[int] = set()
    for (even, odd), _ in polynomial.terms.items():
        names = even + odd
        if any(name not in assignments for name in names):
            raise ValueError('a generator lacks a declared quantum number')
        values.add(sum(assignments[name] for name in names))
    return values


def audit_map_quantum_numbers(
    differential: JetDifferential,
) -> tuple[bool, bool, int, int]:
    dimension_ok = True
    for name in differential.geometric_map_names:
        image = differential.generator_images[name]
        if image.is_zero:
            continue
        dimensions = _polynomial_quantum_numbers(
            image, differential.mass_dimensions
        )
        dimension_ok = dimension_ok and dimensions == {
            differential.mass_dimensions[name]
        }
    ghost_number_ok = True
    for name, image in differential.generator_images.items():
        if image.is_zero:
            continue
        ghost_numbers = _polynomial_quantum_numbers(
            image, differential.ghost_numbers
        )
        ghost_number_ok = ghost_number_ok and ghost_numbers == {
            differential.ghost_numbers[name] + 1
        }
    return (
        dimension_ok,
        ghost_number_ok,
        len(differential.geometric_map_names),
        len(differential.generator_images),
    )


@dataclass(frozen=True)
class FourScalarClassicalBrstJetReceipt:
    contract_sha256: str
    primary_source: str
    primary_source_url: str
    primary_source_date: str
    source_items: tuple[str, ...]
    source_relation: str
    source_contains_four_reference_scalars: bool
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_contract_verified: bool
    sign_convention: str
    graded_leibniz: str
    m1_field_content: str
    matter_scalar_labels: tuple[str, ...]
    reference_scalar_labels: tuple[str, ...]
    scalar_labels: tuple[str, ...]
    matter_scalar_field_count: int
    reference_scalar_field_count: int
    scalar_field_count: int
    spacetime_dimension: int
    maximum_jet_order: int
    generator_count: int
    odd_generator_count: int
    even_generator_count: int
    base_nilpotency_component_count: int
    nilpotency_residual_term_counts: tuple[tuple[str, int], ...]
    maximum_nilpotency_residual_term_count: int
    first_transformation_term_counts: tuple[tuple[str, int], ...]
    locked_base_transformation_mismatch_term_count: int
    required_first_transformations_nonzero: bool
    auxiliary_transformations_zero: bool
    metric_symmetry_preserved: bool
    scalar_multiplet_preserved: bool
    dimension_audit_map_count: int
    ghost_number_audit_map_count: int
    all_geometric_map_dimensions_correct: bool
    all_map_ghost_numbers_correct: bool
    wrong_ghost_sign_residual_term_count: int
    commuting_ghost_residual_term_count: int
    ungraded_leibniz_residual_term_count: int
    unsymmetrized_second_jet_residual_term_count: int
    missing_scalar_transport_mismatch_term_count: int
    missing_metric_lie_slot_mismatch_term_count: int
    broken_doublet_residual_term_count: int
    wrong_reference_scalar_multiplicity_rejected: bool
    missing_matter_scalar_rejected: bool
    wrong_scalar_multiplicity_rejected: bool
    action_density_invariance_computed: bool
    gauge_fixing_fermion_constructed: bool
    bv_antifields_constructed: bool
    classical_master_equation_computed: bool
    quantum_master_equation_computed: bool
    loop_anomaly_cancellation_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    relational_observable_interpretation_proved: bool
    second_jet_generator_images_defined: bool
    claim_ceiling: str
    derivation_status: str
    declared_exact_classical_brst_jet_gate_passed: bool


def _residual_term_total(
    residuals: Iterable[tuple[str, SparseSuperPolynomial]],
) -> int:
    return sum(polynomial.term_count for _, polynomial in residuals)


def evaluate_four_scalar_classical_brst_jet_gate(
) -> FourScalarClassicalBrstJetReceipt:
    contract = four_scalar_classical_brst_jet_contract()
    validate_contract(contract)
    validate_e70_a_contract(m2_admission_closure_contract())

    normal = build_jet_differential()
    residuals = nilpotency_residuals(normal)
    residual_counts = tuple(
        (name, polynomial.term_count) for name, polynomial in residuals
    )
    maximum_residual = max(count for _, count in residual_counts)
    first_counts = tuple(
        (
            name,
            normal.generator_images[name].term_count,
        )
        for name in normal.base_field_names
    )
    first_count_map = dict(first_counts)
    locked_targets = locked_base_transformation_targets()
    locked_base_mismatch = sum(
        (
            normal.generator_images[name] - locked_targets[name]
        ).term_count
        for name in normal.base_field_names
    )
    required_nonzero_names = tuple(
        name
        for name in normal.base_field_names
        if not name.startswith('B')
    )
    required_nonzero = all(
        first_count_map[name] > 0 for name in required_nonzero_names
    )
    auxiliary_zero = all(
        first_count_map[_auxiliary_name(mu)] == 0 for mu in range(4)
    )
    metric_symmetry = all(
        _metric_name(mu, nu) == _metric_name(nu, mu)
        and normal.generator_images[_metric_name(mu, nu)]
        == normal.generator_images[_metric_name(nu, mu)]
        for mu in range(4)
        for nu in range(4)
    )
    scalar_multiplet = (
        tuple(name for name in normal.base_field_names if name in SCALAR_LABELS)
        == SCALAR_LABELS
        and all(first_count_map[label] == 4 for label in SCALAR_LABELS)
    )
    (
        dimensions_ok,
        ghost_numbers_ok,
        dimension_map_count,
        ghost_number_map_count,
    ) = audit_map_quantum_numbers(normal)

    wrong_sign = build_jet_differential(ghost_transport_sign=-1)
    wrong_sign_total = _residual_term_total(nilpotency_residuals(wrong_sign))

    commuting = build_jet_differential(ghosts_are_odd=False)
    commuting_total = _residual_term_total(nilpotency_residuals(commuting))

    ungraded_total = _residual_term_total(
        nilpotency_residuals(normal, graded=False)
    )

    unsymmetrized = build_jet_differential(
        symmetric_second_jets=False
    )
    unsymmetrized_total = _residual_term_total(
        nilpotency_residuals(unsymmetrized)
    )

    missing_scalar = build_jet_differential(scalar_transport=False)
    missing_scalar_mismatch = sum(
        (
            locked_targets[label]
            - missing_scalar.generator_images[label]
        ).term_count
        for label in SCALAR_LABELS
    )

    missing_metric = build_jet_differential(metric_lie_slots=(True, False))
    missing_metric_mismatch = sum(
        (
            locked_targets[_metric_name(mu, nu)]
            - missing_metric.generator_images[_metric_name(mu, nu)]
        ).term_count
        for mu in range(4)
        for nu in range(mu, 4)
    )

    broken_doublet = build_jet_differential(broken_auxiliary=True)
    broken_doublet_residuals = dict(nilpotency_residuals(broken_doublet))
    broken_doublet_total = sum(
        broken_doublet_residuals[_antighost_name(mu)].term_count
        + broken_doublet_residuals[_auxiliary_name(mu)].term_count
        for mu in range(4)
    )

    wrong_reference_scalar_multiplicity_rejected = False
    try:
        validate_contract(
            replace(
                contract,
                reference_scalar_labels=contract.reference_scalar_labels[:-1],
                scalar_labels=(
                    contract.matter_scalar_labels
                    + contract.reference_scalar_labels[:-1]
                ),
            )
        )
    except ValueError:
        wrong_reference_scalar_multiplicity_rejected = True

    missing_matter_scalar_rejected = False
    try:
        validate_contract(
            replace(
                contract,
                matter_scalar_labels=(),
                scalar_labels=contract.reference_scalar_labels,
            )
        )
    except ValueError:
        missing_matter_scalar_rejected = True

    wrong_scalar_multiplicity_rejected = (
        wrong_reference_scalar_multiplicity_rejected
        and missing_matter_scalar_rejected
    )

    unsupported = (
        contract.action_density_invariance_computed,
        contract.gauge_fixing_fermion_constructed,
        contract.bv_antifields_constructed,
        contract.classical_master_equation_computed,
        contract.quantum_master_equation_computed,
        contract.loop_anomaly_cancellation_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.relational_observable_interpretation_proved,
    )
    passed = all(
        (
            len(contract.matter_scalar_labels) == 1,
            len(contract.reference_scalar_labels) == 4,
            len(contract.scalar_labels) == 5,
            len(normal.generator_names) == 293,
            len(normal.odd_generator_names) == 64,
            len(residual_counts) == 27,
            maximum_residual == 0,
            locked_base_mismatch == 0,
            required_nonzero,
            auxiliary_zero,
            metric_symmetry,
            scalar_multiplet,
            dimensions_ok,
            ghost_numbers_ok,
            dimension_map_count == 95,
            ghost_number_map_count == 103,
            wrong_sign_total > 0,
            commuting_total > 0,
            ungraded_total > 0,
            unsymmetrized_total > 0,
            missing_scalar_mismatch > 0,
            missing_metric_mismatch > 0,
            broken_doublet_total > 0,
            wrong_reference_scalar_multiplicity_rejected,
            missing_matter_scalar_rejected,
            wrong_scalar_multiplicity_rejected,
            normal.symmetric_second_jets,
            normal.ghosts_are_odd,
            not any(unsupported),
        )
    )
    return FourScalarClassicalBrstJetReceipt(
        contract_sha256=contract.contract_sha256,
        primary_source=contract.primary_source,
        primary_source_url=contract.primary_source_url,
        primary_source_date=contract.primary_source_date,
        source_items=contract.source_items,
        source_relation=contract.source_relation,
        source_contains_four_reference_scalars=(
            contract.source_contains_four_reference_scalars
        ),
        upstream_hashes=contract.upstream_hashes,
        upstream_contract_verified=True,
        sign_convention=contract.sign_convention,
        graded_leibniz=contract.graded_leibniz,
        m1_field_content=contract.m1_field_content,
        matter_scalar_labels=contract.matter_scalar_labels,
        reference_scalar_labels=contract.reference_scalar_labels,
        scalar_labels=contract.scalar_labels,
        matter_scalar_field_count=len(contract.matter_scalar_labels),
        reference_scalar_field_count=len(contract.reference_scalar_labels),
        scalar_field_count=len(contract.scalar_labels),
        spacetime_dimension=contract.spacetime_dimension,
        maximum_jet_order=contract.maximum_jet_order,
        generator_count=len(normal.generator_names),
        odd_generator_count=len(normal.odd_generator_names),
        even_generator_count=(
            len(normal.generator_names) - len(normal.odd_generator_names)
        ),
        base_nilpotency_component_count=len(residual_counts),
        nilpotency_residual_term_counts=residual_counts,
        maximum_nilpotency_residual_term_count=maximum_residual,
        first_transformation_term_counts=first_counts,
        locked_base_transformation_mismatch_term_count=locked_base_mismatch,
        required_first_transformations_nonzero=required_nonzero,
        auxiliary_transformations_zero=auxiliary_zero,
        metric_symmetry_preserved=metric_symmetry,
        scalar_multiplet_preserved=scalar_multiplet,
        dimension_audit_map_count=dimension_map_count,
        ghost_number_audit_map_count=ghost_number_map_count,
        all_geometric_map_dimensions_correct=dimensions_ok,
        all_map_ghost_numbers_correct=ghost_numbers_ok,
        wrong_ghost_sign_residual_term_count=wrong_sign_total,
        commuting_ghost_residual_term_count=commuting_total,
        ungraded_leibniz_residual_term_count=ungraded_total,
        unsymmetrized_second_jet_residual_term_count=unsymmetrized_total,
        missing_scalar_transport_mismatch_term_count=(
            missing_scalar_mismatch
        ),
        missing_metric_lie_slot_mismatch_term_count=missing_metric_mismatch,
        broken_doublet_residual_term_count=broken_doublet_total,
        wrong_reference_scalar_multiplicity_rejected=(
            wrong_reference_scalar_multiplicity_rejected
        ),
        missing_matter_scalar_rejected=missing_matter_scalar_rejected,
        wrong_scalar_multiplicity_rejected=(
            wrong_scalar_multiplicity_rejected
        ),
        action_density_invariance_computed=(
            contract.action_density_invariance_computed
        ),
        gauge_fixing_fermion_constructed=(
            contract.gauge_fixing_fermion_constructed
        ),
        bv_antifields_constructed=contract.bv_antifields_constructed,
        classical_master_equation_computed=(
            contract.classical_master_equation_computed
        ),
        quantum_master_equation_computed=(
            contract.quantum_master_equation_computed
        ),
        loop_anomaly_cancellation_computed=(
            contract.loop_anomaly_cancellation_computed
        ),
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        relational_observable_interpretation_proved=(
            contract.relational_observable_interpretation_proved
        ),
        second_jet_generator_images_defined=(
            contract.second_jet_generator_images_defined
        ),
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_exact_classical_brst_jet_gate_passed=passed,
    )
