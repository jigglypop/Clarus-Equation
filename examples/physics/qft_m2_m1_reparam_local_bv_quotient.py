'''Bounded local-functional BV quotient witness aligned with M1 scalar labels.

This module implements an exact one-dimensional reparametrization model with
five normalized scalar coordinates labelled chi, X0, ..., X3, their canonical
momenta, a lapse, the reparametrization ghost, and a nonminimal doublet.  It
uses actual jet Euler derivatives and a local-functional BV antibracket.  The
nonzero horizontal currents are retained and tested.

The construction is a convention-adapted polynomial toy.  It is not the 4D
M1 Einstein--Hilbert functional, an unbounded variational bicomplex, a global
boundary theorem, a BV measure, a QME, or quantum M2 evidence.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from typing import Mapping

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
    polynomial_sum,
)
from examples.physics.qft_m2_m1_bv_master_admission import (
    CONTRACT_SHA256 as E70_D_HASH,
    bv_left_derivative,
    bv_right_derivative,
    evaluate_m1_bv_master_admission_gate,
    m1_bv_master_admission_contract,
    validate_contract as validate_e70_d_contract,
)


PRIMARY_SOURCE = 'hep-th/0506098'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0506098'
LOCAL_FUNCTIONAL_SOURCE = 'hep-th/0002245v3'
LOCAL_FUNCTIONAL_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0002245v3'
SECONDARY_SOURCE = 'arXiv:2206.00780v2'
SECONDARY_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
SCALAR_LABELS = ('chi', 'X0', 'X1', 'X2', 'X3')
MAXIMUM_JET_ORDER = 4
NORMALIZATION = (
    't/t_ref, q/q_ref, p/p_ref, lapse/lapse_ref, and every coefficient are '
    'dimensionless exact coordinates of the finite witness'
)
MODEL_RELATION = (
    'standard BV and local-functional conventions are imported from the '
    'sources; the five-label first-order reparametrization model is a '
    'convention-adapted polynomial toy, not a literal source or 4D M1 action'
)
SOURCE_ITEMS = (
    'Fuster--Henneaux--Maas Sec. 3.4/Eqs. (3.12)--(3.13): Euler derivative and total-derivative criterion',
    'Fuster--Henneaux--Maas Eqs. (4.4)--(4.7): antibracket, sF=(S,F), and CME',
    'Barnich--Brandt--Henneaux v3: local BRST cohomology with antifields',
    'Prinz v2 Proposition 3.3/Eq. (35): left classical diffeomorphism-BRST nilpotency',
)
ANTIBRACKET_CONVENTION = (
    '(F,G)=integral sum_A[E_R,Phi(F) E_L,Phi*(G)-'
    'E_R,Phi*(F) E_L,Phi(G)]; sF=(S,F); '
    'S1=sum_A (-1)^parity(Phi_A) Phi_A* sPhi_A in star-left ordering'
)
EULER_CONVENTION = (
    'E_z(f)=sum_k=0^4 (-D)^k partial f/partial z_(k), with exact '
    'left/right exterior derivatives and terminal-jet rejection'
)
CLAIM_CEILING = (
    'exact bounded one-dimensional local-jet Euler calculus, retained '
    'horizontal currents, and a convention-adapted five-label '
    'reparametrization BV functional satisfying the CME modulo D; no 4D M1 '
    'functional, unbounded variational bicomplex, boundary completion, BV '
    'measure, QME, continuum ST, physical Hilbert, HDA, quantum M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-D', E70_D_HASH),)
CONTRACT_SHA256 = (
    '28ba17e093c42eb9dc67c79c93bae05c09b6db87c3e6576e45ba1ac02b0863c6'
)


class JetOrderExceeded(ValueError):
    '''Raised rather than silently truncating a total derivative.'''


@dataclass(frozen=True)
class LocalBVVariableSpec:
    name: str
    role: str
    parity: int
    ghost_number: int
    antifield_number: int


def local_field_specs() -> tuple[LocalBVVariableSpec, ...]:
    specs: list[LocalBVVariableSpec] = []
    for label in SCALAR_LABELS:
        specs.append(
            LocalBVVariableSpec(f'q_{label}', 'normalized scalar coordinate', 0, 0, 0)
        )
    for label in SCALAR_LABELS:
        specs.append(
            LocalBVVariableSpec(f'p_{label}', 'normalized scalar momentum', 0, 0, 0)
        )
    specs.extend(
        (
            LocalBVVariableSpec('n', 'one-dimensional lapse density', 0, 0, 0),
            LocalBVVariableSpec('c', 'reparametrization ghost', 1, 1, 0),
            LocalBVVariableSpec('barc', 'nonminimal antighost', 1, -1, 0),
            LocalBVVariableSpec('B', 'nonminimal auxiliary', 0, 0, 0),
        )
    )
    return tuple(specs)


def local_antifield_specs() -> tuple[LocalBVVariableSpec, ...]:
    return tuple(
        LocalBVVariableSpec(
            f'{field.name}_star',
            f'antifield dual to {field.role}',
            (field.parity + 1) % 2,
            -field.ghost_number - 1,
            1,
        )
        for field in local_field_specs()
    )


ALL_VARIABLE_SPECS = local_field_specs() + local_antifield_specs()
SPEC_BY_NAME = {spec.name: spec for spec in ALL_VARIABLE_SPECS}


def jet_name(base_name: str, order: int) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown local BV variable {base_name}')
    if order < 0 or order > MAXIMUM_JET_ORDER:
        raise JetOrderExceeded(
            f'jet order {order} is outside the locked range 0..{MAXIMUM_JET_ORDER}'
        )
    return f'{base_name}__{order}'


JET_LOOKUP = {
    jet_name(spec.name, order): (spec, order)
    for spec in ALL_VARIABLE_SPECS
    for order in range(MAXIMUM_JET_ORDER + 1)
}


def generator(base_name: str, order: int = 0) -> SparseSuperPolynomial:
    spec = SPEC_BY_NAME[base_name]
    return SparseSuperPolynomial.generator(
        jet_name(base_name, order),
        odd=bool(spec.parity),
    )


def polynomial_maximum_jet_order(polynomial: SparseSuperPolynomial) -> int:
    maximum = 0
    for even_names, odd_names in polynomial.terms:
        for name in even_names + odd_names:
            if name not in JET_LOOKUP:
                raise ValueError(f'unregistered jet generator {name}')
            maximum = max(maximum, JET_LOOKUP[name][1])
    return maximum


def total_derivative(
    polynomial: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    '''Apply the even horizontal derivative D without terminal truncation.'''

    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        for index, name in enumerate(even_names):
            spec, order = JET_LOOKUP[name]
            if order == MAXIMUM_JET_ORDER:
                raise JetOrderExceeded(f'D requires {spec.name} jet order {order + 1}')
            remaining_even = even_names[:index] + even_names[index + 1 :]
            contribution = (
                SparseSuperPolynomial.monomial(
                    even=remaining_even,
                    odd=odd_names,
                )
                * generator(spec.name, order + 1)
            )
            result += coefficient * contribution
        for index, name in enumerate(odd_names):
            spec, order = JET_LOOKUP[name]
            if order == MAXIMUM_JET_ORDER:
                raise JetOrderExceeded(f'D requires {spec.name} jet order {order + 1}')
            prefix = SparseSuperPolynomial.monomial(
                even=even_names,
                odd=odd_names[:index],
            )
            suffix = SparseSuperPolynomial.monomial(odd=odd_names[index + 1 :])
            contribution = prefix * generator(spec.name, order + 1) * suffix
            result += coefficient * contribution
    return result


def iterated_total_derivative(
    polynomial: SparseSuperPolynomial,
    order: int,
) -> SparseSuperPolynomial:
    if order < 0:
        raise ValueError('total derivative order must be nonnegative')
    result = polynomial
    for _ in range(order):
        result = total_derivative(result)
    return result


def jet_partial_derivative(
    polynomial: SparseSuperPolynomial,
    base_name: str,
    order: int,
    *,
    side: str,
) -> SparseSuperPolynomial:
    spec = SPEC_BY_NAME[base_name]
    variable = jet_name(base_name, order)
    if side == 'left':
        return bv_left_derivative(polynomial, variable, odd=bool(spec.parity))
    if side == 'right':
        return bv_right_derivative(polynomial, variable, odd=bool(spec.parity))
    raise ValueError('jet derivative side must be left or right')


def euler_derivative(
    density: SparseSuperPolynomial,
    base_name: str,
    *,
    side: str,
) -> SparseSuperPolynomial:
    '''Exact bounded Euler derivative sum_k (-D)^k partial/partial z_k.'''

    contributions: list[SparseSuperPolynomial] = []
    for order in range(MAXIMUM_JET_ORDER + 1):
        partial = jet_partial_derivative(
            density,
            base_name,
            order,
            side=side,
        )
        if partial.is_zero:
            continue
        integrated = iterated_total_derivative(partial, order)
        contributions.append((-1 if order % 2 else 1) * integrated)
    return polynomial_sum(contributions)


def local_bv_antibracket_density(
    left: SparseSuperPolynomial,
    right: SparseSuperPolynomial,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    if second_term_sign not in (-1, 1):
        raise ValueError('the second local antibracket sign must be plus or minus one')
    result = SparseSuperPolynomial.zero()
    for field in local_field_specs():
        antifield_name = f'{field.name}_star'
        result += (
            euler_derivative(left, field.name, side='right')
            * euler_derivative(right, antifield_name, side='left')
        )
        result += second_term_sign * (
            euler_derivative(left, antifield_name, side='right')
            * euler_derivative(right, field.name, side='left')
        )
    return result


@dataclass(frozen=True)
class ReparamLocalBVModel:
    hamiltonian_density_scalar: SparseSuperPolynomial
    classical_density: SparseSuperPolynomial
    classical_boundary_current: SparseSuperPolynomial
    transformations: Mapping[str, SparseSuperPolynomial]
    antifield_density: SparseSuperPolynomial
    extended_density: SparseSuperPolynomial


def reparam_local_bv_model(
    *,
    ghost_sign: int = 1,
    include_lapse_weight_term: bool = True,
    include_ghost_antifield_term: bool = True,
    parity_signed_antifields: bool = True,
) -> ReparamLocalBVModel:
    if ghost_sign not in (-1, 1):
        raise ValueError('ghost sign must be plus or minus one')
    c0 = generator('c')
    c1 = generator('c', 1)
    n0 = generator('n')
    n1 = generator('n', 1)
    hamiltonian = Fraction(1, 2) * polynomial_sum(
        generator(f'p_{label}') * generator(f'p_{label}')
        + generator(f'q_{label}') * generator(f'q_{label}')
        for label in SCALAR_LABELS
    )
    classical = polynomial_sum(
        generator(f'p_{label}') * generator(f'q_{label}', 1)
        for label in SCALAR_LABELS
    ) - n0 * hamiltonian

    transformations: dict[str, SparseSuperPolynomial] = {}
    for label in SCALAR_LABELS:
        transformations[f'q_{label}'] = c0 * generator(f'q_{label}', 1)
        transformations[f'p_{label}'] = c0 * generator(f'p_{label}', 1)
    transformations['n'] = c0 * n1
    if include_lapse_weight_term:
        transformations['n'] += c1 * n0
    transformations['c'] = ghost_sign * c0 * c1
    transformations['barc'] = generator('B')
    transformations['B'] = SparseSuperPolynomial.zero()

    antifield_terms: list[SparseSuperPolynomial] = []
    for field in local_field_specs():
        if field.name == 'c' and not include_ghost_antifield_term:
            continue
        antifield_terms.append(
            (-1 if parity_signed_antifields and field.parity else 1)
            * generator(f'{field.name}_star')
            * transformations[field.name]
        )
    antifield_density = polynomial_sum(antifield_terms)
    return ReparamLocalBVModel(
        hamiltonian_density_scalar=hamiltonian,
        classical_density=classical,
        classical_boundary_current=c0 * classical,
        transformations=transformations,
        antifield_density=antifield_density,
        extended_density=classical + antifield_density,
    )


def apply_local_brst(
    polynomial: SparseSuperPolynomial,
    transformations: Mapping[str, SparseSuperPolynomial],
) -> SparseSuperPolynomial:
    '''Apply the odd evolutionary differential, prolonging by D on jets.'''

    present_names = {
        name
        for even_names, odd_names in polynomial.terms
        for name in even_names + odd_names
    }
    images: dict[str, SparseSuperPolynomial] = {}
    for name in present_names:
        if name not in JET_LOOKUP:
            raise ValueError(f'unregistered BRST jet generator {name}')
        spec, order = JET_LOOKUP[name]
        if spec.antifield_number:
            raise ValueError('field BRST prolongation does not act on antifields here')
        images[name] = iterated_total_derivative(
            transformations[spec.name],
            order,
        )
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        even_shell = SparseSuperPolynomial.monomial(even=even_names)
        odd_shell = SparseSuperPolynomial.monomial(odd=odd_names)
        for index, name in enumerate(even_names):
            if name not in images:
                raise ValueError(f'BRST image unavailable for {name}')
            remaining = even_names[:index] + even_names[index + 1 :]
            result += coefficient * (
                SparseSuperPolynomial.monomial(even=remaining)
                * images[name]
                * odd_shell
            )
        for index, name in enumerate(odd_names):
            if name not in images:
                raise ValueError(f'BRST image unavailable for {name}')
            remaining = odd_names[:index] + odd_names[index + 1 :]
            sign = -1 if index % 2 else 1
            result += coefficient * sign * (
                even_shell
                * images[name]
                * SparseSuperPolynomial.monomial(odd=remaining)
            )
    return result


def polynomial_parity(polynomial: SparseSuperPolynomial) -> int:
    parities = {
        len(odd_names) % 2
        for _, odd_names in polynomial.terms
    }
    if not parities:
        return 0
    if len(parities) != 1:
        raise ValueError('the polynomial is not parity homogeneous')
    return parities.pop()


def antifield_number_components(
    polynomial: SparseSuperPolynomial,
) -> Mapping[int, SparseSuperPolynomial]:
    terms_by_number: dict[int, dict[tuple[tuple[str, ...], tuple[str, ...]], Fraction]] = {}
    for key, coefficient in polynomial.terms.items():
        even_names, odd_names = key
        number = sum(
            JET_LOOKUP[name][0].antifield_number
            for name in even_names + odd_names
        )
        terms_by_number.setdefault(number, {})[key] = coefficient
    return {
        number: SparseSuperPolynomial(terms)
        for number, terms in terms_by_number.items()
    }


def all_euler_residuals(
    density: SparseSuperPolynomial,
) -> tuple[tuple[str, SparseSuperPolynomial], ...]:
    return tuple(
        (spec.name, euler_derivative(density, spec.name, side='left'))
        for spec in ALL_VARIABLE_SPECS
    )


def graded_antisymmetry_residual(
    left: SparseSuperPolynomial,
    right: SparseSuperPolynomial,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    phase = -1 if (
        (polynomial_parity(left) + 1)
        * (polynomial_parity(right) + 1)
    ) % 2 else 1
    return (
        local_bv_antibracket_density(
            left,
            right,
            second_term_sign=second_term_sign,
        )
        + phase * local_bv_antibracket_density(
            right,
            left,
            second_term_sign=second_term_sign,
        )
    )


def graded_jacobi_residual(
    first: SparseSuperPolynomial,
    second: SparseSuperPolynomial,
    third: SparseSuperPolynomial,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    items = ((first, second, third), (second, third, first), (third, first, second))
    residual = SparseSuperPolynomial.zero()
    for left, middle, right in items:
        phase = -1 if (
            (polynomial_parity(left) + 1)
            * (polynomial_parity(right) + 1)
        ) % 2 else 1
        residual += phase * local_bv_antibracket_density(
            left,
            local_bv_antibracket_density(
                middle,
                right,
                second_term_sign=second_term_sign,
            ),
            second_term_sign=second_term_sign,
        )
    return residual


def classical_noether_identity(
    model: ReparamLocalBVModel,
) -> SparseSuperPolynomial:
    residual = SparseSuperPolynomial.zero()
    for label in SCALAR_LABELS:
        residual += (
            euler_derivative(model.classical_density, f'q_{label}', side='left')
            * generator(f'q_{label}', 1)
        )
        residual += (
            euler_derivative(model.classical_density, f'p_{label}', side='left')
            * generator(f'p_{label}', 1)
        )
    lapse_euler = euler_derivative(model.classical_density, 'n', side='left')
    residual += lapse_euler * generator('n', 1)
    residual -= total_derivative(lapse_euler * generator('n'))
    return residual


def locked_master_boundary_current(
    model: ReparamLocalBVModel,
) -> SparseSuperPolynomial:
    return -(
        generator('c')
        * generator('n')
        * model.hamiltonian_density_scalar
    )


def master_density(
    model: ReparamLocalBVModel,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    return Fraction(1, 2) * local_bv_antibracket_density(
        model.extended_density,
        model.extended_density,
        second_term_sign=second_term_sign,
    )


def derived_transformation_mismatch(
    model: ReparamLocalBVModel,
) -> SparseSuperPolynomial:
    return polynomial_sum(
        local_bv_antibracket_density(
            model.extended_density,
            generator(field.name),
        )
        - model.transformations[field.name]
        for field in local_field_specs()
    )


def bad_missing_lapse_classical_density(
    model: ReparamLocalBVModel,
) -> SparseSuperPolynomial:
    return (
        model.classical_density
        + generator('n') * model.hamiltonian_density_scalar
        - model.hamiltonian_density_scalar
    )


@dataclass(frozen=True)
class M1ReparamLocalBVQuotientContract:
    primary_source: str
    primary_source_url: str
    local_functional_source: str
    local_functional_source_url: str
    secondary_source: str
    secondary_source_url: str
    source_items: tuple[str, ...]
    model_relation: str
    normalization: str
    scalar_labels: tuple[str, ...]
    maximum_jet_order: int
    antibracket_convention: str
    euler_convention: str
    field_specs: tuple[LocalBVVariableSpec, ...]
    antifield_specs: tuple[LocalBVVariableSpec, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    bounded_jet_euler_calculus_constructed: bool
    terminal_jet_rejection_enforced: bool
    bounded_local_functional_antibracket_constructed: bool
    classical_noether_identity_computed: bool
    nonzero_horizontal_currents_retained: bool
    reparam_toy_cme_mod_dh_computed: bool
    explicit_master_current_constructed: bool
    graded_identities_sampled: bool
    silent_terminal_truncation_allowed: bool
    open_boundary_action_invariance_proved: bool
    unbounded_jet_closure_proved: bool
    general_local_functional_theorem_proved: bool
    four_dimensional_m1_action_used: bool
    full_m1_antifield_functional_constructed: bool
    full_m1_classical_master_equation_computed: bool
    boundary_completion_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_reparam_local_bv_quotient_contract() -> M1ReparamLocalBVQuotientContract:
    return M1ReparamLocalBVQuotientContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        local_functional_source=LOCAL_FUNCTIONAL_SOURCE,
        local_functional_source_url=LOCAL_FUNCTIONAL_SOURCE_URL,
        secondary_source=SECONDARY_SOURCE,
        secondary_source_url=SECONDARY_SOURCE_URL,
        source_items=SOURCE_ITEMS,
        model_relation=MODEL_RELATION,
        normalization=NORMALIZATION,
        scalar_labels=SCALAR_LABELS,
        maximum_jet_order=MAXIMUM_JET_ORDER,
        antibracket_convention=ANTIBRACKET_CONVENTION,
        euler_convention=EULER_CONVENTION,
        field_specs=local_field_specs(),
        antifield_specs=local_antifield_specs(),
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        bounded_jet_euler_calculus_constructed=True,
        terminal_jet_rejection_enforced=True,
        bounded_local_functional_antibracket_constructed=True,
        classical_noether_identity_computed=True,
        nonzero_horizontal_currents_retained=True,
        reparam_toy_cme_mod_dh_computed=True,
        explicit_master_current_constructed=True,
        graded_identities_sampled=True,
        silent_terminal_truncation_allowed=False,
        open_boundary_action_invariance_proved=False,
        unbounded_jet_closure_proved=False,
        general_local_functional_theorem_proved=False,
        four_dimensional_m1_action_used=False,
        full_m1_antifield_functional_constructed=False,
        full_m1_classical_master_equation_computed=False,
        boundary_completion_proved=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_bounded_reparam_local_bv_cme_mod_dh_not_four_dimensional_m1'
        ),
    )


def _serialize_local_spec(spec: LocalBVVariableSpec) -> str:
    return ':'.join(
        (
            spec.name,
            spec.role,
            str(spec.parity),
            str(spec.ghost_number),
            str(spec.antifield_number),
        )
    )


def canonical_contract_payload(
    contract: M1ReparamLocalBVQuotientContract,
) -> str:
    comma = chr(44)
    flags = comma.join(
        f'{name}:{getattr(contract, name)}'
        for name in (
            'bounded_jet_euler_calculus_constructed',
            'terminal_jet_rejection_enforced',
            'bounded_local_functional_antibracket_constructed',
            'classical_noether_identity_computed',
            'nonzero_horizontal_currents_retained',
            'reparam_toy_cme_mod_dh_computed',
            'explicit_master_current_constructed',
            'graded_identities_sampled',
            'silent_terminal_truncation_allowed',
            'open_boundary_action_invariance_proved',
            'unbounded_jet_closure_proved',
            'general_local_functional_theorem_proved',
            'four_dimensional_m1_action_used',
            'full_m1_antifield_functional_constructed',
            'full_m1_classical_master_equation_computed',
            'boundary_completion_proved',
            'functional_measure_computed',
            'quantum_master_equation_computed',
            'continuum_loop_st_computed',
            'positive_physical_hilbert_proved',
            'quantum_hda_m2_proved',
            'm3_relational_observables_unlocked',
        )
    )
    return '|'.join(
        (
            f'primary={contract.primary_source}',
            f'primary_url={contract.primary_source_url}',
            f'local_source={contract.local_functional_source}',
            f'local_source_url={contract.local_functional_source_url}',
            f'secondary={contract.secondary_source}',
            f'secondary_url={contract.secondary_source_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'model_relation={contract.model_relation}',
            f'normalization={contract.normalization}',
            f'labels={comma.join(contract.scalar_labels)}',
            f'max_jet={contract.maximum_jet_order}',
            f'antibracket={contract.antibracket_convention}',
            f'euler={contract.euler_convention}',
            f'fields={comma.join(_serialize_local_spec(x) for x in contract.field_specs)}',
            f'antifields={comma.join(_serialize_local_spec(x) for x in contract.antifield_specs)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(
    contract: M1ReparamLocalBVQuotientContract,
) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M1ReparamLocalBVQuotientContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.local_functional_source == LOCAL_FUNCTIONAL_SOURCE,
        contract.local_functional_source_url == LOCAL_FUNCTIONAL_SOURCE_URL,
        contract.secondary_source == SECONDARY_SOURCE,
        contract.secondary_source_url == SECONDARY_SOURCE_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.model_relation == MODEL_RELATION,
        contract.normalization == NORMALIZATION,
        contract.scalar_labels == SCALAR_LABELS,
        contract.maximum_jet_order == MAXIMUM_JET_ORDER,
        contract.antibracket_convention == ANTIBRACKET_CONVENTION,
        contract.euler_convention == EULER_CONVENTION,
        contract.field_specs == local_field_specs(),
        contract.antifield_specs == local_antifield_specs(),
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == 'exact_bounded_reparam_local_bv_cme_mod_dh_not_four_dimensional_m1',
    )
    if not all(frozen):
        raise ValueError('local BV source, basis, convention, or status lock changed')
    if len(contract.field_specs) != 14 or len(contract.antifield_specs) != 14:
        raise ValueError('the reparametrization witness requires 14 canonical pairs')
    for field, antifield in zip(
        contract.field_specs,
        contract.antifield_specs,
        strict=True,
    ):
        if (
            antifield.name != f'{field.name}_star'
            or antifield.parity != (field.parity + 1) % 2
            or antifield.ghost_number != -field.ghost_number - 1
            or antifield.antifield_number != 1
        ):
            raise ValueError('local field-antifield grading lock changed')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('bounded local BV contract hash mismatch')
    required_true = (
        contract.bounded_jet_euler_calculus_constructed,
        contract.terminal_jet_rejection_enforced,
        contract.bounded_local_functional_antibracket_constructed,
        contract.classical_noether_identity_computed,
        contract.nonzero_horizontal_currents_retained,
        contract.reparam_toy_cme_mod_dh_computed,
        contract.explicit_master_current_constructed,
        contract.graded_identities_sampled,
    )
    unsupported = (
        contract.silent_terminal_truncation_allowed,
        contract.open_boundary_action_invariance_proved,
        contract.unbounded_jet_closure_proved,
        contract.general_local_functional_theorem_proved,
        contract.four_dimensional_m1_action_used,
        contract.full_m1_antifield_functional_constructed,
        contract.full_m1_classical_master_equation_computed,
        contract.boundary_completion_proved,
        contract.functional_measure_computed,
        contract.quantum_master_equation_computed,
        contract.continuum_loop_st_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('bounded local BV claim flags changed')


def _nonzero_euler_count(density: SparseSuperPolynomial) -> int:
    return sum(not residual.is_zero for _, residual in all_euler_residuals(density))


def _maximum_euler_term_count(density: SparseSuperPolynomial) -> int:
    return max(
        residual.term_count
        for _, residual in all_euler_residuals(density)
    )


@dataclass(frozen=True)
class M1ReparamLocalBVQuotientReceipt:
    contract_sha256: str
    primary_source: str
    primary_source_url: str
    local_functional_source: str
    local_functional_source_url: str
    secondary_source: str
    secondary_source_url: str
    source_items: tuple[str, ...]
    model_relation: str
    normalization: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_d_verified: bool
    field_count: int
    antifield_count: int
    canonical_pair_count: int
    bounded_jet_generator_count: int
    bounded_even_jet_generator_count: int
    bounded_odd_jet_generator_count: int
    classical_density_term_count: int
    classical_boundary_current_term_count: int
    classical_variation_term_count: int
    classical_current_derivative_term_count: int
    classical_density_identity_mismatch_term_count: int
    classical_noether_identity_residual_term_count: int
    base_nilpotency_component_count: int
    base_nilpotency_maximum_residual_term_count: int
    derived_transformation_component_count: int
    derived_transformation_mismatch_term_count: int
    canonical_field_star_residual_term_count: int
    canonical_star_field_residual_term_count: int
    odd_left_right_derivative_mismatch_term_count: int
    total_derivative_fixture_term_count: int
    total_derivative_fixture_euler_maximum_term_count: int
    graded_antisymmetry_residual_term_count: int
    jacobi_nonzero_nested_bracket_count: int
    graded_jacobi_residual_term_count: int
    master_density_term_count: int
    master_boundary_current_term_count: int
    master_current_derivative_term_count: int
    master_current_mismatch_term_count: int
    master_euler_audit_count: int
    master_euler_maximum_residual_term_count: int
    master_afn0_term_count: int
    master_afn1_term_count: int
    bad_missing_lapse_identity_mismatch_term_count: int
    bad_missing_lapse_nonzero_euler_count: int
    missing_lapse_weight_identity_mismatch_term_count: int
    missing_lapse_weight_nonzero_euler_count: int
    wrong_ghost_sign_nonzero_nilpotency_component_count: int
    wrong_ghost_sign_maximum_nilpotency_residual_term_count: int
    wrong_ghost_sign_master_nonzero_euler_count: int
    omitted_ghost_antifield_master_term_count: int
    omitted_ghost_antifield_master_nonzero_euler_count: int
    uniform_plus_antifield_transformation_mismatch_term_count: int
    uniform_plus_antifield_master_nonzero_euler_count: int
    wrong_antibracket_canonical_residual_term_count: int
    wrong_antibracket_antisymmetry_residual_term_count: int
    wrong_antibracket_jacobi_residual_term_count: int
    naive_partial_vs_euler_difference_term_count: int
    terminal_jet_derivative_rejected: bool
    bounded_jet_euler_calculus_constructed: bool
    terminal_jet_rejection_enforced: bool
    bounded_local_functional_antibracket_constructed: bool
    classical_noether_identity_computed: bool
    nonzero_horizontal_currents_retained: bool
    reparam_toy_cme_mod_dh_computed: bool
    explicit_master_current_constructed: bool
    graded_identities_sampled: bool
    silent_terminal_truncation_allowed: bool
    open_boundary_action_invariance_proved: bool
    unbounded_jet_closure_proved: bool
    general_local_functional_theorem_proved: bool
    four_dimensional_m1_action_used: bool
    full_m1_antifield_functional_constructed: bool
    full_m1_classical_master_equation_computed: bool
    boundary_completion_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_reparam_local_bv_quotient_gate_passed: bool


def evaluate_m1_reparam_local_bv_quotient_gate() -> M1ReparamLocalBVQuotientReceipt:
    contract = m1_reparam_local_bv_quotient_contract()
    validate_contract(contract)
    upstream_contract = m1_bv_master_admission_contract()
    validate_e70_d_contract(upstream_contract)
    upstream_receipt = evaluate_m1_bv_master_admission_gate()

    model = reparam_local_bv_model()
    classical_variation = apply_local_brst(
        model.classical_density,
        model.transformations,
    )
    classical_current_derivative = total_derivative(
        model.classical_boundary_current
    )
    classical_mismatch = classical_variation - classical_current_derivative
    noether_residual = classical_noether_identity(model)
    nilpotency_residuals = tuple(
        apply_local_brst(image, model.transformations)
        for image in model.transformations.values()
    )
    derived_mismatch = derived_transformation_mismatch(model)

    q = generator('q_chi')
    q1 = generator('q_chi', 1)
    q_star = generator('q_chi_star')
    c = generator('c')
    c1 = generator('c', 1)
    c_star = generator('c_star')
    one = SparseSuperPolynomial.scalar(1)
    canonical_field_star = local_bv_antibracket_density(q, q_star) - one
    canonical_star_field = local_bv_antibracket_density(q_star, q) + one
    two_odd = c * q_star
    odd_left_right_mismatch = (
        jet_partial_derivative(two_odd, 'c', 0, side='left')
        + jet_partial_derivative(two_odd, 'c', 0, side='right')
    )

    total_derivative_fixture = total_derivative(
        generator('q_chi') * generator('p_chi')
    )
    sample_first = q_star * c * q1
    sample_second = -(c_star * c * c1)
    sample_third = q
    antisymmetry_residual = graded_antisymmetry_residual(
        sample_first,
        sample_second,
    )
    nested_brackets = (
        local_bv_antibracket_density(
            sample_first,
            local_bv_antibracket_density(sample_second, sample_third),
        ),
        local_bv_antibracket_density(
            sample_second,
            local_bv_antibracket_density(sample_third, sample_first),
        ),
        local_bv_antibracket_density(
            sample_third,
            local_bv_antibracket_density(sample_first, sample_second),
        ),
    )
    jacobi_residual = graded_jacobi_residual(
        sample_first,
        sample_second,
        sample_third,
    )

    master = master_density(model)
    master_current = locked_master_boundary_current(model)
    master_current_derivative = total_derivative(master_current)
    master_mismatch = master - master_current_derivative
    master_euler = all_euler_residuals(master)
    master_components = antifield_number_components(master)

    bad_density = bad_missing_lapse_classical_density(model)
    bad_density_mismatch = (
        apply_local_brst(bad_density, model.transformations)
        - total_derivative(c * bad_density)
    )
    missing_weight_model = reparam_local_bv_model(
        include_lapse_weight_term=False
    )
    missing_weight_mismatch = (
        apply_local_brst(
            missing_weight_model.classical_density,
            missing_weight_model.transformations,
        )
        - total_derivative(
            c * missing_weight_model.classical_density
        )
    )
    wrong_ghost_model = reparam_local_bv_model(ghost_sign=-1)
    wrong_ghost_nilpotency = tuple(
        apply_local_brst(image, wrong_ghost_model.transformations)
        for image in wrong_ghost_model.transformations.values()
    )
    wrong_ghost_master = master_density(wrong_ghost_model)
    omitted_ghost_model = reparam_local_bv_model(
        include_ghost_antifield_term=False
    )
    omitted_ghost_master = master_density(omitted_ghost_model)
    uniform_plus_model = reparam_local_bv_model(
        parity_signed_antifields=False
    )
    uniform_plus_master = master_density(uniform_plus_model)
    wrong_canonical = (
        local_bv_antibracket_density(
            q_star,
            q,
            second_term_sign=1,
        )
        + one
    )
    wrong_antisymmetry = graded_antisymmetry_residual(
        sample_first,
        sample_second,
        second_term_sign=1,
    )
    wrong_jacobi = graded_jacobi_residual(
        sample_first,
        sample_second,
        sample_third,
        second_term_sign=1,
    )
    naive_partial = jet_partial_derivative(
        model.classical_density,
        'q_chi',
        0,
        side='left',
    )
    exact_euler = euler_derivative(
        model.classical_density,
        'q_chi',
        side='left',
    )
    terminal_rejected = False
    try:
        total_derivative(generator('q_chi', MAXIMUM_JET_ORDER))
    except JetOrderExceeded:
        terminal_rejected = True

    unsupported = (
        contract.silent_terminal_truncation_allowed,
        contract.open_boundary_action_invariance_proved,
        contract.unbounded_jet_closure_proved,
        contract.general_local_functional_theorem_proved,
        contract.four_dimensional_m1_action_used,
        contract.full_m1_antifield_functional_constructed,
        contract.full_m1_classical_master_equation_computed,
        contract.boundary_completion_proved,
        contract.functional_measure_computed,
        contract.quantum_master_equation_computed,
        contract.continuum_loop_st_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    passed = all(
        (
            upstream_receipt.declared_m1_bv_master_admission_gate_passed,
            len(contract.field_specs) == 14,
            len(contract.antifield_specs) == 14,
            len(JET_LOOKUP) == 140,
            classical_mismatch.is_zero,
            not model.classical_boundary_current.is_zero,
            noether_residual.is_zero,
            max(x.term_count for x in nilpotency_residuals) == 0,
            derived_mismatch.is_zero,
            canonical_field_star.is_zero,
            canonical_star_field.is_zero,
            odd_left_right_mismatch.is_zero,
            _maximum_euler_term_count(total_derivative_fixture) == 0,
            antisymmetry_residual.is_zero,
            sum(not x.is_zero for x in nested_brackets) == 2,
            jacobi_residual.is_zero,
            not master.is_zero,
            not master_current.is_zero,
            master_mismatch.is_zero,
            max(x.term_count for _, x in master_euler) == 0,
            master_components.get(0, SparseSuperPolynomial.zero()).term_count == 30,
            master_components.get(1, SparseSuperPolynomial.zero()).is_zero,
            bad_density_mismatch.term_count > 0,
            _nonzero_euler_count(bad_density_mismatch) > 0,
            missing_weight_mismatch.term_count > 0,
            _nonzero_euler_count(missing_weight_mismatch) > 0,
            sum(not x.is_zero for x in wrong_ghost_nilpotency) > 0,
            _nonzero_euler_count(wrong_ghost_master) > 0,
            _nonzero_euler_count(omitted_ghost_master) > 0,
            derived_transformation_mismatch(uniform_plus_model).term_count > 0,
            _nonzero_euler_count(uniform_plus_master) > 0,
            wrong_canonical.term_count > 0,
            wrong_antisymmetry.term_count > 0,
            wrong_jacobi.term_count > 0,
            (naive_partial - exact_euler).term_count > 0,
            terminal_rejected,
            not any(unsupported),
        )
    )
    field_even = sum(spec.parity == 0 for spec in contract.field_specs)
    antifield_even = sum(spec.parity == 0 for spec in contract.antifield_specs)
    return M1ReparamLocalBVQuotientReceipt(
        contract_sha256=contract.contract_sha256,
        primary_source=contract.primary_source,
        primary_source_url=contract.primary_source_url,
        local_functional_source=contract.local_functional_source,
        local_functional_source_url=contract.local_functional_source_url,
        secondary_source=contract.secondary_source,
        secondary_source_url=contract.secondary_source_url,
        source_items=contract.source_items,
        model_relation=contract.model_relation,
        normalization=contract.normalization,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_d_verified=True,
        field_count=len(contract.field_specs),
        antifield_count=len(contract.antifield_specs),
        canonical_pair_count=len(contract.field_specs),
        bounded_jet_generator_count=len(JET_LOOKUP),
        bounded_even_jet_generator_count=(field_even + antifield_even) * 5,
        bounded_odd_jet_generator_count=(28 - field_even - antifield_even) * 5,
        classical_density_term_count=model.classical_density.term_count,
        classical_boundary_current_term_count=(
            model.classical_boundary_current.term_count
        ),
        classical_variation_term_count=classical_variation.term_count,
        classical_current_derivative_term_count=(
            classical_current_derivative.term_count
        ),
        classical_density_identity_mismatch_term_count=classical_mismatch.term_count,
        classical_noether_identity_residual_term_count=noether_residual.term_count,
        base_nilpotency_component_count=len(nilpotency_residuals),
        base_nilpotency_maximum_residual_term_count=max(
            x.term_count for x in nilpotency_residuals
        ),
        derived_transformation_component_count=len(model.transformations),
        derived_transformation_mismatch_term_count=derived_mismatch.term_count,
        canonical_field_star_residual_term_count=canonical_field_star.term_count,
        canonical_star_field_residual_term_count=canonical_star_field.term_count,
        odd_left_right_derivative_mismatch_term_count=(
            odd_left_right_mismatch.term_count
        ),
        total_derivative_fixture_term_count=total_derivative_fixture.term_count,
        total_derivative_fixture_euler_maximum_term_count=(
            _maximum_euler_term_count(total_derivative_fixture)
        ),
        graded_antisymmetry_residual_term_count=antisymmetry_residual.term_count,
        jacobi_nonzero_nested_bracket_count=sum(
            not x.is_zero for x in nested_brackets
        ),
        graded_jacobi_residual_term_count=jacobi_residual.term_count,
        master_density_term_count=master.term_count,
        master_boundary_current_term_count=master_current.term_count,
        master_current_derivative_term_count=master_current_derivative.term_count,
        master_current_mismatch_term_count=master_mismatch.term_count,
        master_euler_audit_count=len(master_euler),
        master_euler_maximum_residual_term_count=max(
            x.term_count for _, x in master_euler
        ),
        master_afn0_term_count=master_components.get(
            0,
            SparseSuperPolynomial.zero(),
        ).term_count,
        master_afn1_term_count=master_components.get(
            1,
            SparseSuperPolynomial.zero(),
        ).term_count,
        bad_missing_lapse_identity_mismatch_term_count=bad_density_mismatch.term_count,
        bad_missing_lapse_nonzero_euler_count=_nonzero_euler_count(
            bad_density_mismatch
        ),
        missing_lapse_weight_identity_mismatch_term_count=(
            missing_weight_mismatch.term_count
        ),
        missing_lapse_weight_nonzero_euler_count=_nonzero_euler_count(
            missing_weight_mismatch
        ),
        wrong_ghost_sign_nonzero_nilpotency_component_count=sum(
            not x.is_zero for x in wrong_ghost_nilpotency
        ),
        wrong_ghost_sign_maximum_nilpotency_residual_term_count=max(
            x.term_count for x in wrong_ghost_nilpotency
        ),
        wrong_ghost_sign_master_nonzero_euler_count=_nonzero_euler_count(
            wrong_ghost_master
        ),
        omitted_ghost_antifield_master_term_count=omitted_ghost_master.term_count,
        omitted_ghost_antifield_master_nonzero_euler_count=_nonzero_euler_count(
            omitted_ghost_master
        ),
        uniform_plus_antifield_transformation_mismatch_term_count=(
            derived_transformation_mismatch(uniform_plus_model).term_count
        ),
        uniform_plus_antifield_master_nonzero_euler_count=_nonzero_euler_count(
            uniform_plus_master
        ),
        wrong_antibracket_canonical_residual_term_count=wrong_canonical.term_count,
        wrong_antibracket_antisymmetry_residual_term_count=(
            wrong_antisymmetry.term_count
        ),
        wrong_antibracket_jacobi_residual_term_count=wrong_jacobi.term_count,
        naive_partial_vs_euler_difference_term_count=(
            naive_partial - exact_euler
        ).term_count,
        terminal_jet_derivative_rejected=terminal_rejected,
        bounded_jet_euler_calculus_constructed=(
            contract.bounded_jet_euler_calculus_constructed
        ),
        terminal_jet_rejection_enforced=contract.terminal_jet_rejection_enforced,
        bounded_local_functional_antibracket_constructed=(
            contract.bounded_local_functional_antibracket_constructed
        ),
        classical_noether_identity_computed=(
            contract.classical_noether_identity_computed
        ),
        nonzero_horizontal_currents_retained=(
            contract.nonzero_horizontal_currents_retained
        ),
        reparam_toy_cme_mod_dh_computed=contract.reparam_toy_cme_mod_dh_computed,
        explicit_master_current_constructed=(
            contract.explicit_master_current_constructed
        ),
        graded_identities_sampled=contract.graded_identities_sampled,
        silent_terminal_truncation_allowed=(
            contract.silent_terminal_truncation_allowed
        ),
        open_boundary_action_invariance_proved=(
            contract.open_boundary_action_invariance_proved
        ),
        unbounded_jet_closure_proved=contract.unbounded_jet_closure_proved,
        general_local_functional_theorem_proved=(
            contract.general_local_functional_theorem_proved
        ),
        four_dimensional_m1_action_used=contract.four_dimensional_m1_action_used,
        full_m1_antifield_functional_constructed=(
            contract.full_m1_antifield_functional_constructed
        ),
        full_m1_classical_master_equation_computed=(
            contract.full_m1_classical_master_equation_computed
        ),
        boundary_completion_proved=contract.boundary_completion_proved,
        functional_measure_computed=contract.functional_measure_computed,
        quantum_master_equation_computed=(
            contract.quantum_master_equation_computed
        ),
        continuum_loop_st_computed=contract.continuum_loop_st_computed,
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        m3_relational_observables_unlocked=(
            contract.m3_relational_observables_unlocked
        ),
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_m1_reparam_local_bv_quotient_gate_passed=passed,
    )
