'''BV field-antifield and formal master-decomposition admission for M1.

The module locks all 27 E70-B base fields to antifields, calibrates a finite
canonical BV antibracket with exact left/right derivatives, and separates the
antifield-number-zero action residual from the antifield-number-one
nilpotency residual.  A nontrivial finite toy solves the classical master
equation, while independent bad-action and broken-doublet controls fail.

For M1 itself this remains an admission ledger.  E70-C is a bulk
type/naturality certificate rather than a coordinate-jet functional
variation, and E70-B does not define an all-jet evolutionary differential.
No M1 antifield functional, variational bicomplex, boundary quotient, or full
M1 classical master equation is constructed here.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    CONTRACT_SHA256 as E70_B_HASH,
    SparseSuperPolynomial,
    build_jet_differential,
    evaluate_four_scalar_classical_brst_jet_gate,
    four_scalar_classical_brst_jet_contract,
    validate_contract as validate_e70_b_contract,
)
from examples.physics.qft_m2_m1_classical_action_gaugefixing import (
    CONTRACT_SHA256 as E70_C_HASH,
    evaluate_m1_classical_action_gaugefixing_gate,
    m1_classical_action_gaugefixing_contract,
    validate_contract as validate_e70_c_contract,
)


PRIMARY_SOURCE = 'hep-th/0506098'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0506098'
SECONDARY_SOURCE = 'arXiv:2206.00780v2'
SECONDARY_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
SOURCE_ITEMS = (
    'Fuster--Henneaux--Maas: field-antifield parity, ghost number, antibracket, and CME',
    'Prinz v2 Proposition 3.3/Eq. (35): classical diffeomorphism-BRST nilpotency',
    'Prinz v2 Remark 2.14: BV/renormalization and quantum ST lifting remain separate',
)
SOURCE_RELATION = (
    'the BV conventions are imported from the antifield review; Prinz v2 '
    'supplies the perturbative BRST differential but not a full antifield '
    'action or M1 CME; this M1 ledger is convention-adapted, not a literal '
    'source transcription'
)
ANTIBRACKET_CONVENTION = (
    '(F,G)=sum_i[(d_R F/d Phi_i)(d_L G/d Phi_i*)-'
    '(d_R F/d Phi_i*)(d_L G/d Phi_i)]; sF=(S,F) with the standard left '
    'graded Leibniz rule'
)
ANTIFIELD_RULE = (
    'parity(Phi*)=parity(Phi)+1 mod 2; '
    'gh(Phi*)=-gh(Phi)-1; afn(Phi*)=1'
)
FORMAL_MASTER_DECOMPOSITION = (
    '1/2(S_ext,S_ext)=sS0 at antifield number zero plus '
    'sum_i (-1)^parity(Phi_i) Phi_i* s^2 Phi_i at antifield number one, '
    'only after a declared '
    'local functional variational calculus and horizontal-boundary quotient'
)
CLAIM_CEILING = (
    'exact 27-pair M1 field-antifield type ledger and exact finite canonical '
    'standard-left BV toy calibration including a nonzero ghost map, plus a '
    'formal parity-signed AFN0/AFN1 master-residual admission '
    'split; no M1 antifield functional, jet variational antibracket, local '
    'functional boundary quotient, full M1 CME, BV measure, QME, loop ST, '
    'physical Hilbert, HDA, quantum M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-B', E70_B_HASH), ('E70-C', E70_C_HASH))
CONTRACT_SHA256 = (
    '90397427cad532820a1d301a416e7c417a07ae92fdb0862fd9ccf481e277fb58'
)


@dataclass(frozen=True)
class BVVariableSpec:
    name: str
    tensor_type: str
    parity: int
    ghost_number: int
    mass_dimension: int
    antifield_number: int
    maximum_jet_order: int


def m1_base_field_specs() -> tuple[BVVariableSpec, ...]:
    specs: list[BVVariableSpec] = []
    for mu in range(4):
        specs.append(
            BVVariableSpec(f'c{mu}', 'contravariant ghost vector', 1, 1, -1, 0, 0)
        )
    specs.append(BVVariableSpec('chi', 'matter scalar', 0, 0, 1, 0, 0))
    for label in ('X0', 'X1', 'X2', 'X3'):
        specs.append(BVVariableSpec(label, 'reference scalar', 0, 0, 0, 0, 0))
    for mu in range(4):
        for nu in range(mu, 4):
            specs.append(
                BVVariableSpec(
                    f'g{mu}{nu}',
                    'symmetric covariant metric component',
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            )
    for mu in range(4):
        specs.append(
            BVVariableSpec(f'barc{mu}', 'antighost covector', 1, -1, 3, 0, 0)
        )
    for mu in range(4):
        specs.append(
            BVVariableSpec(f'B{mu}', 'Nakanishi--Lautrup covector', 0, 0, 3, 0, 0)
        )
    return tuple(specs)


def antifield_spec(field: BVVariableSpec) -> BVVariableSpec:
    return BVVariableSpec(
        name=f'{field.name}_star',
        tensor_type=f'dual density to {field.tensor_type}',
        parity=(field.parity + 1) % 2,
        ghost_number=-field.ghost_number - 1,
        mass_dimension=4 - field.mass_dimension,
        antifield_number=1,
        maximum_jet_order=0,
    )


def m1_antifield_specs() -> tuple[BVVariableSpec, ...]:
    return tuple(antifield_spec(field) for field in m1_base_field_specs())


@dataclass(frozen=True)
class M1BVMasterAdmissionContract:
    primary_source: str
    primary_source_url: str
    secondary_source: str
    secondary_source_url: str
    source_items: tuple[str, ...]
    source_relation: str
    antibracket_convention: str
    antifield_rule: str
    formal_master_decomposition: str
    base_field_specs: tuple[BVVariableSpec, ...]
    antifield_specs: tuple[BVVariableSpec, ...]
    claim_ceiling: str
    upstream_hashes: tuple[tuple[str, str], ...]
    contract_sha256: str
    antifield_ledger_constructed: bool
    finite_canonical_antibracket_calibrated: bool
    finite_toy_classical_master_equation_computed: bool
    formal_m1_master_residual_decomposition_admitted: bool
    full_m1_antifield_functional_constructed: bool
    jet_antifield_variational_calculus_constructed: bool
    local_functional_boundary_quotient_constructed: bool
    boundary_completion_proved: bool
    full_m1_classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_bv_master_admission_contract() -> M1BVMasterAdmissionContract:
    return M1BVMasterAdmissionContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        secondary_source=SECONDARY_SOURCE,
        secondary_source_url=SECONDARY_SOURCE_URL,
        source_items=SOURCE_ITEMS,
        source_relation=SOURCE_RELATION,
        antibracket_convention=ANTIBRACKET_CONVENTION,
        antifield_rule=ANTIFIELD_RULE,
        formal_master_decomposition=FORMAL_MASTER_DECOMPOSITION,
        base_field_specs=m1_base_field_specs(),
        antifield_specs=m1_antifield_specs(),
        claim_ceiling=CLAIM_CEILING,
        upstream_hashes=UPSTREAM_HASHES,
        contract_sha256=CONTRACT_SHA256,
        antifield_ledger_constructed=True,
        finite_canonical_antibracket_calibrated=True,
        finite_toy_classical_master_equation_computed=True,
        formal_m1_master_residual_decomposition_admitted=True,
        full_m1_antifield_functional_constructed=False,
        jet_antifield_variational_calculus_constructed=False,
        local_functional_boundary_quotient_constructed=False,
        boundary_completion_proved=False,
        full_m1_classical_master_equation_computed=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_antifield_ledger_and_standard_left_finite_bv_toy_m1_cme_incomplete'
        ),
    )


def _serialize_spec(spec: BVVariableSpec) -> str:
    return ':'.join(
        (
            spec.name,
            spec.tensor_type,
            str(spec.parity),
            str(spec.ghost_number),
            str(spec.mass_dimension),
            str(spec.antifield_number),
            str(spec.maximum_jet_order),
        )
    )


def canonical_contract_payload(contract: M1BVMasterAdmissionContract) -> str:
    comma = chr(44)
    upstream = comma.join(
        f'{name}:{value}' for name, value in contract.upstream_hashes
    )
    flags = comma.join(
        f'{name}:{getattr(contract, name)}'
        for name in (
            'antifield_ledger_constructed',
            'finite_canonical_antibracket_calibrated',
            'finite_toy_classical_master_equation_computed',
            'formal_m1_master_residual_decomposition_admitted',
            'full_m1_antifield_functional_constructed',
            'jet_antifield_variational_calculus_constructed',
            'local_functional_boundary_quotient_constructed',
            'boundary_completion_proved',
            'full_m1_classical_master_equation_computed',
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
            f'secondary={contract.secondary_source}',
            f'secondary_url={contract.secondary_source_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_relation={contract.source_relation}',
            f'antibracket={contract.antibracket_convention}',
            f'antifield_rule={contract.antifield_rule}',
            f'decomposition={contract.formal_master_decomposition}',
            f'fields={comma.join(_serialize_spec(x) for x in contract.base_field_specs)}',
            f'antifields={comma.join(_serialize_spec(x) for x in contract.antifield_specs)}',
            f'ceiling={contract.claim_ceiling}',
            f'upstream={upstream}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(contract: M1BVMasterAdmissionContract) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M1BVMasterAdmissionContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.secondary_source == SECONDARY_SOURCE,
        contract.secondary_source_url == SECONDARY_SOURCE_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.source_relation == SOURCE_RELATION,
        contract.antibracket_convention == ANTIBRACKET_CONVENTION,
        contract.antifield_rule == ANTIFIELD_RULE,
        contract.formal_master_decomposition == FORMAL_MASTER_DECOMPOSITION,
        contract.base_field_specs == m1_base_field_specs(),
        contract.antifield_specs == m1_antifield_specs(),
        contract.claim_ceiling == CLAIM_CEILING,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.derivation_status
        == 'exact_antifield_ledger_and_standard_left_finite_bv_toy_m1_cme_incomplete',
    )
    if not all(frozen):
        raise ValueError('BV source, convention, ledger, or status lock changed')
    if len(contract.base_field_specs) != 27 or len(contract.antifield_specs) != 27:
        raise ValueError('the full M1 BV ledger requires 27 field-antifield pairs')
    for field, star in zip(
        contract.base_field_specs,
        contract.antifield_specs,
        strict=True,
    ):
        if (
            star.name != f'{field.name}_star'
            or star.parity != (field.parity + 1) % 2
            or star.ghost_number != -field.ghost_number - 1
            or star.mass_dimension + field.mass_dimension != 4
            or star.antifield_number != 1
        ):
            raise ValueError('antifield parity, degree, or dimension rule changed')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('M1 BV admission contract hash mismatch')
    required_true = (
        contract.antifield_ledger_constructed,
        contract.finite_canonical_antibracket_calibrated,
        contract.finite_toy_classical_master_equation_computed,
        contract.formal_m1_master_residual_decomposition_admitted,
    )
    unsupported = (
        contract.full_m1_antifield_functional_constructed,
        contract.jet_antifield_variational_calculus_constructed,
        contract.local_functional_boundary_quotient_constructed,
        contract.boundary_completion_proved,
        contract.full_m1_classical_master_equation_computed,
        contract.functional_measure_computed,
        contract.quantum_master_equation_computed,
        contract.continuum_loop_st_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('BV admission claim flags changed')


def _even(name: str) -> SparseSuperPolynomial:
    return SparseSuperPolynomial.generator(name, odd=False)


def _odd(name: str) -> SparseSuperPolynomial:
    return SparseSuperPolynomial.generator(name, odd=True)


def bv_left_derivative(
    polynomial: SparseSuperPolynomial,
    variable: str,
    *,
    odd: bool,
) -> SparseSuperPolynomial:
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        names = odd_names if odd else even_names
        for index, name in enumerate(names):
            if name != variable:
                continue
            if odd:
                remaining_odd = odd_names[:index] + odd_names[index + 1 :]
                sign = -1 if index % 2 else 1
                term = SparseSuperPolynomial.monomial(
                    even=even_names,
                    odd=remaining_odd,
                )
            else:
                remaining_even = even_names[:index] + even_names[index + 1 :]
                sign = 1
                term = SparseSuperPolynomial.monomial(
                    even=remaining_even,
                    odd=odd_names,
                )
            result += coefficient * sign * term
    return result


def bv_right_derivative(
    polynomial: SparseSuperPolynomial,
    variable: str,
    *,
    odd: bool,
) -> SparseSuperPolynomial:
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        names = odd_names if odd else even_names
        for index, name in enumerate(names):
            if name != variable:
                continue
            if odd:
                remaining_odd = odd_names[:index] + odd_names[index + 1 :]
                sign = -1 if (len(odd_names) - 1 - index) % 2 else 1
                term = SparseSuperPolynomial.monomial(
                    even=even_names,
                    odd=remaining_odd,
                )
            else:
                remaining_even = even_names[:index] + even_names[index + 1 :]
                sign = 1
                term = SparseSuperPolynomial.monomial(
                    even=remaining_even,
                    odd=odd_names,
                )
            result += coefficient * sign * term
    return result


@dataclass(frozen=True)
class CanonicalBVPair:
    field: str
    antifield: str
    field_is_odd: bool


def bv_antibracket(
    left: SparseSuperPolynomial,
    right: SparseSuperPolynomial,
    pairs: tuple[CanonicalBVPair, ...],
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    if second_term_sign not in (-1, 1):
        raise ValueError('the second antibracket sign must be plus or minus one')
    result = SparseSuperPolynomial.zero()
    for pair in pairs:
        field_odd = pair.field_is_odd
        antifield_odd = not field_odd
        result += (
            bv_right_derivative(left, pair.field, odd=field_odd)
            * bv_left_derivative(right, pair.antifield, odd=antifield_odd)
        )
        result += second_term_sign * (
            bv_right_derivative(left, pair.antifield, odd=antifield_odd)
            * bv_left_derivative(right, pair.field, odd=field_odd)
        )
    return result


@dataclass(frozen=True)
class FiniteBVToyAlgebra:
    pairs: tuple[CanonicalBVPair, ...]
    action: SparseSuperPolynomial
    master_residual: SparseSuperPolynomial
    canonical_field_star_residual: SparseSuperPolynomial
    canonical_star_field_residual: SparseSuperPolynomial
    transformation_mismatch: SparseSuperPolynomial
    bad_action_master_residual: SparseSuperPolynomial
    broken_doublet_master_residual: SparseSuperPolynomial
    wrong_antibracket_sign_residual: SparseSuperPolynomial
    wrong_odd_antifield_sign_transformation_mismatch: SparseSuperPolynomial


def finite_bv_toy_algebra() -> FiniteBVToyAlgebra:
    pairs = (
        CanonicalBVPair('x', 'x_star', False),
        CanonicalBVPair('y', 'y_star', False),
        CanonicalBVPair('c', 'c_star', True),
        CanonicalBVPair('barc', 'barc_star', True),
        CanonicalBVPair('B', 'B_star', False),
        CanonicalBVPair('u', 'u_star', True),
        CanonicalBVPair('v', 'v_star', True),
    )
    x = _even('x')
    y = _even('y')
    c = _odd('c')
    barc = _odd('barc')
    b = _even('B')
    x_star = _odd('x_star')
    y_star = _odd('y_star')
    barc_star = _even('barc_star')
    u = _odd('u')
    v = _odd('v')
    u_star = _even('u_star')

    invariant_action = Fraction(1, 2) * (x * x + y * y)
    antifield_action = (
        x_star * c * y
        - y_star * c * x
        - barc_star * b
        - u_star * u * v
    )
    action = invariant_action + antifield_action
    master_residual = Fraction(1, 2) * bv_antibracket(
        action,
        action,
        pairs,
    )

    one = SparseSuperPolynomial.scalar(1)
    canonical_field_star = bv_antibracket(x, x_star, pairs) - one
    canonical_star_field = bv_antibracket(x_star, x, pairs) + one
    locked_transformations = {
        'x': c * y,
        'y': -(c * x),
        'c': SparseSuperPolynomial.zero(),
        'barc': b,
        'B': SparseSuperPolynomial.zero(),
        'u': u * v,
        'v': SparseSuperPolynomial.zero(),
    }
    variables = {
        'x': x,
        'y': y,
        'c': c,
        'barc': barc,
        'B': b,
        'u': u,
        'v': v,
    }
    transformation_mismatch = SparseSuperPolynomial.zero()
    for name, variable in variables.items():
        transformation_mismatch += (
            bv_antibracket(action, variable, pairs)
            - locked_transformations[name]
        )

    wrong_odd_sign_action = (
        action
        + 2 * barc_star * b
        + 2 * u_star * u * v
    )
    wrong_odd_sign_mismatch = SparseSuperPolynomial.zero()
    for name in ('barc', 'u'):
        wrong_odd_sign_mismatch += (
            bv_antibracket(wrong_odd_sign_action, variables[name], pairs)
            - locked_transformations[name]
        )

    bad_action = Fraction(1, 2) * x * x + antifield_action
    bad_action_master = Fraction(1, 2) * bv_antibracket(
        bad_action,
        bad_action,
        pairs,
    )

    j = _odd('J')
    b_star = _odd('B_star')
    broken_pairs = pairs + (CanonicalBVPair('J', 'J_star', True),)
    broken_action = action + b_star * j
    broken_master = Fraction(1, 2) * bv_antibracket(
        broken_action,
        broken_action,
        broken_pairs,
    )

    wrong_star_field = bv_antibracket(
        x_star,
        x,
        pairs,
        second_term_sign=1,
    )
    wrong_sign_residual = wrong_star_field + one
    return FiniteBVToyAlgebra(
        pairs=pairs,
        action=action,
        master_residual=master_residual,
        canonical_field_star_residual=canonical_field_star,
        canonical_star_field_residual=canonical_star_field,
        transformation_mismatch=transformation_mismatch,
        bad_action_master_residual=bad_action_master,
        broken_doublet_master_residual=broken_master,
        wrong_antibracket_sign_residual=wrong_sign_residual,
        wrong_odd_antifield_sign_transformation_mismatch=(
            wrong_odd_sign_mismatch
        ),
    )


@dataclass(frozen=True)
class M1BVMasterAdmissionReceipt:
    contract_sha256: str
    primary_source: str
    primary_source_url: str
    secondary_source: str
    secondary_source_url: str
    source_items: tuple[str, ...]
    source_relation: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_contracts_verified: bool
    antibracket_convention: str
    antifield_rule: str
    base_field_count: int
    antifield_count: int
    base_even_count: int
    base_odd_count: int
    antifield_even_count: int
    antifield_odd_count: int
    base_name_coverage_mismatch_count: int
    antifield_name_coverage_mismatch_count: int
    antifield_rule_mismatch_count: int
    antifield_action_term_type_audit_count: int
    antifield_action_term_type_mismatch_count: int
    omitted_field_antifield_pair_rejected: bool
    wrong_antifield_parity_rejected: bool
    finite_toy_canonical_pair_count: int
    finite_toy_field_star_calibration_residual_term_count: int
    finite_toy_star_field_calibration_residual_term_count: int
    finite_toy_transformation_mismatch_term_count: int
    finite_toy_master_residual_term_count: int
    bad_action_master_residual_term_count: int
    broken_doublet_master_residual_term_count: int
    wrong_antibracket_sign_residual_term_count: int
    wrong_odd_antifield_sign_transformation_mismatch_term_count: int
    upstream_base_nilpotency_component_count: int
    upstream_base_nilpotency_maximum_residual_term_count: int
    upstream_bulk_type_naturality_term_count: int
    upstream_bulk_type_naturality_maximum_residual: int
    upstream_boundary_flux_retained: bool
    upstream_full_coordinate_jet_action_variation_computed: bool
    formal_afn0_input_status: str
    formal_afn1_input_status: str
    formal_master_residual_input_count: int
    antifield_ledger_constructed: bool
    finite_canonical_antibracket_calibrated: bool
    finite_toy_classical_master_equation_computed: bool
    formal_m1_master_residual_decomposition_admitted: bool
    full_m1_antifield_functional_constructed: bool
    jet_antifield_variational_calculus_constructed: bool
    local_functional_boundary_quotient_constructed: bool
    boundary_completion_proved: bool
    full_m1_classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_bv_master_admission_gate_passed: bool


def evaluate_m1_bv_master_admission_gate() -> M1BVMasterAdmissionReceipt:
    contract = m1_bv_master_admission_contract()
    validate_contract(contract)

    e70_b_contract = four_scalar_classical_brst_jet_contract()
    validate_e70_b_contract(e70_b_contract)
    e70_b_receipt = evaluate_four_scalar_classical_brst_jet_gate()
    e70_c_contract = m1_classical_action_gaugefixing_contract()
    validate_e70_c_contract(e70_c_contract)
    e70_c_receipt = evaluate_m1_classical_action_gaugefixing_gate()

    fields = contract.base_field_specs
    antifields = contract.antifield_specs
    differential = build_jet_differential()
    expected_field_names = tuple(differential.base_field_names)
    ledger_field_names = tuple(spec.name for spec in fields)
    base_name_mismatch = len(
        set(expected_field_names).symmetric_difference(ledger_field_names)
    )
    expected_antifield_names = tuple(f'{name}_star' for name in expected_field_names)
    ledger_antifield_names = tuple(spec.name for spec in antifields)
    antifield_name_mismatch = len(
        set(expected_antifield_names).symmetric_difference(ledger_antifield_names)
    )
    antifield_rule_mismatch = sum(
        (
            star.parity != (field.parity + 1) % 2
            or star.ghost_number != -field.ghost_number - 1
            or star.antifield_number != 1
            or star.mass_dimension + field.mass_dimension != 4
        )
        for field, star in zip(fields, antifields, strict=True)
    )
    term_type_mismatch = sum(
        (
            (star.parity + field.parity + 1) % 2 != 0
            or star.ghost_number + field.ghost_number + 1 != 0
            or star.mass_dimension + field.mass_dimension != 4
        )
        for field, star in zip(fields, antifields, strict=True)
    )

    omitted_pair_rejected = False
    try:
        validate_contract(
            replace(
                contract,
                base_field_specs=contract.base_field_specs[:-1],
                antifield_specs=contract.antifield_specs[:-1],
            )
        )
    except ValueError:
        omitted_pair_rejected = True

    wrong_parity_rejected = False
    wrong_first_star = replace(
        contract.antifield_specs[0],
        parity=contract.base_field_specs[0].parity,
    )
    try:
        validate_contract(
            replace(
                contract,
                antifield_specs=(wrong_first_star,) + contract.antifield_specs[1:],
            )
        )
    except ValueError:
        wrong_parity_rejected = True

    toy = finite_bv_toy_algebra()
    nilpotency_maximum = max(
        count for _, count in e70_b_receipt.nilpotency_residual_term_counts
    )
    type_naturality_maximum = max(
        e70_c_receipt.maximum_bulk_dimension_residual,
        e70_c_receipt.maximum_bulk_density_weight_residual,
    )
    unsupported = (
        contract.full_m1_antifield_functional_constructed,
        contract.jet_antifield_variational_calculus_constructed,
        contract.local_functional_boundary_quotient_constructed,
        contract.boundary_completion_proved,
        contract.full_m1_classical_master_equation_computed,
        contract.functional_measure_computed,
        contract.quantum_master_equation_computed,
        contract.continuum_loop_st_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    passed = all(
        (
            e70_b_receipt.declared_exact_classical_brst_jet_gate_passed,
            e70_c_receipt.declared_m1_classical_action_gaugefixing_gate_passed,
            len(fields) == 27,
            len(antifields) == 27,
            base_name_mismatch == 0,
            antifield_name_mismatch == 0,
            antifield_rule_mismatch == 0,
            term_type_mismatch == 0,
            omitted_pair_rejected,
            wrong_parity_rejected,
            toy.canonical_field_star_residual.is_zero,
            toy.canonical_star_field_residual.is_zero,
            toy.transformation_mismatch.is_zero,
            toy.master_residual.is_zero,
            toy.bad_action_master_residual.term_count > 0,
            toy.broken_doublet_master_residual.term_count > 0,
            toy.wrong_antibracket_sign_residual.term_count > 0,
            toy.wrong_odd_antifield_sign_transformation_mismatch.term_count > 0,
            e70_b_receipt.base_nilpotency_component_count == 27,
            nilpotency_maximum == 0,
            e70_c_receipt.bulk_term_count == 6,
            type_naturality_maximum == 0,
            e70_c_receipt.boundary_flux_retained,
            not e70_c_receipt.full_coordinate_jet_action_variation_computed,
            not any(unsupported),
        )
    )
    return M1BVMasterAdmissionReceipt(
        contract_sha256=contract.contract_sha256,
        primary_source=contract.primary_source,
        primary_source_url=contract.primary_source_url,
        secondary_source=contract.secondary_source,
        secondary_source_url=contract.secondary_source_url,
        source_items=contract.source_items,
        source_relation=contract.source_relation,
        upstream_hashes=contract.upstream_hashes,
        upstream_contracts_verified=True,
        antibracket_convention=contract.antibracket_convention,
        antifield_rule=contract.antifield_rule,
        base_field_count=len(fields),
        antifield_count=len(antifields),
        base_even_count=sum(spec.parity == 0 for spec in fields),
        base_odd_count=sum(spec.parity == 1 for spec in fields),
        antifield_even_count=sum(spec.parity == 0 for spec in antifields),
        antifield_odd_count=sum(spec.parity == 1 for spec in antifields),
        base_name_coverage_mismatch_count=base_name_mismatch,
        antifield_name_coverage_mismatch_count=antifield_name_mismatch,
        antifield_rule_mismatch_count=antifield_rule_mismatch,
        antifield_action_term_type_audit_count=len(fields),
        antifield_action_term_type_mismatch_count=term_type_mismatch,
        omitted_field_antifield_pair_rejected=omitted_pair_rejected,
        wrong_antifield_parity_rejected=wrong_parity_rejected,
        finite_toy_canonical_pair_count=len(toy.pairs),
        finite_toy_field_star_calibration_residual_term_count=(
            toy.canonical_field_star_residual.term_count
        ),
        finite_toy_star_field_calibration_residual_term_count=(
            toy.canonical_star_field_residual.term_count
        ),
        finite_toy_transformation_mismatch_term_count=(
            toy.transformation_mismatch.term_count
        ),
        finite_toy_master_residual_term_count=toy.master_residual.term_count,
        bad_action_master_residual_term_count=(
            toy.bad_action_master_residual.term_count
        ),
        broken_doublet_master_residual_term_count=(
            toy.broken_doublet_master_residual.term_count
        ),
        wrong_antibracket_sign_residual_term_count=(
            toy.wrong_antibracket_sign_residual.term_count
        ),
        wrong_odd_antifield_sign_transformation_mismatch_term_count=(
            toy.wrong_odd_antifield_sign_transformation_mismatch.term_count
        ),
        upstream_base_nilpotency_component_count=(
            e70_b_receipt.base_nilpotency_component_count
        ),
        upstream_base_nilpotency_maximum_residual_term_count=(
            nilpotency_maximum
        ),
        upstream_bulk_type_naturality_term_count=(
            e70_c_receipt.bulk_term_count
        ),
        upstream_bulk_type_naturality_maximum_residual=(
            type_naturality_maximum
        ),
        upstream_boundary_flux_retained=e70_c_receipt.boundary_flux_retained,
        upstream_full_coordinate_jet_action_variation_computed=(
            e70_c_receipt.full_coordinate_jet_action_variation_computed
        ),
        formal_afn0_input_status=(
            'six_bulk_type_naturality_residuals_zero_boundary_flux_retained'
        ),
        formal_afn1_input_status='twenty_seven_base_nilpotency_residuals_zero',
        formal_master_residual_input_count=33,
        antifield_ledger_constructed=contract.antifield_ledger_constructed,
        finite_canonical_antibracket_calibrated=(
            contract.finite_canonical_antibracket_calibrated
        ),
        finite_toy_classical_master_equation_computed=(
            contract.finite_toy_classical_master_equation_computed
        ),
        formal_m1_master_residual_decomposition_admitted=(
            contract.formal_m1_master_residual_decomposition_admitted
        ),
        full_m1_antifield_functional_constructed=(
            contract.full_m1_antifield_functional_constructed
        ),
        jet_antifield_variational_calculus_constructed=(
            contract.jet_antifield_variational_calculus_constructed
        ),
        local_functional_boundary_quotient_constructed=(
            contract.local_functional_boundary_quotient_constructed
        ),
        boundary_completion_proved=contract.boundary_completion_proved,
        full_m1_classical_master_equation_computed=(
            contract.full_m1_classical_master_equation_computed
        ),
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
        declared_m1_bv_master_admission_gate_passed=passed,
    )
