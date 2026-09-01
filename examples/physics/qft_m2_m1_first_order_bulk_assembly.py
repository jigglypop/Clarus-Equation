'''Normalized first-order classical bulk assembly for the full M1 field content.

The gate combines the Palatini curvature sector, the metric-density
compatibility constraint, chi plus four reference scalars, and the
nonminimal BRST catalog.  Seven unit densities are checked separately before
a distinct-prime mixed fixture is used.  The primes are algebraic sentinels,
not phenomenological parameter values.

No global boundary action, full set of Euler equations, BV functional, CME,
QME, physical Hilbert space, or quantum M2 is claimed.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
from typing import Mapping

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
    polynomial_sum,
)
from examples.physics.qft_m2_m1_4d_densitized_scalar_bv import (
    DensitizedBVVariableSpec,
    SCALAR_LABELS,
)
from examples.physics.qft_m2_m1_classical_action_gaugefixing import (
    CONTRACT_SHA256 as E70_C_HASH,
    evaluate_m1_classical_action_gaugefixing_gate,
    m1_classical_action_gaugefixing_contract,
    validate_contract as validate_e70_c_contract,
)
from examples.physics.qft_m2_m1_metric_density_compatibility_bv import (
    compatibility_field_specs,
    metric_density_compatibility_bv_model,
)
from examples.physics.qft_m2_m1_palatini_connection_eom import (
    CONTRACT_SHA256 as E70_I_HASH,
    CONNECTION_KEYS,
    canonical_connection_euler_component,
    evaluate_m1_palatini_connection_eom_gate,
    m1_palatini_connection_eom_contract,
    validate_contract as validate_e70_i_contract,
)
from examples.physics.qft_m2_m1_palatini_curvature_brst import (
    DIMENSION,
    MAXIMUM_TOTAL_JET_ORDER,
    PalatiniJetOrderExceeded,
    _gamma_name,
    _h_name,
    palatini_curvature_brst_model,
)


PRIMARY_SOURCE = 'hep-th/0609219'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0609219'
DENSITY_BRST_SOURCE = 'arXiv:2206.00780v2'
DENSITY_BRST_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
FIRST_ORDER_LOOP_CAVEAT_SOURCE = 'arXiv:1706.02622v7'
FIRST_ORDER_LOOP_CAVEAT_SOURCE_URL = 'https://arxiv.org/html/1706.02622v7'
M1_ACTION_EDITION = (
    'paper/6_최신_연구/37_QFT_M0_M2_공변_기준장과_고전_제약.md Eq. (37.1)'
)
SOURCE_ITEMS = (
    'hep-th/0609219: d>2 independent metric/affine first-order Einstein--Hilbert canonical background',
    'arXiv:2206.00780v2: diffeomorphism BRST and density-weight structure, not the literal M1 Palatini model',
    'arXiv:1706.02622v7: first-order gravity plus scalar loop comparison, with Levi--Civita background caveat',
    'M1 Eq. (37.1): internal canonical six-structure action edition used for this assembly',
)
SOURCE_BOUNDARY = (
    'the six M1 structures and determinant multiplier are a convention-adapted '
    'assembly: minimally coupled scalars use h and rho but do not directly '
    'couple to Gamma; this must not be generalized to arbitrary metric-affine '
    'matter or promoted to global first/second-order or quantum equivalence'
)
NORMALIZATION = (
    'all fields, coordinates, derivatives, and coefficients are normalized '
    'to dimensionless exact polynomial coordinates; distinct primes mark '
    'independent unit densities and are not physical parameter estimates'
)
PHYSICAL_PARAMETER_DICTIONARY = (
    'a_R=M_P^2/2, a_Lambda=M_P^2 Lambda, a_chiK=1, a_m=m^2, '
    'a_4=lambda, a_X=mu_X^2 before reference-scale normalization; a_C is '
    'an auxiliary compatibility multiplier normalization'
)
CLAIM_CEILING = (
    'normalized bounded four-dimensional first-order classical M1 bulk '
    'assembly with all six action structures, determinant multiplier, merged '
    '69-field BRST catalog, seven separate density identities, retained '
    'current, Gamma-Euler factorization, and ell constraint equation; no '
    'phenomenological coefficient inference, all-field Euler system, '
    'Palatini/GHY or global boundary principle, full M1 BV/CME, measure, QME, '
    'ST, Hilbert, HDA M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-C', E70_C_HASH), ('E70-I', E70_I_HASH))
CONTRACT_SHA256 = (
    '89837f5e904dd2cd37e8e67c2b71c06b940490d3446b6805c5a1081fe655cce1'
)


class FirstOrderBulkJetOrderExceeded(ValueError):
    '''Raised instead of silently truncating a first-order M1 jet.'''


MultiIndex = tuple[int, int, int, int]
ZERO_MULTIINDEX: MultiIndex = (0, 0, 0, 0)


def multiindices_up_to(maximum_order: int) -> tuple[MultiIndex, ...]:
    values = (
        index
        for index in product(range(maximum_order + 1), repeat=DIMENSION)
        if sum(index) <= maximum_order
    )
    return tuple(sorted(values, key=lambda item: (sum(item), item)))


MULTIINDICES = multiindices_up_to(MAXIMUM_TOTAL_JET_ORDER)


def first_order_bulk_field_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    gamma_specs = tuple(
        DensitizedBVVariableSpec(
            _gamma_name(upper, lower_left, lower_right),
            'torsion-free affine connection component',
            0,
            0,
            0,
            0,
        )
        for upper in range(DIMENSION)
        for lower_left in range(DIMENSION)
        for lower_right in range(lower_left, DIMENSION)
    )
    return compatibility_field_specs() + gamma_specs


FIELD_SPECS = first_order_bulk_field_specs()
SPEC_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}


def jet_name(base_name: str, multiindex: MultiIndex) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown first-order bulk variable {base_name}')
    if len(multiindex) != DIMENSION or min(multiindex) < 0:
        raise ValueError('multi-index must contain four nonnegative entries')
    if sum(multiindex) > MAXIMUM_TOTAL_JET_ORDER:
        raise FirstOrderBulkJetOrderExceeded(
            f'multi-index {multiindex} exceeds order {MAXIMUM_TOTAL_JET_ORDER}'
        )
    return f'{base_name}__{multiindex[0]}_{multiindex[1]}_{multiindex[2]}_{multiindex[3]}'


JET_LOOKUP = {
    jet_name(spec.name, multiindex): (spec, multiindex)
    for spec in FIELD_SPECS
    for multiindex in MULTIINDICES
}


def unit_multiindex(direction: int) -> MultiIndex:
    if direction < 0 or direction >= DIMENSION:
        raise ValueError('direction must be in 0..3')
    return tuple(1 if index == direction else 0 for index in range(DIMENSION))  # type: ignore[return-value]


def add_multiindex(left: MultiIndex, right: MultiIndex) -> MultiIndex:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def generator(
    base_name: str,
    multiindex: MultiIndex = ZERO_MULTIINDEX,
) -> SparseSuperPolynomial:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown first-order bulk variable {base_name}')
    spec = SPEC_BY_NAME[base_name]
    return SparseSuperPolynomial.generator(
        jet_name(base_name, multiindex),
        odd=bool(spec.parity),
    )


def horizontal_derivative(
    polynomial: SparseSuperPolynomial,
    direction: int,
) -> SparseSuperPolynomial:
    increment = unit_multiindex(direction)
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        for index, name in enumerate(even_names):
            spec, multiindex = JET_LOOKUP[name]
            next_index = add_multiindex(multiindex, increment)
            if sum(next_index) > MAXIMUM_TOTAL_JET_ORDER:
                raise FirstOrderBulkJetOrderExceeded(
                    f'D_{direction} requires {spec.name} jet {next_index}'
                )
            remaining = even_names[:index] + even_names[index + 1 :]
            result += coefficient * (
                SparseSuperPolynomial.monomial(even=remaining, odd=odd_names)
                * generator(spec.name, next_index)
            )
        for index, name in enumerate(odd_names):
            spec, multiindex = JET_LOOKUP[name]
            next_index = add_multiindex(multiindex, increment)
            if sum(next_index) > MAXIMUM_TOTAL_JET_ORDER:
                raise FirstOrderBulkJetOrderExceeded(
                    f'D_{direction} requires {spec.name} jet {next_index}'
                )
            prefix = SparseSuperPolynomial.monomial(
                even=even_names,
                odd=odd_names[:index],
            )
            suffix = SparseSuperPolynomial.monomial(odd=odd_names[index + 1 :])
            result += coefficient * prefix * generator(spec.name, next_index) * suffix
    return result


def multi_total_derivative(
    polynomial: SparseSuperPolynomial,
    multiindex: MultiIndex,
) -> SparseSuperPolynomial:
    result = polynomial
    for direction, count in enumerate(multiindex):
        for _ in range(count):
            result = horizontal_derivative(result, direction)
    return result


def divergence(currents: tuple[SparseSuperPolynomial, ...]) -> SparseSuperPolynomial:
    return polynomial_sum(
        horizontal_derivative(current, direction)
        for direction, current in enumerate(currents)
    )


def lift_external_polynomial(
    polynomial: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    names = {
        name
        for even_names, odd_names in polynomial.terms
        for name in even_names + odd_names
    }
    unknown = names.difference(JET_LOOKUP)
    if unknown:
        raise ValueError(f'external polynomial contains unknown jets {sorted(unknown)}')
    return SparseSuperPolynomial(polynomial.terms)


def apply_bulk_brst(
    polynomial: SparseSuperPolynomial,
    transformations: Mapping[str, SparseSuperPolynomial],
) -> SparseSuperPolynomial:
    present_names = {
        name
        for even_names, odd_names in polynomial.terms
        for name in even_names + odd_names
    }
    images: dict[str, SparseSuperPolynomial] = {}
    for name in present_names:
        spec, multiindex = JET_LOOKUP[name]
        images[name] = multi_total_derivative(
            transformations[spec.name],
            multiindex,
        )
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        even_shell = SparseSuperPolynomial.monomial(even=even_names)
        odd_shell = SparseSuperPolynomial.monomial(odd=odd_names)
        for index, name in enumerate(even_names):
            remaining = even_names[:index] + even_names[index + 1 :]
            result += coefficient * (
                SparseSuperPolynomial.monomial(even=remaining)
                * images[name]
                * odd_shell
            )
        for index, name in enumerate(odd_names):
            remaining = odd_names[:index] + odd_names[index + 1 :]
            sign = -1 if index % 2 else 1
            result += coefficient * sign * (
                even_shell
                * images[name]
                * SparseSuperPolynomial.monomial(odd=remaining)
            )
    return result


def even_jet_partial_derivative(
    polynomial: SparseSuperPolynomial,
    base_name: str,
    multiindex: MultiIndex,
) -> SparseSuperPolynomial:
    target = jet_name(base_name, multiindex)
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        multiplicity = even_names.count(target)
        if not multiplicity:
            continue
        position = even_names.index(target)
        remaining = even_names[:position] + even_names[position + 1 :]
        result += SparseSuperPolynomial.monomial(
            even=remaining,
            odd=odd_names,
            coefficient=coefficient * multiplicity,
        )
    return result


def even_euler_derivative(
    density: SparseSuperPolynomial,
    base_name: str,
) -> SparseSuperPolynomial:
    return polynomial_sum(
        (-1 if sum(multiindex) % 2 else 1)
        * multi_total_derivative(partial, multiindex)
        for multiindex in MULTIINDICES
        if not (
            partial := even_jet_partial_derivative(
                density,
                base_name,
                multiindex,
            )
        ).is_zero
    )


@dataclass(frozen=True)
class NormalizedCoefficientFixture:
    gravity: Fraction
    cosmological: Fraction
    chi_kinetic: Fraction
    chi_mass: Fraction
    chi_quartic: Fraction
    reference_kinetic: Fraction
    compatibility: Fraction


COEFFICIENT_FIXTURE = NormalizedCoefficientFixture(
    Fraction(2),
    Fraction(3),
    Fraction(5),
    Fraction(7),
    Fraction(11),
    Fraction(13),
    Fraction(17),
)


def validate_coefficient_fixture(fixture: NormalizedCoefficientFixture) -> None:
    values = tuple(fixture.__dict__.values())
    if fixture != COEFFICIENT_FIXTURE:
        raise ValueError('normalized coefficient fixture changed')
    if any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError('coefficient sentinels must be positive and distinct')


@dataclass(frozen=True)
class FirstOrderBulkAssemblyModel:
    unit_densities: tuple[tuple[str, SparseSuperPolynomial], ...]
    classical_density: SparseSuperPolynomial
    boundary_current: tuple[SparseSuperPolynomial, ...]
    transformations: Mapping[str, SparseSuperPolynomial]
    shared_transformation_mismatches: tuple[SparseSuperPolynomial, ...]
    compatibility_constraint: SparseSuperPolynomial


def first_order_bulk_assembly_model(
    fixture: NormalizedCoefficientFixture = COEFFICIENT_FIXTURE,
) -> FirstOrderBulkAssemblyModel:
    validate_coefficient_fixture(fixture)
    compatibility_model = metric_density_compatibility_bv_model()
    palatini_model = palatini_curvature_brst_model()

    transformations = {
        name: lift_external_polynomial(image)
        for name, image in compatibility_model.transformations.items()
    }
    shared_names = tuple(
        name
        for name in palatini_model.transformations
        if name in transformations
    )
    shared_mismatches = tuple(
        transformations[name]
        - lift_external_polynomial(palatini_model.transformations[name])
        for name in shared_names
    )
    transformations.update(
        {
            name: lift_external_polynomial(image)
            for name, image in palatini_model.transformations.items()
            if name.startswith('Gamma')
        }
    )

    chi = generator('phi_chi')
    palatini_density = lift_external_polynomial(palatini_model.palatini_density)
    cosmological_density = -generator('rho')
    chi_kinetic_density = -Fraction(1, 2) * polynomial_sum(
        generator(_h_name(mu, nu))
        * generator('phi_chi', unit_multiindex(mu))
        * generator('phi_chi', unit_multiindex(nu))
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    chi_mass_density = -Fraction(1, 2) * generator('rho') * chi * chi
    chi_quartic_density = (
        -Fraction(1, 24) * generator('rho') * chi * chi * chi * chi
    )
    reference_density = -Fraction(1, 2) * polynomial_sum(
        generator(_h_name(mu, nu))
        * generator(f'phi_{label}', unit_multiindex(mu))
        * generator(f'phi_{label}', unit_multiindex(nu))
        for label in SCALAR_LABELS
        if label != 'chi'
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    compatibility_constraint = lift_external_polynomial(
        compatibility_model.compatibility_constraint
    )
    compatibility_density = generator('ell') * compatibility_constraint
    units = (
        ('einstein_hilbert_palatini', palatini_density),
        ('cosmological_constant', cosmological_density),
        ('chi_kinetic', chi_kinetic_density),
        ('chi_mass', chi_mass_density),
        ('chi_quartic', chi_quartic_density),
        ('reference_kinetic', reference_density),
        ('metric_density_compatibility', compatibility_density),
    )
    coefficients = tuple(fixture.__dict__.values())
    classical_density = polynomial_sum(
        coefficient * density
        for coefficient, (_, density) in zip(coefficients, units, strict=True)
    )
    return FirstOrderBulkAssemblyModel(
        unit_densities=units,
        classical_density=classical_density,
        boundary_current=tuple(
            generator(f'c{mu}') * classical_density
            for mu in range(DIMENSION)
        ),
        transformations=transformations,
        shared_transformation_mismatches=shared_mismatches,
        compatibility_constraint=compatibility_constraint,
    )


@dataclass(frozen=True)
class M1FirstOrderBulkAssemblyContract:
    primary_source: str
    primary_source_url: str
    density_brst_source: str
    density_brst_source_url: str
    first_order_loop_caveat_source: str
    first_order_loop_caveat_source_url: str
    m1_action_edition: str
    source_items: tuple[str, ...]
    source_boundary: str
    normalization: str
    physical_parameter_dictionary: str
    dimension: int
    maximum_total_jet_order: int
    field_specs: tuple[DensitizedBVVariableSpec, ...]
    coefficient_fixture: NormalizedCoefficientFixture
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    m1_six_bulk_structures_assembled: bool
    compatibility_multiplier_assembled: bool
    shared_brst_maps_matched: bool
    all_69_base_nilpotency_computed: bool
    seven_unit_density_identities_computed: bool
    mixed_density_total_divergence_computed: bool
    nonzero_boundary_current_retained: bool
    gamma_euler_factorization_computed: bool
    ell_constraint_euler_computed: bool
    live_negative_controls_computed: bool
    physical_coefficients_inferred_from_fixture: bool
    arbitrary_metric_affine_matter_generalized: bool
    all_field_euler_equations_computed: bool
    palatini_boundary_term_constructed: bool
    ghy_boundary_term_used: bool
    global_first_second_order_equivalence_proved: bool
    full_m1_bv_functional_constructed: bool
    classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


_CONTRACT_FLAG_NAMES = (
    'm1_six_bulk_structures_assembled',
    'compatibility_multiplier_assembled',
    'shared_brst_maps_matched',
    'all_69_base_nilpotency_computed',
    'seven_unit_density_identities_computed',
    'mixed_density_total_divergence_computed',
    'nonzero_boundary_current_retained',
    'gamma_euler_factorization_computed',
    'ell_constraint_euler_computed',
    'live_negative_controls_computed',
    'physical_coefficients_inferred_from_fixture',
    'arbitrary_metric_affine_matter_generalized',
    'all_field_euler_equations_computed',
    'palatini_boundary_term_constructed',
    'ghy_boundary_term_used',
    'global_first_second_order_equivalence_proved',
    'full_m1_bv_functional_constructed',
    'classical_master_equation_computed',
    'functional_measure_computed',
    'quantum_master_equation_computed',
    'continuum_loop_st_computed',
    'positive_physical_hilbert_proved',
    'quantum_hda_m2_proved',
    'm3_relational_observables_unlocked',
)


def m1_first_order_bulk_assembly_contract() -> M1FirstOrderBulkAssemblyContract:
    return M1FirstOrderBulkAssemblyContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        density_brst_source=DENSITY_BRST_SOURCE,
        density_brst_source_url=DENSITY_BRST_SOURCE_URL,
        first_order_loop_caveat_source=FIRST_ORDER_LOOP_CAVEAT_SOURCE,
        first_order_loop_caveat_source_url=FIRST_ORDER_LOOP_CAVEAT_SOURCE_URL,
        m1_action_edition=M1_ACTION_EDITION,
        source_items=SOURCE_ITEMS,
        source_boundary=SOURCE_BOUNDARY,
        normalization=NORMALIZATION,
        physical_parameter_dictionary=PHYSICAL_PARAMETER_DICTIONARY,
        dimension=DIMENSION,
        maximum_total_jet_order=MAXIMUM_TOTAL_JET_ORDER,
        field_specs=FIELD_SPECS,
        coefficient_fixture=COEFFICIENT_FIXTURE,
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        m1_six_bulk_structures_assembled=True,
        compatibility_multiplier_assembled=True,
        shared_brst_maps_matched=True,
        all_69_base_nilpotency_computed=True,
        seven_unit_density_identities_computed=True,
        mixed_density_total_divergence_computed=True,
        nonzero_boundary_current_retained=True,
        gamma_euler_factorization_computed=True,
        ell_constraint_euler_computed=True,
        live_negative_controls_computed=True,
        physical_coefficients_inferred_from_fixture=False,
        arbitrary_metric_affine_matter_generalized=False,
        all_field_euler_equations_computed=False,
        palatini_boundary_term_constructed=False,
        ghy_boundary_term_used=False,
        global_first_second_order_equivalence_proved=False,
        full_m1_bv_functional_constructed=False,
        classical_master_equation_computed=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_normalized_first_order_full_m1_classical_bulk_assembly_'
            'not_boundary_bv_or_quantum_m2'
        ),
    )


def _serialize_field_spec(spec: DensitizedBVVariableSpec) -> str:
    return ':'.join(
        (
            spec.name,
            spec.role,
            str(spec.parity),
            str(spec.ghost_number),
            str(spec.antifield_number),
            str(spec.density_weight),
        )
    )


def _serialize_coefficients(fixture: NormalizedCoefficientFixture) -> str:
    return ','.join(
        f'{name}:{value}' for name, value in fixture.__dict__.items()
    )


def canonical_contract_payload(contract: M1FirstOrderBulkAssemblyContract) -> str:
    comma = chr(44)
    flags = comma.join(
        f'{name}:{getattr(contract, name)}' for name in _CONTRACT_FLAG_NAMES
    )
    return '|'.join(
        (
            f'primary={contract.primary_source}',
            f'primary_url={contract.primary_source_url}',
            f'density_brst={contract.density_brst_source}',
            f'density_brst_url={contract.density_brst_source_url}',
            f'loop_caveat={contract.first_order_loop_caveat_source}',
            f'loop_caveat_url={contract.first_order_loop_caveat_source_url}',
            f'm1_action={contract.m1_action_edition}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_boundary={contract.source_boundary}',
            f'normalization={contract.normalization}',
            f'physical_dictionary={contract.physical_parameter_dictionary}',
            f'dimension={contract.dimension}',
            f'max_total_jet={contract.maximum_total_jet_order}',
            f'fields={comma.join(_serialize_field_spec(x) for x in contract.field_specs)}',
            f'coefficients={_serialize_coefficients(contract.coefficient_fixture)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(contract: M1FirstOrderBulkAssemblyContract) -> str:
    return hashlib.sha256(canonical_contract_payload(contract).encode('utf-8')).hexdigest()


def validate_contract(contract: M1FirstOrderBulkAssemblyContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.density_brst_source == DENSITY_BRST_SOURCE,
        contract.density_brst_source_url == DENSITY_BRST_SOURCE_URL,
        contract.first_order_loop_caveat_source == FIRST_ORDER_LOOP_CAVEAT_SOURCE,
        contract.first_order_loop_caveat_source_url == FIRST_ORDER_LOOP_CAVEAT_SOURCE_URL,
        contract.m1_action_edition == M1_ACTION_EDITION,
        contract.source_items == SOURCE_ITEMS,
        contract.source_boundary == SOURCE_BOUNDARY,
        contract.normalization == NORMALIZATION,
        contract.physical_parameter_dictionary == PHYSICAL_PARAMETER_DICTIONARY,
        contract.dimension == DIMENSION,
        contract.maximum_total_jet_order == MAXIMUM_TOTAL_JET_ORDER,
        contract.field_specs == FIELD_SPECS,
        contract.coefficient_fixture == COEFFICIENT_FIXTURE,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == (
            'exact_normalized_first_order_full_m1_classical_bulk_assembly_'
            'not_boundary_bv_or_quantum_m2'
        ),
    )
    if not all(frozen):
        raise ValueError('first-order M1 bulk source, basis, or status lock changed')
    validate_coefficient_fixture(contract.coefficient_fixture)
    if len(contract.field_specs) != 69:
        raise ValueError('first-order M1 bulk catalog requires 69 base fields')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('first-order M1 bulk contract hash mismatch')
    if not all(getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[:10]):
        raise ValueError('required first-order M1 bulk flag disabled')
    if any(getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]):
        raise ValueError('unsupported first-order M1 bulk claim promoted')


@dataclass(frozen=True)
class UnitDensityReceipt:
    name: str
    density_term_count: int
    variation_term_count: int
    current_term_count: int
    current_divergence_term_count: int
    identity_mismatch_term_count: int


@dataclass(frozen=True)
class M1FirstOrderBulkAssemblyReceipt:
    contract_sha256: str
    source_boundary: str
    normalization: str
    physical_parameter_dictionary: str
    coefficient_fixture: NormalizedCoefficientFixture
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_c_verified: bool
    upstream_e70_i_verified: bool
    base_field_count: int
    even_base_field_count: int
    odd_base_field_count: int
    multiindex_count: int
    bounded_jet_generator_count: int
    bounded_even_jet_generator_count: int
    bounded_odd_jet_generator_count: int
    transformation_component_count: int
    shared_transformation_component_count: int
    shared_transformation_nonzero_mismatch_count: int
    base_nilpotency_component_count: int
    base_nilpotency_nonzero_component_count: int
    base_nilpotency_maximum_residual_term_count: int
    unit_density_receipts: tuple[UnitDensityReceipt, ...]
    classical_density_term_count: int
    classical_variation_term_count: int
    boundary_current_term_count: int
    boundary_current_divergence_term_count: int
    classical_identity_mismatch_term_count: int
    gamma_euler_component_count: int
    gamma_euler_total_term_count: int
    gamma_euler_nonzero_factorization_mismatch_count: int
    gamma_euler_maximum_factorization_mismatch_term_count: int
    ell_euler_term_count: int
    ell_constraint_mismatch_term_count: int
    wrong_gravity_factor_nonzero_gamma_component_count: int
    wrong_gravity_factor_total_mismatch_term_count: int
    wrong_gravity_factor_maximum_mismatch_term_count: int
    direct_gamma_matter_contamination_nonzero_component_count: int
    direct_gamma_matter_contamination_total_term_count: int
    omitted_ell_map_compatibility_mismatch_term_count: int
    wrong_cosmological_scalar_mismatch_term_count: int
    perturbed_shared_map_mismatch_term_count: int
    omitted_boundary_current_residual_term_count: int
    invalid_coefficient_fixture_rejected: bool
    terminal_jet_derivative_rejected: bool
    m1_six_bulk_structures_assembled: bool
    compatibility_multiplier_assembled: bool
    shared_brst_maps_matched: bool
    all_69_base_nilpotency_computed: bool
    seven_unit_density_identities_computed: bool
    mixed_density_total_divergence_computed: bool
    nonzero_boundary_current_retained: bool
    gamma_euler_factorization_computed: bool
    ell_constraint_euler_computed: bool
    live_negative_controls_computed: bool
    physical_coefficients_inferred_from_fixture: bool
    arbitrary_metric_affine_matter_generalized: bool
    all_field_euler_equations_computed: bool
    palatini_boundary_term_constructed: bool
    ghy_boundary_term_used: bool
    global_first_second_order_equivalence_proved: bool
    full_m1_bv_functional_constructed: bool
    classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_first_order_bulk_assembly_gate_passed: bool


@lru_cache(maxsize=1)
def evaluate_m1_first_order_bulk_assembly_gate(
) -> M1FirstOrderBulkAssemblyReceipt:
    contract = m1_first_order_bulk_assembly_contract()
    validate_contract(contract)
    e70_c_contract = m1_classical_action_gaugefixing_contract()
    validate_e70_c_contract(e70_c_contract)
    e70_c_receipt = evaluate_m1_classical_action_gaugefixing_gate()
    upstream_e70_c_verified = (
        e70_c_receipt.declared_m1_classical_action_gaugefixing_gate_passed
    )
    e70_i_contract = m1_palatini_connection_eom_contract()
    validate_e70_i_contract(e70_i_contract)
    e70_i_receipt = evaluate_m1_palatini_connection_eom_gate()
    upstream_e70_i_verified = (
        e70_i_receipt.declared_m1_palatini_connection_eom_gate_passed
    )

    model = first_order_bulk_assembly_model()
    unit_receipts: list[UnitDensityReceipt] = []
    for name, density in model.unit_densities:
        variation = apply_bulk_brst(density, model.transformations)
        current = tuple(
            generator(f'c{mu}') * density for mu in range(DIMENSION)
        )
        current_divergence = divergence(current)
        unit_receipts.append(
            UnitDensityReceipt(
                name=name,
                density_term_count=density.term_count,
                variation_term_count=variation.term_count,
                current_term_count=sum(value.term_count for value in current),
                current_divergence_term_count=current_divergence.term_count,
                identity_mismatch_term_count=(
                    variation - current_divergence
                ).term_count,
            )
        )

    classical_variation = apply_bulk_brst(
        model.classical_density,
        model.transformations,
    )
    boundary_divergence = divergence(model.boundary_current)
    classical_mismatch = classical_variation - boundary_divergence
    nilpotency_residuals = tuple(
        apply_bulk_brst(image, model.transformations)
        for image in model.transformations.values()
    )

    gamma_euler_values = tuple(
        even_euler_derivative(
            model.classical_density,
            _gamma_name(*key),
        )
        for key in CONNECTION_KEYS
    )
    gamma_euler_mismatches = tuple(
        direct
        - COEFFICIENT_FIXTURE.gravity
        * lift_external_polynomial(canonical_connection_euler_component(key))
        for key, direct in zip(
            CONNECTION_KEYS,
            gamma_euler_values,
            strict=True,
        )
    )
    ell_euler = even_euler_derivative(model.classical_density, 'ell')
    ell_mismatch = (
        ell_euler
        - COEFFICIENT_FIXTURE.compatibility
        * model.compatibility_constraint
    )

    wrong_gravity_factor = tuple(
        direct
        - Fraction(3)
        * lift_external_polynomial(canonical_connection_euler_component(key))
        for key, direct in zip(
            CONNECTION_KEYS,
            gamma_euler_values,
            strict=True,
        )
    )
    contaminated_density = (
        model.classical_density
        + generator('Gamma0_00')
        * generator('phi_chi')
        * generator('phi_chi')
    )
    contamination_residuals = tuple(
        even_euler_derivative(contaminated_density, _gamma_name(*key))
        - COEFFICIENT_FIXTURE.gravity
        * lift_external_polynomial(canonical_connection_euler_component(key))
        for key in CONNECTION_KEYS
    )

    omitted_ell_transformations = dict(model.transformations)
    omitted_ell_transformations['ell'] = SparseSuperPolynomial.zero()
    compatibility_density = model.unit_densities[-1][1]
    omitted_ell_variation = apply_bulk_brst(
        compatibility_density,
        omitted_ell_transformations,
    )
    compatibility_current = tuple(
        generator(f'c{mu}') * compatibility_density
        for mu in range(DIMENSION)
    )
    omitted_ell_mismatch = (
        omitted_ell_variation - divergence(compatibility_current)
    )

    wrong_cosmological_density = -SparseSuperPolynomial.scalar(1)
    wrong_cosmological_mismatch = (
        apply_bulk_brst(
            wrong_cosmological_density,
            model.transformations,
        )
        - divergence(
            tuple(
                generator(f'c{mu}') * wrong_cosmological_density
                for mu in range(DIMENSION)
            )
        )
    )
    perturbed_shared_map = (
        model.shared_transformation_mismatches[0]
        + SparseSuperPolynomial.scalar(1)
    )

    invalid_fixture_rejected = False
    try:
        first_order_bulk_assembly_model(
            NormalizedCoefficientFixture(
                Fraction(0),
                Fraction(3),
                Fraction(5),
                Fraction(7),
                Fraction(11),
                Fraction(13),
                Fraction(17),
            )
        )
    except ValueError:
        invalid_fixture_rejected = True
    terminal_rejected = False
    try:
        horizontal_derivative(
            generator(
                'Gamma0_00',
                (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0),
            ),
            0,
        )
    except FirstOrderBulkJetOrderExceeded:
        terminal_rejected = True

    even_base_count = sum(spec.parity == 0 for spec in FIELD_SPECS)
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]
    )
    expected_unit_counts = (
        ('einstein_hilbert_palatini', 276, 4032, 1104, 4032, 0),
        ('cosmological_constant', 1, 8, 4, 8, 0),
        ('chi_kinetic', 10, 144, 40, 144, 0),
        ('chi_mass', 1, 12, 4, 12, 0),
        ('chi_quartic', 1, 12, 4, 12, 0),
        ('reference_kinetic', 40, 576, 160, 576, 0),
        ('metric_density_compatibility', 18, 372, 72, 372, 0),
    )
    observed_unit_counts = tuple(
        (
            value.name,
            value.density_term_count,
            value.variation_term_count,
            value.current_term_count,
            value.current_divergence_term_count,
            value.identity_mismatch_term_count,
        )
        for value in unit_receipts
    )
    passed = all(
        (
            upstream_e70_c_verified,
            upstream_e70_i_verified,
            len(FIELD_SPECS) == 69,
            even_base_count == 61,
            len(FIELD_SPECS) - even_base_count == 8,
            len(MULTIINDICES) == 35,
            len(JET_LOOKUP) == 2415,
            even_base_count * len(MULTIINDICES) == 2135,
            (len(FIELD_SPECS) - even_base_count) * len(MULTIINDICES)
            == 280,
            len(model.transformations) == 69,
            len(model.shared_transformation_mismatches) == 14,
            max(
                value.term_count
                for value in model.shared_transformation_mismatches
            )
            == 0,
            len(nilpotency_residuals) == 69,
            max(value.term_count for value in nilpotency_residuals) == 0,
            observed_unit_counts == expected_unit_counts,
            model.classical_density.term_count == 347,
            classical_variation.term_count == 5156,
            sum(value.term_count for value in model.boundary_current) == 1388,
            boundary_divergence.term_count == 5156,
            classical_mismatch.is_zero,
            len(gamma_euler_values) == 40,
            sum(value.term_count for value in gamma_euler_values) == 456,
            max(value.term_count for value in gamma_euler_mismatches) == 0,
            ell_euler.term_count == 18,
            ell_mismatch.is_zero,
            sum(not value.is_zero for value in wrong_gravity_factor) == 40,
            sum(value.term_count for value in wrong_gravity_factor) == 456,
            max(value.term_count for value in wrong_gravity_factor) == 17,
            sum(not value.is_zero for value in contamination_residuals) == 1,
            sum(value.term_count for value in contamination_residuals) == 1,
            omitted_ell_mismatch.term_count == 144,
            wrong_cosmological_mismatch.term_count == 4,
            perturbed_shared_map.term_count == 1,
            classical_variation.term_count == 5156,
            invalid_fixture_rejected,
            terminal_rejected,
            not any(unsupported),
        )
    )

    return M1FirstOrderBulkAssemblyReceipt(
        contract_sha256=contract.contract_sha256,
        source_boundary=contract.source_boundary,
        normalization=contract.normalization,
        physical_parameter_dictionary=contract.physical_parameter_dictionary,
        coefficient_fixture=contract.coefficient_fixture,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_c_verified=upstream_e70_c_verified,
        upstream_e70_i_verified=upstream_e70_i_verified,
        base_field_count=len(FIELD_SPECS),
        even_base_field_count=even_base_count,
        odd_base_field_count=len(FIELD_SPECS) - even_base_count,
        multiindex_count=len(MULTIINDICES),
        bounded_jet_generator_count=len(JET_LOOKUP),
        bounded_even_jet_generator_count=even_base_count * len(MULTIINDICES),
        bounded_odd_jet_generator_count=(
            len(JET_LOOKUP) - even_base_count * len(MULTIINDICES)
        ),
        transformation_component_count=len(model.transformations),
        shared_transformation_component_count=(
            len(model.shared_transformation_mismatches)
        ),
        shared_transformation_nonzero_mismatch_count=sum(
            not value.is_zero
            for value in model.shared_transformation_mismatches
        ),
        base_nilpotency_component_count=len(nilpotency_residuals),
        base_nilpotency_nonzero_component_count=sum(
            not value.is_zero for value in nilpotency_residuals
        ),
        base_nilpotency_maximum_residual_term_count=max(
            value.term_count for value in nilpotency_residuals
        ),
        unit_density_receipts=tuple(unit_receipts),
        classical_density_term_count=model.classical_density.term_count,
        classical_variation_term_count=classical_variation.term_count,
        boundary_current_term_count=sum(
            value.term_count for value in model.boundary_current
        ),
        boundary_current_divergence_term_count=boundary_divergence.term_count,
        classical_identity_mismatch_term_count=classical_mismatch.term_count,
        gamma_euler_component_count=len(gamma_euler_values),
        gamma_euler_total_term_count=sum(
            value.term_count for value in gamma_euler_values
        ),
        gamma_euler_nonzero_factorization_mismatch_count=sum(
            not value.is_zero for value in gamma_euler_mismatches
        ),
        gamma_euler_maximum_factorization_mismatch_term_count=max(
            value.term_count for value in gamma_euler_mismatches
        ),
        ell_euler_term_count=ell_euler.term_count,
        ell_constraint_mismatch_term_count=ell_mismatch.term_count,
        wrong_gravity_factor_nonzero_gamma_component_count=sum(
            not value.is_zero for value in wrong_gravity_factor
        ),
        wrong_gravity_factor_total_mismatch_term_count=sum(
            value.term_count for value in wrong_gravity_factor
        ),
        wrong_gravity_factor_maximum_mismatch_term_count=max(
            value.term_count for value in wrong_gravity_factor
        ),
        direct_gamma_matter_contamination_nonzero_component_count=sum(
            not value.is_zero for value in contamination_residuals
        ),
        direct_gamma_matter_contamination_total_term_count=sum(
            value.term_count for value in contamination_residuals
        ),
        omitted_ell_map_compatibility_mismatch_term_count=(
            omitted_ell_mismatch.term_count
        ),
        wrong_cosmological_scalar_mismatch_term_count=(
            wrong_cosmological_mismatch.term_count
        ),
        perturbed_shared_map_mismatch_term_count=perturbed_shared_map.term_count,
        omitted_boundary_current_residual_term_count=(
            classical_variation.term_count
        ),
        invalid_coefficient_fixture_rejected=invalid_fixture_rejected,
        terminal_jet_derivative_rejected=terminal_rejected,
        m1_six_bulk_structures_assembled=(
            contract.m1_six_bulk_structures_assembled
        ),
        compatibility_multiplier_assembled=(
            contract.compatibility_multiplier_assembled
        ),
        shared_brst_maps_matched=contract.shared_brst_maps_matched,
        all_69_base_nilpotency_computed=(
            contract.all_69_base_nilpotency_computed
        ),
        seven_unit_density_identities_computed=(
            contract.seven_unit_density_identities_computed
        ),
        mixed_density_total_divergence_computed=(
            contract.mixed_density_total_divergence_computed
        ),
        nonzero_boundary_current_retained=(
            contract.nonzero_boundary_current_retained
        ),
        gamma_euler_factorization_computed=(
            contract.gamma_euler_factorization_computed
        ),
        ell_constraint_euler_computed=contract.ell_constraint_euler_computed,
        live_negative_controls_computed=contract.live_negative_controls_computed,
        physical_coefficients_inferred_from_fixture=(
            contract.physical_coefficients_inferred_from_fixture
        ),
        arbitrary_metric_affine_matter_generalized=(
            contract.arbitrary_metric_affine_matter_generalized
        ),
        all_field_euler_equations_computed=(
            contract.all_field_euler_equations_computed
        ),
        palatini_boundary_term_constructed=(
            contract.palatini_boundary_term_constructed
        ),
        ghy_boundary_term_used=contract.ghy_boundary_term_used,
        global_first_second_order_equivalence_proved=(
            contract.global_first_second_order_equivalence_proved
        ),
        full_m1_bv_functional_constructed=(
            contract.full_m1_bv_functional_constructed
        ),
        classical_master_equation_computed=(
            contract.classical_master_equation_computed
        ),
        functional_measure_computed=contract.functional_measure_computed,
        quantum_master_equation_computed=contract.quantum_master_equation_computed,
        continuum_loop_st_computed=contract.continuum_loop_st_computed,
        positive_physical_hilbert_proved=contract.positive_physical_hilbert_proved,
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        m3_relational_observables_unlocked=(
            contract.m3_relational_observables_unlocked
        ),
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_m1_first_order_bulk_assembly_gate_passed=passed,
    )
