'''Exact classical density and gauge-fermion gate for the full M1 model.

This gate has two deliberately separate certificates.  First, it checks that
each term in the M1 bulk Lagrangian is a mass-dimension-four scalar and hence
becomes a weight-one density after multiplication by sqrt(-g).  Its BRST
variation is then the explicit horizontal divergence d_mu(c^mu L), with the
boundary flux retained.  This is a type/naturality certificate, not a new
coordinate-jet expansion of R and sqrt(-g).  Second, an abstract coordinate-gauge fermion is
expanded with the exact exterior algebra used by E70-B and compared with an
independently assembled gauge-fixing-plus-ghost density.

The gate does not vary the GHY term, discard an open-boundary flux, introduce
BV antifields, or compute a CME, QME, loop Slavnov--Taylor identity, physical
Hilbert space, or quantum hypersurface-deformation algebra.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from typing import Mapping

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    CONTRACT_SHA256 as E70_B_HASH,
    MATTER_SCALAR_LABELS,
    M1_FIELD_CONTENT,
    REFERENCE_SCALAR_LABELS,
    SparseSuperPolynomial,
    apply_odd_derivation,
    build_jet_differential,
    evaluate_four_scalar_classical_brst_jet_gate,
    four_scalar_classical_brst_jet_contract,
    polynomial_sum,
    validate_contract as validate_e70_b_contract,
)


PRIMARY_SOURCE = 'arXiv:2206.00780v2'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
PRIMARY_SOURCE_DATE = '2025-06-01'
SOURCE_ITEMS = (
    'Lemma 3.5 and Eq. (38): BRST action on a scalar density of weight w',
    'Proposition 3.6 and Eq. (40)--(41): de Donder gauge fixing from P sigma',
    'Remark 2.14: quantum lifting additionally requires anomaly-free ST identities',
)
SOURCE_RELATION = (
    'Prinz v2 supplies the rescaled perturbative density and de Donder '
    'structures; the unscaled covariant full-M1 chi-plus-four-X bulk typing '
    'and abstract F^mu gauge family are convention-adapted, not literal '
    'source transcriptions'
)
BULK_ACTION_DENSITY = (
    'sqrt(-g)[M_P^2/2(R-2 Lambda)-1/2(nabla chi)^2-'
    'm^2 chi^2/2-lambda chi^4/4!-mu_X^2/2 delta_AB '
    '(nabla X^A)(nabla X^B)]'
)
BULK_TERM_SPECS = (
    ('einstein_hilbert', 2, 2, 4, 1),
    ('cosmological_constant', 4, 0, 4, 1),
    ('chi_kinetic', 0, 4, 4, 1),
    ('chi_mass_potential', 2, 2, 4, 1),
    ('chi_quartic_potential', 0, 4, 4, 1),
    ('reference_scalar_kinetic', 2, 2, 4, 1),
)
DIMENSION_CONVENTION = (
    '[partial]=1; [g]=[X]=0; [chi]=1; [M_P^2]=[mu_X^2]=2; '
    '[Lambda]=[m^2]=2; [lambda]=0; every bulk scalar term has mass '
    'dimension four and sqrt(-g) is a dimensionless weight-one density'
)
DENSITY_RULE = (
    's L_w=c^rho partial_rho L_w+w(partial_rho c^rho)L_w; '
    'for w=1 this is partial_rho(c^rho L_1)'
)
GAUGE_FERMION = (
    'Psi=int d4x bar_c_mu(F^mu+alpha B^mu/2); '
    'sPsi=int d4x[B_mu F^mu+alpha B_mu B^mu/2-bar_c_mu sF^mu]'
)
GAUGE_DIMENSION_CONVENTION = (
    '[F]=[sF]=1; [bar_c]=[B]=3; [alpha]=-2; s is dimensionless; '
    'the coordinate-density integrands have mass dimension four'
)
BOUNDARY_CONVENTION = (
    'bulk local-functional identity modulo horizontal divergence; retain '
    'J^rho=c^rho L and its open-boundary flux; no GHY variation or boundary '
    'cancellation is computed'
)
CLAIM_CEILING = (
    'exact full-M1 bulk type/naturality density certificate modulo an explicit '
    'horizontal divergence, plus exact abstract coordinate-gauge-fermion '
    'expansion; no full coordinate-jet action variation, boundary-completed '
    'action theorem, derived M1 gauge condition, BV CME, QME, functional '
    'measure, loop ST, physical Hilbert, HDA, quantum M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-B', E70_B_HASH),)
CONTRACT_SHA256 = (
    '55f0984aed5db1682d6f0e84bdea2520a9a54197fdcc87ab06b679cc763def41'
)


@dataclass(frozen=True)
class M1ClassicalActionGaugeFixingContract:
    primary_source: str
    primary_source_url: str
    primary_source_date: str
    source_items: tuple[str, ...]
    source_relation: str
    source_contains_m1_chi_plus_four_x: bool
    m1_field_content: str
    matter_scalar_labels: tuple[str, ...]
    reference_scalar_labels: tuple[str, ...]
    bulk_action_density: str
    bulk_term_specs: tuple[tuple[str, int, int, int, int], ...]
    dimension_convention: str
    density_rule: str
    gauge_fermion: str
    gauge_dimension_convention: str
    boundary_convention: str
    claim_ceiling: str
    upstream_hashes: tuple[tuple[str, str], ...]
    contract_sha256: str
    bulk_density_type_naturality_certificate_computed: bool
    full_coordinate_jet_action_variation_computed: bool
    boundary_flux_retained: bool
    boundary_discarded: bool
    ghy_boundary_variation_computed: bool
    integrated_action_invariance_proved: bool
    gauge_fixing_fermion_constructed: bool
    gauge_fixing_brst_exactness_computed: bool
    m1_gauge_condition_derived: bool
    bv_antifields_constructed: bool
    classical_master_equation_computed: bool
    quantum_master_equation_computed: bool
    functional_measure_computed: bool
    loop_st_anomaly_cancellation_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_classical_action_gaugefixing_contract(
) -> M1ClassicalActionGaugeFixingContract:
    return M1ClassicalActionGaugeFixingContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        primary_source_date=PRIMARY_SOURCE_DATE,
        source_items=SOURCE_ITEMS,
        source_relation=SOURCE_RELATION,
        source_contains_m1_chi_plus_four_x=False,
        m1_field_content=M1_FIELD_CONTENT,
        matter_scalar_labels=MATTER_SCALAR_LABELS,
        reference_scalar_labels=REFERENCE_SCALAR_LABELS,
        bulk_action_density=BULK_ACTION_DENSITY,
        bulk_term_specs=BULK_TERM_SPECS,
        dimension_convention=DIMENSION_CONVENTION,
        density_rule=DENSITY_RULE,
        gauge_fermion=GAUGE_FERMION,
        gauge_dimension_convention=GAUGE_DIMENSION_CONVENTION,
        boundary_convention=BOUNDARY_CONVENTION,
        claim_ceiling=CLAIM_CEILING,
        upstream_hashes=UPSTREAM_HASHES,
        contract_sha256=CONTRACT_SHA256,
        bulk_density_type_naturality_certificate_computed=True,
        full_coordinate_jet_action_variation_computed=False,
        boundary_flux_retained=True,
        boundary_discarded=False,
        ghy_boundary_variation_computed=False,
        integrated_action_invariance_proved=False,
        gauge_fixing_fermion_constructed=True,
        gauge_fixing_brst_exactness_computed=True,
        m1_gauge_condition_derived=False,
        bv_antifields_constructed=False,
        classical_master_equation_computed=False,
        quantum_master_equation_computed=False,
        functional_measure_computed=False,
        loop_st_anomaly_cancellation_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_bulk_type_naturality_mod_dh_and_abstract_gauge_fermion_only'
        ),
    )


def canonical_contract_payload(
    contract: M1ClassicalActionGaugeFixingContract,
) -> str:
    comma = chr(44)
    terms = comma.join(
        ':'.join(str(item) for item in spec)
        for spec in contract.bulk_term_specs
    )
    upstream = comma.join(
        f'{name}:{value}' for name, value in contract.upstream_hashes
    )
    flag_names = (
        'bulk_density_type_naturality_certificate_computed',
        'full_coordinate_jet_action_variation_computed',
        'boundary_flux_retained',
        'boundary_discarded',
        'ghy_boundary_variation_computed',
        'integrated_action_invariance_proved',
        'gauge_fixing_fermion_constructed',
        'gauge_fixing_brst_exactness_computed',
        'm1_gauge_condition_derived',
        'bv_antifields_constructed',
        'classical_master_equation_computed',
        'quantum_master_equation_computed',
        'functional_measure_computed',
        'loop_st_anomaly_cancellation_computed',
        'positive_physical_hilbert_proved',
        'quantum_hda_m2_proved',
        'm3_relational_observables_unlocked',
    )
    flags = comma.join(
        f'{name}:{getattr(contract, name)}' for name in flag_names
    )
    return '|'.join(
        (
            f'source={contract.primary_source}',
            f'url={contract.primary_source_url}',
            f'date={contract.primary_source_date}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_relation={contract.source_relation}',
            f'source_has_m1={contract.source_contains_m1_chi_plus_four_x}',
            f'field_content={contract.m1_field_content}',
            f'matter={comma.join(contract.matter_scalar_labels)}',
            f'reference={comma.join(contract.reference_scalar_labels)}',
            f'bulk={contract.bulk_action_density}',
            f'terms={terms}',
            f'dimensions={contract.dimension_convention}',
            f'density={contract.density_rule}',
            f'gauge_fermion={contract.gauge_fermion}',
            f'gauge_dimensions={contract.gauge_dimension_convention}',
            f'boundary={contract.boundary_convention}',
            f'ceiling={contract.claim_ceiling}',
            f'upstream={upstream}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(
    contract: M1ClassicalActionGaugeFixingContract,
) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M1ClassicalActionGaugeFixingContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.primary_source_date == PRIMARY_SOURCE_DATE,
        contract.source_items == SOURCE_ITEMS,
        contract.source_relation == SOURCE_RELATION,
        not contract.source_contains_m1_chi_plus_four_x,
        contract.m1_field_content == M1_FIELD_CONTENT,
        contract.matter_scalar_labels == MATTER_SCALAR_LABELS,
        contract.reference_scalar_labels == REFERENCE_SCALAR_LABELS,
        contract.bulk_action_density == BULK_ACTION_DENSITY,
        contract.bulk_term_specs == BULK_TERM_SPECS,
        contract.dimension_convention == DIMENSION_CONVENTION,
        contract.density_rule == DENSITY_RULE,
        contract.gauge_fermion == GAUGE_FERMION,
        contract.gauge_dimension_convention == GAUGE_DIMENSION_CONVENTION,
        contract.boundary_convention == BOUNDARY_CONVENTION,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.derivation_status
        == 'exact_bulk_type_naturality_mod_dh_and_abstract_gauge_fermion_only',
    )
    if not all(frozen):
        raise ValueError('M1 action/gauge-fixing source or fixture lock changed')
    if (
        len(contract.matter_scalar_labels) != 1
        or len(contract.reference_scalar_labels) != 4
        or len(contract.bulk_term_specs) != 6
        or any(
            spec[1] + spec[2] != spec[3] or spec[3:] != (4, 1)
            for spec in contract.bulk_term_specs
        )
    ):
        raise ValueError('full M1 field, dimension, or density-weight fixture changed')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('M1 action/gauge-fixing contract hash mismatch')
    required_true = (
        contract.bulk_density_type_naturality_certificate_computed,
        contract.boundary_flux_retained,
        contract.gauge_fixing_fermion_constructed,
        contract.gauge_fixing_brst_exactness_computed,
    )
    unsupported = (
        contract.full_coordinate_jet_action_variation_computed,
        contract.boundary_discarded,
        contract.ghy_boundary_variation_computed,
        contract.integrated_action_invariance_proved,
        contract.m1_gauge_condition_derived,
        contract.bv_antifields_constructed,
        contract.classical_master_equation_computed,
        contract.quantum_master_equation_computed,
        contract.functional_measure_computed,
        contract.loop_st_anomaly_cancellation_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('classical density/gauge-fixing claim flags changed')


def density_weight_residual_coefficient(weight: int) -> int:
    '''Coefficient of (partial.c)L after subtracting partial(c L).'''

    return weight - 1


def bulk_dimension_residuals(
    specs: tuple[tuple[str, int, int, int, int], ...] = BULK_TERM_SPECS,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, coefficient_dimension + scalar_dimension - 4)
        for name, coefficient_dimension, scalar_dimension, _, _ in specs
    )


def bulk_density_residuals(
    specs: tuple[tuple[str, int, int, int, int], ...] = BULK_TERM_SPECS,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, density_weight_residual_coefficient(weight))
        for name, _, _, _, weight in specs
    )


def apply_even_derivation(
    polynomial: SparseSuperPolynomial,
    generator_images: Mapping[str, SparseSuperPolynomial],
) -> SparseSuperPolynomial:
    '''Apply an even derivation to the sparse super-polynomial algebra.'''

    result = SparseSuperPolynomial.zero()
    for (even, odd), coefficient in polynomial.terms.items():
        odd_shell = SparseSuperPolynomial.monomial(odd=odd)
        for index, name in enumerate(even):
            if name not in generator_images:
                raise ValueError(f'missing even-derivation image for {name}')
            remaining = even[:index] + even[index + 1 :]
            contribution = (
                SparseSuperPolynomial.monomial(even=remaining)
                * generator_images[name]
                * odd_shell
            )
            result += coefficient * contribution
        even_shell = SparseSuperPolynomial.monomial(even=even)
        for index, name in enumerate(odd):
            if name not in generator_images:
                raise ValueError(f'missing even-derivation image for {name}')
            prefix = SparseSuperPolynomial.monomial(odd=odd[:index])
            suffix = SparseSuperPolynomial.monomial(odd=odd[index + 1 :])
            contribution = (
                even_shell * prefix * generator_images[name] * suffix
            )
            result += coefficient * contribution
    return result


def partial_even_generator(
    polynomial: SparseSuperPolynomial,
    generator_name: str,
) -> SparseSuperPolynomial:
    '''Formal partial derivative with respect to one commuting generator.'''

    result = SparseSuperPolynomial.zero()
    for (even, odd), coefficient in polynomial.terms.items():
        for index, name in enumerate(even):
            if name == generator_name:
                remaining = even[:index] + even[index + 1 :]
                result += coefficient * SparseSuperPolynomial.monomial(
                    even=remaining,
                    odd=odd,
                )
    return result


def _even(name: str) -> SparseSuperPolynomial:
    return SparseSuperPolynomial.generator(name, odd=False)


def _odd(name: str) -> SparseSuperPolynomial:
    return SparseSuperPolynomial.generator(name, odd=True)


def _mass_scale_cubed() -> SparseSuperPolynomial:
    scale = _even('M')
    return scale * scale * scale


@dataclass(frozen=True)
class BadCoordinateDensityAlgebra:
    density: SparseSuperPolynomial
    variation: SparseSuperPolynomial
    locked_variation: SparseSuperPolynomial
    second_variation: SparseSuperPolynomial
    euler_chi_residual: SparseSuperPolynomial
    locked_euler_chi_residual: SparseSuperPolynomial


def bad_coordinate_density_algebra() -> BadCoordinateDensityAlgebra:
    '''A dimension-four but weight-zero density with nilpotent BRST action.'''

    differential = build_jet_differential()
    images = dict(differential.generator_images)
    images['M'] = SparseSuperPolynomial.zero()
    scale_cubed = _mass_scale_cubed()
    density = scale_cubed * _even('chi')
    variation = apply_odd_derivation(density, images)
    locked_variation = polynomial_sum(
        scale_cubed * _odd(f'c{mu}') * _even(f'dchi_{mu}')
        for mu in range(4)
    )
    second_variation = apply_odd_derivation(variation, images)

    euler_residual = SparseSuperPolynomial.zero()
    for mu in range(4):
        momentum = partial_even_generator(variation, f'dchi_{mu}')
        horizontal_images = {
            'M': SparseSuperPolynomial.zero(),
            **{
                f'c{nu}': _odd(f'dc{nu}_{mu}')
                for nu in range(4)
            },
        }
        euler_residual -= apply_even_derivation(
            momentum,
            horizontal_images,
        )
    locked_euler = -polynomial_sum(
        scale_cubed * _odd(f'dc{mu}_{mu}') for mu in range(4)
    )
    return BadCoordinateDensityAlgebra(
        density=density,
        variation=variation,
        locked_variation=locked_variation,
        second_variation=second_variation,
        euler_chi_residual=euler_residual,
        locked_euler_chi_residual=locked_euler,
    )


@dataclass(frozen=True)
class GaugeFermionAlgebra:
    fermion_components: tuple[SparseSuperPolynomial, ...]
    expanded_components: tuple[SparseSuperPolynomial, ...]
    locked_target_components: tuple[SparseSuperPolynomial, ...]
    second_variation_components: tuple[SparseSuperPolynomial, ...]
    wrong_sign_targets: tuple[SparseSuperPolynomial, ...]
    omitted_auxiliary_square_targets: tuple[SparseSuperPolynomial, ...]
    commuting_ghost_expansions: tuple[SparseSuperPolynomial, ...]
    commuting_ghost_targets: tuple[SparseSuperPolynomial, ...]
    broken_doublet_second_variations: tuple[SparseSuperPolynomial, ...]


def gauge_fermion_algebra() -> GaugeFermionAlgebra:
    zero = SparseSuperPolynomial.zero()
    alpha = _even('alpha')
    images: dict[str, SparseSuperPolynomial] = {'alpha': zero}
    broken_images: dict[str, SparseSuperPolynomial] = {'alpha': zero}
    commuting_images: dict[str, SparseSuperPolynomial] = {'alpha': zero}
    fermions: list[SparseSuperPolynomial] = []
    expanded: list[SparseSuperPolynomial] = []
    targets: list[SparseSuperPolynomial] = []
    wrong_targets: list[SparseSuperPolynomial] = []
    omitted_targets: list[SparseSuperPolynomial] = []
    commuting_fermions: list[SparseSuperPolynomial] = []
    commuting_targets: list[SparseSuperPolynomial] = []

    for mu in range(4):
        f_name = f'F{mu}'
        b_name = f'B{mu}'
        bar_name = f'barc{mu}'
        k_name = f'K{mu}'
        j_name = f'J{mu}'
        images.update(
            {
                f_name: _odd(k_name),
                b_name: zero,
                bar_name: _even(b_name),
                k_name: zero,
            }
        )
        broken_images.update(
            {
                f_name: _odd(k_name),
                b_name: _odd(j_name),
                bar_name: _even(b_name),
                k_name: zero,
                j_name: zero,
            }
        )
        commuting_images.update(
            {
                f_name: _even(k_name),
                b_name: zero,
                bar_name: _even(b_name),
                k_name: zero,
            }
        )
        fermions.append(
            _odd(bar_name) * (_even(f_name) + Fraction(1, 2) * alpha * _even(b_name))
        )
        targets.append(
            _even(b_name) * _even(f_name)
            + Fraction(1, 2) * alpha * _even(b_name) * _even(b_name)
            - _odd(bar_name) * _odd(k_name)
        )
        wrong_targets.append(
            _even(b_name) * _even(f_name)
            + Fraction(1, 2) * alpha * _even(b_name) * _even(b_name)
            + _odd(bar_name) * _odd(k_name)
        )
        omitted_targets.append(
            _even(b_name) * _even(f_name) - _odd(bar_name) * _odd(k_name)
        )
        commuting_fermions.append(
            _even(bar_name)
            * (_even(f_name) + Fraction(1, 2) * alpha * _even(b_name))
        )
        commuting_targets.append(
            _even(b_name) * _even(f_name)
            + Fraction(1, 2) * alpha * _even(b_name) * _even(b_name)
            - _even(bar_name) * _even(k_name)
        )

    expanded = [apply_odd_derivation(item, images) for item in fermions]
    second = [apply_odd_derivation(item, images) for item in expanded]
    commuting_expanded = [
        apply_odd_derivation(item, commuting_images)
        for item in commuting_fermions
    ]
    broken_first = [
        apply_odd_derivation(item, broken_images) for item in fermions
    ]
    broken_second = [
        apply_odd_derivation(item, broken_images) for item in broken_first
    ]
    return GaugeFermionAlgebra(
        fermion_components=tuple(fermions),
        expanded_components=tuple(expanded),
        locked_target_components=tuple(targets),
        second_variation_components=tuple(second),
        wrong_sign_targets=tuple(wrong_targets),
        omitted_auxiliary_square_targets=tuple(omitted_targets),
        commuting_ghost_expansions=tuple(commuting_expanded),
        commuting_ghost_targets=tuple(commuting_targets),
        broken_doublet_second_variations=tuple(broken_second),
    )


def polynomial_has_quantum_numbers(
    polynomial: SparseSuperPolynomial,
    mass_dimensions: Mapping[str, int],
    ghost_numbers: Mapping[str, int],
    *,
    expected_mass_dimension: int,
    expected_ghost_number: int,
) -> bool:
    for even, odd in polynomial.terms:
        names = even + odd
        if any(
            name not in mass_dimensions or name not in ghost_numbers
            for name in names
        ):
            return False
        if sum(mass_dimensions[name] for name in names) != expected_mass_dimension:
            return False
        if sum(ghost_numbers[name] for name in names) != expected_ghost_number:
            return False
    return True


def gauge_quantum_number_audit(
    algebra: GaugeFermionAlgebra,
) -> tuple[bool, bool, int]:
    dimensions = {'alpha': -2}
    ghost_numbers = {'alpha': 0}
    for mu in range(4):
        dimensions.update(
            {
                f'F{mu}': 1,
                f'B{mu}': 3,
                f'barc{mu}': 3,
                f'K{mu}': 1,
            }
        )
        ghost_numbers.update(
            {
                f'F{mu}': 0,
                f'B{mu}': 0,
                f'barc{mu}': -1,
                f'K{mu}': 1,
            }
        )
    fermion_ok = all(
        polynomial_has_quantum_numbers(
            item,
            dimensions,
            ghost_numbers,
            expected_mass_dimension=4,
            expected_ghost_number=-1,
        )
        for item in algebra.fermion_components
    )
    target_ok = all(
        polynomial_has_quantum_numbers(
            item,
            dimensions,
            ghost_numbers,
            expected_mass_dimension=4,
            expected_ghost_number=0,
        )
        for item in algebra.locked_target_components
    )
    return fermion_ok, target_ok, 8


@dataclass(frozen=True)
class M1ClassicalActionGaugeFixingReceipt:
    contract_sha256: str
    primary_source: str
    primary_source_url: str
    primary_source_date: str
    source_items: tuple[str, ...]
    source_relation: str
    source_contains_m1_chi_plus_four_x: bool
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_contract_verified: bool
    upstream_classical_brst_gate_passed: bool
    upstream_base_nilpotency_component_count: int
    m1_field_content: str
    matter_scalar_labels: tuple[str, ...]
    reference_scalar_labels: tuple[str, ...]
    bulk_action_density: str
    bulk_term_specs: tuple[tuple[str, int, int, int, int], ...]
    bulk_term_count: int
    bulk_dimension_residuals: tuple[tuple[str, int], ...]
    maximum_bulk_dimension_residual: int
    bulk_density_weight_residuals: tuple[tuple[str, int], ...]
    maximum_bulk_density_weight_residual: int
    bulk_density_type_naturality_certificate_computed: bool
    full_coordinate_jet_action_variation_computed: bool
    bulk_boundary_current_rule: str
    bulk_boundary_current_component_count: int
    boundary_flux_retained: bool
    boundary_discarded: bool
    ghy_boundary_variation_computed: bool
    integrated_action_invariance_proved: bool
    dropped_sqrt_g_density_residual_term_count: int
    missing_reference_scale_dimension_residual: int
    omitted_bulk_term_rejected: bool
    bad_coordinate_density_mass_dimension: int
    bad_coordinate_density_weight: int
    bad_density_variation_term_count: int
    bad_density_locked_variation_mismatch_term_count: int
    bad_density_second_variation_term_count: int
    bad_density_euler_chi_residual_term_count: int
    bad_density_locked_euler_mismatch_term_count: int
    bad_density_not_horizontal_divergence: bool
    nilpotency_does_not_imply_action_invariance: bool
    gauge_fermion: str
    gauge_fermion_component_count: int
    gauge_fixing_locked_mismatch_term_count: int
    gauge_fixing_second_variation_term_count: int
    gauge_quantum_number_audit_map_count: int
    gauge_fermion_dimensions_correct: bool
    gauge_fixed_density_ghost_numbers_correct: bool
    wrong_gauge_ghost_sign_mismatch_term_count: int
    omitted_auxiliary_square_mismatch_term_count: int
    commuting_gauge_ghost_mismatch_term_count: int
    broken_gauge_doublet_second_variation_term_count: int
    gauge_fixing_fermion_constructed: bool
    gauge_fixing_brst_exactness_computed: bool
    m1_gauge_condition_derived: bool
    gauge_fixed_density_diffeomorphism_covariance_proved: bool
    bv_antifields_constructed: bool
    classical_master_equation_computed: bool
    quantum_master_equation_computed: bool
    functional_measure_computed: bool
    loop_st_anomaly_cancellation_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_classical_action_gaugefixing_gate_passed: bool


def evaluate_m1_classical_action_gaugefixing_gate(
) -> M1ClassicalActionGaugeFixingReceipt:
    contract = m1_classical_action_gaugefixing_contract()
    validate_contract(contract)
    upstream_contract = four_scalar_classical_brst_jet_contract()
    validate_e70_b_contract(upstream_contract)
    upstream_receipt = evaluate_four_scalar_classical_brst_jet_gate()

    dimension_residuals = bulk_dimension_residuals(contract.bulk_term_specs)
    density_residuals = bulk_density_residuals(contract.bulk_term_specs)
    maximum_dimension_residual = max(
        abs(value) for _, value in dimension_residuals
    )
    maximum_density_residual = max(
        abs(value) for _, value in density_residuals
    )
    dropped_sqrt_g_residual = sum(
        density_weight_residual_coefficient(0) != 0
        for _ in contract.bulk_term_specs
    )
    reference_spec = next(
        spec
        for spec in contract.bulk_term_specs
        if spec[0] == 'reference_scalar_kinetic'
    )
    missing_reference_scale_residual = reference_spec[2] - 4

    omitted_bulk_term_rejected = False
    try:
        validate_contract(
            replace(
                contract,
                bulk_term_specs=contract.bulk_term_specs[:-1],
            )
        )
    except ValueError:
        omitted_bulk_term_rejected = True

    bad = bad_coordinate_density_algebra()
    bad_variation_mismatch = (
        bad.variation - bad.locked_variation
    ).term_count
    bad_euler_mismatch = (
        bad.euler_chi_residual - bad.locked_euler_chi_residual
    ).term_count
    bad_not_divergence = (
        bad.euler_chi_residual.term_count > 0
        and bad_euler_mismatch == 0
    )
    nilpotency_not_invariance = (
        upstream_receipt.declared_exact_classical_brst_jet_gate_passed
        and bad.second_variation.is_zero
        and bad_not_divergence
    )

    gauge = gauge_fermion_algebra()
    gauge_mismatch = sum(
        (expanded - target).term_count
        for expanded, target in zip(
            gauge.expanded_components,
            gauge.locked_target_components,
        )
    )
    gauge_second = sum(
        item.term_count for item in gauge.second_variation_components
    )
    gauge_dimensions_ok, gauge_ghost_numbers_ok, gauge_audit_count = (
        gauge_quantum_number_audit(gauge)
    )
    wrong_sign_mismatch = sum(
        (expanded - target).term_count
        for expanded, target in zip(
            gauge.expanded_components,
            gauge.wrong_sign_targets,
        )
    )
    omitted_auxiliary_mismatch = sum(
        (expanded - target).term_count
        for expanded, target in zip(
            gauge.expanded_components,
            gauge.omitted_auxiliary_square_targets,
        )
    )
    commuting_mismatch = sum(
        (expanded - target).term_count
        for expanded, target in zip(
            gauge.commuting_ghost_expansions,
            gauge.commuting_ghost_targets,
        )
    )
    broken_doublet_second = sum(
        item.term_count
        for item in gauge.broken_doublet_second_variations
    )

    unsupported = (
        contract.full_coordinate_jet_action_variation_computed,
        contract.boundary_discarded,
        contract.ghy_boundary_variation_computed,
        contract.integrated_action_invariance_proved,
        contract.m1_gauge_condition_derived,
        contract.bv_antifields_constructed,
        contract.classical_master_equation_computed,
        contract.quantum_master_equation_computed,
        contract.functional_measure_computed,
        contract.loop_st_anomaly_cancellation_computed,
        contract.positive_physical_hilbert_proved,
        contract.quantum_hda_m2_proved,
        contract.m3_relational_observables_unlocked,
    )
    passed = all(
        (
            upstream_receipt.declared_exact_classical_brst_jet_gate_passed,
            upstream_receipt.base_nilpotency_component_count == 27,
            len(contract.matter_scalar_labels) == 1,
            len(contract.reference_scalar_labels) == 4,
            len(contract.bulk_term_specs) == 6,
            maximum_dimension_residual == 0,
            maximum_density_residual == 0,
            dropped_sqrt_g_residual == 6,
            missing_reference_scale_residual != 0,
            omitted_bulk_term_rejected,
            contract.boundary_flux_retained,
            not contract.boundary_discarded,
            bad_variation_mismatch == 0,
            bad.second_variation.is_zero,
            bad.euler_chi_residual.term_count == 4,
            bad_euler_mismatch == 0,
            bad_not_divergence,
            nilpotency_not_invariance,
            len(gauge.fermion_components) == 4,
            gauge_mismatch == 0,
            gauge_second == 0,
            gauge_dimensions_ok,
            gauge_ghost_numbers_ok,
            gauge_audit_count == 8,
            wrong_sign_mismatch > 0,
            omitted_auxiliary_mismatch > 0,
            commuting_mismatch > 0,
            broken_doublet_second > 0,
            not any(unsupported),
        )
    )
    return M1ClassicalActionGaugeFixingReceipt(
        contract_sha256=contract.contract_sha256,
        primary_source=contract.primary_source,
        primary_source_url=contract.primary_source_url,
        primary_source_date=contract.primary_source_date,
        source_items=contract.source_items,
        source_relation=contract.source_relation,
        source_contains_m1_chi_plus_four_x=(
            contract.source_contains_m1_chi_plus_four_x
        ),
        upstream_hashes=contract.upstream_hashes,
        upstream_contract_verified=True,
        upstream_classical_brst_gate_passed=(
            upstream_receipt.declared_exact_classical_brst_jet_gate_passed
        ),
        upstream_base_nilpotency_component_count=(
            upstream_receipt.base_nilpotency_component_count
        ),
        m1_field_content=contract.m1_field_content,
        matter_scalar_labels=contract.matter_scalar_labels,
        reference_scalar_labels=contract.reference_scalar_labels,
        bulk_action_density=contract.bulk_action_density,
        bulk_term_specs=contract.bulk_term_specs,
        bulk_term_count=len(contract.bulk_term_specs),
        bulk_dimension_residuals=dimension_residuals,
        maximum_bulk_dimension_residual=maximum_dimension_residual,
        bulk_density_weight_residuals=density_residuals,
        maximum_bulk_density_weight_residual=maximum_density_residual,
        bulk_density_type_naturality_certificate_computed=(
            contract.bulk_density_type_naturality_certificate_computed
        ),
        full_coordinate_jet_action_variation_computed=(
            contract.full_coordinate_jet_action_variation_computed
        ),
        bulk_boundary_current_rule='J_i^rho=c^rho L_i',
        bulk_boundary_current_component_count=(
            4 * len(contract.bulk_term_specs)
        ),
        boundary_flux_retained=contract.boundary_flux_retained,
        boundary_discarded=contract.boundary_discarded,
        ghy_boundary_variation_computed=(
            contract.ghy_boundary_variation_computed
        ),
        integrated_action_invariance_proved=(
            contract.integrated_action_invariance_proved
        ),
        dropped_sqrt_g_density_residual_term_count=(
            dropped_sqrt_g_residual
        ),
        missing_reference_scale_dimension_residual=(
            missing_reference_scale_residual
        ),
        omitted_bulk_term_rejected=omitted_bulk_term_rejected,
        bad_coordinate_density_mass_dimension=4,
        bad_coordinate_density_weight=0,
        bad_density_variation_term_count=bad.variation.term_count,
        bad_density_locked_variation_mismatch_term_count=(
            bad_variation_mismatch
        ),
        bad_density_second_variation_term_count=(
            bad.second_variation.term_count
        ),
        bad_density_euler_chi_residual_term_count=(
            bad.euler_chi_residual.term_count
        ),
        bad_density_locked_euler_mismatch_term_count=bad_euler_mismatch,
        bad_density_not_horizontal_divergence=bad_not_divergence,
        nilpotency_does_not_imply_action_invariance=(
            nilpotency_not_invariance
        ),
        gauge_fermion=contract.gauge_fermion,
        gauge_fermion_component_count=len(gauge.fermion_components),
        gauge_fixing_locked_mismatch_term_count=gauge_mismatch,
        gauge_fixing_second_variation_term_count=gauge_second,
        gauge_quantum_number_audit_map_count=gauge_audit_count,
        gauge_fermion_dimensions_correct=gauge_dimensions_ok,
        gauge_fixed_density_ghost_numbers_correct=(
            gauge_ghost_numbers_ok
        ),
        wrong_gauge_ghost_sign_mismatch_term_count=wrong_sign_mismatch,
        omitted_auxiliary_square_mismatch_term_count=(
            omitted_auxiliary_mismatch
        ),
        commuting_gauge_ghost_mismatch_term_count=commuting_mismatch,
        broken_gauge_doublet_second_variation_term_count=(
            broken_doublet_second
        ),
        gauge_fixing_fermion_constructed=(
            contract.gauge_fixing_fermion_constructed
        ),
        gauge_fixing_brst_exactness_computed=(
            contract.gauge_fixing_brst_exactness_computed
        ),
        m1_gauge_condition_derived=contract.m1_gauge_condition_derived,
        gauge_fixed_density_diffeomorphism_covariance_proved=False,
        bv_antifields_constructed=contract.bv_antifields_constructed,
        classical_master_equation_computed=(
            contract.classical_master_equation_computed
        ),
        quantum_master_equation_computed=(
            contract.quantum_master_equation_computed
        ),
        functional_measure_computed=contract.functional_measure_computed,
        loop_st_anomaly_cancellation_computed=(
            contract.loop_st_anomaly_cancellation_computed
        ),
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        m3_relational_observables_unlocked=(
            contract.m3_relational_observables_unlocked
        ),
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_m1_classical_action_gaugefixing_gate_passed=passed,
    )
