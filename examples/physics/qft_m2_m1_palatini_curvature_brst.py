'''Finite-jet Palatini curvature and affine-connection BRST witness.

The module uses an independent symmetric weight-one h^{mu nu}, a torsion-free
affine connection Gamma^lambda_{mu nu}, and the diffeomorphism ghost.  It
checks affine BRST nilpotency, Ricci covariance, and the local density identity
for L_P=h^{mu nu}R_{mu nu}(Gamma).

It does not derive the connection equation, Levi--Civita equivalence, GHY or
global boundary completion, the scalar/full M1 BV functional, QME, or M2.
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
from examples.physics.qft_m2_m1_metric_density_compatibility_bv import (
    CONTRACT_SHA256 as E70_G_HASH,
    evaluate_m1_metric_density_compatibility_bv_gate,
    m1_metric_density_compatibility_bv_contract,
    validate_contract as validate_e70_g_contract,
)


DIMENSION = 4
MAXIMUM_TOTAL_JET_ORDER = 3
PRIMARY_SOURCE = 'hep-th/0609219'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0609219'
AFFINE_SOURCE = 'arXiv:1005.3001'
AFFINE_SOURCE_URL = 'https://arxiv.org/abs/1005.3001'
TWO_DIMENSIONAL_PRECEDENT = 'hep-th/0503231'
TWO_DIMENSIONAL_PRECEDENT_URL = 'https://arxiv.org/abs/hep-th/0503231'
SOURCE_BOUNDARY = (
    'the sources support first-order Einstein--Hilbert variables and affine '
    'connection structure; the covariant diffeomorphism BRST convention below '
    'is a finite-jet adaptation and is not identified with every canonical '
    'constraint-generated gauge transformation in those sources'
)
PALATINI_DENSITY = 'L_P=h^{mu nu} R_{mu nu}(Gamma)'
NORMALIZATION = (
    'h, Gamma, ghosts, coordinates, and coefficients are normalized to '
    'dimensionless exact polynomial coordinates; the omitted M_P^2/2 factor '
    'does not affect the covariance identities'
)
CLAIM_CEILING = (
    'bounded four-dimensional torsion-free affine BRST, Ricci covariance, '
    'and first-order Palatini bulk-density covariance with retained local '
    'current; no connection EOM, Levi--Civita or first/second-order '
    'equivalence, GHY/global boundary completion, scalar/full M1 BV/CME, '
    'functional measure, QME, continuum ST, physical Hilbert, quantum HDA '
    'M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-G', E70_G_HASH),)
SOURCE_ITEMS = (
    'hep-th/0609219: d>2 first-order Einstein--Hilbert canonical variables and constraints',
    'arXiv:1005.3001: d>2 affine-connection first-order constraint structure',
    'hep-th/0503231: two-dimensional h-Gamma precedent and warning that canonical gauge maps need not equal diffeomorphisms',
)
AFFINE_BRST_CONVENTION = (
    'sGamma^lambda_mn=c^rho partial_rho Gamma^lambda_mn-'
    'Gamma^rho_mn partial_rho c^lambda+Gamma^lambda_rho,n '
    'partial_m c^rho+Gamma^lambda_m,rho partial_n c^rho+'
    'partial_m partial_n c^lambda; sc^lambda=c^rho partial_rho c^lambda'
)
CONTRACT_SHA256 = (
    'ac12325f300917aeeacb55d12dfed0ac28718fb265b044e34c4581f2c48db5b2'
)


class PalatiniJetOrderExceeded(ValueError):
    '''Raised instead of silently truncating a Palatini jet.'''


MultiIndex = tuple[int, int, int, int]
ZERO_MULTIINDEX: MultiIndex = (0, 0, 0, 0)


def multiindices_up_to(maximum_order: int) -> tuple[MultiIndex, ...]:
    if maximum_order < 0:
        raise ValueError('maximum multi-index order must be nonnegative')
    values = (
        index
        for index in product(range(maximum_order + 1), repeat=DIMENSION)
        if sum(index) <= maximum_order
    )
    return tuple(sorted(values, key=lambda item: (sum(item), item)))


MULTIINDICES = multiindices_up_to(MAXIMUM_TOTAL_JET_ORDER)


@dataclass(frozen=True)
class PalatiniVariableSpec:
    name: str
    role: str
    parity: int
    ghost_number: int
    geometric_density_weight: int | None


def _h_name(mu: int, nu: int) -> str:
    left, right = sorted((mu, nu))
    return f'h{left}{right}'


def _gamma_name(upper: int, lower_left: int, lower_right: int) -> str:
    left, right = sorted((lower_left, lower_right))
    return f'Gamma{upper}_{left}{right}'


def palatini_variable_specs() -> tuple[PalatiniVariableSpec, ...]:
    specs: list[PalatiniVariableSpec] = []
    for mu in range(DIMENSION):
        for nu in range(mu, DIMENSION):
            specs.append(
                PalatiniVariableSpec(
                    _h_name(mu, nu),
                    'symmetric contravariant tensor density component',
                    0,
                    0,
                    1,
                )
            )
    for upper in range(DIMENSION):
        for lower_left in range(DIMENSION):
            for lower_right in range(lower_left, DIMENSION):
                specs.append(
                    PalatiniVariableSpec(
                        _gamma_name(upper, lower_left, lower_right),
                        'torsion-free affine connection component',
                        0,
                        0,
                        None,
                    )
                )
    for mu in range(DIMENSION):
        specs.append(
            PalatiniVariableSpec(
                f'c{mu}',
                'contravariant diffeomorphism ghost',
                1,
                1,
                0,
            )
        )
    return tuple(specs)


VARIABLE_SPECS = palatini_variable_specs()
SPEC_BY_NAME = {spec.name: spec for spec in VARIABLE_SPECS}


def jet_name(base_name: str, multiindex: MultiIndex) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown Palatini variable {base_name}')
    if len(multiindex) != DIMENSION or min(multiindex) < 0:
        raise ValueError('multi-index must contain four nonnegative entries')
    if sum(multiindex) > MAXIMUM_TOTAL_JET_ORDER:
        raise PalatiniJetOrderExceeded(
            f'multi-index {multiindex} exceeds total order '
            f'{MAXIMUM_TOTAL_JET_ORDER}'
        )
    return f'{base_name}__{multiindex[0]}_{multiindex[1]}_{multiindex[2]}_{multiindex[3]}'


JET_LOOKUP = {
    jet_name(spec.name, multiindex): (spec, multiindex)
    for spec in VARIABLE_SPECS
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
                raise PalatiniJetOrderExceeded(
                    f'D_{direction} requires {spec.name} jet {next_index}'
                )
            remaining = even_names[:index] + even_names[index + 1 :]
            result += coefficient * (
                SparseSuperPolynomial.monomial(
                    even=remaining,
                    odd=odd_names,
                )
                * generator(spec.name, next_index)
            )
        for index, name in enumerate(odd_names):
            spec, multiindex = JET_LOOKUP[name]
            next_index = add_multiindex(multiindex, increment)
            if sum(next_index) > MAXIMUM_TOTAL_JET_ORDER:
                raise PalatiniJetOrderExceeded(
                    f'D_{direction} requires {spec.name} jet {next_index}'
                )
            prefix = SparseSuperPolynomial.monomial(
                even=even_names,
                odd=odd_names[:index],
            )
            suffix = SparseSuperPolynomial.monomial(odd=odd_names[index + 1 :])
            result += coefficient * (
                prefix * generator(spec.name, next_index) * suffix
            )
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


def divergence(
    currents: tuple[SparseSuperPolynomial, ...],
) -> SparseSuperPolynomial:
    if len(currents) != DIMENSION:
        raise ValueError('a four-current must have four components')
    return polynomial_sum(
        horizontal_derivative(current, direction)
        for direction, current in enumerate(currents)
    )


@dataclass(frozen=True)
class PalatiniCurvatureBRSTModel:
    transformations: Mapping[str, SparseSuperPolynomial]
    ricci_components: Mapping[tuple[int, int], SparseSuperPolynomial]
    palatini_density: SparseSuperPolynomial
    boundary_current: tuple[SparseSuperPolynomial, ...]


def ricci_components(
    *,
    second_derivative_coefficient: int = -1,
    first_product_coefficient: int = 1,
    second_product_coefficient: int = -1,
) -> Mapping[tuple[int, int], SparseSuperPolynomial]:
    return {
        (mu, nu): polynomial_sum(
            horizontal_derivative(
                generator(_gamma_name(lam, nu, mu)),
                lam,
            )
            + second_derivative_coefficient
            * horizontal_derivative(
                generator(_gamma_name(lam, lam, mu)),
                nu,
            )
            for lam in range(DIMENSION)
        )
        + first_product_coefficient
        * polynomial_sum(
            generator(_gamma_name(lam, lam, sigma))
            * generator(_gamma_name(sigma, nu, mu))
            for lam in range(DIMENSION)
            for sigma in range(DIMENSION)
        )
        + second_product_coefficient
        * polynomial_sum(
            generator(_gamma_name(lam, nu, sigma))
            * generator(_gamma_name(sigma, lam, mu))
            for lam in range(DIMENSION)
            for sigma in range(DIMENSION)
        )
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    }


def palatini_curvature_brst_model(
    *,
    ghost_sign: int = 1,
    h_density_weight: int = 1,
    gamma_inhomogeneous_coefficient: int = 1,
    upper_transport_coefficient: int = -1,
    include_first_lower_transport: bool = True,
    include_second_lower_transport: bool = True,
    ricci_second_derivative_coefficient: int = -1,
    ricci_first_product_coefficient: int = 1,
    ricci_second_product_coefficient: int = -1,
) -> PalatiniCurvatureBRSTModel:
    if ghost_sign not in (-1, 1):
        raise ValueError('ghost sign must be plus or minus one')
    if gamma_inhomogeneous_coefficient not in (-1, 0, 1):
        raise ValueError('Gamma inhomogeneous coefficient must be -1, 0, or 1')
    if upper_transport_coefficient not in (-1, 1):
        raise ValueError('upper connection transport coefficient must be +/-1')

    transformations: dict[str, SparseSuperPolynomial] = {}
    divergence_ghost = polynomial_sum(
        generator(f'c{rho}', unit_multiindex(rho))
        for rho in range(DIMENSION)
    )
    for mu in range(DIMENSION):
        for nu in range(mu, DIMENSION):
            name = _h_name(mu, nu)
            transformations[name] = (
                polynomial_sum(
                    generator(f'c{rho}')
                    * generator(name, unit_multiindex(rho))
                    - generator(_h_name(rho, nu))
                    * generator(f'c{mu}', unit_multiindex(rho))
                    - generator(_h_name(mu, rho))
                    * generator(f'c{nu}', unit_multiindex(rho))
                    for rho in range(DIMENSION)
                )
                + h_density_weight * generator(name) * divergence_ghost
            )
    for upper in range(DIMENSION):
        for lower_left in range(DIMENSION):
            for lower_right in range(lower_left, DIMENSION):
                name = _gamma_name(upper, lower_left, lower_right)
                image = polynomial_sum(
                    generator(f'c{rho}')
                    * generator(name, unit_multiindex(rho))
                    + upper_transport_coefficient
                    * generator(_gamma_name(rho, lower_left, lower_right))
                    * generator(f'c{upper}', unit_multiindex(rho))
                    for rho in range(DIMENSION)
                )
                if include_first_lower_transport:
                    image += polynomial_sum(
                        generator(_gamma_name(upper, rho, lower_right))
                        * generator(f'c{rho}', unit_multiindex(lower_left))
                        for rho in range(DIMENSION)
                    )
                if include_second_lower_transport:
                    image += polynomial_sum(
                        generator(_gamma_name(upper, lower_left, rho))
                        * generator(f'c{rho}', unit_multiindex(lower_right))
                        for rho in range(DIMENSION)
                    )
                if gamma_inhomogeneous_coefficient:
                    image += gamma_inhomogeneous_coefficient * generator(
                        f'c{upper}',
                        add_multiindex(
                            unit_multiindex(lower_left),
                            unit_multiindex(lower_right),
                        ),
                    )
                transformations[name] = image
    for mu in range(DIMENSION):
        transformations[f'c{mu}'] = ghost_sign * polynomial_sum(
            generator(f'c{rho}')
            * generator(f'c{mu}', unit_multiindex(rho))
            for rho in range(DIMENSION)
        )

    ricci = ricci_components(
        second_derivative_coefficient=ricci_second_derivative_coefficient,
        first_product_coefficient=ricci_first_product_coefficient,
        second_product_coefficient=ricci_second_product_coefficient,
    )
    density = polynomial_sum(
        generator(_h_name(mu, nu)) * ricci[(mu, nu)]
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    return PalatiniCurvatureBRSTModel(
        transformations=transformations,
        ricci_components=ricci,
        palatini_density=density,
        boundary_current=tuple(
            generator(f'c{mu}') * density
            for mu in range(DIMENSION)
        ),
    )


def apply_palatini_brst(
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
        if name not in JET_LOOKUP:
            raise ValueError(f'unregistered Palatini jet generator {name}')
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


def ricci_covariance_target(
    ricci: Mapping[tuple[int, int], SparseSuperPolynomial],
    mu: int,
    nu: int,
) -> SparseSuperPolynomial:
    return (
        polynomial_sum(
            generator(f'c{rho}')
            * horizontal_derivative(ricci[(mu, nu)], rho)
            for rho in range(DIMENSION)
        )
        + polynomial_sum(
            ricci[(rho, nu)]
            * generator(f'c{rho}', unit_multiindex(mu))
            for rho in range(DIMENSION)
        )
        + polynomial_sum(
            ricci[(mu, rho)]
            * generator(f'c{rho}', unit_multiindex(nu))
            for rho in range(DIMENSION)
        )
    )


@dataclass(frozen=True)
class M1PalatiniCurvatureBRSTContract:
    primary_source: str
    primary_source_url: str
    affine_source: str
    affine_source_url: str
    two_dimensional_precedent: str
    two_dimensional_precedent_url: str
    source_items: tuple[str, ...]
    source_boundary: str
    palatini_density: str
    normalization: str
    affine_brst_convention: str
    dimension: int
    maximum_total_jet_order: int
    variable_specs: tuple[PalatiniVariableSpec, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    torsion_free_connection_basis_constructed: bool
    affine_second_ghost_derivative_included: bool
    affine_brst_nilpotency_computed: bool
    ricci_tensor_covariance_computed: bool
    nonsymmetric_ricci_fixture_retained: bool
    palatini_bulk_density_constructed: bool
    palatini_density_total_divergence_computed: bool
    nonzero_boundary_current_retained: bool
    live_negative_controls_computed: bool
    silent_terminal_truncation_allowed: bool
    metric_density_constraint_included: bool
    connection_equation_derived: bool
    metric_compatibility_derived: bool
    levi_civita_connection_derived: bool
    first_second_order_equivalence_proved: bool
    canonical_gauge_generator_equated_to_diffeomorphism: bool
    ghy_boundary_term_used: bool
    global_boundary_completion_proved: bool
    scalar_full_m1_action_assembled: bool
    full_m1_bv_functional_constructed: bool
    classical_master_equation_computed: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_palatini_curvature_brst_contract() -> M1PalatiniCurvatureBRSTContract:
    return M1PalatiniCurvatureBRSTContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        affine_source=AFFINE_SOURCE,
        affine_source_url=AFFINE_SOURCE_URL,
        two_dimensional_precedent=TWO_DIMENSIONAL_PRECEDENT,
        two_dimensional_precedent_url=TWO_DIMENSIONAL_PRECEDENT_URL,
        source_items=SOURCE_ITEMS,
        source_boundary=SOURCE_BOUNDARY,
        palatini_density=PALATINI_DENSITY,
        normalization=NORMALIZATION,
        affine_brst_convention=AFFINE_BRST_CONVENTION,
        dimension=DIMENSION,
        maximum_total_jet_order=MAXIMUM_TOTAL_JET_ORDER,
        variable_specs=VARIABLE_SPECS,
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        torsion_free_connection_basis_constructed=True,
        affine_second_ghost_derivative_included=True,
        affine_brst_nilpotency_computed=True,
        ricci_tensor_covariance_computed=True,
        nonsymmetric_ricci_fixture_retained=True,
        palatini_bulk_density_constructed=True,
        palatini_density_total_divergence_computed=True,
        nonzero_boundary_current_retained=True,
        live_negative_controls_computed=True,
        silent_terminal_truncation_allowed=False,
        metric_density_constraint_included=False,
        connection_equation_derived=False,
        metric_compatibility_derived=False,
        levi_civita_connection_derived=False,
        first_second_order_equivalence_proved=False,
        canonical_gauge_generator_equated_to_diffeomorphism=False,
        ghy_boundary_term_used=False,
        global_boundary_completion_proved=False,
        scalar_full_m1_action_assembled=False,
        full_m1_bv_functional_constructed=False,
        classical_master_equation_computed=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_bounded_4d_torsion_free_affine_ricci_palatini_density_'
            'not_connection_eom_levi_civita_full_m1_bv_or_quantum_m2'
        ),
    )


def _serialize_spec(spec: PalatiniVariableSpec) -> str:
    return ':'.join(
        (
            spec.name,
            spec.role,
            str(spec.parity),
            str(spec.ghost_number),
            str(spec.geometric_density_weight),
        )
    )


_CONTRACT_FLAG_NAMES = (
    'torsion_free_connection_basis_constructed',
    'affine_second_ghost_derivative_included',
    'affine_brst_nilpotency_computed',
    'ricci_tensor_covariance_computed',
    'nonsymmetric_ricci_fixture_retained',
    'palatini_bulk_density_constructed',
    'palatini_density_total_divergence_computed',
    'nonzero_boundary_current_retained',
    'live_negative_controls_computed',
    'silent_terminal_truncation_allowed',
    'metric_density_constraint_included',
    'connection_equation_derived',
    'metric_compatibility_derived',
    'levi_civita_connection_derived',
    'first_second_order_equivalence_proved',
    'canonical_gauge_generator_equated_to_diffeomorphism',
    'ghy_boundary_term_used',
    'global_boundary_completion_proved',
    'scalar_full_m1_action_assembled',
    'full_m1_bv_functional_constructed',
    'classical_master_equation_computed',
    'functional_measure_computed',
    'quantum_master_equation_computed',
    'continuum_loop_st_computed',
    'positive_physical_hilbert_proved',
    'quantum_hda_m2_proved',
    'm3_relational_observables_unlocked',
)


def canonical_contract_payload(contract: M1PalatiniCurvatureBRSTContract) -> str:
    comma = chr(44)
    flags = comma.join(
        f'{name}:{getattr(contract, name)}'
        for name in _CONTRACT_FLAG_NAMES
    )
    return '|'.join(
        (
            f'primary={contract.primary_source}',
            f'primary_url={contract.primary_source_url}',
            f'affine={contract.affine_source}',
            f'affine_url={contract.affine_source_url}',
            f'two_d={contract.two_dimensional_precedent}',
            f'two_d_url={contract.two_dimensional_precedent_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_boundary={contract.source_boundary}',
            f'palatini={contract.palatini_density}',
            f'normalization={contract.normalization}',
            f'affine_brst={contract.affine_brst_convention}',
            f'dimension={contract.dimension}',
            f'max_total_jet={contract.maximum_total_jet_order}',
            f'variables={comma.join(_serialize_spec(x) for x in contract.variable_specs)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(contract: M1PalatiniCurvatureBRSTContract) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M1PalatiniCurvatureBRSTContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.affine_source == AFFINE_SOURCE,
        contract.affine_source_url == AFFINE_SOURCE_URL,
        contract.two_dimensional_precedent == TWO_DIMENSIONAL_PRECEDENT,
        contract.two_dimensional_precedent_url == TWO_DIMENSIONAL_PRECEDENT_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.source_boundary == SOURCE_BOUNDARY,
        contract.palatini_density == PALATINI_DENSITY,
        contract.normalization == NORMALIZATION,
        contract.affine_brst_convention == AFFINE_BRST_CONVENTION,
        contract.dimension == DIMENSION,
        contract.maximum_total_jet_order == MAXIMUM_TOTAL_JET_ORDER,
        contract.variable_specs == VARIABLE_SPECS,
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == (
            'exact_bounded_4d_torsion_free_affine_ricci_palatini_density_'
            'not_connection_eom_levi_civita_full_m1_bv_or_quantum_m2'
        ),
    )
    if not all(frozen):
        raise ValueError('Palatini curvature source, basis, or status lock changed')
    if len(contract.variable_specs) != 54:
        raise ValueError('Palatini witness requires 54 base variables')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('Palatini curvature contract hash mismatch')
    required_true = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[:9]
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[9:]
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('Palatini curvature claim flags changed')


@dataclass(frozen=True)
class M1PalatiniCurvatureBRSTReceipt:
    contract_sha256: str
    source_boundary: str
    normalization: str
    affine_brst_convention: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_g_verified: bool
    h_component_count: int
    torsion_free_connection_component_count: int
    ghost_component_count: int
    base_variable_count: int
    multiindex_count: int
    bounded_jet_generator_count: int
    bounded_even_jet_generator_count: int
    bounded_odd_jet_generator_count: int
    torsion_free_name_symmetry_locked: bool
    unsymmetrized_connection_name_rejected: bool
    affine_map_component_count: int
    affine_inhomogeneous_second_ghost_component_count: int
    base_nilpotency_component_count: int
    base_nilpotency_nonzero_component_count: int
    base_nilpotency_maximum_residual_term_count: int
    ricci_component_count: int
    ricci_total_term_count: int
    ricci_minimum_component_term_count: int
    ricci_maximum_component_term_count: int
    nonsymmetric_ricci_component_count: int
    nonsymmetric_ricci_total_term_count: int
    ricci_covariance_nonzero_component_count: int
    ricci_covariance_maximum_residual_term_count: int
    palatini_density_term_count: int
    palatini_variation_term_count: int
    boundary_current_term_count: int
    boundary_current_divergence_term_count: int
    palatini_density_identity_mismatch_term_count: int
    omitted_inhomogeneous_nonzero_nilpotency_component_count: int
    omitted_inhomogeneous_ricci_nonzero_component_count: int
    omitted_inhomogeneous_ricci_maximum_residual_term_count: int
    omitted_inhomogeneous_density_mismatch_term_count: int
    omitted_lower_transport_nonzero_nilpotency_component_count: int
    omitted_lower_transport_ricci_nonzero_component_count: int
    omitted_lower_transport_density_mismatch_term_count: int
    wrong_upper_transport_nonzero_nilpotency_component_count: int
    wrong_upper_transport_ricci_nonzero_component_count: int
    wrong_upper_transport_density_mismatch_term_count: int
    missing_h_weight_density_mismatch_term_count: int
    wrong_ghost_sign_nonzero_nilpotency_component_count: int
    wrong_ghost_sign_maximum_nilpotency_residual_term_count: int
    wrong_ricci_derivative_nonzero_covariance_component_count: int
    wrong_ricci_derivative_density_mismatch_term_count: int
    wrong_ricci_product_nonzero_covariance_component_count: int
    wrong_ricci_product_density_mismatch_term_count: int
    terminal_jet_derivative_rejected: bool
    torsion_free_connection_basis_constructed: bool
    affine_second_ghost_derivative_included: bool
    affine_brst_nilpotency_computed: bool
    ricci_tensor_covariance_computed: bool
    nonsymmetric_ricci_fixture_retained: bool
    palatini_bulk_density_constructed: bool
    palatini_density_total_divergence_computed: bool
    nonzero_boundary_current_retained: bool
    live_negative_controls_computed: bool
    silent_terminal_truncation_allowed: bool
    metric_density_constraint_included: bool
    connection_equation_derived: bool
    metric_compatibility_derived: bool
    levi_civita_connection_derived: bool
    first_second_order_equivalence_proved: bool
    canonical_gauge_generator_equated_to_diffeomorphism: bool
    ghy_boundary_term_used: bool
    global_boundary_completion_proved: bool
    scalar_full_m1_action_assembled: bool
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
    declared_m1_palatini_curvature_brst_gate_passed: bool


def _nilpotency_residuals(
    model: PalatiniCurvatureBRSTModel,
) -> tuple[SparseSuperPolynomial, ...]:
    return tuple(
        apply_palatini_brst(image, model.transformations)
        for image in model.transformations.values()
    )


def _ricci_covariance_residuals(
    model: PalatiniCurvatureBRSTModel,
) -> tuple[SparseSuperPolynomial, ...]:
    return tuple(
        apply_palatini_brst(
            model.ricci_components[(mu, nu)],
            model.transformations,
        )
        - ricci_covariance_target(model.ricci_components, mu, nu)
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )


def _palatini_density_mismatch(
    model: PalatiniCurvatureBRSTModel,
) -> SparseSuperPolynomial:
    return (
        apply_palatini_brst(
            model.palatini_density,
            model.transformations,
        )
        - divergence(model.boundary_current)
    )


@lru_cache(maxsize=1)
def evaluate_m1_palatini_curvature_brst_gate(
) -> M1PalatiniCurvatureBRSTReceipt:
    contract = m1_palatini_curvature_brst_contract()
    validate_contract(contract)
    upstream_contract = m1_metric_density_compatibility_bv_contract()
    validate_e70_g_contract(upstream_contract)
    upstream_receipt = evaluate_m1_metric_density_compatibility_bv_gate()
    upstream_verified = (
        upstream_receipt.declared_m1_metric_density_compatibility_bv_gate_passed
    )

    model = palatini_curvature_brst_model()
    nilpotency_residuals = _nilpotency_residuals(model)
    ricci_residuals = _ricci_covariance_residuals(model)
    ricci_antisymmetry_fixtures = tuple(
        model.ricci_components[(mu, nu)]
        - model.ricci_components[(nu, mu)]
        for mu in range(DIMENSION)
        for nu in range(mu + 1, DIMENSION)
    )
    palatini_variation = apply_palatini_brst(
        model.palatini_density,
        model.transformations,
    )
    boundary_divergence = divergence(model.boundary_current)
    density_mismatch = palatini_variation - boundary_divergence

    torsion_free_name_symmetry = all(
        _gamma_name(upper, lower_left, lower_right)
        == _gamma_name(upper, lower_right, lower_left)
        for upper in range(DIMENSION)
        for lower_left in range(DIMENSION)
        for lower_right in range(DIMENSION)
    )
    unsymmetrized_name_rejected = False
    try:
        generator('Gamma0_10')
    except (KeyError, ValueError):
        unsymmetrized_name_rejected = True

    no_inhomogeneous = palatini_curvature_brst_model(
        gamma_inhomogeneous_coefficient=0
    )
    inhomogeneous_differences = tuple(
        model.transformations[name] - no_inhomogeneous.transformations[name]
        for name in model.transformations
        if name.startswith('Gamma')
    )
    no_inhomogeneous_nilpotency = _nilpotency_residuals(no_inhomogeneous)
    no_inhomogeneous_ricci = _ricci_covariance_residuals(no_inhomogeneous)
    no_inhomogeneous_density = _palatini_density_mismatch(no_inhomogeneous)

    omitted_lower = palatini_curvature_brst_model(
        include_first_lower_transport=False
    )
    omitted_lower_nilpotency = _nilpotency_residuals(omitted_lower)
    omitted_lower_ricci = _ricci_covariance_residuals(omitted_lower)
    omitted_lower_density = _palatini_density_mismatch(omitted_lower)

    wrong_upper = palatini_curvature_brst_model(
        upper_transport_coefficient=1
    )
    wrong_upper_nilpotency = _nilpotency_residuals(wrong_upper)
    wrong_upper_ricci = _ricci_covariance_residuals(wrong_upper)
    wrong_upper_density = _palatini_density_mismatch(wrong_upper)

    missing_h_weight = palatini_curvature_brst_model(h_density_weight=0)
    missing_h_weight_density = _palatini_density_mismatch(missing_h_weight)

    wrong_ghost_sign = palatini_curvature_brst_model(ghost_sign=-1)
    wrong_ghost_nilpotency = _nilpotency_residuals(wrong_ghost_sign)

    wrong_ricci_derivative = palatini_curvature_brst_model(
        ricci_second_derivative_coefficient=1
    )
    wrong_ricci_derivative_residuals = _ricci_covariance_residuals(
        wrong_ricci_derivative
    )
    wrong_ricci_derivative_density = _palatini_density_mismatch(
        wrong_ricci_derivative
    )

    wrong_ricci_product = palatini_curvature_brst_model(
        ricci_second_product_coefficient=1
    )
    wrong_ricci_product_residuals = _ricci_covariance_residuals(
        wrong_ricci_product
    )
    wrong_ricci_product_density = _palatini_density_mismatch(
        wrong_ricci_product
    )

    terminal_rejected = False
    try:
        horizontal_derivative(
            generator(
                'Gamma0_00',
                (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0),
            ),
            0,
        )
    except PalatiniJetOrderExceeded:
        terminal_rejected = True

    h_specs = tuple(spec for spec in VARIABLE_SPECS if spec.name.startswith('h'))
    gamma_specs = tuple(
        spec for spec in VARIABLE_SPECS if spec.name.startswith('Gamma')
    )
    ghost_specs = tuple(
        spec for spec in VARIABLE_SPECS if spec.name.startswith('c')
    )
    even_base_count = sum(spec.parity == 0 for spec in VARIABLE_SPECS)
    ricci_term_counts = tuple(
        component.term_count for component in model.ricci_components.values()
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[9:]
    )
    passed = all(
        (
            upstream_verified,
            len(h_specs) == 10,
            len(gamma_specs) == 40,
            len(ghost_specs) == 4,
            len(VARIABLE_SPECS) == 54,
            len(MULTIINDICES) == 35,
            len(JET_LOOKUP) == 1890,
            even_base_count * len(MULTIINDICES) == 1750,
            (len(VARIABLE_SPECS) - even_base_count) * len(MULTIINDICES)
            == 140,
            torsion_free_name_symmetry,
            unsymmetrized_name_rejected,
            len(model.transformations) == 54,
            len(inhomogeneous_differences) == 40,
            all(value.term_count == 1 for value in inhomogeneous_differences),
            len(nilpotency_residuals) == 54,
            max(value.term_count for value in nilpotency_residuals) == 0,
            len(model.ricci_components) == 16,
            sum(ricci_term_counts) == 396,
            min(ricci_term_counts) == 24,
            max(ricci_term_counts) == 27,
            len(ricci_antisymmetry_fixtures) == 6,
            all(not value.is_zero for value in ricci_antisymmetry_fixtures),
            sum(value.term_count for value in ricci_antisymmetry_fixtures)
            == 48,
            max(value.term_count for value in ricci_residuals) == 0,
            model.palatini_density.term_count == 276,
            palatini_variation.term_count == 4032,
            sum(value.term_count for value in model.boundary_current) == 1104,
            boundary_divergence.term_count == 4032,
            density_mismatch.is_zero,
            sum(not value.is_zero for value in no_inhomogeneous_nilpotency)
            == 0,
            sum(not value.is_zero for value in no_inhomogeneous_ricci) == 16,
            max(value.term_count for value in no_inhomogeneous_ricci) == 39,
            no_inhomogeneous_density.term_count == 372,
            sum(not value.is_zero for value in omitted_lower_nilpotency) == 40,
            sum(not value.is_zero for value in omitted_lower_ricci) == 16,
            omitted_lower_density.term_count == 1786,
            sum(not value.is_zero for value in wrong_upper_nilpotency) == 40,
            sum(not value.is_zero for value in wrong_upper_ricci) == 16,
            wrong_upper_density.term_count == 2040,
            missing_h_weight_density.term_count == 1104,
            sum(not value.is_zero for value in wrong_ghost_nilpotency) == 50,
            max(value.term_count for value in wrong_ghost_nilpotency) == 121,
            sum(
                not value.is_zero
                for value in wrong_ricci_derivative_residuals
            )
            == 16,
            wrong_ricci_derivative_density.term_count == 200,
            sum(
                not value.is_zero for value in wrong_ricci_product_residuals
            )
            == 16,
            wrong_ricci_product_density.term_count == 256,
            terminal_rejected,
            not any(unsupported),
        )
    )

    return M1PalatiniCurvatureBRSTReceipt(
        contract_sha256=contract.contract_sha256,
        source_boundary=contract.source_boundary,
        normalization=contract.normalization,
        affine_brst_convention=contract.affine_brst_convention,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_g_verified=upstream_verified,
        h_component_count=len(h_specs),
        torsion_free_connection_component_count=len(gamma_specs),
        ghost_component_count=len(ghost_specs),
        base_variable_count=len(VARIABLE_SPECS),
        multiindex_count=len(MULTIINDICES),
        bounded_jet_generator_count=len(JET_LOOKUP),
        bounded_even_jet_generator_count=even_base_count * len(MULTIINDICES),
        bounded_odd_jet_generator_count=(
            len(JET_LOOKUP) - even_base_count * len(MULTIINDICES)
        ),
        torsion_free_name_symmetry_locked=torsion_free_name_symmetry,
        unsymmetrized_connection_name_rejected=unsymmetrized_name_rejected,
        affine_map_component_count=len(model.transformations),
        affine_inhomogeneous_second_ghost_component_count=(
            len(inhomogeneous_differences)
        ),
        base_nilpotency_component_count=len(nilpotency_residuals),
        base_nilpotency_nonzero_component_count=sum(
            not value.is_zero for value in nilpotency_residuals
        ),
        base_nilpotency_maximum_residual_term_count=max(
            value.term_count for value in nilpotency_residuals
        ),
        ricci_component_count=len(model.ricci_components),
        ricci_total_term_count=sum(ricci_term_counts),
        ricci_minimum_component_term_count=min(ricci_term_counts),
        ricci_maximum_component_term_count=max(ricci_term_counts),
        nonsymmetric_ricci_component_count=sum(
            not value.is_zero for value in ricci_antisymmetry_fixtures
        ),
        nonsymmetric_ricci_total_term_count=sum(
            value.term_count for value in ricci_antisymmetry_fixtures
        ),
        ricci_covariance_nonzero_component_count=sum(
            not value.is_zero for value in ricci_residuals
        ),
        ricci_covariance_maximum_residual_term_count=max(
            value.term_count for value in ricci_residuals
        ),
        palatini_density_term_count=model.palatini_density.term_count,
        palatini_variation_term_count=palatini_variation.term_count,
        boundary_current_term_count=sum(
            value.term_count for value in model.boundary_current
        ),
        boundary_current_divergence_term_count=boundary_divergence.term_count,
        palatini_density_identity_mismatch_term_count=density_mismatch.term_count,
        omitted_inhomogeneous_nonzero_nilpotency_component_count=sum(
            not value.is_zero for value in no_inhomogeneous_nilpotency
        ),
        omitted_inhomogeneous_ricci_nonzero_component_count=sum(
            not value.is_zero for value in no_inhomogeneous_ricci
        ),
        omitted_inhomogeneous_ricci_maximum_residual_term_count=max(
            value.term_count for value in no_inhomogeneous_ricci
        ),
        omitted_inhomogeneous_density_mismatch_term_count=(
            no_inhomogeneous_density.term_count
        ),
        omitted_lower_transport_nonzero_nilpotency_component_count=sum(
            not value.is_zero for value in omitted_lower_nilpotency
        ),
        omitted_lower_transport_ricci_nonzero_component_count=sum(
            not value.is_zero for value in omitted_lower_ricci
        ),
        omitted_lower_transport_density_mismatch_term_count=(
            omitted_lower_density.term_count
        ),
        wrong_upper_transport_nonzero_nilpotency_component_count=sum(
            not value.is_zero for value in wrong_upper_nilpotency
        ),
        wrong_upper_transport_ricci_nonzero_component_count=sum(
            not value.is_zero for value in wrong_upper_ricci
        ),
        wrong_upper_transport_density_mismatch_term_count=(
            wrong_upper_density.term_count
        ),
        missing_h_weight_density_mismatch_term_count=(
            missing_h_weight_density.term_count
        ),
        wrong_ghost_sign_nonzero_nilpotency_component_count=sum(
            not value.is_zero for value in wrong_ghost_nilpotency
        ),
        wrong_ghost_sign_maximum_nilpotency_residual_term_count=max(
            value.term_count for value in wrong_ghost_nilpotency
        ),
        wrong_ricci_derivative_nonzero_covariance_component_count=sum(
            not value.is_zero for value in wrong_ricci_derivative_residuals
        ),
        wrong_ricci_derivative_density_mismatch_term_count=(
            wrong_ricci_derivative_density.term_count
        ),
        wrong_ricci_product_nonzero_covariance_component_count=sum(
            not value.is_zero for value in wrong_ricci_product_residuals
        ),
        wrong_ricci_product_density_mismatch_term_count=(
            wrong_ricci_product_density.term_count
        ),
        terminal_jet_derivative_rejected=terminal_rejected,
        torsion_free_connection_basis_constructed=(
            contract.torsion_free_connection_basis_constructed
        ),
        affine_second_ghost_derivative_included=(
            contract.affine_second_ghost_derivative_included
        ),
        affine_brst_nilpotency_computed=contract.affine_brst_nilpotency_computed,
        ricci_tensor_covariance_computed=contract.ricci_tensor_covariance_computed,
        nonsymmetric_ricci_fixture_retained=(
            contract.nonsymmetric_ricci_fixture_retained
        ),
        palatini_bulk_density_constructed=contract.palatini_bulk_density_constructed,
        palatini_density_total_divergence_computed=(
            contract.palatini_density_total_divergence_computed
        ),
        nonzero_boundary_current_retained=contract.nonzero_boundary_current_retained,
        live_negative_controls_computed=contract.live_negative_controls_computed,
        silent_terminal_truncation_allowed=(
            contract.silent_terminal_truncation_allowed
        ),
        metric_density_constraint_included=contract.metric_density_constraint_included,
        connection_equation_derived=contract.connection_equation_derived,
        metric_compatibility_derived=contract.metric_compatibility_derived,
        levi_civita_connection_derived=contract.levi_civita_connection_derived,
        first_second_order_equivalence_proved=(
            contract.first_second_order_equivalence_proved
        ),
        canonical_gauge_generator_equated_to_diffeomorphism=(
            contract.canonical_gauge_generator_equated_to_diffeomorphism
        ),
        ghy_boundary_term_used=contract.ghy_boundary_term_used,
        global_boundary_completion_proved=contract.global_boundary_completion_proved,
        scalar_full_m1_action_assembled=contract.scalar_full_m1_action_assembled,
        full_m1_bv_functional_constructed=contract.full_m1_bv_functional_constructed,
        classical_master_equation_computed=contract.classical_master_equation_computed,
        functional_measure_computed=contract.functional_measure_computed,
        quantum_master_equation_computed=contract.quantum_master_equation_computed,
        continuum_loop_st_computed=contract.continuum_loop_st_computed,
        positive_physical_hilbert_proved=contract.positive_physical_hilbert_proved,
        quantum_hda_m2_proved=contract.quantum_hda_m2_proved,
        m3_relational_observables_unlocked=contract.m3_relational_observables_unlocked,
        claim_ceiling=contract.claim_ceiling,
        derivation_status=contract.derivation_status,
        declared_m1_palatini_curvature_brst_gate_passed=passed,
    )
