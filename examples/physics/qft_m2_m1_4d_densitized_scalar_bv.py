'''Four-dimensional densitized scalar-sector local BV witness.

The module uses five normalized scalar labels (chi and four X labels), an
independent symmetric contravariant tensor density h^{mu nu}, and an
independent scalar density rho.  It implements exact symmetric multi-index
jets, horizontal derivatives, Euler derivatives, and the standard-left BV
antibracket for a polynomial 4D diffeomorphism toy.

Neither h nor rho is constrained to equal a metric-derived density.  There
is no inverse-metric/determinant relation, Einstein--Hilbert term, 4D M1
functional, global boundary completion, BV measure, QME, or quantum M2 here.
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
from examples.physics.qft_m2_m1_bv_master_admission import (
    bv_left_derivative,
    bv_right_derivative,
)
from examples.physics.qft_m2_m1_reparam_local_bv_quotient import (
    CONTRACT_SHA256 as E70_E_HASH,
    evaluate_m1_reparam_local_bv_quotient_gate,
    m1_reparam_local_bv_quotient_contract,
    validate_contract as validate_e70_e_contract,
)


DIMENSION = 4
MAXIMUM_TOTAL_JET_ORDER = 3
SCALAR_LABELS = ('chi', 'X0', 'X1', 'X2', 'X3')
PRIMARY_SOURCE = 'hep-th/0506098'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0506098'
LOCAL_FUNCTIONAL_SOURCE = 'hep-th/0002245v3'
LOCAL_FUNCTIONAL_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0002245v3'
DIFFEOMORPHISM_SOURCE = 'arXiv:2206.00780v2'
DIFFEOMORPHISM_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
NORMALIZATION = (
    'all coordinates, scalar fields, h, rho, ghosts, antifields, and '
    'coefficients are divided by declared reference scales and represented '
    'as dimensionless exact coordinates of this finite witness'
)
MODEL_RELATION = (
    'h^{mu nu} and rho are independent weight-one polynomial density '
    'variables; no h=sqrt(-g)g^{-1}, rho=sqrt(-g), determinant, inverse '
    'metric, or Einstein--Hilbert relation is imposed'
)
SOURCE_ITEMS = (
    'Fuster--Henneaux--Maas Sec. 3.4/Eqs. (3.12)--(3.13): Euler derivative and total-derivative criterion',
    'Fuster--Henneaux--Maas Eqs. (4.4)--(4.7): antibracket, sF=(S,F), and CME',
    'Barnich--Brandt--Henneaux v3: local BRST cohomology with antifields',
    'Prinz v2 Lemma 3.5/Eq. (38): a weight-one density has a total-derivative BRST variation',
)
SOURCE_BOUNDARY = (
    'the h and rho Lie-derivative formulas are convention-adapted density '
    'geometry; Prinz supplies the weight-one density identity, not this '
    'independent-h polynomial toy literally'
)
ANTIBRACKET_CONVENTION = (
    '(F,G)=integral sum_A[E_R,Phi(F) E_L,Phi*(G)-'
    'E_R,Phi*(F) E_L,Phi(G)]; sF=(S,F); '
    'S1=sum_A (-1)^parity(Phi_A) Phi_A* sPhi_A in star-left ordering'
)
EULER_CONVENTION = (
    'E_z(f)=sum_|I|<=3 (-D)^I partial f/partial z_I over symmetric '
    'four-dimensional multi-indices, with exact left/right derivatives '
    'and terminal-jet rejection'
)
CLAIM_CEILING = (
    'exact bounded four-dimensional multi-index Euler calculus and a '
    'local d_H-retained BV quotient for a polynomial toy with five scalars '
    'and independent weight-one h^{mu nu} and rho; no metric determinant '
    'relation, Einstein--Hilbert or full M1 action, global boundary '
    'completion, unbounded variational bicomplex, BV measure, QME, continuum '
    'ST, physical Hilbert space, quantum HDA M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-E', E70_E_HASH),)
CONTRACT_SHA256 = (
    '832ccc6d55f444038d14c1a91740bbd7d0ef0da63b5165888ef030055d2572f5'
)


class MultiJetOrderExceeded(ValueError):
    '''Raised instead of silently truncating a horizontal derivative.'''


MultiIndex = tuple[int, int, int, int]


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
class DensitizedBVVariableSpec:
    name: str
    role: str
    parity: int
    ghost_number: int
    antifield_number: int
    density_weight: int


def _h_name(mu: int, nu: int) -> str:
    left, right = sorted((mu, nu))
    return f'h{left}{right}'


def densitized_field_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    specs: list[DensitizedBVVariableSpec] = []
    for label in SCALAR_LABELS:
        specs.append(
            DensitizedBVVariableSpec(
                f'phi_{label}',
                'normalized scalar',
                0,
                0,
                0,
                0,
            )
        )
    for mu in range(DIMENSION):
        for nu in range(mu, DIMENSION):
            specs.append(
                DensitizedBVVariableSpec(
                    _h_name(mu, nu),
                    'symmetric contravariant rank-two density component',
                    0,
                    0,
                    0,
                    1,
                )
            )
    specs.append(
        DensitizedBVVariableSpec(
            'rho',
            'independent scalar density',
            0,
            0,
            0,
            1,
        )
    )
    for mu in range(DIMENSION):
        specs.append(
            DensitizedBVVariableSpec(
                f'c{mu}',
                'contravariant diffeomorphism ghost',
                1,
                1,
                0,
                0,
            )
        )
    for mu in range(DIMENSION):
        specs.append(
            DensitizedBVVariableSpec(
                f'barc{mu}',
                'nonminimal antighost component',
                1,
                -1,
                0,
                0,
            )
        )
    for mu in range(DIMENSION):
        specs.append(
            DensitizedBVVariableSpec(
                f'B{mu}',
                'nonminimal auxiliary component',
                0,
                0,
                0,
                0,
            )
        )
    return tuple(specs)


def densitized_antifield_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    return tuple(
        DensitizedBVVariableSpec(
            f'{field.name}_star',
            f'antifield dual to {field.role}',
            (field.parity + 1) % 2,
            -field.ghost_number - 1,
            1,
            1 - field.density_weight,
        )
        for field in densitized_field_specs()
    )


ALL_VARIABLE_SPECS = densitized_field_specs() + densitized_antifield_specs()
SPEC_BY_NAME = {spec.name: spec for spec in ALL_VARIABLE_SPECS}


def jet_name(base_name: str, multiindex: MultiIndex) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown densitized BV variable {base_name}')
    if len(multiindex) != DIMENSION or min(multiindex) < 0:
        raise ValueError('multi-index must contain four nonnegative entries')
    if sum(multiindex) > MAXIMUM_TOTAL_JET_ORDER:
        raise MultiJetOrderExceeded(
            f'multi-index {multiindex} exceeds total order {MAXIMUM_TOTAL_JET_ORDER}'
        )
    return f'{base_name}__{multiindex[0]}_{multiindex[1]}_{multiindex[2]}_{multiindex[3]}'


JET_LOOKUP = {
    jet_name(spec.name, multiindex): (spec, multiindex)
    for spec in ALL_VARIABLE_SPECS
    for multiindex in MULTIINDICES
}


ZERO_MULTIINDEX: MultiIndex = (0, 0, 0, 0)


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
                raise MultiJetOrderExceeded(
                    f'D_{direction} requires {spec.name} jet {next_index}'
                )
            remaining_even = even_names[:index] + even_names[index + 1 :]
            result += coefficient * (
                SparseSuperPolynomial.monomial(
                    even=remaining_even,
                    odd=odd_names,
                )
                * generator(spec.name, next_index)
            )
        for index, name in enumerate(odd_names):
            spec, multiindex = JET_LOOKUP[name]
            next_index = add_multiindex(multiindex, increment)
            if sum(next_index) > MAXIMUM_TOTAL_JET_ORDER:
                raise MultiJetOrderExceeded(
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


def jet_partial_derivative(
    polynomial: SparseSuperPolynomial,
    base_name: str,
    multiindex: MultiIndex,
    *,
    side: str,
) -> SparseSuperPolynomial:
    spec = SPEC_BY_NAME[base_name]
    variable = jet_name(base_name, multiindex)
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
    contributions: list[SparseSuperPolynomial] = []
    for multiindex in MULTIINDICES:
        partial = jet_partial_derivative(
            density,
            base_name,
            multiindex,
            side=side,
        )
        if partial.is_zero:
            continue
        integrated = multi_total_derivative(partial, multiindex)
        contributions.append((-1 if sum(multiindex) % 2 else 1) * integrated)
    return polynomial_sum(contributions)


def local_bv_antibracket_density(
    left: SparseSuperPolynomial,
    right: SparseSuperPolynomial,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    if second_term_sign not in (-1, 1):
        raise ValueError('second antibracket sign must be plus or minus one')
    result = SparseSuperPolynomial.zero()
    for field in densitized_field_specs():
        star = f'{field.name}_star'
        result += (
            euler_derivative(left, field.name, side='right')
            * euler_derivative(right, star, side='left')
        )
        result += second_term_sign * (
            euler_derivative(left, star, side='right')
            * euler_derivative(right, field.name, side='left')
        )
    return result


@dataclass(frozen=True)
class DensitizedScalarBVModel:
    scalar_potential: SparseSuperPolynomial
    classical_density: SparseSuperPolynomial
    classical_boundary_current: tuple[SparseSuperPolynomial, ...]
    transformations: Mapping[str, SparseSuperPolynomial]
    antifield_density: SparseSuperPolynomial
    extended_density: SparseSuperPolynomial


def densitized_scalar_bv_model(
    *,
    ghost_sign: int = 1,
    include_h_density_trace: bool = True,
    include_rho_density_trace: bool = True,
    include_second_h_index_transport: bool = True,
    include_ghost_antifield_terms: bool = True,
    parity_signed_antifields: bool = True,
    potential_coupled_to_rho: bool = True,
) -> DensitizedScalarBVModel:
    '''Construct the locked five-scalar independent-density polynomial toy.'''

    if ghost_sign not in (-1, 1):
        raise ValueError('ghost sign must be plus or minus one')
    phi_chi = generator('phi_chi')
    potential = (
        Fraction(1, 2) * phi_chi * phi_chi
        + Fraction(1, 4) * phi_chi * phi_chi * phi_chi * phi_chi
    )
    kinetic = -Fraction(1, 2) * polynomial_sum(
        generator(_h_name(mu, nu))
        * generator(f'phi_{label}', unit_multiindex(mu))
        * generator(f'phi_{label}', unit_multiindex(nu))
        for label in SCALAR_LABELS
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    potential_density = (
        generator('rho') * potential
        if potential_coupled_to_rho
        else potential
    )
    classical_density = kinetic - potential_density

    transformations: dict[str, SparseSuperPolynomial] = {}
    for label in SCALAR_LABELS:
        name = f'phi_{label}'
        transformations[name] = polynomial_sum(
            generator(f'c{rho}')
            * generator(name, unit_multiindex(rho))
            for rho in range(DIMENSION)
        )
    for mu in range(DIMENSION):
        for nu in range(mu, DIMENSION):
            name = _h_name(mu, nu)
            image = polynomial_sum(
                generator(f'c{rho}')
                * generator(name, unit_multiindex(rho))
                - generator(_h_name(rho, nu))
                * generator(f'c{mu}', unit_multiindex(rho))
                for rho in range(DIMENSION)
            )
            if include_second_h_index_transport:
                image -= polynomial_sum(
                    generator(_h_name(mu, rho))
                    * generator(f'c{nu}', unit_multiindex(rho))
                    for rho in range(DIMENSION)
                )
            if include_h_density_trace:
                image += generator(name) * polynomial_sum(
                    generator(f'c{rho}', unit_multiindex(rho))
                    for rho in range(DIMENSION)
                )
            transformations[name] = image
    rho_image = polynomial_sum(
        generator(f'c{mu}') * generator('rho', unit_multiindex(mu))
        for mu in range(DIMENSION)
    )
    if include_rho_density_trace:
        rho_image += generator('rho') * polynomial_sum(
            generator(f'c{mu}', unit_multiindex(mu))
            for mu in range(DIMENSION)
        )
    transformations['rho'] = rho_image
    for mu in range(DIMENSION):
        transformations[f'c{mu}'] = ghost_sign * polynomial_sum(
            generator(f'c{rho}')
            * generator(f'c{mu}', unit_multiindex(rho))
            for rho in range(DIMENSION)
        )
        transformations[f'barc{mu}'] = generator(f'B{mu}')
        transformations[f'B{mu}'] = SparseSuperPolynomial.zero()

    antifield_terms: list[SparseSuperPolynomial] = []
    for field in densitized_field_specs():
        if field.name.startswith('c') and not include_ghost_antifield_terms:
            continue
        coefficient = -1 if parity_signed_antifields and field.parity else 1
        antifield_terms.append(
            coefficient
            * generator(f'{field.name}_star')
            * transformations[field.name]
        )
    antifield_density = polynomial_sum(antifield_terms)
    return DensitizedScalarBVModel(
        scalar_potential=potential,
        classical_density=classical_density,
        classical_boundary_current=tuple(
            generator(f'c{mu}') * classical_density
            for mu in range(DIMENSION)
        ),
        transformations=transformations,
        antifield_density=antifield_density,
        extended_density=classical_density + antifield_density,
    )


def apply_densitized_brst(
    polynomial: SparseSuperPolynomial,
    transformations: Mapping[str, SparseSuperPolynomial],
) -> SparseSuperPolynomial:
    '''Apply the left odd evolutionary differential with exact prolongation.'''

    present_names = {
        name
        for even_names, odd_names in polynomial.terms
        for name in even_names + odd_names
    }
    images: dict[str, SparseSuperPolynomial] = {}
    for name in present_names:
        if name not in JET_LOOKUP:
            raise ValueError(f'unregistered BRST jet generator {name}')
        spec, multiindex = JET_LOOKUP[name]
        if spec.antifield_number:
            raise ValueError('field BRST prolongation does not act on antifields')
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


def polynomial_parity(polynomial: SparseSuperPolynomial) -> int:
    parities = {len(odd_names) % 2 for _, odd_names in polynomial.terms}
    if not parities:
        return 0
    if len(parities) != 1:
        raise ValueError('the polynomial is not parity homogeneous')
    return parities.pop()


def polynomial_maximum_total_jet_order(
    polynomial: SparseSuperPolynomial,
) -> int:
    maximum = 0
    for even_names, odd_names in polynomial.terms:
        for name in even_names + odd_names:
            if name not in JET_LOOKUP:
                raise ValueError(f'unregistered jet generator {name}')
            maximum = max(maximum, sum(JET_LOOKUP[name][1]))
    return maximum


def polynomial_density_weights(
    polynomial: SparseSuperPolynomial,
) -> frozenset[int]:
    return frozenset(
        sum(JET_LOOKUP[name][0].density_weight for name in even_names + odd_names)
        for even_names, odd_names in polynomial.terms
    )


def polynomial_ghost_numbers(
    polynomial: SparseSuperPolynomial,
) -> frozenset[int]:
    return frozenset(
        sum(JET_LOOKUP[name][0].ghost_number for name in even_names + odd_names)
        for even_names, odd_names in polynomial.terms
    )


def antifield_number_components(
    polynomial: SparseSuperPolynomial,
) -> Mapping[int, SparseSuperPolynomial]:
    grouped: dict[
        int,
        dict[tuple[tuple[str, ...], tuple[str, ...]], Fraction],
    ] = {}
    for key, coefficient in polynomial.terms.items():
        even_names, odd_names = key
        number = sum(
            JET_LOOKUP[name][0].antifield_number
            for name in even_names + odd_names
        )
        grouped.setdefault(number, {})[key] = coefficient
    return {
        number: SparseSuperPolynomial(terms)
        for number, terms in grouped.items()
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
        + phase
        * local_bv_antibracket_density(
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
    residual = SparseSuperPolynomial.zero()
    for left, middle, right in (
        (first, second, third),
        (second, third, first),
        (third, first, second),
    ):
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


def master_density(
    model: DensitizedScalarBVModel,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    return Fraction(1, 2) * local_bv_antibracket_density(
        model.extended_density,
        model.extended_density,
        second_term_sign=second_term_sign,
    )


def derived_transformation_mismatch(
    model: DensitizedScalarBVModel,
) -> SparseSuperPolynomial:
    return polynomial_sum(
        local_bv_antibracket_density(
            model.extended_density,
            generator(field.name),
        )
        - model.transformations[field.name]
        for field in densitized_field_specs()
    )


def locked_master_boundary_current(
    model: DensitizedScalarBVModel,
) -> tuple[SparseSuperPolynomial, ...]:
    currents: list[SparseSuperPolynomial] = []
    for mu in range(DIMENSION):
        theta_correction = polynomial_sum(
            generator(_h_name(mu, nu))
            * generator(f'phi_{label}', unit_multiindex(nu))
            * polynomial_sum(
                generator(f'c{rho}')
                * generator(f'phi_{label}', unit_multiindex(rho))
                for rho in range(DIMENSION)
            )
            for label in SCALAR_LABELS
            for nu in range(DIMENSION)
        )
        currents.append(
            generator(f'c{mu}') * model.classical_density
            + theta_correction
        )
    return tuple(currents)


def _candidate_afn1_boundary_current(
    model: DensitizedScalarBVModel,
    *,
    side: str,
    derivative_first: bool,
    parity_sign: bool,
) -> tuple[SparseSuperPolynomial, ...]:
    currents: list[SparseSuperPolynomial] = []
    for mu in range(DIMENSION):
        terms: list[SparseSuperPolynomial] = []
        for field in densitized_field_specs():
            derivative = jet_partial_derivative(
                model.antifield_density,
                field.name,
                unit_multiindex(mu),
                side=side,
            )
            image = model.transformations[field.name]
            term = derivative * image if derivative_first else image * derivative
            if parity_sign and field.parity:
                term = -term
            terms.append(term)
        currents.append(polynomial_sum(terms))
    return tuple(currents)


def _subtract_multiindex(
    multiindex: MultiIndex,
    direction: int,
) -> MultiIndex:
    if multiindex[direction] <= 0:
        raise ValueError('cannot subtract an absent multi-index direction')
    values = list(multiindex)
    values[direction] -= 1
    return tuple(values)  # type: ignore[return-value]


def _integration_by_parts_current(
    base_name: str,
    multiindex: MultiIndex,
    coefficient: SparseSuperPolynomial,
) -> tuple[SparseSuperPolynomial, ...]:
    '''Return K for z_I p = (-1)^|I| z D_I p + div K.'''

    if sum(multiindex) == 0 or coefficient.is_zero:
        return tuple(SparseSuperPolynomial.zero() for _ in range(DIMENSION))
    direction = next(
        index for index, count in enumerate(multiindex) if count
    )
    reduced = _subtract_multiindex(multiindex, direction)
    nested = _integration_by_parts_current(
        base_name,
        reduced,
        horizontal_derivative(coefficient, direction),
    )
    currents = [-component for component in nested]
    currents[direction] += generator(base_name, reduced) * coefficient
    return tuple(currents)


def variational_homotopy_current(
    density: SparseSuperPolynomial,
) -> tuple[SparseSuperPolynomial, ...]:
    '''Construct the scaling-homotopy current of a positive-degree density.

    Each monomial is first divided by its total polynomial degree.  Euler's
    homogeneous identity followed by deterministic integration by parts then
    gives density = sum_z z E_z(H density) + div(current).  In particular, a
    variationally trivial density is reconstructed as an explicit divergence.
    '''

    scaled_terms: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        Fraction,
    ] = {}
    for key, coefficient in density.terms.items():
        degree = len(key[0]) + len(key[1])
        if degree == 0:
            raise ValueError('the variational homotopy rejects constants')
        scaled_terms[key] = coefficient / degree
    scaled = SparseSuperPolynomial(scaled_terms)
    currents = [SparseSuperPolynomial.zero() for _ in range(DIMENSION)]
    for spec in ALL_VARIABLE_SPECS:
        for multiindex in MULTIINDICES:
            partial = jet_partial_derivative(
                scaled,
                spec.name,
                multiindex,
                side='left',
            )
            if partial.is_zero:
                continue
            contribution = _integration_by_parts_current(
                spec.name,
                multiindex,
                partial,
            )
            for direction in range(DIMENSION):
                currents[direction] += contribution[direction]
    return tuple(currents)


def variational_homotopy_euler_remainder(
    density: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    scaled_terms = {
        key: coefficient / (len(key[0]) + len(key[1]))
        for key, coefficient in density.terms.items()
        if len(key[0]) + len(key[1])
    }
    if len(scaled_terms) != len(density.terms):
        raise ValueError('the variational homotopy rejects constants')
    scaled = SparseSuperPolynomial(scaled_terms)
    return polynomial_sum(
        generator(spec.name)
        * euler_derivative(scaled, spec.name, side='left')
        for spec in ALL_VARIABLE_SPECS
    )


def full_master_boundary_current(
    model: DensitizedScalarBVModel,
    master: SparseSuperPolynomial | None = None,
) -> tuple[SparseSuperPolynomial, ...]:
    '''Combine the analytic AFN0 Noether current and explicit AFN1 improvement.'''

    density = master if master is not None else master_density(model)
    components = antifield_number_components(density)
    unsupported = set(components).difference((0, 1))
    if unsupported:
        raise ValueError(f'unexpected master antifield numbers {sorted(unsupported)}')
    afn1 = components.get(1, SparseSuperPolynomial.zero())
    improvement = variational_homotopy_current(afn1)
    return tuple(
        analytic + correction
        for analytic, correction in zip(
            locked_master_boundary_current(model),
            improvement,
            strict=True,
        )
    )


def _rename_polynomial_generators(
    polynomial: SparseSuperPolynomial,
    mapping: Mapping[str, str],
) -> SparseSuperPolynomial:
    result = SparseSuperPolynomial.zero()
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        term = SparseSuperPolynomial.scalar(coefficient)
        for name in even_names + odd_names:
            target = mapping.get(name, name)
            if target not in JET_LOOKUP:
                raise ValueError(f'generator permutation produced {target}')
            target_spec = JET_LOOKUP[target][0]
            term *= SparseSuperPolynomial.generator(
                target,
                odd=bool(target_spec.parity),
            )
        result += term
    return result


def scalar_label_permutation_image(
    polynomial: SparseSuperPolynomial,
    permutation: Mapping[str, str],
) -> SparseSuperPolynomial:
    if set(permutation) != set(SCALAR_LABELS) or set(permutation.values()) != set(
        SCALAR_LABELS
    ):
        raise ValueError('scalar-label permutation must be a bijection')
    names: dict[str, str] = {}
    for jet, (spec, multiindex) in JET_LOOKUP.items():
        target_base = spec.name
        for label in SCALAR_LABELS:
            field = f'phi_{label}'
            star = f'{field}_star'
            if spec.name == field:
                target_base = f'phi_{permutation[label]}'
                break
            if spec.name == star:
                target_base = f'phi_{permutation[label]}_star'
                break
        names[jet] = jet_name(target_base, multiindex)
    return _rename_polynomial_generators(polynomial, names)


def spacetime_axis_permutation_image(
    polynomial: SparseSuperPolynomial,
    permutation: tuple[int, int, int, int],
) -> SparseSuperPolynomial:
    if tuple(sorted(permutation)) != tuple(range(DIMENSION)):
        raise ValueError('axis permutation must be a bijection of 0..3')

    def rename_base(name: str) -> str:
        starred = name.endswith('_star')
        core = name[:-5] if starred else name
        if core.startswith('h') and len(core) == 3:
            target = _h_name(permutation[int(core[1])], permutation[int(core[2])])
        elif core.startswith('barc') and core[4:].isdigit():
            target = f'barc{permutation[int(core[4:])]}'
        elif core.startswith('c') and core[1:].isdigit():
            target = f'c{permutation[int(core[1:])]}'
        elif core.startswith('B') and core[1:].isdigit():
            target = f'B{permutation[int(core[1:])]}'
        else:
            target = core
        return f'{target}_star' if starred else target

    names: dict[str, str] = {}
    for jet, (spec, multiindex) in JET_LOOKUP.items():
        permuted_index = [0] * DIMENSION
        for old_direction, count in enumerate(multiindex):
            permuted_index[permutation[old_direction]] = count
        names[jet] = jet_name(
            rename_base(spec.name),
            tuple(permuted_index),  # type: ignore[arg-type]
        )
    return _rename_polynomial_generators(polynomial, names)


def nonperiodic_boundary_fixture() -> tuple[Fraction, Fraction, Fraction]:
    '''Endpoint receipt for K^0(x^0)=(x^0)^2 on the unit interval.'''

    lower = Fraction(0)
    upper = Fraction(1)
    return lower, upper, upper * upper - lower * lower


def bad_missing_density_potential(
    model: DensitizedScalarBVModel,
) -> SparseSuperPolynomial:
    return (
        model.classical_density
        + generator('rho') * model.scalar_potential
        - model.scalar_potential
    )


@dataclass(frozen=True)
class M14DDensitizedScalarBVContract:
    primary_source: str
    primary_source_url: str
    local_functional_source: str
    local_functional_source_url: str
    diffeomorphism_source: str
    diffeomorphism_source_url: str
    source_items: tuple[str, ...]
    source_boundary: str
    model_relation: str
    normalization: str
    dimension: int
    scalar_labels: tuple[str, ...]
    maximum_total_jet_order: int
    antibracket_convention: str
    euler_convention: str
    field_specs: tuple[DensitizedBVVariableSpec, ...]
    antifield_specs: tuple[DensitizedBVVariableSpec, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    four_dimensional_densitized_scalar_toy_constructed: bool
    exact_multiindex_euler_calculus_constructed: bool
    terminal_jet_rejection_enforced: bool
    local_functional_antibracket_constructed: bool
    density_and_ghost_gradings_computed: bool
    classical_density_identity_computed: bool
    base_brst_nilpotency_computed: bool
    explicit_afn0_noether_current_constructed: bool
    explicit_afn1_homotopy_current_constructed: bool
    scalar_toy_cme_mod_dh_computed: bool
    basis_permutation_covariance_sampled: bool
    nonperiodic_boundary_retained: bool
    graded_identities_sampled: bool
    silent_terminal_truncation_allowed: bool
    h_metric_determinant_relation_imposed: bool
    rho_metric_determinant_relation_imposed: bool
    einstein_hilbert_action_used: bool
    full_m1_functional_constructed: bool
    global_boundary_completion_proved: bool
    unbounded_variational_bicomplex_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_4d_densitized_scalar_bv_contract() -> M14DDensitizedScalarBVContract:
    return M14DDensitizedScalarBVContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        local_functional_source=LOCAL_FUNCTIONAL_SOURCE,
        local_functional_source_url=LOCAL_FUNCTIONAL_SOURCE_URL,
        diffeomorphism_source=DIFFEOMORPHISM_SOURCE,
        diffeomorphism_source_url=DIFFEOMORPHISM_SOURCE_URL,
        source_items=SOURCE_ITEMS,
        source_boundary=SOURCE_BOUNDARY,
        model_relation=MODEL_RELATION,
        normalization=NORMALIZATION,
        dimension=DIMENSION,
        scalar_labels=SCALAR_LABELS,
        maximum_total_jet_order=MAXIMUM_TOTAL_JET_ORDER,
        antibracket_convention=ANTIBRACKET_CONVENTION,
        euler_convention=EULER_CONVENTION,
        field_specs=densitized_field_specs(),
        antifield_specs=densitized_antifield_specs(),
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        four_dimensional_densitized_scalar_toy_constructed=True,
        exact_multiindex_euler_calculus_constructed=True,
        terminal_jet_rejection_enforced=True,
        local_functional_antibracket_constructed=True,
        density_and_ghost_gradings_computed=True,
        classical_density_identity_computed=True,
        base_brst_nilpotency_computed=True,
        explicit_afn0_noether_current_constructed=True,
        explicit_afn1_homotopy_current_constructed=True,
        scalar_toy_cme_mod_dh_computed=True,
        basis_permutation_covariance_sampled=True,
        nonperiodic_boundary_retained=True,
        graded_identities_sampled=True,
        silent_terminal_truncation_allowed=False,
        h_metric_determinant_relation_imposed=False,
        rho_metric_determinant_relation_imposed=False,
        einstein_hilbert_action_used=False,
        full_m1_functional_constructed=False,
        global_boundary_completion_proved=False,
        unbounded_variational_bicomplex_proved=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_bounded_4d_independent_density_scalar_bv_cme_mod_dh_'
            'not_metric_eh_full_m1_or_quantum_m2'
        ),
    )


def _serialize_densitized_spec(spec: DensitizedBVVariableSpec) -> str:
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


_CONTRACT_FLAG_NAMES = (
    'four_dimensional_densitized_scalar_toy_constructed',
    'exact_multiindex_euler_calculus_constructed',
    'terminal_jet_rejection_enforced',
    'local_functional_antibracket_constructed',
    'density_and_ghost_gradings_computed',
    'classical_density_identity_computed',
    'base_brst_nilpotency_computed',
    'explicit_afn0_noether_current_constructed',
    'explicit_afn1_homotopy_current_constructed',
    'scalar_toy_cme_mod_dh_computed',
    'basis_permutation_covariance_sampled',
    'nonperiodic_boundary_retained',
    'graded_identities_sampled',
    'silent_terminal_truncation_allowed',
    'h_metric_determinant_relation_imposed',
    'rho_metric_determinant_relation_imposed',
    'einstein_hilbert_action_used',
    'full_m1_functional_constructed',
    'global_boundary_completion_proved',
    'unbounded_variational_bicomplex_proved',
    'functional_measure_computed',
    'quantum_master_equation_computed',
    'continuum_loop_st_computed',
    'positive_physical_hilbert_proved',
    'quantum_hda_m2_proved',
    'm3_relational_observables_unlocked',
)


def canonical_contract_payload(
    contract: M14DDensitizedScalarBVContract,
) -> str:
    comma = chr(44)
    flags = comma.join(
        f'{name}:{getattr(contract, name)}'
        for name in _CONTRACT_FLAG_NAMES
    )
    return '|'.join(
        (
            f'primary={contract.primary_source}',
            f'primary_url={contract.primary_source_url}',
            f'local_source={contract.local_functional_source}',
            f'local_source_url={contract.local_functional_source_url}',
            f'diffeomorphism_source={contract.diffeomorphism_source}',
            f'diffeomorphism_source_url={contract.diffeomorphism_source_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_boundary={contract.source_boundary}',
            f'model_relation={contract.model_relation}',
            f'normalization={contract.normalization}',
            f'dimension={contract.dimension}',
            f'labels={comma.join(contract.scalar_labels)}',
            f'max_total_jet={contract.maximum_total_jet_order}',
            f'antibracket={contract.antibracket_convention}',
            f'euler={contract.euler_convention}',
            f'fields={comma.join(_serialize_densitized_spec(x) for x in contract.field_specs)}',
            f'antifields={comma.join(_serialize_densitized_spec(x) for x in contract.antifield_specs)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(
    contract: M14DDensitizedScalarBVContract,
) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M14DDensitizedScalarBVContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.local_functional_source == LOCAL_FUNCTIONAL_SOURCE,
        contract.local_functional_source_url == LOCAL_FUNCTIONAL_SOURCE_URL,
        contract.diffeomorphism_source == DIFFEOMORPHISM_SOURCE,
        contract.diffeomorphism_source_url == DIFFEOMORPHISM_SOURCE_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.source_boundary == SOURCE_BOUNDARY,
        contract.model_relation == MODEL_RELATION,
        contract.normalization == NORMALIZATION,
        contract.dimension == DIMENSION,
        contract.scalar_labels == SCALAR_LABELS,
        contract.maximum_total_jet_order == MAXIMUM_TOTAL_JET_ORDER,
        contract.antibracket_convention == ANTIBRACKET_CONVENTION,
        contract.euler_convention == EULER_CONVENTION,
        contract.field_specs == densitized_field_specs(),
        contract.antifield_specs == densitized_antifield_specs(),
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == (
            'exact_bounded_4d_independent_density_scalar_bv_cme_mod_dh_'
            'not_metric_eh_full_m1_or_quantum_m2'
        ),
    )
    if not all(frozen):
        raise ValueError('4D densitized scalar BV source, basis, or status lock changed')
    if len(contract.field_specs) != 28 or len(contract.antifield_specs) != 28:
        raise ValueError('the 4D densitized witness requires 28 canonical pairs')
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
            or antifield.density_weight != 1 - field.density_weight
        ):
            raise ValueError('field-antifield parity, ghost, AFN, or weight lock changed')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('4D densitized scalar BV contract hash mismatch')
    required_true = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[:13]
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[13:]
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('4D scalar-toy BV claim flags changed')


def _nonzero_euler_count(density: SparseSuperPolynomial) -> int:
    return sum(not residual.is_zero for _, residual in all_euler_residuals(density))


def _maximum_euler_term_count(density: SparseSuperPolynomial) -> int:
    return max(
        residual.term_count
        for _, residual in all_euler_residuals(density)
    )


@dataclass(frozen=True)
class M14DDensitizedScalarBVReceipt:
    contract_sha256: str
    primary_source: str
    primary_source_url: str
    local_functional_source: str
    local_functional_source_url: str
    diffeomorphism_source: str
    diffeomorphism_source_url: str
    source_items: tuple[str, ...]
    source_boundary: str
    model_relation: str
    normalization: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_e_verified: bool
    dimension: int
    scalar_count: int
    symmetric_h_component_count: int
    field_count: int
    antifield_count: int
    canonical_pair_count: int
    multiindex_count: int
    bounded_jet_generator_count: int
    bounded_even_jet_generator_count: int
    bounded_odd_jet_generator_count: int
    potential_term_count: int
    classical_density_term_count: int
    antifield_density_term_count: int
    extended_density_term_count: int
    classical_density_weights: tuple[int, ...]
    antifield_density_weights: tuple[int, ...]
    extended_density_ghost_numbers: tuple[int, ...]
    classical_variation_term_count: int
    classical_current_term_count: int
    classical_current_divergence_term_count: int
    classical_identity_mismatch_term_count: int
    base_nilpotency_component_count: int
    base_nilpotency_nonzero_component_count: int
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
    master_density_maximum_total_jet_order: int
    master_density_weights: tuple[int, ...]
    master_density_ghost_numbers: tuple[int, ...]
    master_afn0_term_count: int
    master_afn1_term_count: int
    analytic_afn0_current_term_count: int
    analytic_afn0_current_divergence_term_count: int
    analytic_afn0_mismatch_term_count: int
    homotopy_afn1_current_term_count: int
    homotopy_afn1_current_divergence_term_count: int
    homotopy_afn1_remainder_term_count: int
    homotopy_afn1_identity_mismatch_term_count: int
    homotopy_afn1_direct_mismatch_term_count: int
    full_master_current_term_count: int
    full_master_current_divergence_term_count: int
    full_master_current_mismatch_term_count: int
    master_euler_audit_count: int
    master_euler_nonzero_count: int
    master_euler_maximum_residual_term_count: int
    scalar_label_permutation_mismatch_term_count: int
    spacetime_axis_permutation_mismatch_term_count: int
    symmetric_h_transpose_name_locked: bool
    nonperiodic_boundary_lower_value: Fraction
    nonperiodic_boundary_upper_value: Fraction
    nonperiodic_boundary_endpoint_difference: Fraction
    missing_h_weight_identity_mismatch_term_count: int
    missing_h_weight_phi_euler_term_count: int
    missing_rho_weight_identity_mismatch_term_count: int
    missing_rho_weight_phi_euler_term_count: int
    missing_second_h_index_identity_mismatch_term_count: int
    missing_second_h_index_nonzero_nilpotency_component_count: int
    bad_density_potential_identity_mismatch_term_count: int
    bad_density_potential_phi_euler_term_count: int
    wrong_ghost_sign_nonzero_nilpotency_component_count: int
    wrong_ghost_sign_maximum_nilpotency_residual_term_count: int
    wrong_ghost_sign_master_phi_euler_term_count: int
    omitted_ghost_antifield_transformation_mismatch_term_count: int
    omitted_ghost_antifield_master_phi_euler_term_count: int
    uniform_plus_antifield_transformation_mismatch_term_count: int
    uniform_plus_antifield_master_phi_euler_term_count: int
    wrong_antibracket_canonical_residual_term_count: int
    wrong_antibracket_antisymmetry_residual_term_count: int
    wrong_antibracket_jacobi_residual_term_count: int
    naive_partial_vs_euler_difference_term_count: int
    terminal_jet_derivative_rejected: bool
    four_dimensional_densitized_scalar_toy_constructed: bool
    exact_multiindex_euler_calculus_constructed: bool
    terminal_jet_rejection_enforced: bool
    local_functional_antibracket_constructed: bool
    density_and_ghost_gradings_computed: bool
    classical_density_identity_computed: bool
    base_brst_nilpotency_computed: bool
    explicit_afn0_noether_current_constructed: bool
    explicit_afn1_homotopy_current_constructed: bool
    scalar_toy_cme_mod_dh_computed: bool
    basis_permutation_covariance_sampled: bool
    nonperiodic_boundary_retained: bool
    graded_identities_sampled: bool
    silent_terminal_truncation_allowed: bool
    h_metric_determinant_relation_imposed: bool
    rho_metric_determinant_relation_imposed: bool
    einstein_hilbert_action_used: bool
    full_m1_functional_constructed: bool
    global_boundary_completion_proved: bool
    unbounded_variational_bicomplex_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_4d_densitized_scalar_bv_gate_passed: bool


@lru_cache(maxsize=1)
def evaluate_m1_4d_densitized_scalar_bv_gate() -> M14DDensitizedScalarBVReceipt:
    contract = m1_4d_densitized_scalar_bv_contract()
    validate_contract(contract)
    upstream_contract = m1_reparam_local_bv_quotient_contract()
    validate_e70_e_contract(upstream_contract)
    upstream_receipt = evaluate_m1_reparam_local_bv_quotient_gate()
    upstream_verified = (
        upstream_receipt.declared_m1_reparam_local_bv_quotient_gate_passed
    )

    model = densitized_scalar_bv_model()
    classical_variation = apply_densitized_brst(
        model.classical_density,
        model.transformations,
    )
    classical_current_divergence = divergence(
        model.classical_boundary_current
    )
    classical_mismatch = classical_variation - classical_current_divergence
    nilpotency_residuals = tuple(
        apply_densitized_brst(image, model.transformations)
        for image in model.transformations.values()
    )
    derived_mismatch = derived_transformation_mismatch(model)

    phi = generator('phi_chi')
    phi_one = generator('phi_chi', unit_multiindex(0))
    phi_star = generator('phi_chi_star')
    ghost = generator('c0')
    ghost_one = generator('c0', unit_multiindex(0))
    ghost_star = generator('c0_star')
    one = SparseSuperPolynomial.scalar(1)
    canonical_field_star = local_bv_antibracket_density(phi, phi_star) - one
    canonical_star_field = local_bv_antibracket_density(phi_star, phi) + one
    two_odd = ghost * phi_star
    odd_left_right_mismatch = (
        jet_partial_derivative(
            two_odd,
            'c0',
            ZERO_MULTIINDEX,
            side='left',
        )
        + jet_partial_derivative(
            two_odd,
            'c0',
            ZERO_MULTIINDEX,
            side='right',
        )
    )
    total_derivative_fixture = horizontal_derivative(
        generator('phi_chi') * generator('h00'),
        0,
    )
    sample_first = phi_star * ghost * phi_one
    sample_second = -(ghost_star * ghost * ghost_one)
    sample_third = phi
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
    master_components = antifield_number_components(master)
    afn0 = master_components.get(0, SparseSuperPolynomial.zero())
    afn1 = master_components.get(1, SparseSuperPolynomial.zero())
    analytic_current = locked_master_boundary_current(model)
    analytic_divergence = divergence(analytic_current)
    homotopy_current = variational_homotopy_current(afn1)
    homotopy_divergence = divergence(homotopy_current)
    homotopy_remainder = variational_homotopy_euler_remainder(afn1)
    homotopy_identity_mismatch = (
        afn1 - homotopy_remainder - homotopy_divergence
    )
    full_current = tuple(
        analytic + improvement
        for analytic, improvement in zip(
            analytic_current,
            homotopy_current,
            strict=True,
        )
    )
    full_divergence = divergence(full_current)
    full_master_mismatch = master - full_divergence
    master_euler = all_euler_residuals(master)

    scalar_permutation = {
        'chi': 'chi',
        'X0': 'X3',
        'X1': 'X1',
        'X2': 'X2',
        'X3': 'X0',
    }
    scalar_permutation_mismatch = (
        scalar_label_permutation_image(
            model.extended_density,
            scalar_permutation,
        )
        - model.extended_density
    )
    axis_permutation_mismatch = (
        spacetime_axis_permutation_image(
            model.extended_density,
            (1, 0, 2, 3),
        )
        - model.extended_density
    )
    boundary_lower, boundary_upper, boundary_difference = (
        nonperiodic_boundary_fixture()
    )

    missing_h_weight = densitized_scalar_bv_model(
        include_h_density_trace=False
    )
    missing_h_weight_mismatch = (
        apply_densitized_brst(
            missing_h_weight.classical_density,
            missing_h_weight.transformations,
        )
        - divergence(missing_h_weight.classical_boundary_current)
    )
    missing_rho_weight = densitized_scalar_bv_model(
        include_rho_density_trace=False
    )
    missing_rho_weight_mismatch = (
        apply_densitized_brst(
            missing_rho_weight.classical_density,
            missing_rho_weight.transformations,
        )
        - divergence(missing_rho_weight.classical_boundary_current)
    )
    missing_second_index = densitized_scalar_bv_model(
        include_second_h_index_transport=False
    )
    missing_second_index_mismatch = (
        apply_densitized_brst(
            missing_second_index.classical_density,
            missing_second_index.transformations,
        )
        - divergence(missing_second_index.classical_boundary_current)
    )
    missing_second_index_nilpotency = tuple(
        apply_densitized_brst(image, missing_second_index.transformations)
        for image in missing_second_index.transformations.values()
    )
    bad_potential = densitized_scalar_bv_model(
        potential_coupled_to_rho=False
    )
    bad_potential_mismatch = (
        apply_densitized_brst(
            bad_potential.classical_density,
            bad_potential.transformations,
        )
        - divergence(bad_potential.classical_boundary_current)
    )
    wrong_ghost = densitized_scalar_bv_model(ghost_sign=-1)
    wrong_ghost_nilpotency = tuple(
        apply_densitized_brst(image, wrong_ghost.transformations)
        for image in wrong_ghost.transformations.values()
    )
    wrong_ghost_master = master_density(wrong_ghost)
    omitted_ghost = densitized_scalar_bv_model(
        include_ghost_antifield_terms=False
    )
    omitted_ghost_master = master_density(omitted_ghost)
    uniform_plus = densitized_scalar_bv_model(
        parity_signed_antifields=False
    )
    uniform_plus_master = master_density(uniform_plus)

    wrong_canonical = (
        local_bv_antibracket_density(
            phi_star,
            phi,
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
        'phi_chi',
        ZERO_MULTIINDEX,
        side='left',
    )
    exact_euler = euler_derivative(
        model.classical_density,
        'phi_chi',
        side='left',
    )
    terminal_rejected = False
    try:
        horizontal_derivative(
            generator('phi_chi', (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0)),
            0,
        )
    except MultiJetOrderExceeded:
        terminal_rejected = True

    total_derivative_fixture_euler_maximum = _maximum_euler_term_count(
        total_derivative_fixture
    )
    missing_h_weight_phi_euler = euler_derivative(
        missing_h_weight_mismatch,
        'phi_chi',
        side='left',
    )
    missing_rho_weight_phi_euler = euler_derivative(
        missing_rho_weight_mismatch,
        'phi_chi',
        side='left',
    )
    bad_potential_phi_euler = euler_derivative(
        bad_potential_mismatch,
        'phi_chi',
        side='left',
    )
    wrong_ghost_master_phi_euler = euler_derivative(
        wrong_ghost_master,
        'phi_chi',
        side='left',
    )
    omitted_ghost_mismatch = derived_transformation_mismatch(omitted_ghost)
    omitted_ghost_master_phi_euler = euler_derivative(
        omitted_ghost_master,
        'phi_chi',
        side='left',
    )
    uniform_plus_mismatch = derived_transformation_mismatch(uniform_plus)
    uniform_plus_master_phi_euler = euler_derivative(
        uniform_plus_master,
        'phi_chi',
        side='left',
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[13:]
    )
    field_even = sum(spec.parity == 0 for spec in contract.field_specs)
    antifield_even = sum(
        spec.parity == 0 for spec in contract.antifield_specs
    )
    passed = all(
        (
            upstream_verified,
            len(contract.field_specs) == 28,
            len(contract.antifield_specs) == 28,
            len(MULTIINDICES) == 35,
            len(JET_LOOKUP) == 1960,
            (field_even + antifield_even) * len(MULTIINDICES) == 980,
            model.scalar_potential.term_count == 2,
            model.classical_density.term_count == 52,
            model.antifield_density.term_count == 164,
            polynomial_density_weights(model.classical_density) == frozenset((1,)),
            polynomial_density_weights(model.antifield_density) == frozenset((1,)),
            polynomial_ghost_numbers(model.extended_density) == frozenset((0,)),
            classical_mismatch.is_zero,
            not all(current.is_zero for current in model.classical_boundary_current),
            max(x.term_count for x in nilpotency_residuals) == 0,
            derived_mismatch.is_zero,
            canonical_field_star.is_zero,
            canonical_star_field.is_zero,
            odd_left_right_mismatch.is_zero,
            total_derivative_fixture_euler_maximum == 0,
            antisymmetry_residual.is_zero,
            sum(not x.is_zero for x in nested_brackets) == 2,
            jacobi_residual.is_zero,
            master.term_count == 2604,
            polynomial_density_weights(master) == frozenset((1,)),
            polynomial_ghost_numbers(master) == frozenset((1,)),
            afn0.term_count == 984,
            afn1.term_count == 1620,
            (afn0 - analytic_divergence).is_zero,
            homotopy_remainder.is_zero,
            homotopy_identity_mismatch.is_zero,
            (afn1 - homotopy_divergence).is_zero,
            full_master_mismatch.is_zero,
            max(x.term_count for _, x in master_euler) == 0,
            scalar_permutation_mismatch.is_zero,
            axis_permutation_mismatch.is_zero,
            _h_name(1, 0) == _h_name(0, 1),
            boundary_difference == 1,
            missing_h_weight_mismatch.term_count > 0,
            missing_h_weight_phi_euler.term_count > 0,
            missing_rho_weight_mismatch.term_count > 0,
            missing_rho_weight_phi_euler.term_count > 0,
            missing_second_index_mismatch.term_count > 0,
            sum(not x.is_zero for x in missing_second_index_nilpotency) > 0,
            bad_potential_mismatch.term_count > 0,
            bad_potential_phi_euler.term_count > 0,
            sum(not x.is_zero for x in wrong_ghost_nilpotency) > 0,
            wrong_ghost_master_phi_euler.term_count > 0,
            omitted_ghost_mismatch.term_count > 0,
            omitted_ghost_master_phi_euler.term_count > 0,
            uniform_plus_mismatch.term_count > 0,
            uniform_plus_master_phi_euler.term_count > 0,
            wrong_canonical.term_count > 0,
            wrong_antisymmetry.term_count > 0,
            wrong_jacobi.term_count > 0,
            (naive_partial - exact_euler).term_count > 0,
            terminal_rejected,
            not any(unsupported),
        )
    )
    return M14DDensitizedScalarBVReceipt(
        contract_sha256=contract.contract_sha256,
        primary_source=contract.primary_source,
        primary_source_url=contract.primary_source_url,
        local_functional_source=contract.local_functional_source,
        local_functional_source_url=contract.local_functional_source_url,
        diffeomorphism_source=contract.diffeomorphism_source,
        diffeomorphism_source_url=contract.diffeomorphism_source_url,
        source_items=contract.source_items,
        source_boundary=contract.source_boundary,
        model_relation=contract.model_relation,
        normalization=contract.normalization,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_e_verified=upstream_verified,
        dimension=contract.dimension,
        scalar_count=len(contract.scalar_labels),
        symmetric_h_component_count=DIMENSION * (DIMENSION + 1) // 2,
        field_count=len(contract.field_specs),
        antifield_count=len(contract.antifield_specs),
        canonical_pair_count=len(contract.field_specs),
        multiindex_count=len(MULTIINDICES),
        bounded_jet_generator_count=len(JET_LOOKUP),
        bounded_even_jet_generator_count=(
            (field_even + antifield_even) * len(MULTIINDICES)
        ),
        bounded_odd_jet_generator_count=(
            len(JET_LOOKUP)
            - (field_even + antifield_even) * len(MULTIINDICES)
        ),
        potential_term_count=model.scalar_potential.term_count,
        classical_density_term_count=model.classical_density.term_count,
        antifield_density_term_count=model.antifield_density.term_count,
        extended_density_term_count=model.extended_density.term_count,
        classical_density_weights=tuple(
            sorted(polynomial_density_weights(model.classical_density))
        ),
        antifield_density_weights=tuple(
            sorted(polynomial_density_weights(model.antifield_density))
        ),
        extended_density_ghost_numbers=tuple(
            sorted(polynomial_ghost_numbers(model.extended_density))
        ),
        classical_variation_term_count=classical_variation.term_count,
        classical_current_term_count=sum(
            current.term_count for current in model.classical_boundary_current
        ),
        classical_current_divergence_term_count=(
            classical_current_divergence.term_count
        ),
        classical_identity_mismatch_term_count=classical_mismatch.term_count,
        base_nilpotency_component_count=len(nilpotency_residuals),
        base_nilpotency_nonzero_component_count=sum(
            not residual.is_zero for residual in nilpotency_residuals
        ),
        base_nilpotency_maximum_residual_term_count=max(
            residual.term_count for residual in nilpotency_residuals
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
            total_derivative_fixture_euler_maximum
        ),
        graded_antisymmetry_residual_term_count=antisymmetry_residual.term_count,
        jacobi_nonzero_nested_bracket_count=sum(
            not bracket.is_zero for bracket in nested_brackets
        ),
        graded_jacobi_residual_term_count=jacobi_residual.term_count,
        master_density_term_count=master.term_count,
        master_density_maximum_total_jet_order=(
            polynomial_maximum_total_jet_order(master)
        ),
        master_density_weights=tuple(
            sorted(polynomial_density_weights(master))
        ),
        master_density_ghost_numbers=tuple(
            sorted(polynomial_ghost_numbers(master))
        ),
        master_afn0_term_count=afn0.term_count,
        master_afn1_term_count=afn1.term_count,
        analytic_afn0_current_term_count=sum(
            current.term_count for current in analytic_current
        ),
        analytic_afn0_current_divergence_term_count=(
            analytic_divergence.term_count
        ),
        analytic_afn0_mismatch_term_count=(afn0 - analytic_divergence).term_count,
        homotopy_afn1_current_term_count=sum(
            current.term_count for current in homotopy_current
        ),
        homotopy_afn1_current_divergence_term_count=(
            homotopy_divergence.term_count
        ),
        homotopy_afn1_remainder_term_count=homotopy_remainder.term_count,
        homotopy_afn1_identity_mismatch_term_count=(
            homotopy_identity_mismatch.term_count
        ),
        homotopy_afn1_direct_mismatch_term_count=(
            afn1 - homotopy_divergence
        ).term_count,
        full_master_current_term_count=sum(
            current.term_count for current in full_current
        ),
        full_master_current_divergence_term_count=full_divergence.term_count,
        full_master_current_mismatch_term_count=full_master_mismatch.term_count,
        master_euler_audit_count=len(master_euler),
        master_euler_nonzero_count=sum(
            not residual.is_zero for _, residual in master_euler
        ),
        master_euler_maximum_residual_term_count=max(
            residual.term_count for _, residual in master_euler
        ),
        scalar_label_permutation_mismatch_term_count=(
            scalar_permutation_mismatch.term_count
        ),
        spacetime_axis_permutation_mismatch_term_count=(
            axis_permutation_mismatch.term_count
        ),
        symmetric_h_transpose_name_locked=_h_name(1, 0) == _h_name(0, 1),
        nonperiodic_boundary_lower_value=boundary_lower,
        nonperiodic_boundary_upper_value=boundary_upper,
        nonperiodic_boundary_endpoint_difference=boundary_difference,
        missing_h_weight_identity_mismatch_term_count=(
            missing_h_weight_mismatch.term_count
        ),
        missing_h_weight_phi_euler_term_count=(
            missing_h_weight_phi_euler.term_count
        ),
        missing_rho_weight_identity_mismatch_term_count=(
            missing_rho_weight_mismatch.term_count
        ),
        missing_rho_weight_phi_euler_term_count=(
            missing_rho_weight_phi_euler.term_count
        ),
        missing_second_h_index_identity_mismatch_term_count=(
            missing_second_index_mismatch.term_count
        ),
        missing_second_h_index_nonzero_nilpotency_component_count=sum(
            not residual.is_zero
            for residual in missing_second_index_nilpotency
        ),
        bad_density_potential_identity_mismatch_term_count=(
            bad_potential_mismatch.term_count
        ),
        bad_density_potential_phi_euler_term_count=(
            bad_potential_phi_euler.term_count
        ),
        wrong_ghost_sign_nonzero_nilpotency_component_count=sum(
            not residual.is_zero for residual in wrong_ghost_nilpotency
        ),
        wrong_ghost_sign_maximum_nilpotency_residual_term_count=max(
            residual.term_count for residual in wrong_ghost_nilpotency
        ),
        wrong_ghost_sign_master_phi_euler_term_count=(
            wrong_ghost_master_phi_euler.term_count
        ),
        omitted_ghost_antifield_transformation_mismatch_term_count=(
            omitted_ghost_mismatch.term_count
        ),
        omitted_ghost_antifield_master_phi_euler_term_count=(
            omitted_ghost_master_phi_euler.term_count
        ),
        uniform_plus_antifield_transformation_mismatch_term_count=(
            uniform_plus_mismatch.term_count
        ),
        uniform_plus_antifield_master_phi_euler_term_count=(
            uniform_plus_master_phi_euler.term_count
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
        four_dimensional_densitized_scalar_toy_constructed=(
            contract.four_dimensional_densitized_scalar_toy_constructed
        ),
        exact_multiindex_euler_calculus_constructed=(
            contract.exact_multiindex_euler_calculus_constructed
        ),
        terminal_jet_rejection_enforced=contract.terminal_jet_rejection_enforced,
        local_functional_antibracket_constructed=(
            contract.local_functional_antibracket_constructed
        ),
        density_and_ghost_gradings_computed=(
            contract.density_and_ghost_gradings_computed
        ),
        classical_density_identity_computed=(
            contract.classical_density_identity_computed
        ),
        base_brst_nilpotency_computed=contract.base_brst_nilpotency_computed,
        explicit_afn0_noether_current_constructed=(
            contract.explicit_afn0_noether_current_constructed
        ),
        explicit_afn1_homotopy_current_constructed=(
            contract.explicit_afn1_homotopy_current_constructed
        ),
        scalar_toy_cme_mod_dh_computed=contract.scalar_toy_cme_mod_dh_computed,
        basis_permutation_covariance_sampled=(
            contract.basis_permutation_covariance_sampled
        ),
        nonperiodic_boundary_retained=contract.nonperiodic_boundary_retained,
        graded_identities_sampled=contract.graded_identities_sampled,
        silent_terminal_truncation_allowed=(
            contract.silent_terminal_truncation_allowed
        ),
        h_metric_determinant_relation_imposed=(
            contract.h_metric_determinant_relation_imposed
        ),
        rho_metric_determinant_relation_imposed=(
            contract.rho_metric_determinant_relation_imposed
        ),
        einstein_hilbert_action_used=contract.einstein_hilbert_action_used,
        full_m1_functional_constructed=contract.full_m1_functional_constructed,
        global_boundary_completion_proved=(
            contract.global_boundary_completion_proved
        ),
        unbounded_variational_bicomplex_proved=(
            contract.unbounded_variational_bicomplex_proved
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
        declared_m1_4d_densitized_scalar_bv_gate_passed=passed,
    )
