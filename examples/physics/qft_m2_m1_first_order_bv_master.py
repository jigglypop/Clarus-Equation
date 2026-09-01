'''Bounded local BV master witness for the normalized first-order M1 bulk.

The module lifts the E70-J torsion-free Palatini bulk assembly to one
antifield for each of its 69 fields.  Exact four-dimensional symmetric
multi-index jets are retained through total order four.  The extra order is
required by the Euler audit of master-density terms containing second ghost
and second antifield jets; order three is deliberately not called a CME gate.

The result is a local-functional, modulo-total-divergence classical witness.
It is not a global Palatini/BV--BFV boundary theory, a BV measure, a quantum
master equation, a Slavnov--Taylor restoration, a physical Hilbert-space
construction, or quantum M2.
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
    CONTRACT_SHA256 as E70_D_HASH,
    bv_left_derivative,
    bv_right_derivative,
)
from examples.physics.qft_m2_m1_4d_densitized_scalar_bv import (
    DensitizedBVVariableSpec,
)
from examples.physics.qft_m2_m1_first_order_bulk_assembly import (
    CONTRACT_SHA256 as E70_J_HASH,
    FIELD_SPECS as BULK_FIELD_SPECS,
    evaluate_m1_first_order_bulk_assembly_gate,
    first_order_bulk_assembly_model,
    m1_first_order_bulk_assembly_contract,
    validate_contract as validate_e70_j_contract,
)


DIMENSION = 4
MAXIMUM_TOTAL_JET_ORDER = 4
RAW_BRACKET_MINIMUM_JET_ORDER = 3
PRIMARY_SOURCE = 'hep-th/0506098'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0506098'
LOCAL_FUNCTIONAL_SOURCE = 'hep-th/0002245v3'
LOCAL_FUNCTIONAL_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0002245v3'
FIRST_ORDER_BV_SOURCE = 'arXiv:1707.06328'
FIRST_ORDER_BV_SOURCE_URL = 'https://arxiv.org/abs/1707.06328'
PALATINI_DIFF_SOURCE = 'arXiv:0708.3300'
PALATINI_DIFF_SOURCE_URL = 'https://arxiv.org/abs/0708.3300'
PALATINI_BRST_SOURCE = 'hep-th/0005011'
PALATINI_BRST_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0005011'
SOURCE_ITEMS = (
    'Fuster--Henneaux--Maas Sec. 3.4 and Eqs. (4.4)--(4.7): Euler derivatives, antibracket, sF=(S,F), and CME',
    'Barnich--Brandt--Henneaux v3: local BRST cohomology and local functionals modulo total divergences',
    'Cattaneo--Schiavina: BV--BFV structure for tetrad/connection Palatini--Cartan--Holst, not this literal metric-density model',
    'Samanta and Piguet: first-order Palatini diffeomorphism/BRST structure including the affine second derivative of the ghost',
)
SOURCE_BOUNDARY = (
    'the cited first-order gravity sources justify the structural adaptation, '
    'not a literal derivation of the E70-J metric-density plus five-scalar '
    'model; torsion freedom is imposed before variation and unrestricted '
    'projective, torsion, global-boundary, and open-algebra sectors are absent'
)
NORMALIZATION = (
    'the exact distinct-prime E70-J coefficient fixture and every field, jet, '
    'antifield, and coefficient are dimensionless algebraic sentinels; they '
    'are not phenomenological parameter estimates'
)
ANTIBRACKET_CONVENTION = (
    '(F,G)=sum_A[E_R,Phi(F)E_L,Phi*(G)-E_R,Phi*(F)E_L,Phi(G)]; '
    'sF=(S,F); S1=sum_A (-1)^parity(Phi_A) Phi_A* sPhi_A in star-left ordering'
)
EULER_CONVENTION = (
    'E_z(f)=sum_|I|<=4 (-D)^I partial f/partial z_I over symmetric '
    'four-dimensional multi-indices with exact left/right derivatives and '
    'terminal-jet rejection; order four, not order three, closes the full '
    'Euler audit of the second-ghost-jet master density'
)
CLAIM_CEILING = (
    'exact normalized bounded torsion-free first-order M1 local BV witness '
    'with 69 canonical pairs, order-four Euler calculus, all base BRST maps, '
    'and the classical master density retained as an explicit local '
    'divergence; no unrestricted metric-affine/projective theorem, global '
    'Palatini or BV--BFV boundary completion, unbounded variational '
    'bicomplex, BV measure, QME, continuum ST or anomaly cancellation, '
    'positive physical Hilbert space, quantum HDA M2, or M3 unlock'
)
UPSTREAM_HASHES = (('E70-D', E70_D_HASH), ('E70-J', E70_J_HASH))
CONTRACT_SHA256 = 'TO_BE_COMPUTED'


class FirstOrderBVJetOrderExceeded(ValueError):
    '''Raised instead of silently truncating a first-order BV jet.'''


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


def first_order_bv_field_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    return tuple(BULK_FIELD_SPECS)


def first_order_bv_antifield_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    return tuple(
        DensitizedBVVariableSpec(
            f'{field.name}_star',
            f'antifield dual to {field.role}',
            (field.parity + 1) % 2,
            -field.ghost_number - 1,
            1,
            1 - field.density_weight,
        )
        for field in first_order_bv_field_specs()
    )


FIELD_SPECS = first_order_bv_field_specs()
ANTIFIELD_SPECS = first_order_bv_antifield_specs()
ALL_VARIABLE_SPECS = FIELD_SPECS + ANTIFIELD_SPECS
SPEC_BY_NAME = {spec.name: spec for spec in ALL_VARIABLE_SPECS}


def jet_name(base_name: str, multiindex: MultiIndex) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown first-order BV variable {base_name}')
    if len(multiindex) != DIMENSION or min(multiindex) < 0:
        raise ValueError('multi-index must contain four nonnegative entries')
    if sum(multiindex) > MAXIMUM_TOTAL_JET_ORDER:
        raise FirstOrderBVJetOrderExceeded(
            f'multi-index {multiindex} exceeds total order '
            f'{MAXIMUM_TOTAL_JET_ORDER}'
        )
    return f'{base_name}__{multiindex[0]}_{multiindex[1]}_{multiindex[2]}_{multiindex[3]}'


JET_LOOKUP = {
    jet_name(spec.name, multiindex): (spec, multiindex)
    for spec in ALL_VARIABLE_SPECS
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
        raise ValueError(f'unknown first-order BV variable {base_name}')
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
                raise FirstOrderBVJetOrderExceeded(
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
                raise FirstOrderBVJetOrderExceeded(
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


def divergence(
    currents: tuple[SparseSuperPolynomial, ...],
) -> SparseSuperPolynomial:
    if len(currents) != DIMENSION:
        raise ValueError('a four-current must have four components')
    return polynomial_sum(
        horizontal_derivative(current, direction)
        for direction, current in enumerate(currents)
    )


def lift_bulk_polynomial(
    polynomial: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    names = {
        name
        for even_names, odd_names in polynomial.terms
        for name in even_names + odd_names
    }
    unknown = names.difference(JET_LOOKUP)
    if unknown:
        raise ValueError(f'bulk polynomial contains unknown jets {sorted(unknown)}')
    return SparseSuperPolynomial(polynomial.terms)


def _present_multiindices(
    polynomial: SparseSuperPolynomial,
    base_name: str,
) -> tuple[MultiIndex, ...]:
    present: set[MultiIndex] = set()
    for even_names, odd_names in polynomial.terms:
        for name in even_names + odd_names:
            spec, multiindex = JET_LOOKUP[name]
            if spec.name == base_name:
                present.add(multiindex)
    return tuple(sorted(present, key=lambda item: (sum(item), item)))


def jet_partial_derivative(
    polynomial: SparseSuperPolynomial,
    base_name: str,
    multiindex: MultiIndex,
    *,
    side: str,
) -> SparseSuperPolynomial:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown first-order BV variable {base_name}')
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
    for multiindex in _present_multiindices(density, base_name):
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
    same = left is right
    cache: dict[tuple[str, str, str], SparseSuperPolynomial] = {}

    def derivative(
        density: SparseSuperPolynomial,
        label: str,
        base_name: str,
        side: str,
    ) -> SparseSuperPolynomial:
        key = ('same' if same else label, base_name, side)
        if key not in cache:
            cache[key] = euler_derivative(density, base_name, side=side)
        return cache[key]

    for field in FIELD_SPECS:
        star = f'{field.name}_star'
        result += (
            derivative(left, 'left', field.name, 'right')
            * derivative(right, 'right', star, 'left')
        )
        result += second_term_sign * (
            derivative(left, 'left', star, 'right')
            * derivative(right, 'right', field.name, 'left')
        )
    return result


def _drop_affine_second_ghost_derivatives(
    polynomial: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    terms = {}
    for key, coefficient in polynomial.terms.items():
        names = key[0] + key[1]
        has_second_ghost_jet = any(
            JET_LOOKUP[name][0].name.startswith('c')
            and sum(JET_LOOKUP[name][1]) == 2
            for name in names
        )
        if not has_second_ghost_jet:
            terms[key] = coefficient
    return SparseSuperPolynomial(terms)


@dataclass(frozen=True)
class FirstOrderBVModel:
    classical_density: SparseSuperPolynomial
    classical_boundary_current: tuple[SparseSuperPolynomial, ...]
    transformations: Mapping[str, SparseSuperPolynomial]
    antifield_density: SparseSuperPolynomial
    extended_density: SparseSuperPolynomial


def first_order_bv_model(
    *,
    parity_signed_antifields: bool = True,
    omitted_antifields: frozenset[str] = frozenset(),
    ghost_sign: int = 1,
    include_affine_second_ghost_derivatives: bool = True,
) -> FirstOrderBVModel:
    if ghost_sign not in (-1, 1):
        raise ValueError('ghost sign must be plus or minus one')
    unknown = omitted_antifields.difference(spec.name for spec in FIELD_SPECS)
    if unknown:
        raise ValueError(f'unknown omitted antifields {sorted(unknown)}')
    bulk = first_order_bulk_assembly_model()
    transformations = {
        name: lift_bulk_polynomial(image)
        for name, image in bulk.transformations.items()
    }
    if ghost_sign == -1:
        for mu in range(DIMENSION):
            name = f'c{mu}'
            transformations[name] = -transformations[name]
    if not include_affine_second_ghost_derivatives:
        for name in tuple(transformations):
            if name.startswith('Gamma'):
                transformations[name] = _drop_affine_second_ghost_derivatives(
                    transformations[name]
                )

    antifield_terms: list[SparseSuperPolynomial] = []
    for field in FIELD_SPECS:
        if field.name in omitted_antifields:
            continue
        coefficient = -1 if parity_signed_antifields and field.parity else 1
        antifield_terms.append(
            coefficient
            * generator(f'{field.name}_star')
            * transformations[field.name]
        )
    classical_density = lift_bulk_polynomial(bulk.classical_density)
    antifield_density = polynomial_sum(antifield_terms)
    return FirstOrderBVModel(
        classical_density=classical_density,
        classical_boundary_current=tuple(
            lift_bulk_polynomial(component)
            for component in bulk.boundary_current
        ),
        transformations=transformations,
        antifield_density=antifield_density,
        extended_density=classical_density + antifield_density,
    )


def apply_field_brst(
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


def master_density(
    model: FirstOrderBVModel,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    return Fraction(1, 2) * local_bv_antibracket_density(
        model.extended_density,
        model.extended_density,
        second_term_sign=second_term_sign,
    )


def derived_transformation_residuals(
    model: FirstOrderBVModel,
) -> tuple[tuple[str, SparseSuperPolynomial], ...]:
    residuals: list[tuple[str, SparseSuperPolynomial]] = []
    for field in FIELD_SPECS:
        generated = -euler_derivative(
            model.extended_density,
            f'{field.name}_star',
            side='right',
        )
        residuals.append(
            (field.name, generated - model.transformations[field.name])
        )
    return tuple(residuals)


def canonical_pair_residuals() -> tuple[SparseSuperPolynomial, ...]:
    one = SparseSuperPolynomial.scalar(1)
    samples = ('phi_chi', 'c0', 'Gamma0_01', 'ell', 'barc0', 'B0')
    residuals: list[SparseSuperPolynomial] = []
    for name in samples:
        field = generator(name)
        star = generator(f'{name}_star')
        residuals.append(local_bv_antibracket_density(field, star) - one)
        residuals.append(local_bv_antibracket_density(star, field) + one)
    return tuple(residuals)


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
    direction = next(index for index, count in enumerate(multiindex) if count)
    reduced = _subtract_multiindex(multiindex, direction)
    nested = _integration_by_parts_current(
        base_name,
        reduced,
        horizontal_derivative(coefficient, direction),
    )
    currents = [-component for component in nested]
    currents[direction] += generator(base_name, reduced) * coefficient
    return tuple(currents)


def _homogeneous_scaling_density(
    density: SparseSuperPolynomial,
) -> SparseSuperPolynomial:
    scaled_terms = {}
    for key, coefficient in density.terms.items():
        degree = len(key[0]) + len(key[1])
        if degree == 0:
            raise ValueError('the variational homotopy rejects constants')
        scaled_terms[key] = coefficient / degree
    return SparseSuperPolynomial(scaled_terms)


def variational_homotopy_current(
    density: SparseSuperPolynomial,
) -> tuple[SparseSuperPolynomial, ...]:
    '''Construct the exact scaling-homotopy current of a positive-degree density.'''

    scaled = _homogeneous_scaling_density(density)
    currents = [SparseSuperPolynomial.zero() for _ in range(DIMENSION)]
    for spec in ALL_VARIABLE_SPECS:
        for multiindex in _present_multiindices(scaled, spec.name):
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
    scaled = _homogeneous_scaling_density(density)
    return polynomial_sum(
        generator(spec.name)
        * euler_derivative(scaled, spec.name, side='left')
        for spec in ALL_VARIABLE_SPECS
    )


def nonperiodic_boundary_fixture() -> tuple[Fraction, Fraction, Fraction]:
    '''A retained endpoint-flux fixture for J^0=x^0 after de-normalization.'''

    lower = Fraction(0)
    upper = Fraction(1)
    return lower, upper, upper - lower
