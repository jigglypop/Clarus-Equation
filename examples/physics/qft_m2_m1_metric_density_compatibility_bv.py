'''Metric-density compatibility bridge above the E70-F 4D scalar BV toy.

The module adjoins an even scalar-density multiplier ell of weight -1 and
tests the four-dimensional polynomial constraint

    C = det(h^{mu nu}) + rho^2 = 0.

On a nondegenerate positive-rho Lorentzian patch this is the determinant
compatibility condition for h^{mu nu}=sqrt(-g) g^{mu nu} and rho=sqrt(-g).
It does not introduce curvature, the Einstein--Hilbert action, a global
metric reconstruction, a BV measure, a QME, or quantum M2 evidence.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import permutations, product
from typing import Mapping

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
    polynomial_sum,
)
from examples.physics.qft_m2_m1_bv_master_admission import (
    bv_left_derivative,
    bv_right_derivative,
)
from examples.physics.qft_m2_m1_4d_densitized_scalar_bv import (
    CONTRACT_SHA256 as E70_F_HASH,
    DIMENSION,
    SCALAR_LABELS,
    DensitizedBVVariableSpec,
    _h_name,
    densitized_field_specs,
    evaluate_m1_4d_densitized_scalar_bv_gate,
    m1_4d_densitized_scalar_bv_contract,
    validate_contract as validate_e70_f_contract,
)


MAXIMUM_TOTAL_JET_ORDER = 3
PRIMARY_SOURCE = 'hep-th/0506098'
PRIMARY_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0506098'
LOCAL_FUNCTIONAL_SOURCE = 'hep-th/0002245v3'
LOCAL_FUNCTIONAL_SOURCE_URL = 'https://arxiv.org/abs/hep-th/0002245v3'
DENSITY_SOURCE = 'arXiv:2206.00780v2'
DENSITY_SOURCE_URL = 'https://arxiv.org/abs/2206.00780v2'
METRIC_DENSITY_PRECEDENT = 'hep-th/0501204'
METRIC_DENSITY_PRECEDENT_URL = 'https://arxiv.org/abs/hep-th/0501204'
PARAMETRIZATION_SOURCE = 'arXiv:1605.00454'
PARAMETRIZATION_SOURCE_URL = 'https://arxiv.org/abs/1605.00454'
SOURCE_BOUNDARY = (
    'hep-th/0501204 is a two-dimensional metric-density precedent and '
    'arXiv:1605.00454 supports densitized metric parametrizations; neither '
    'literally supplies the four-dimensional determinant identity, which is '
    'derived algebraically from h=rho g^{-1} and rho=sqrt(-det g)'
)
NORMALIZATION = (
    'h, rho, ell, all fields, ghosts, antifields, coordinates, and '
    'coefficients are normalized to dimensionless exact polynomial '
    'coordinates; this is not a physical measure or renormalization choice'
)
COMPATIBILITY_RELATION = (
    'C=det(h)+rho^2 has scalar-density weight two in four dimensions; '
    'ell has weight minus one so ell*C is a weight-one action density'
)
UPSTREAM_HASHES = (('E70-F', E70_F_HASH),)
SOURCE_ITEMS = (
    'Fuster--Henneaux--Maas Sec. 3.4 and Eqs. (4.4)--(4.7): Euler quotient and standard-left BV convention',
    'Barnich--Brandt--Henneaux v3: local BRST cohomology with antifields',
    'Prinz v2 Lemma 3.5/Eq. (38): scalar-density BRST weight identity',
    'hep-th/0501204: two-dimensional metric-density canonical precedent only',
    'arXiv:1605.00454: densitized metric parametrization precedent only',
)
ANTIBRACKET_CONVENTION = (
    '(F,G)=integral sum_A[E_R,Phi(F) E_L,Phi*(G)-'
    'E_R,Phi*(F) E_L,Phi(G)]; sF=(S,F); '
    'star-left coefficient is (-1)^parity(Phi)'
)
CLAIM_CEILING = (
    'bounded four-dimensional polynomial metric-density determinant '
    'compatibility and conditional rho-positive local metric reconstruction, '
    'with a weight-minus-one multiplier local BV quotient and explicit '
    'retained currents; no curvature, Einstein--Hilbert or GHY term, global '
    'metric patching, time orientation, full M1 functional/CME, BV measure, '
    'QME, continuum ST, physical Hilbert, quantum HDA M2, or M3 unlock'
)
CONTRACT_SHA256 = (
    '6dcb7db9da4c85c6f67d06c566d50352fa76ea52151b62f1f5f41c87c4e5dc9b'
)


class CompatibilityJetOrderExceeded(ValueError):
    '''Raised instead of silently truncating a compatibility jet.'''


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


def compatibility_field_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    return densitized_field_specs() + (
        DensitizedBVVariableSpec(
            'ell',
            'weight-minus-one metric-density compatibility multiplier',
            0,
            0,
            0,
            -1,
        ),
    )


def compatibility_antifield_specs() -> tuple[DensitizedBVVariableSpec, ...]:
    return tuple(
        DensitizedBVVariableSpec(
            f'{field.name}_star',
            f'antifield dual to {field.role}',
            (field.parity + 1) % 2,
            -field.ghost_number - 1,
            1,
            1 - field.density_weight,
        )
        for field in compatibility_field_specs()
    )


ALL_VARIABLE_SPECS = compatibility_field_specs() + compatibility_antifield_specs()
SPEC_BY_NAME = {spec.name: spec for spec in ALL_VARIABLE_SPECS}


def jet_name(base_name: str, multiindex: MultiIndex) -> str:
    if base_name not in SPEC_BY_NAME:
        raise ValueError(f'unknown compatibility BV variable {base_name}')
    if len(multiindex) != DIMENSION or min(multiindex) < 0:
        raise ValueError('multi-index must contain four nonnegative entries')
    if sum(multiindex) > MAXIMUM_TOTAL_JET_ORDER:
        raise CompatibilityJetOrderExceeded(
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
                raise CompatibilityJetOrderExceeded(
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
                raise CompatibilityJetOrderExceeded(
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
    for field in compatibility_field_specs():
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


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _polynomial_product(
    factors: tuple[SparseSuperPolynomial, ...],
) -> SparseSuperPolynomial:
    result = SparseSuperPolynomial.scalar(1)
    for factor in factors:
        result *= factor
    return result


def matrix_determinant(
    matrix: tuple[tuple[SparseSuperPolynomial, ...], ...],
) -> SparseSuperPolynomial:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError('determinant requires a nonempty square matrix')
    return polynomial_sum(
        _permutation_sign(permutation)
        * _polynomial_product(
            tuple(matrix[row][permutation[row]] for row in range(size))
        )
        for permutation in permutations(range(size))
    )


def h_matrix() -> tuple[tuple[SparseSuperPolynomial, ...], ...]:
    return tuple(
        tuple(generator(_h_name(mu, nu)) for nu in range(DIMENSION))
        for mu in range(DIMENSION)
    )


def h_determinant() -> SparseSuperPolynomial:
    return matrix_determinant(h_matrix())


def h_adjugate() -> tuple[tuple[SparseSuperPolynomial, ...], ...]:
    matrix = h_matrix()
    adjugate: list[tuple[SparseSuperPolynomial, ...]] = []
    for row in range(DIMENSION):
        values: list[SparseSuperPolynomial] = []
        for column in range(DIMENSION):
            minor = tuple(
                tuple(
                    matrix[source_row][source_column]
                    for source_column in range(DIMENSION)
                    if source_column != row
                )
                for source_row in range(DIMENSION)
                if source_row != column
            )
            values.append(
                (-1 if (row + column) % 2 else 1)
                * matrix_determinant(minor)
            )
        adjugate.append(tuple(values))
    return tuple(adjugate)


def adjugate_identity_residuals() -> tuple[SparseSuperPolynomial, ...]:
    matrix = h_matrix()
    adjugate = h_adjugate()
    determinant = h_determinant()
    return tuple(
        polynomial_sum(
            matrix[mu][rho] * adjugate[rho][nu]
            for rho in range(DIMENSION)
        )
        - (determinant if mu == nu else SparseSuperPolynomial.zero())
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )


@dataclass(frozen=True)
class MetricDensityCompatibilityBVModel:
    scalar_potential: SparseSuperPolynomial
    scalar_density: SparseSuperPolynomial
    determinant_density: SparseSuperPolynomial
    compatibility_constraint: SparseSuperPolynomial
    compatibility_density: SparseSuperPolynomial
    classical_density: SparseSuperPolynomial
    classical_boundary_current: tuple[SparseSuperPolynomial, ...]
    transformations: Mapping[str, SparseSuperPolynomial]
    antifield_density: SparseSuperPolynomial
    extended_density: SparseSuperPolynomial


def metric_density_compatibility_bv_model(
    *,
    determinant_rho_sign: int = 1,
    ghost_sign: int = 1,
    h_density_weight: int = 1,
    rho_density_weight: int = 1,
    ell_density_weight: int = -1,
    include_second_h_index_transport: bool = True,
    include_ghost_antifield_terms: bool = True,
    include_ell_antifield_term: bool = True,
    parity_signed_antifields: bool = True,
) -> MetricDensityCompatibilityBVModel:
    if determinant_rho_sign not in (-1, 1):
        raise ValueError('determinant-rho sign must be plus or minus one')
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
    scalar_density = kinetic - generator('rho') * potential
    determinant = h_determinant()
    constraint = (
        determinant
        + determinant_rho_sign * generator('rho') * generator('rho')
    )
    compatibility_density = generator('ell') * constraint
    classical_density = scalar_density + compatibility_density

    transformations: dict[str, SparseSuperPolynomial] = {}
    divergence_ghost = polynomial_sum(
        generator(f'c{rho}', unit_multiindex(rho))
        for rho in range(DIMENSION)
    )
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
            image += h_density_weight * generator(name) * divergence_ghost
            transformations[name] = image
    transformations['rho'] = polynomial_sum(
        generator(f'c{mu}') * generator('rho', unit_multiindex(mu))
        for mu in range(DIMENSION)
    ) + rho_density_weight * generator('rho') * divergence_ghost
    for mu in range(DIMENSION):
        transformations[f'c{mu}'] = ghost_sign * polynomial_sum(
            generator(f'c{rho}')
            * generator(f'c{mu}', unit_multiindex(rho))
            for rho in range(DIMENSION)
        )
        transformations[f'barc{mu}'] = generator(f'B{mu}')
        transformations[f'B{mu}'] = SparseSuperPolynomial.zero()
    transformations['ell'] = polynomial_sum(
        generator(f'c{mu}') * generator('ell', unit_multiindex(mu))
        for mu in range(DIMENSION)
    ) + ell_density_weight * generator('ell') * divergence_ghost

    antifield_terms: list[SparseSuperPolynomial] = []
    for field in compatibility_field_specs():
        if field.name.startswith('c') and not include_ghost_antifield_terms:
            continue
        if field.name == 'ell' and not include_ell_antifield_term:
            continue
        coefficient = -1 if parity_signed_antifields and field.parity else 1
        antifield_terms.append(
            coefficient
            * generator(f'{field.name}_star')
            * transformations[field.name]
        )
    antifield_density = polynomial_sum(antifield_terms)
    return MetricDensityCompatibilityBVModel(
        scalar_potential=potential,
        scalar_density=scalar_density,
        determinant_density=determinant,
        compatibility_constraint=constraint,
        compatibility_density=compatibility_density,
        classical_density=classical_density,
        classical_boundary_current=tuple(
            generator(f'c{mu}') * classical_density
            for mu in range(DIMENSION)
        ),
        transformations=transformations,
        antifield_density=antifield_density,
        extended_density=classical_density + antifield_density,
    )


def apply_compatibility_brst(
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
            raise ValueError(f'unregistered compatibility jet generator {name}')
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


def density_transformation_target(
    density: SparseSuperPolynomial,
    weight: int,
) -> SparseSuperPolynomial:
    transport = polynomial_sum(
        generator(f'c{mu}') * horizontal_derivative(density, mu)
        for mu in range(DIMENSION)
    )
    trace = polynomial_sum(
        generator(f'c{mu}', unit_multiindex(mu))
        for mu in range(DIMENSION)
    )
    return transport + weight * trace * density


def polynomial_parity(polynomial: SparseSuperPolynomial) -> int:
    parities = {len(odd_names) % 2 for _, odd_names in polynomial.terms}
    if not parities:
        return 0
    if len(parities) != 1:
        raise ValueError('the polynomial is not parity homogeneous')
    return parities.pop()


def component_factor_weight_sums(
    polynomial: SparseSuperPolynomial,
) -> frozenset[int]:
    '''Return raw field-factor weights, not tensor-contraction density weights.

    In particular det(h) contains two implicit alternating tensors whose
    Jacobian contribution lowers the geometric weight from four to two.  Its
    true weight is therefore certified by the explicit BRST covariance
    residual rather than by this diagnostic sum.
    '''

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


def polynomial_maximum_total_jet_order(
    polynomial: SparseSuperPolynomial,
) -> int:
    return max(
        (
            sum(JET_LOOKUP[name][1])
            for even_names, odd_names in polynomial.terms
            for name in even_names + odd_names
        ),
        default=0,
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
    model: MetricDensityCompatibilityBVModel,
    *,
    second_term_sign: int = -1,
) -> SparseSuperPolynomial:
    return Fraction(1, 2) * local_bv_antibracket_density(
        model.extended_density,
        model.extended_density,
        second_term_sign=second_term_sign,
    )


def derived_transformation_mismatch(
    model: MetricDensityCompatibilityBVModel,
) -> SparseSuperPolynomial:
    return polynomial_sum(
        local_bv_antibracket_density(
            model.extended_density,
            generator(field.name),
        )
        - model.transformations[field.name]
        for field in compatibility_field_specs()
    )


def analytic_afn0_master_current(
    model: MetricDensityCompatibilityBVModel,
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


def variational_homotopy_current(
    density: SparseSuperPolynomial,
) -> tuple[SparseSuperPolynomial, ...]:
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


NumericMatrix = tuple[tuple[Fraction, ...], ...]


def numeric_matrix_determinant(matrix: NumericMatrix) -> Fraction:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError('numeric determinant requires a square matrix')
    return sum(
        Fraction(_permutation_sign(permutation))
        * _numeric_product(
            tuple(matrix[row][permutation[row]] for row in range(size))
        )
        for permutation in permutations(range(size))
    )


def _numeric_product(values: tuple[Fraction, ...]) -> Fraction:
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def numeric_matrix_inverse(matrix: NumericMatrix) -> NumericMatrix:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError('numeric inverse requires a square matrix')
    augmented = [
        list(row)
        + [Fraction(1 if row_index == column else 0) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError('numeric matrix is singular')
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(
                        augmented[row],
                        augmented[column],
                        strict=True,
                    )
                ]
    return tuple(
        tuple(row[size:])
        for row in augmented
    )


def numeric_matrix_product(left: NumericMatrix, right: NumericMatrix) -> NumericMatrix:
    size = len(left)
    if (
        size == 0
        or any(len(row) != size for row in left)
        or len(right) != size
        or any(len(row) != size for row in right)
    ):
        raise ValueError('numeric product requires equal square matrices')
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


@dataclass(frozen=True)
class MetricPatchReconstructionReceipt:
    rho: Fraction
    h_determinant: Fraction
    compatibility_residual: Fraction
    g_contravariant: NumericMatrix
    g_covariant: NumericMatrix
    inverse_product_maximum_residual: Fraction
    g_covariant_determinant: Fraction
    g_covariant_determinant_residual: Fraction
    g_contravariant_determinant: Fraction
    g_contravariant_determinant_residual: Fraction
    positive_density_orientation_branch: bool
    real_symmetric_nondegenerate_lorentzian_inertia: bool
    time_orientation_selected: bool
    global_patch_reconstruction_proved: bool


def reconstruct_metric_patch(
    h_values: NumericMatrix,
    rho_value: Fraction | int,
) -> MetricPatchReconstructionReceipt:
    rho = Fraction(rho_value)
    if len(h_values) != DIMENSION or any(
        len(row) != DIMENSION for row in h_values
    ):
        raise ValueError('metric-density patch requires a 4 by 4 matrix')
    h = tuple(tuple(Fraction(value) for value in row) for row in h_values)
    if any(h[mu][nu] != h[nu][mu] for mu in range(DIMENSION) for nu in range(DIMENSION)):
        raise ValueError('metric-density patch requires symmetric h')
    if rho <= 0:
        raise ValueError('metric reconstruction requires the rho>0 orientation patch')
    determinant = numeric_matrix_determinant(h)
    compatibility_residual = determinant + rho * rho
    if compatibility_residual:
        raise ValueError('metric-density determinant compatibility failed')
    inverse_h = numeric_matrix_inverse(h)
    g_contravariant = tuple(
        tuple(value / rho for value in row)
        for row in h
    )
    g_covariant = tuple(
        tuple(rho * value for value in row)
        for row in inverse_h
    )
    product_matrix = numeric_matrix_product(g_contravariant, g_covariant)
    inverse_residual = max(
        abs(product_matrix[mu][nu] - Fraction(1 if mu == nu else 0))
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    covariant_determinant = numeric_matrix_determinant(g_covariant)
    contravariant_determinant = numeric_matrix_determinant(g_contravariant)
    return MetricPatchReconstructionReceipt(
        rho=rho,
        h_determinant=determinant,
        compatibility_residual=compatibility_residual,
        g_contravariant=g_contravariant,
        g_covariant=g_covariant,
        inverse_product_maximum_residual=inverse_residual,
        g_covariant_determinant=covariant_determinant,
        g_covariant_determinant_residual=(
            covariant_determinant + rho * rho
        ),
        g_contravariant_determinant=contravariant_determinant,
        g_contravariant_determinant_residual=(
            contravariant_determinant + Fraction(1, 1) / (rho * rho)
        ),
        positive_density_orientation_branch=True,
        real_symmetric_nondegenerate_lorentzian_inertia=determinant < 0,
        time_orientation_selected=False,
        global_patch_reconstruction_proved=False,
    )


def evaluate_even_base_polynomial(
    polynomial: SparseSuperPolynomial,
    values: Mapping[str, Fraction | int],
) -> Fraction:
    result = Fraction(0)
    for (even_names, odd_names), coefficient in polynomial.terms.items():
        if odd_names:
            raise ValueError('numeric fixture rejects odd generators')
        term = coefficient
        for jet in even_names:
            spec, multiindex = JET_LOOKUP[jet]
            if multiindex != ZERO_MULTIINDEX or spec.name not in values:
                raise ValueError(f'missing base-only numeric value for {jet}')
            term *= Fraction(values[spec.name])
        result += term
    return result


@dataclass(frozen=True)
class M1MetricDensityCompatibilityBVContract:
    primary_source: str
    primary_source_url: str
    local_functional_source: str
    local_functional_source_url: str
    density_source: str
    density_source_url: str
    metric_density_precedent: str
    metric_density_precedent_url: str
    parametrization_source: str
    parametrization_source_url: str
    source_items: tuple[str, ...]
    source_boundary: str
    normalization: str
    compatibility_relation: str
    dimension: int
    scalar_labels: tuple[str, ...]
    maximum_total_jet_order: int
    antibracket_convention: str
    field_specs: tuple[DensitizedBVVariableSpec, ...]
    antifield_specs: tuple[DensitizedBVVariableSpec, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    claim_ceiling: str
    contract_sha256: str
    determinant_polynomial_constructed: bool
    adjugate_identity_computed: bool
    determinant_weight_two_covariance_computed: bool
    compatibility_ideal_brst_stable: bool
    weight_minus_one_multiplier_constructed: bool
    conditional_positive_rho_metric_reconstruction_computed: bool
    bounded_local_bv_quotient_constructed: bool
    explicit_afn0_and_afn1_currents_constructed: bool
    compatibility_cme_mod_dh_computed: bool
    live_negative_controls_computed: bool
    silent_terminal_truncation_allowed: bool
    rho_zero_patch_allowed: bool
    negative_rho_orientation_branch_admitted: bool
    time_orientation_selected: bool
    global_metric_reconstruction_proved: bool
    curvature_tensor_constructed: bool
    einstein_hilbert_action_used: bool
    ghy_boundary_term_used: bool
    full_m1_functional_constructed: bool
    global_boundary_completion_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    derivation_status: str


def m1_metric_density_compatibility_bv_contract(
) -> M1MetricDensityCompatibilityBVContract:
    return M1MetricDensityCompatibilityBVContract(
        primary_source=PRIMARY_SOURCE,
        primary_source_url=PRIMARY_SOURCE_URL,
        local_functional_source=LOCAL_FUNCTIONAL_SOURCE,
        local_functional_source_url=LOCAL_FUNCTIONAL_SOURCE_URL,
        density_source=DENSITY_SOURCE,
        density_source_url=DENSITY_SOURCE_URL,
        metric_density_precedent=METRIC_DENSITY_PRECEDENT,
        metric_density_precedent_url=METRIC_DENSITY_PRECEDENT_URL,
        parametrization_source=PARAMETRIZATION_SOURCE,
        parametrization_source_url=PARAMETRIZATION_SOURCE_URL,
        source_items=SOURCE_ITEMS,
        source_boundary=SOURCE_BOUNDARY,
        normalization=NORMALIZATION,
        compatibility_relation=COMPATIBILITY_RELATION,
        dimension=DIMENSION,
        scalar_labels=SCALAR_LABELS,
        maximum_total_jet_order=MAXIMUM_TOTAL_JET_ORDER,
        antibracket_convention=ANTIBRACKET_CONVENTION,
        field_specs=compatibility_field_specs(),
        antifield_specs=compatibility_antifield_specs(),
        upstream_hashes=UPSTREAM_HASHES,
        claim_ceiling=CLAIM_CEILING,
        contract_sha256=CONTRACT_SHA256,
        determinant_polynomial_constructed=True,
        adjugate_identity_computed=True,
        determinant_weight_two_covariance_computed=True,
        compatibility_ideal_brst_stable=True,
        weight_minus_one_multiplier_constructed=True,
        conditional_positive_rho_metric_reconstruction_computed=True,
        bounded_local_bv_quotient_constructed=True,
        explicit_afn0_and_afn1_currents_constructed=True,
        compatibility_cme_mod_dh_computed=True,
        live_negative_controls_computed=True,
        silent_terminal_truncation_allowed=False,
        rho_zero_patch_allowed=False,
        negative_rho_orientation_branch_admitted=False,
        time_orientation_selected=False,
        global_metric_reconstruction_proved=False,
        curvature_tensor_constructed=False,
        einstein_hilbert_action_used=False,
        ghy_boundary_term_used=False,
        full_m1_functional_constructed=False,
        global_boundary_completion_proved=False,
        functional_measure_computed=False,
        quantum_master_equation_computed=False,
        continuum_loop_st_computed=False,
        positive_physical_hilbert_proved=False,
        quantum_hda_m2_proved=False,
        m3_relational_observables_unlocked=False,
        derivation_status=(
            'exact_bounded_4d_metric_density_determinant_compatibility_'
            'and_local_bv_not_curvature_eh_full_m1_or_quantum_m2'
        ),
    )


def _serialize_spec(spec: DensitizedBVVariableSpec) -> str:
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
    'determinant_polynomial_constructed',
    'adjugate_identity_computed',
    'determinant_weight_two_covariance_computed',
    'compatibility_ideal_brst_stable',
    'weight_minus_one_multiplier_constructed',
    'conditional_positive_rho_metric_reconstruction_computed',
    'bounded_local_bv_quotient_constructed',
    'explicit_afn0_and_afn1_currents_constructed',
    'compatibility_cme_mod_dh_computed',
    'live_negative_controls_computed',
    'silent_terminal_truncation_allowed',
    'rho_zero_patch_allowed',
    'negative_rho_orientation_branch_admitted',
    'time_orientation_selected',
    'global_metric_reconstruction_proved',
    'curvature_tensor_constructed',
    'einstein_hilbert_action_used',
    'ghy_boundary_term_used',
    'full_m1_functional_constructed',
    'global_boundary_completion_proved',
    'functional_measure_computed',
    'quantum_master_equation_computed',
    'continuum_loop_st_computed',
    'positive_physical_hilbert_proved',
    'quantum_hda_m2_proved',
    'm3_relational_observables_unlocked',
)


def canonical_contract_payload(
    contract: M1MetricDensityCompatibilityBVContract,
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
            f'local={contract.local_functional_source}',
            f'local_url={contract.local_functional_source_url}',
            f'density={contract.density_source}',
            f'density_url={contract.density_source_url}',
            f'metric_precedent={contract.metric_density_precedent}',
            f'metric_precedent_url={contract.metric_density_precedent_url}',
            f'parametrization={contract.parametrization_source}',
            f'parametrization_url={contract.parametrization_source_url}',
            f'source_items={comma.join(contract.source_items)}',
            f'source_boundary={contract.source_boundary}',
            f'normalization={contract.normalization}',
            f'compatibility={contract.compatibility_relation}',
            f'dimension={contract.dimension}',
            f'labels={comma.join(contract.scalar_labels)}',
            f'max_total_jet={contract.maximum_total_jet_order}',
            f'antibracket={contract.antibracket_convention}',
            f'fields={comma.join(_serialize_spec(x) for x in contract.field_specs)}',
            f'antifields={comma.join(_serialize_spec(x) for x in contract.antifield_specs)}',
            f'upstream={comma.join(name + chr(58) + value for name, value in contract.upstream_hashes)}',
            f'ceiling={contract.claim_ceiling}',
            f'flags={flags}',
            f'status={contract.derivation_status}',
        )
    )


def contract_payload_sha256(
    contract: M1MetricDensityCompatibilityBVContract,
) -> str:
    return hashlib.sha256(
        canonical_contract_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M1MetricDensityCompatibilityBVContract) -> None:
    frozen = (
        contract.primary_source == PRIMARY_SOURCE,
        contract.primary_source_url == PRIMARY_SOURCE_URL,
        contract.local_functional_source == LOCAL_FUNCTIONAL_SOURCE,
        contract.local_functional_source_url == LOCAL_FUNCTIONAL_SOURCE_URL,
        contract.density_source == DENSITY_SOURCE,
        contract.density_source_url == DENSITY_SOURCE_URL,
        contract.metric_density_precedent == METRIC_DENSITY_PRECEDENT,
        contract.metric_density_precedent_url == METRIC_DENSITY_PRECEDENT_URL,
        contract.parametrization_source == PARAMETRIZATION_SOURCE,
        contract.parametrization_source_url == PARAMETRIZATION_SOURCE_URL,
        contract.source_items == SOURCE_ITEMS,
        contract.source_boundary == SOURCE_BOUNDARY,
        contract.normalization == NORMALIZATION,
        contract.compatibility_relation == COMPATIBILITY_RELATION,
        contract.dimension == DIMENSION,
        contract.scalar_labels == SCALAR_LABELS,
        contract.maximum_total_jet_order == MAXIMUM_TOTAL_JET_ORDER,
        contract.antibracket_convention == ANTIBRACKET_CONVENTION,
        contract.field_specs == compatibility_field_specs(),
        contract.antifield_specs == compatibility_antifield_specs(),
        contract.upstream_hashes == UPSTREAM_HASHES,
        contract.claim_ceiling == CLAIM_CEILING,
        contract.derivation_status
        == (
            'exact_bounded_4d_metric_density_determinant_compatibility_'
            'and_local_bv_not_curvature_eh_full_m1_or_quantum_m2'
        ),
    )
    if not all(frozen):
        raise ValueError('metric-density compatibility source or basis lock changed')
    if len(contract.field_specs) != 29 or len(contract.antifield_specs) != 29:
        raise ValueError('metric-density compatibility requires 29 canonical pairs')
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
            raise ValueError('compatibility field-antifield grading lock changed')
    if (
        contract.contract_sha256 != CONTRACT_SHA256
        or contract_payload_sha256(contract) != CONTRACT_SHA256
    ):
        raise ValueError('metric-density compatibility contract hash mismatch')
    required_true = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[:10]
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]
    )
    if not all(required_true) or any(unsupported):
        raise ValueError('metric-density compatibility claim flags changed')


@dataclass(frozen=True)
class M1MetricDensityCompatibilityBVReceipt:
    contract_sha256: str
    source_boundary: str
    normalization: str
    compatibility_relation: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_e70_f_verified: bool
    field_count: int
    antifield_count: int
    multiindex_count: int
    bounded_jet_generator_count: int
    bounded_even_jet_generator_count: int
    bounded_odd_jet_generator_count: int
    determinant_term_count: int
    determinant_raw_component_weight_sums: tuple[int, ...]
    adjugate_identity_component_count: int
    adjugate_identity_nonzero_component_count: int
    determinant_variation_term_count: int
    determinant_weight_two_target_term_count: int
    determinant_weight_two_mismatch_term_count: int
    wrong_determinant_weight_four_mismatch_term_count: int
    compatibility_constraint_term_count: int
    compatibility_constraint_weight_two_mismatch_term_count: int
    compatibility_density_term_count: int
    compatibility_density_weight_one_mismatch_term_count: int
    scalar_density_term_count: int
    classical_density_term_count: int
    antifield_density_term_count: int
    extended_density_term_count: int
    classical_variation_term_count: int
    classical_current_term_count: int
    classical_current_divergence_term_count: int
    classical_identity_mismatch_term_count: int
    base_nilpotency_component_count: int
    base_nilpotency_nonzero_component_count: int
    derived_transformation_component_count: int
    derived_transformation_mismatch_term_count: int
    canonical_field_star_residual_term_count: int
    canonical_star_field_residual_term_count: int
    odd_left_right_derivative_mismatch_term_count: int
    graded_antisymmetry_residual_term_count: int
    jacobi_nonzero_nested_bracket_count: int
    graded_jacobi_residual_term_count: int
    master_density_term_count: int
    master_density_maximum_total_jet_order: int
    master_density_ghost_numbers: tuple[int, ...]
    master_afn0_term_count: int
    master_afn1_term_count: int
    analytic_afn0_current_term_count: int
    compatibility_afn0_current_increment_term_count: int
    analytic_afn0_current_divergence_term_count: int
    analytic_afn0_mismatch_term_count: int
    homotopy_afn1_current_term_count: int
    homotopy_afn1_current_divergence_term_count: int
    homotopy_afn1_remainder_term_count: int
    homotopy_afn1_direct_mismatch_term_count: int
    full_master_current_term_count: int
    full_master_current_divergence_term_count: int
    full_master_current_mismatch_term_count: int
    master_euler_audit_count: int
    master_euler_nonzero_count: int
    master_euler_maximum_residual_term_count: int
    reconstruction_rho: Fraction
    reconstruction_h_determinant: Fraction
    reconstruction_inverse_product_maximum_residual: Fraction
    reconstruction_g_covariant_determinant_residual: Fraction
    reconstruction_g_contravariant_determinant_residual: Fraction
    reconstruction_lorentzian_inertia: bool
    reconstruction_time_orientation_selected: bool
    reconstruction_global_patch_proved: bool
    correct_constraint_numeric_residual: Fraction
    wrong_sign_constraint_numeric_residual: Fraction
    missing_h_weight_covariance_mismatch_term_count: int
    ell_weight_zero_density_mismatch_term_count: int
    ell_weight_zero_classical_mismatch_term_count: int
    ell_weight_zero_euler_ell_term_count: int
    omitted_ell_antifield_transformation_mismatch_term_count: int
    omitted_ell_antifield_master_ell_euler_term_count: int
    wrong_antibracket_canonical_residual_term_count: int
    wrong_antibracket_antisymmetry_residual_term_count: int
    wrong_antibracket_jacobi_residual_term_count: int
    rho_zero_patch_rejected: bool
    negative_rho_patch_rejected: bool
    incompatible_determinant_patch_rejected: bool
    nonsymmetric_h_patch_rejected: bool
    terminal_jet_derivative_rejected: bool
    determinant_polynomial_constructed: bool
    adjugate_identity_computed: bool
    determinant_weight_two_covariance_computed: bool
    compatibility_ideal_brst_stable: bool
    weight_minus_one_multiplier_constructed: bool
    conditional_positive_rho_metric_reconstruction_computed: bool
    bounded_local_bv_quotient_constructed: bool
    explicit_afn0_and_afn1_currents_constructed: bool
    compatibility_cme_mod_dh_computed: bool
    live_negative_controls_computed: bool
    silent_terminal_truncation_allowed: bool
    rho_zero_patch_allowed: bool
    negative_rho_orientation_branch_admitted: bool
    time_orientation_selected: bool
    global_metric_reconstruction_proved: bool
    curvature_tensor_constructed: bool
    einstein_hilbert_action_used: bool
    ghy_boundary_term_used: bool
    full_m1_functional_constructed: bool
    global_boundary_completion_proved: bool
    functional_measure_computed: bool
    quantum_master_equation_computed: bool
    continuum_loop_st_computed: bool
    positive_physical_hilbert_proved: bool
    quantum_hda_m2_proved: bool
    m3_relational_observables_unlocked: bool
    claim_ceiling: str
    derivation_status: str
    declared_m1_metric_density_compatibility_bv_gate_passed: bool


@lru_cache(maxsize=1)
def evaluate_m1_metric_density_compatibility_bv_gate(
) -> M1MetricDensityCompatibilityBVReceipt:
    contract = m1_metric_density_compatibility_bv_contract()
    validate_contract(contract)
    upstream_contract = m1_4d_densitized_scalar_bv_contract()
    validate_e70_f_contract(upstream_contract)
    upstream_receipt = evaluate_m1_4d_densitized_scalar_bv_gate()
    upstream_verified = (
        upstream_receipt.declared_m1_4d_densitized_scalar_bv_gate_passed
    )

    model = metric_density_compatibility_bv_model()
    adjugate_residuals = adjugate_identity_residuals()
    determinant_variation = apply_compatibility_brst(
        model.determinant_density,
        model.transformations,
    )
    determinant_weight_two_target = density_transformation_target(
        model.determinant_density,
        2,
    )
    determinant_weight_two_mismatch = (
        determinant_variation - determinant_weight_two_target
    )
    wrong_determinant_weight_four_mismatch = (
        determinant_variation
        - density_transformation_target(model.determinant_density, 4)
    )
    constraint_variation = apply_compatibility_brst(
        model.compatibility_constraint,
        model.transformations,
    )
    constraint_mismatch = (
        constraint_variation
        - density_transformation_target(model.compatibility_constraint, 2)
    )
    compatibility_density_variation = apply_compatibility_brst(
        model.compatibility_density,
        model.transformations,
    )
    compatibility_density_mismatch = (
        compatibility_density_variation
        - density_transformation_target(model.compatibility_density, 1)
    )
    classical_variation = apply_compatibility_brst(
        model.classical_density,
        model.transformations,
    )
    classical_current_divergence = divergence(
        model.classical_boundary_current
    )
    classical_mismatch = classical_variation - classical_current_divergence
    nilpotency_residuals = tuple(
        apply_compatibility_brst(image, model.transformations)
        for image in model.transformations.values()
    )
    derived_mismatch = derived_transformation_mismatch(model)

    ell = generator('ell')
    ell_one = generator('ell', unit_multiindex(0))
    ell_star = generator('ell_star')
    ghost = generator('c0')
    ghost_one = generator('c0', unit_multiindex(0))
    ghost_star = generator('c0_star')
    one = SparseSuperPolynomial.scalar(1)
    canonical_field_star = local_bv_antibracket_density(ell, ell_star) - one
    canonical_star_field = local_bv_antibracket_density(ell_star, ell) + one
    two_odd = ghost * ell_star
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
    sample_first = ell_star * ghost * ell_one
    sample_second = -(ghost_star * ghost * ghost_one)
    sample_third = ell
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
    analytic_current = analytic_afn0_master_current(model)
    analytic_divergence = divergence(analytic_current)
    homotopy_current = variational_homotopy_current(afn1)
    homotopy_divergence = divergence(homotopy_current)
    homotopy_remainder = variational_homotopy_euler_remainder(afn1)
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

    sample_h: NumericMatrix = (
        (Fraction(-4), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    reconstruction = reconstruct_metric_patch(sample_h, Fraction(4))
    fixture_values: dict[str, Fraction] = {'rho': Fraction(4)}
    for mu in range(DIMENSION):
        for nu in range(mu, DIMENSION):
            fixture_values[_h_name(mu, nu)] = sample_h[mu][nu]
    correct_constraint_numeric = evaluate_even_base_polynomial(
        model.compatibility_constraint,
        fixture_values,
    )
    wrong_sign_model = metric_density_compatibility_bv_model(
        determinant_rho_sign=-1
    )
    wrong_sign_constraint_numeric = evaluate_even_base_polynomial(
        wrong_sign_model.compatibility_constraint,
        fixture_values,
    )

    missing_h_weight = metric_density_compatibility_bv_model(
        h_density_weight=0
    )
    missing_h_weight_covariance_mismatch = (
        apply_compatibility_brst(
            missing_h_weight.determinant_density,
            missing_h_weight.transformations,
        )
        - density_transformation_target(
            missing_h_weight.determinant_density,
            2,
        )
    )
    ell_weight_zero = metric_density_compatibility_bv_model(
        ell_density_weight=0
    )
    ell_weight_zero_density_mismatch = (
        apply_compatibility_brst(
            ell_weight_zero.compatibility_density,
            ell_weight_zero.transformations,
        )
        - density_transformation_target(
            ell_weight_zero.compatibility_density,
            1,
        )
    )
    ell_weight_zero_classical_mismatch = (
        apply_compatibility_brst(
            ell_weight_zero.classical_density,
            ell_weight_zero.transformations,
        )
        - divergence(ell_weight_zero.classical_boundary_current)
    )
    ell_weight_zero_euler_ell = euler_derivative(
        ell_weight_zero_classical_mismatch,
        'ell',
        side='left',
    )
    omitted_ell = metric_density_compatibility_bv_model(
        include_ell_antifield_term=False
    )
    omitted_ell_mismatch = derived_transformation_mismatch(omitted_ell)
    omitted_ell_master = master_density(omitted_ell)
    omitted_ell_master_euler = euler_derivative(
        omitted_ell_master,
        'ell',
        side='left',
    )

    wrong_canonical = (
        local_bv_antibracket_density(
            ell_star,
            ell,
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

    def reconstruction_rejected(h: NumericMatrix, rho: Fraction) -> bool:
        try:
            reconstruct_metric_patch(h, rho)
        except ValueError:
            return True
        return False

    rho_zero_rejected = reconstruction_rejected(sample_h, Fraction(0))
    negative_rho_rejected = reconstruction_rejected(sample_h, Fraction(-4))
    incompatible_h: NumericMatrix = (
        (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    incompatible_rejected = reconstruction_rejected(
        incompatible_h,
        Fraction(4),
    )
    nonsymmetric_h: NumericMatrix = (
        (Fraction(-4), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    nonsymmetric_rejected = reconstruction_rejected(
        nonsymmetric_h,
        Fraction(4),
    )
    terminal_rejected = False
    try:
        horizontal_derivative(
            generator('ell', (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0)),
            0,
        )
    except CompatibilityJetOrderExceeded:
        terminal_rejected = True

    field_even = sum(spec.parity == 0 for spec in contract.field_specs)
    antifield_even = sum(
        spec.parity == 0 for spec in contract.antifield_specs
    )
    unsupported = tuple(
        getattr(contract, name) for name in _CONTRACT_FLAG_NAMES[10:]
    )
    passed = all(
        (
            upstream_verified,
            len(contract.field_specs) == 29,
            len(contract.antifield_specs) == 29,
            len(MULTIINDICES) == 35,
            len(JET_LOOKUP) == 2030,
            (field_even + antifield_even) * len(MULTIINDICES) == 1015,
            model.determinant_density.term_count == 17,
            component_factor_weight_sums(model.determinant_density)
            == frozenset((4,)),
            max(residual.term_count for residual in adjugate_residuals) == 0,
            determinant_weight_two_mismatch.is_zero,
            wrong_determinant_weight_four_mismatch.term_count > 0,
            model.compatibility_constraint.term_count == 18,
            constraint_mismatch.is_zero,
            compatibility_density_mismatch.is_zero,
            model.scalar_density.term_count == 52,
            model.classical_density.term_count == 70,
            model.antifield_density.term_count == 172,
            classical_mismatch.is_zero,
            max(residual.term_count for residual in nilpotency_residuals) == 0,
            derived_mismatch.is_zero,
            canonical_field_star.is_zero,
            canonical_star_field.is_zero,
            odd_left_right_mismatch.is_zero,
            antisymmetry_residual.is_zero,
            sum(not value.is_zero for value in nested_brackets) == 2,
            jacobi_residual.is_zero,
            master.term_count == 3036,
            polynomial_ghost_numbers(master) == frozenset((1,)),
            afn0.term_count == 1356,
            afn1.term_count == 1680,
            (afn0 - analytic_divergence).is_zero,
            homotopy_remainder.is_zero,
            (afn1 - homotopy_divergence).is_zero,
            full_master_mismatch.is_zero,
            max(residual.term_count for _, residual in master_euler) == 0,
            reconstruction.compatibility_residual == 0,
            reconstruction.inverse_product_maximum_residual == 0,
            reconstruction.g_covariant_determinant_residual == 0,
            reconstruction.g_contravariant_determinant_residual == 0,
            reconstruction.real_symmetric_nondegenerate_lorentzian_inertia,
            not reconstruction.time_orientation_selected,
            not reconstruction.global_patch_reconstruction_proved,
            correct_constraint_numeric == 0,
            wrong_sign_constraint_numeric != 0,
            missing_h_weight_covariance_mismatch.term_count > 0,
            ell_weight_zero_density_mismatch.term_count > 0,
            ell_weight_zero_classical_mismatch.term_count > 0,
            ell_weight_zero_euler_ell.term_count > 0,
            omitted_ell_mismatch.term_count > 0,
            omitted_ell_master_euler.term_count > 0,
            wrong_canonical.term_count > 0,
            wrong_antisymmetry.term_count > 0,
            wrong_jacobi.term_count > 0,
            rho_zero_rejected,
            negative_rho_rejected,
            incompatible_rejected,
            nonsymmetric_rejected,
            terminal_rejected,
            not any(unsupported),
        )
    )
    return M1MetricDensityCompatibilityBVReceipt(
        contract_sha256=contract.contract_sha256,
        source_boundary=contract.source_boundary,
        normalization=contract.normalization,
        compatibility_relation=contract.compatibility_relation,
        upstream_hashes=contract.upstream_hashes,
        upstream_e70_f_verified=upstream_verified,
        field_count=len(contract.field_specs),
        antifield_count=len(contract.antifield_specs),
        multiindex_count=len(MULTIINDICES),
        bounded_jet_generator_count=len(JET_LOOKUP),
        bounded_even_jet_generator_count=(
            (field_even + antifield_even) * len(MULTIINDICES)
        ),
        bounded_odd_jet_generator_count=(
            len(JET_LOOKUP)
            - (field_even + antifield_even) * len(MULTIINDICES)
        ),
        determinant_term_count=model.determinant_density.term_count,
        determinant_raw_component_weight_sums=tuple(
            sorted(component_factor_weight_sums(model.determinant_density))
        ),
        adjugate_identity_component_count=len(adjugate_residuals),
        adjugate_identity_nonzero_component_count=sum(
            not residual.is_zero for residual in adjugate_residuals
        ),
        determinant_variation_term_count=determinant_variation.term_count,
        determinant_weight_two_target_term_count=(
            determinant_weight_two_target.term_count
        ),
        determinant_weight_two_mismatch_term_count=(
            determinant_weight_two_mismatch.term_count
        ),
        wrong_determinant_weight_four_mismatch_term_count=(
            wrong_determinant_weight_four_mismatch.term_count
        ),
        compatibility_constraint_term_count=(
            model.compatibility_constraint.term_count
        ),
        compatibility_constraint_weight_two_mismatch_term_count=(
            constraint_mismatch.term_count
        ),
        compatibility_density_term_count=model.compatibility_density.term_count,
        compatibility_density_weight_one_mismatch_term_count=(
            compatibility_density_mismatch.term_count
        ),
        scalar_density_term_count=model.scalar_density.term_count,
        classical_density_term_count=model.classical_density.term_count,
        antifield_density_term_count=model.antifield_density.term_count,
        extended_density_term_count=model.extended_density.term_count,
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
        derived_transformation_component_count=len(model.transformations),
        derived_transformation_mismatch_term_count=derived_mismatch.term_count,
        canonical_field_star_residual_term_count=canonical_field_star.term_count,
        canonical_star_field_residual_term_count=canonical_star_field.term_count,
        odd_left_right_derivative_mismatch_term_count=(
            odd_left_right_mismatch.term_count
        ),
        graded_antisymmetry_residual_term_count=antisymmetry_residual.term_count,
        jacobi_nonzero_nested_bracket_count=sum(
            not value.is_zero for value in nested_brackets
        ),
        graded_jacobi_residual_term_count=jacobi_residual.term_count,
        master_density_term_count=master.term_count,
        master_density_maximum_total_jet_order=(
            polynomial_maximum_total_jet_order(master)
        ),
        master_density_ghost_numbers=tuple(
            sorted(polynomial_ghost_numbers(master))
        ),
        master_afn0_term_count=afn0.term_count,
        master_afn1_term_count=afn1.term_count,
        analytic_afn0_current_term_count=sum(
            current.term_count for current in analytic_current
        ),
        compatibility_afn0_current_increment_term_count=sum(
            (generator(f'c{mu}') * model.compatibility_density).term_count
            for mu in range(DIMENSION)
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
        reconstruction_rho=reconstruction.rho,
        reconstruction_h_determinant=reconstruction.h_determinant,
        reconstruction_inverse_product_maximum_residual=(
            reconstruction.inverse_product_maximum_residual
        ),
        reconstruction_g_covariant_determinant_residual=(
            reconstruction.g_covariant_determinant_residual
        ),
        reconstruction_g_contravariant_determinant_residual=(
            reconstruction.g_contravariant_determinant_residual
        ),
        reconstruction_lorentzian_inertia=(
            reconstruction.real_symmetric_nondegenerate_lorentzian_inertia
        ),
        reconstruction_time_orientation_selected=(
            reconstruction.time_orientation_selected
        ),
        reconstruction_global_patch_proved=(
            reconstruction.global_patch_reconstruction_proved
        ),
        correct_constraint_numeric_residual=correct_constraint_numeric,
        wrong_sign_constraint_numeric_residual=wrong_sign_constraint_numeric,
        missing_h_weight_covariance_mismatch_term_count=(
            missing_h_weight_covariance_mismatch.term_count
        ),
        ell_weight_zero_density_mismatch_term_count=(
            ell_weight_zero_density_mismatch.term_count
        ),
        ell_weight_zero_classical_mismatch_term_count=(
            ell_weight_zero_classical_mismatch.term_count
        ),
        ell_weight_zero_euler_ell_term_count=(
            ell_weight_zero_euler_ell.term_count
        ),
        omitted_ell_antifield_transformation_mismatch_term_count=(
            omitted_ell_mismatch.term_count
        ),
        omitted_ell_antifield_master_ell_euler_term_count=(
            omitted_ell_master_euler.term_count
        ),
        wrong_antibracket_canonical_residual_term_count=wrong_canonical.term_count,
        wrong_antibracket_antisymmetry_residual_term_count=(
            wrong_antisymmetry.term_count
        ),
        wrong_antibracket_jacobi_residual_term_count=wrong_jacobi.term_count,
        rho_zero_patch_rejected=rho_zero_rejected,
        negative_rho_patch_rejected=negative_rho_rejected,
        incompatible_determinant_patch_rejected=incompatible_rejected,
        nonsymmetric_h_patch_rejected=nonsymmetric_rejected,
        terminal_jet_derivative_rejected=terminal_rejected,
        determinant_polynomial_constructed=(
            contract.determinant_polynomial_constructed
        ),
        adjugate_identity_computed=contract.adjugate_identity_computed,
        determinant_weight_two_covariance_computed=(
            contract.determinant_weight_two_covariance_computed
        ),
        compatibility_ideal_brst_stable=(
            contract.compatibility_ideal_brst_stable
        ),
        weight_minus_one_multiplier_constructed=(
            contract.weight_minus_one_multiplier_constructed
        ),
        conditional_positive_rho_metric_reconstruction_computed=(
            contract.conditional_positive_rho_metric_reconstruction_computed
        ),
        bounded_local_bv_quotient_constructed=(
            contract.bounded_local_bv_quotient_constructed
        ),
        explicit_afn0_and_afn1_currents_constructed=(
            contract.explicit_afn0_and_afn1_currents_constructed
        ),
        compatibility_cme_mod_dh_computed=(
            contract.compatibility_cme_mod_dh_computed
        ),
        live_negative_controls_computed=(
            contract.live_negative_controls_computed
        ),
        silent_terminal_truncation_allowed=(
            contract.silent_terminal_truncation_allowed
        ),
        rho_zero_patch_allowed=contract.rho_zero_patch_allowed,
        negative_rho_orientation_branch_admitted=(
            contract.negative_rho_orientation_branch_admitted
        ),
        time_orientation_selected=contract.time_orientation_selected,
        global_metric_reconstruction_proved=(
            contract.global_metric_reconstruction_proved
        ),
        curvature_tensor_constructed=contract.curvature_tensor_constructed,
        einstein_hilbert_action_used=contract.einstein_hilbert_action_used,
        ghy_boundary_term_used=contract.ghy_boundary_term_used,
        full_m1_functional_constructed=contract.full_m1_functional_constructed,
        global_boundary_completion_proved=(
            contract.global_boundary_completion_proved
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
        declared_m1_metric_density_compatibility_bv_gate_passed=passed,
    )
