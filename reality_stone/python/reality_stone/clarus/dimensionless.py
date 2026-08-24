"""Dimensionless bookkeeping for CE gates.

CE calculations should close first on dimensionless ratios.  This module
provides small exact-rational tools for checking that rule without pulling in
symbolic math dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, log, log1p, prod
from typing import Callable, Generic, Iterable, Sequence, TypeVar


DimVector = tuple[Fraction, ...]
T = TypeVar("T")
U = TypeVar("U")


def _frac(x: int | float | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator()


def dim(*exponents: int | float | Fraction) -> DimVector:
    """Build a dimension vector, e.g. ``dim(1, 0, -2)`` for M L^0 T^-2."""

    return tuple(_frac(x) for x in exponents)


DIMENSIONLESS: DimVector = dim(0, 0, 0, 0)
MASS: DimVector = dim(1, 0, 0, 0)
LENGTH: DimVector = dim(0, 1, 0, 0)
TIME: DimVector = dim(0, 0, 1, 0)
TEMPERATURE: DimVector = dim(0, 0, 0, 1)
ENERGY: DimVector = MASS
CURVATURE: DimVector = dim(0, -2, 0, 0)
ACTION: DimVector = dim(1, 2, -1, 0)


@dataclass(frozen=True)
class Quantity:
    """Numeric value with a base-dimension exponent vector."""

    name: str
    value: float
    dims: DimVector = DIMENSIONLESS

    @property
    def dimensionless(self) -> bool:
        return is_dimensionless(self.dims)


@dataclass(frozen=True)
class GateResult(Generic[T]):
    """Small Result/Either value for composing CE gate checks."""

    value: T | None = None
    errors: tuple[str, ...] = ()

    @classmethod
    def ok(cls, value: T) -> "GateResult[T]":
        return cls(value=value)

    @classmethod
    def fail(cls, *errors: str) -> "GateResult[T]":
        return cls(errors=tuple(error for error in errors if error))

    @property
    def passed(self) -> bool:
        return not self.errors

    def map(self, transform: Callable[[T], U]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return GateResult.ok(transform(self.unwrap()))

    def bind(self, transform: Callable[[T], "GateResult[U]"]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return transform(self.unwrap())

    def unwrap(self) -> T:
        if self.errors:
            raise ValueError("; ".join(self.errors))
        if self.value is None:
            raise ValueError("gate result has no value")
        return self.value


def is_dimensionless(dims: Sequence[Fraction]) -> bool:
    return all(_frac(x) == 0 for x in dims)


def same_dimensions(left: Quantity, right: Quantity) -> bool:
    """Return whether two quantities have exactly the same dimension vector."""

    return left.dims == right.dims


def require_same_dimensions(left: Quantity, right: Quantity, *, context: str = "") -> tuple[Quantity, Quantity]:
    """Return like-dimension quantities or raise; numeric values are not compared.

    Equality, subtraction, and a shared reference scale are defined only for
    quantities with the same dimension vector.  This check intentionally does
    not treat an untyped dimensionless ``0`` as a zero of every dimension.
    """

    if not same_dimensions(left, right):
        where = f" for {context}" if context else ""
        raise ValueError(
            f"{left.name} and {right.name} must have the same dimensions{where}; "
            f"dims={left.dims} vs {right.dims}"
        )
    return left, right


def typed_zero(name: str, dims: DimVector) -> Quantity:
    """Construct the zero section with an explicit dimension type.

    Use this for statements such as ``F = 0`` when ``F`` carries units; a bare
    dimensionless zero remains dimensionless and is not interchangeable with it.
    """

    return Quantity(name=name, value=0.0, dims=dims)


def typed_equal(left: Quantity, right: Quantity) -> bool:
    """Check finite value equality after requiring matching dimension vectors."""

    require_same_dimensions(left, right, context="typed equality")
    if not all(isfinite(quantity.value) for quantity in (left, right)):
        raise ValueError("typed equality requires finite values")
    return left.value == right.value


def same_rank_dims(quantities: Iterable[Quantity]) -> int:
    lengths = {len(q.dims) for q in quantities}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent dimension ranks: {sorted(lengths)}")
    return lengths.pop()


def require_dimensionless(quantity: Quantity, *, context: str = "") -> Quantity:
    """Raise if a quantity is not dimensionless."""

    if not quantity.dimensionless:
        where = f" for {context}" if context else ""
        raise ValueError(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")
    return quantity


def linear_equality_defect(left: Quantity, right: Quantity, scale: Quantity) -> float:
    """Return ``|left-right|/scale`` for like dimensions and positive scale.

    The result is dimensionless and invariant if all three numeric values are
    rescaled by the same positive unit conversion.  ``scale`` must be finite
    and strictly positive in the chosen representative unit.
    """

    require_same_dimensions(left, right, context="linear equality defect")
    require_same_dimensions(left, scale, context="linear equality defect scale")
    if not all(isfinite(quantity.value) for quantity in (left, right, scale)):
        raise ValueError("linear equality defect requires finite inputs")
    if scale.value <= 0:
        raise ValueError(f"{scale.name} must be a positive reference scale")
    difference = left.value - right.value
    defect = abs(difference) / scale.value
    if not isfinite(defect):
        raise ValueError("linear equality defect must be finite")
    return defect


def log_equality_defect(left: Quantity, right: Quantity) -> float:
    """Return ``|log(left/right)|`` for positive like-dimension quantities.

    The domain is strictly positive finite values.  It is unit invariant under
    a common positive rescaling of both quantities.
    """

    require_same_dimensions(left, right, context="log equality defect")
    if not all(isfinite(q.value) and q.value > 0 for q in (left, right)):
        raise ValueError("log equality defect requires finite positive inputs")
    if left.value == right.value:
        return 0.0
    larger, smaller = max(left.value, right.value), min(left.value, right.value)
    relative = (larger - smaller) / smaller
    defect = log1p(relative) if isfinite(relative) else log(larger) - log(smaller)
    if not isfinite(defect) or defect <= 0:
        raise ValueError("log equality defect must be finite")
    return defect


def _cholesky_spd(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Validate a finite symmetric positive-definite matrix and factor it."""

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Mahalanobis covariance must be a non-empty square matrix")
    values = [[float(entry) for entry in row] for row in matrix]
    for i in range(n):
        for j in range(n):
            if not isfinite(values[i][j]):
                raise ValueError("Mahalanobis covariance entries must be finite")
            if values[i][j] != values[j][i]:
                raise ValueError("Mahalanobis covariance must be symmetric")
    lower = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            subtotal = 0.0
            for k in range(j):
                product = lower[i][k] * lower[j][k]
                subtotal += product
                if not isfinite(product) or not isfinite(subtotal):
                    raise ValueError("Mahalanobis covariance factorization overflowed")
            if i == j:
                diagonal = values[i][i] - subtotal
                if not isfinite(diagonal) or diagonal <= 0:
                    raise ValueError("Mahalanobis covariance must be positive definite")
                lower[i][j] = diagonal**0.5
                if not isfinite(lower[i][j]):
                    raise ValueError("Mahalanobis covariance factorization overflowed")
            else:
                numerator = values[i][j] - subtotal
                lower[i][j] = numerator / lower[j][j]
                if not isfinite(numerator) or not isfinite(lower[i][j]):
                    raise ValueError("Mahalanobis covariance factorization overflowed")
    return lower


def mahalanobis_equality_defect(
    left: Sequence[Quantity],
    right: Sequence[Quantity],
    scales: Sequence[Quantity],
    covariance: Sequence[Sequence[float]],
) -> float:
    """Return ``r^T covariance^-1 r`` for componentwise normalized residuals.

    Each residual is ``r_i=(left_i-right_i)/scales_i`` and is dimensionless.
    The covariance is therefore a dimensionless, finite, symmetric positive-
    definite matrix; indefinite, singular, non-square, and asymmetric inputs
    are rejected before solving.
    """

    n = len(left)
    if n == 0 or len(right) != n or len(scales) != n:
        raise ValueError("Mahalanobis inputs must be non-empty sequences of equal length")
    residual = [
        linear_equality_defect(a, b, scale) if a.value >= b.value else -linear_equality_defect(a, b, scale)
        for a, b, scale in zip(left, right, scales)
    ]
    lower = _cholesky_spd(covariance)
    if len(lower) != n:
        raise ValueError("Mahalanobis covariance size must match residual length")
    # Solve L y = r; then r^T Sigma^-1 r = y^T y for Sigma = L L^T.
    solved: list[float] = []
    for i, value in enumerate(residual):
        subtotal = 0.0
        for j in range(i):
            product = lower[i][j] * solved[j]
            subtotal += product
            if not isfinite(product) or not isfinite(subtotal):
                raise ValueError("Mahalanobis solve overflowed")
        solution = (value - subtotal) / lower[i][i]
        if not isfinite(solution):
            raise ValueError("Mahalanobis solve overflowed")
        solved.append(solution)
    defect = 0.0
    for value in solved:
        square = value * value
        defect += square
        if not isfinite(square) or not isfinite(defect):
            raise ValueError("Mahalanobis equality defect must be finite")
    return defect


def compensate_beta_for_affine_defect(beta: float, multiplier: float) -> float:
    """Return beta' = beta/multiplier for ``delta'=offset+multiplier*delta``.

    This is a finite-beta algebraic reparameterization: any additive offset
    cancels in normalized Gibbs weights.  Positivity of beta and Gibbs
    normalizability/concentration remain caller policy.
    """

    if not isfinite(beta) or not isfinite(multiplier) or multiplier <= 0:
        raise ValueError("beta and positive defect multiplier must be finite")
    compensated = beta / multiplier
    if not isfinite(compensated):
        raise ValueError("compensated beta must be finite")
    return compensated


def compensate_beta_for_reference_scale(beta: float, scale_multiplier: float) -> float:
    """Return beta' = scale_multiplier*beta when ``S'=scale_multiplier*S``.

    This is a finite-beta algebraic reparameterization; beta positivity and
    Gibbs normalizability/concentration remain caller policy.
    """

    if not isfinite(beta) or not isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("beta and positive scale multiplier must be finite")
    compensated = scale_multiplier * beta
    if not isfinite(compensated):
        raise ValueError("compensated beta must be finite")
    return compensated


def check_dimensionless(quantity: Quantity, *, context: str = "") -> GateResult[Quantity]:
    """Return a composable gate result instead of raising on dimensional failure."""

    if quantity.dimensionless:
        return GateResult.ok(quantity)
    where = f" for {context}" if context else ""
    return GateResult.fail(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")


def audit_dimensionless(
    quantities: Iterable[Quantity],
    *,
    context: str = "",
) -> GateResult[tuple[Quantity, ...]]:
    """Validate many quantities and accumulate every dimensional violation."""

    accepted: list[Quantity] = []
    errors: list[str] = []
    for quantity in quantities:
        check = check_dimensionless(quantity, context=context)
        if check.passed:
            accepted.append(check.unwrap())
        else:
            errors.extend(check.errors)
    if errors:
        return GateResult.fail(*errors)
    return GateResult.ok(tuple(accepted))


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    rows = [row[:] for row in matrix]
    if not rows:
        return rows, []
    n_rows, n_cols = len(rows), len(rows[0])
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = next((i for i in range(r, n_rows) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [x * inv for x in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def nullspace(matrix: Sequence[Sequence[int | float | Fraction]]) -> list[list[Fraction]]:
    """Exact rational basis for ``matrix @ x = 0``."""

    rows = [[_frac(x) for x in row] for row in matrix]
    if not rows:
        return []
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("ragged matrix")
    rref, pivots = _rref(rows)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    basis: list[list[Fraction]] = []
    for free in free_cols:
        vec = [Fraction(0, 1) for _ in range(n_cols)]
        vec[free] = Fraction(1, 1)
        for row_idx, pivot_col in enumerate(pivots):
            vec[pivot_col] = -rref[row_idx][free]
        basis.append(vec)
    return basis


def buckingham_pi_groups(quantities: Sequence[Quantity]) -> list[dict[str, Fraction]]:
    """Return exponent maps for independent dimensionless Pi groups."""

    if not quantities:
        return []
    rank = same_rank_dims(quantities)
    matrix = [[q.dims[row] for q in quantities] for row in range(rank)]
    groups = []
    for vec in nullspace(matrix):
        groups.append({q.name: exponent for q, exponent in zip(quantities, vec) if exponent != 0})
    return groups


def evaluate_group(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> float:
    by_name = {q.name: q for q in quantities}
    missing = sorted(set(exponents) - set(by_name))
    if missing:
        raise KeyError(f"unknown quantities in group: {missing}")
    return prod(by_name[name].value ** float(power) for name, power in exponents.items())


def group_dimension(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> DimVector:
    by_name = {q.name: q for q in quantities}
    rank = same_rank_dims(quantities)
    out = [Fraction(0, 1) for _ in range(rank)]
    for name, power in exponents.items():
        q = by_name[name]
        for i, exponent in enumerate(q.dims):
            out[i] += exponent * power
    return tuple(out)


def nondimensionalize(quantity: Quantity, scales: Sequence[Quantity]) -> Quantity:
    """Divide ``quantity`` by dimensional scales that span its dimension vector."""

    all_q = [*scales, quantity]
    rank = same_rank_dims(all_q)
    matrix = [[scale.dims[row] for scale in scales] for row in range(rank)]
    target = [-quantity.dims[row] for row in range(rank)]
    augmented = [row + [rhs] for row, rhs in zip(matrix, target)]
    rref, pivots = _rref(augmented)
    if len(scales) in pivots:
        raise ValueError(f"{quantity.name} dimensions cannot be spanned by supplied scales")
    powers = [Fraction(0, 1) for _ in scales]
    for row_idx, pivot_col in enumerate(pivots):
        if pivot_col < len(scales):
            powers[pivot_col] = rref[row_idx][-1]
    value = quantity.value * prod(scale.value ** float(power) for scale, power in zip(scales, powers))
    scale_part = " ".join(f"{s.name}^{p}" for s, p in zip(scales, powers) if p)
    name = f"{quantity.name}*{scale_part}" if scale_part else quantity.name
    return Quantity(name=name, value=value, dims=DIMENSIONLESS)


def exp_argument(quantity: Quantity) -> float:
    """Return a value that is safe to place in exp/log-like CE kernels."""

    require_dimensionless(quantity, context="exponential/logarithmic kernel")
    return quantity.value


def exp_arguments(quantities: Iterable[Quantity]) -> GateResult[tuple[float, ...]]:
    """Validate a batch of exp/log arguments and return their raw values."""

    return audit_dimensionless(
        quantities,
        context="exponential/logarithmic kernel",
    ).map(lambda checked: tuple(quantity.value for quantity in checked))
