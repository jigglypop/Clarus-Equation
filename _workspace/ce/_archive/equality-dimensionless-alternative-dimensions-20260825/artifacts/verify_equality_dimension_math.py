"""Small standard-library certificate for the E1--E5 mathematics lane."""
from __future__ import annotations

from fractions import Fraction
from math import comb, exp, isclose, log


def hodge_square(n: int, p: int, negative_signature: int) -> int:
    """Convention: **=(-1)^(p(n-p)+s), s = number of negative directions."""
    return -1 if (p * (n - p) + negative_signature) % 2 else 1


def binomial_solutions(n: int, p: int) -> list[int]:
    if not 0 <= p <= n:
        raise ValueError("form degree outside 0 <= p <= n")
    return [q for q in range(n + 1) if comb(n, p) == comb(n, q)]


def rank(matrix: list[list[Fraction]]) -> int:
    """Exact row rank, sufficient for a fixed dimension matrix stratum."""
    rows = [row[:] for row in matrix]
    if not rows:
        return 0
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [x / scale for x in rows[pivot_row]]
        for i in range(len(rows)):
            if i != pivot_row and rows[i][column]:
                factor = rows[i][column]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[pivot_row])]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def positive_torus_preconditions(
    values: list[float], dimension_matrix: list[list[Fraction]], *, declared_constant_rank: bool
) -> bool:
    """A fixed matrix is constant-rank only after a stratum has been declared."""
    return (
        bool(values)
        and all(value > 0.0 for value in values)
        and bool(dimension_matrix)
        and declared_constant_rank
    )


def affine_on_support(values: list[float], transformed: list[float]) -> bool:
    """Whether a transform has the required affine form on realized levels."""
    if len(values) <= 2:
        return True
    c = (transformed[1] - transformed[0]) / (values[1] - values[0])
    a = transformed[0] - c * values[0]
    return all(isclose(y, a + c * x, rel_tol=0.0, abs_tol=1e-14) for x, y in zip(values, transformed))


def normalized_weights(beta: float, defects: list[float]) -> list[float]:
    weights = [exp(-beta * value) for value in defects]
    total = sum(weights)
    return [weight / total for weight in weights]


def coupling_dimensions(D: int, scalar_power: int) -> tuple[Fraction, Fraction, Fraction]:
    phi = Fraction(D - 2, 2)
    scalar = Fraction(D) - scalar_power * phi
    ym = Fraction(4 - D, 2)
    newton = Fraction(2 - D)
    return scalar, ym, newton


def main() -> None:
    # Characters of R_+^r agree iff their exponent vectors agree: a concrete
    # separating rescaling for u != v is lambda_a=2 in a differing coordinate.
    u, v = (Fraction(1), Fraction(-2)), (Fraction(1), Fraction(-1))
    lam = (1.0, 2.0)
    chi_u = lam[0] ** float(u[0]) * lam[1] ** float(u[1])
    chi_v = lam[0] ** float(v[0]) * lam[1] ** float(v[1])
    assert chi_u != chi_v
    typed_zero_u = (0.0, u)
    typed_zero_v = (0.0, v)
    assert typed_zero_u != typed_zero_v  # zero sections retain their target type
    assert (0.0, u) == typed_zero_u

    # SPD C3 has a strict equality zero set; an indefinite invertible matrix does not.
    r = (1.0, 1.0)
    sigma_spd_inverse = ((1.0, 0.0), (0.0, 1.0))
    sigma_indefinite_inverse = ((1.0, 0.0), (0.0, -1.0))
    q_spd = sum(r[i] * sigma_spd_inverse[i][j] * r[j] for i in range(2) for j in range(2))
    q_indefinite = sum(r[i] * sigma_indefinite_inverse[i][j] * r[j] for i in range(2) for j in range(2))
    assert q_spd > 0.0 and q_indefinite == 0.0

    # Pi quotient preconditions are positive values and a declared fixed-rank stratum.
    matrix = [[Fraction(1), Fraction(0), Fraction(1)], [Fraction(0), Fraction(1), Fraction(-1)]]
    assert rank(matrix) == 2
    assert positive_torus_preconditions([2.0, 3.0, 5.0], matrix, declared_constant_rank=True)
    assert not positive_torus_preconditions([2.0, 3.0, 5.0], matrix, declared_constant_rank=False)
    assert not positive_torus_preconditions([2.0, 0.0, 5.0], matrix, declared_constant_rank=True)
    assert not positive_torus_preconditions([2.0, -3.0, 5.0], matrix, declared_constant_rank=True)

    # C2 changes by only a positive constant under a reference-scale change;
    # compensate beta to obtain exactly the same finite Gibbs weights.
    defects = [0.0, 0.5, 2.0]
    beta, c = 1.7, 3.25
    base = [exp(-beta * x) for x in defects]
    scaled = [exp(-(beta / c) * (c * x)) for x in defects]
    assert all(isclose(x, y, rel_tol=0.0, abs_tol=1e-14) for x, y in zip(base, scaled))
    # The fixed-beta uniqueness statement needs beta != 0 and at least two
    # realized levels.  Uniform beta=0 and constant-defect supports are exact
    # degenerate counterexamples for every positive multiplier.
    uniform = normalized_weights(0.0, defects)
    uniform_scaled = normalized_weights(0.0, [c * x for x in defects])
    assert all(isclose(x, y, rel_tol=0.0, abs_tol=1e-14) for x, y in zip(uniform, uniform_scaled))
    constant = [1.0, 1.0, 1.0]
    constant_base = normalized_weights(beta, constant)
    constant_scaled = normalized_weights(beta, [4.0 + c * x for x in constant])
    assert all(isclose(x, y, rel_tol=0.0, abs_tol=1e-14) for x, y in zip(constant_base, constant_scaled))
    assert normalized_weights(beta, defects) != normalized_weights(beta, [c * x for x in defects])
    nonlinear = [exp(-beta * x * x) for x in defects]
    assert not isclose(base[1] / base[2], nonlinear[1] / nonlinear[2])
    assert log(5.0 / 5.0) == 0.0
    two_level = [0.0, 2.0]
    two_level_square = [x * x for x in two_level]
    assert affine_on_support(two_level, two_level_square)
    beta_two = beta / 2.0  # x^2 = 2x on the realized {0,2} support
    assert all(isclose(exp(-beta * x), exp(-beta_two * y), abs_tol=1e-14) for x, y in zip(two_level, two_level_square))
    three_level = [0.0, 0.5, 2.0]
    assert not affine_on_support(three_level, [x * x for x in three_level])

    # Binomial theorem: only p and n-p occur in the integer domain.
    for n in range(0, 15):
        for p in range(n + 1):
            assert binomial_solutions(n, p) == sorted({p, n - p})
    assert [n for n in range(2, 15) if comb(n, 1) == comb(n, 2)] == [3]
    try:
        binomial_solutions(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("p=1 must be rejected when n=0")

    # Euclidean 4D real 2-form self-duality; Lorentzian 3+1 needs complexification.
    assert hodge_square(4, 2, 0) == 1
    assert hodge_square(4, 2, 1) == -1
    assert hodge_square(2, 1, 1) == 1

    # Engineering dimensions: phi^4/YM marginal in D=4; G marginal in D=2.
    assert coupling_dimensions(4, 4) == (Fraction(0), Fraction(0), Fraction(-2))
    assert coupling_dimensions(6, 3)[0] == 0
    assert coupling_dimensions(2, 4)[2] == 0
    print("OK equality/dimension certificate")


if __name__ == "__main__":
    main()
