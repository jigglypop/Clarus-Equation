"""Finite-time bridge from a compact opening source to cold produced matter.

All densities in this module are normalized by the present critical density and
``n = ln(a)`` is dimensionless.  The construction is intentionally modest:
``q``, the paying reservoir, and its equation of state are *axioms of the
effective model*.  In particular, a microscopic Bogoliubov calculation
``beta_k -> q(n) -> omega_prod0`` is not supplied here and remains an
unfinished physical bridge.

The useful conditional statement is nevertheless exact.  If a non-negative
compact source satisfies

    rho_p' + 3 rho_p = q,
    rho_R' + 3(1+w_R) rho_R = -q,

then their sum has no source term.  This file uses a C1 polynomial bump for
``q`` and evaluates both integrating factors analytically; it does not hide a
quadrature error inside the conservation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


_MOMENT_SERIES_CUTOFF = 4.0
_MOMENT_SERIES_TERMS = 96


def _finite_real(value: object, name: str) -> float:
    """Return a finite real scalar, rejecting booleans and generic inputs."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class FiniteQuenchBridgeConfig:
    """Dimensionless effective-model inputs for one finite opening.

    ``n_star +/- half_width`` is the compact source support and is required to
    finish no later than today.  ``w_reservoir >= -1`` makes the reservoir
    non-phantom; with a non-negative source this also makes its density
    monotone non-increasing towards the future.
    """

    n_star: float
    half_width: float
    omega_prod0: float
    reservoir_present_density: float
    w_reservoir: float
    w_open: float
    n_initial: float = math.log(1.0e-4)
    cold_envelope: str = "nonrelativistic_a_minus_2"

    def __post_init__(self) -> None:
        n_star = _finite_real(self.n_star, "n_star")
        half_width = _finite_real(self.half_width, "half_width")
        omega_prod0 = _finite_real(self.omega_prod0, "omega_prod0")
        reservoir_present_density = _finite_real(
            self.reservoir_present_density, "reservoir_present_density"
        )
        w_reservoir = _finite_real(self.w_reservoir, "w_reservoir")
        w_open = _finite_real(self.w_open, "w_open")
        n_initial = _finite_real(self.n_initial, "n_initial")
        if half_width <= 0.0:
            raise ValueError("half_width must be > 0")
        if n_star + half_width > 0.0:
            raise ValueError("n_star + half_width must be <= 0")
        if omega_prod0 < 0.0:
            raise ValueError("omega_prod0 must be >= 0")
        if reservoir_present_density < 0.0:
            raise ValueError("reservoir_present_density must be >= 0")
        if w_reservoir < -1.0:
            raise ValueError("w_reservoir must be >= -1")
        if w_open < 0.0:
            raise ValueError("w_open must be >= 0")
        if n_initial > n_star - half_width:
            raise ValueError("n_initial must be <= n_star - half_width")
        if self.cold_envelope != "nonrelativistic_a_minus_2":
            raise ValueError(
                "cold_envelope must be 'nonrelativistic_a_minus_2'"
            )
        object.__setattr__(self, "n_star", n_star)
        object.__setattr__(self, "half_width", half_width)
        object.__setattr__(self, "omega_prod0", omega_prod0)
        object.__setattr__(self, "reservoir_present_density", reservoir_present_density)
        object.__setattr__(self, "w_reservoir", w_reservoir)
        object.__setattr__(self, "w_open", w_open)
        object.__setattr__(self, "n_initial", n_initial)

    @property
    def n_minus(self) -> float:
        return self.n_star - self.half_width

    @property
    def n_plus(self) -> float:
        return self.n_star + self.half_width


@dataclass(frozen=True)
class FiniteQuenchCertificate:
    """Immutable numerical record of the conditional bridge invariants."""

    support: tuple[float, float]
    present_abundance_residual: float
    early_reservoir_density: float
    present_reservoir_density: float
    min_reservoir_density: float
    max_sampled_total_continuity_residual: float
    max_sampled_total_continuity_relative_residual: float
    cold_density_error_bound: float
    cold_envelope: str
    dimensionless_roles: tuple[tuple[str, str], ...]


def compact_c1_bump(n: object, n_star: object, half_width: object) -> float:
    """Normalized C1 bump ``psi(n)`` with integral one on its support.

    ``psi = 15/(16 Delta) (1-x^2)^2`` for ``|x| < 1`` and zero otherwise,
    where ``x=(n-n_star)/Delta``.  It and its first derivative vanish at both
    endpoints, hence the compact extension is C1.
    """

    n_value = _finite_real(n, "n")
    center = _finite_real(n_star, "n_star")
    width = _finite_real(half_width, "half_width")
    if width <= 0.0:
        raise ValueError("half_width must be > 0")
    lower = center - width
    upper = center + width
    if n_value <= lower or n_value >= upper:
        return 0.0
    x = (n_value - center) / width
    if abs(x) >= 1.0:
        return 0.0
    return 15.0 * (1.0 - x * x) ** 2 / (16.0 * width)


def compact_c1_bump_derivative(
    n: object,
    n_star: object,
    half_width: object,
) -> float:
    """Exact derivative of the compact C1 bump, including zero endpoints."""

    n_value = _finite_real(n, "n")
    center = _finite_real(n_star, "n_star")
    width = _finite_real(half_width, "half_width")
    if width <= 0.0:
        raise ValueError("half_width must be > 0")
    lower = center - width
    upper = center + width
    if n_value <= lower or n_value >= upper:
        return 0.0
    x = (n_value - center) / width
    if abs(x) >= 1.0:
        return 0.0
    return -15.0 * x * (1.0 - x * x) / (4.0 * width**2)


def compact_c1_cumulative(n: object, n_star: object, half_width: object) -> float:
    """Exact cumulative integral of :func:`compact_c1_bump` from ``-inf``."""

    n_value = _finite_real(n, "n")
    center = _finite_real(n_star, "n_star")
    width = _finite_real(half_width, "half_width")
    if width <= 0.0:
        raise ValueError("half_width must be > 0")
    lower = center - width
    upper = center + width
    if n_value <= lower:
        return 0.0
    if n_value >= upper:
        return 1.0
    x = (n_value - center) / width
    if x <= -1.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    polynomial_integral = x - (2.0 / 3.0) * x**3 + (1.0 / 5.0) * x**5
    return (15.0 / 16.0) * (polynomial_integral + 8.0 / 15.0)


def _exp_decay_moment(power: int, rate: float, length: float) -> float:
    """Return ``integral_0^length y^power exp(-rate*y) dy`` analytically."""

    if rate == 0.0:
        return length ** (power + 1) / (power + 1)
    z = rate * length
    if z < _MOMENT_SERIES_CUTOFF:
        term = length ** (power + 1) / (power + 1)
        terms = [term]
        for index in range(1, _MOMENT_SERIES_TERMS):
            term *= (
                -z
                * (power + index)
                / (index * (power + index + 1))
            )
            terms.append(term)
            if abs(term) <= 2.0 * math.ulp(max(1.0, abs(math.fsum(terms)))):
                break
        return math.fsum(terms)
    if z >= 745.0:
        lower_gamma_fraction = 1.0
    else:
        truncated_exponential = math.fsum(
            z**index / math.factorial(index) for index in range(power + 1)
        )
        lower_gamma_fraction = 1.0 - math.exp(-z) * truncated_exponential
    return (
        math.factorial(power)
        * rate ** (-(power + 1))
        * lower_gamma_fraction
    )


def _scaled_exp_polynomial_integral(
    x_lower: float, x_upper: float, exponent_slope: float
) -> tuple[float, float]:
    """Integrate the bump polynomial without separately forming huge exponentials.

    Returns ``(anchor, core)``, where
    ``core = integral exp(exponent_slope*(x-anchor))*(1-x^2)^2 dx``.
    The anchor is the endpoint with the largest exponent, so the exponential
    inside every moment is at most one.
    """

    if x_upper <= x_lower:
        return x_lower, 0.0
    anchor = x_upper if exponent_slope >= 0.0 else x_lower
    direction = -1.0 if exponent_slope >= 0.0 else 1.0
    length = x_upper - x_lower
    rate = abs(exponent_slope)
    polynomial = (1.0, 0.0, -2.0, 0.0, 1.0)
    shifted_coefficients = []
    for power in range(5):
        coefficient = math.fsum(
            polynomial[degree]
            * math.comb(degree, power)
            * anchor ** (degree - power)
            * direction**power
            for degree in range(power, 5)
        )
        shifted_coefficients.append(coefficient)
    core = math.fsum(
        coefficient * _exp_decay_moment(power, rate, length)
        for power, coefficient in enumerate(shifted_coefficients)
    )
    return anchor, core


class FiniteQuenchBridge:
    """Analytic effective bridge with a compact, finite-time source.

    This is deliberately an effective background construction.  It proves
    conditional continuity once ``q``, ``w_reservoir``, and ``w_open`` have
    been chosen; it does not derive those axioms from a microscopic quench.
    """

    def __init__(self, config: FiniteQuenchBridgeConfig) -> None:
        if not isinstance(config, FiniteQuenchBridgeConfig):
            raise ValueError("config must be a FiniteQuenchBridgeConfig")
        self.config = config

    def source(self, n: object) -> float:
        """Return ``q = omega_prod0 * exp(-3n) * psi(n)``."""

        n_value = _finite_real(n, "n")
        psi = compact_c1_bump(n_value, self.config.n_star, self.config.half_width)
        if psi == 0.0 or self.config.omega_prod0 == 0.0:
            return 0.0
        try:
            value = self.config.omega_prod0 * math.exp(-3.0 * n_value) * psi
        except OverflowError as error:
            raise ValueError("source is not finite on the declared domain") from error
        if not math.isfinite(value):
            raise ValueError("source is not finite on the declared domain")
        return value

    def source_derivative(self, n: object) -> float:
        """Return the exact ``n`` derivative of the compact source."""

        n_value = _finite_real(n, "n")
        psi = compact_c1_bump(
            n_value,
            self.config.n_star,
            self.config.half_width,
        )
        psi_prime = compact_c1_bump_derivative(
            n_value,
            self.config.n_star,
            self.config.half_width,
        )
        if (
            self.config.omega_prod0 == 0.0
            or (psi == 0.0 and psi_prime == 0.0)
        ):
            return 0.0
        try:
            value = (
                self.config.omega_prod0
                * math.exp(-3.0 * n_value)
                * (psi_prime - 3.0 * psi)
            )
        except OverflowError as error:
            raise ValueError(
                "source derivative is not finite on the declared domain"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                "source derivative is not finite on the declared domain"
            )
        return value

    def production_density(self, n: object) -> float:
        """Produced density: zero before, gradual during, exact dust after."""

        n_value = _finite_real(n, "n")
        fraction = compact_c1_cumulative(n_value, self.config.n_star, self.config.half_width)
        if fraction == 0.0 or self.config.omega_prod0 == 0.0:
            return 0.0
        try:
            value = self.config.omega_prod0 * math.exp(-3.0 * n_value) * fraction
        except OverflowError as error:
            raise ValueError(
                "production density is not finite on the declared domain"
            ) from error
        if not math.isfinite(value):
            raise ValueError("production density is not finite on the declared domain")
        return value

    def production_derivative(self, n: object) -> float:
        n_value = _finite_real(n, "n")
        return -3.0 * self.production_density(n_value) + self.source(n_value)

    def _weighted_source_integral(self, lower: float, upper: float) -> float:
        """Return integral lower..upper of exp(3(1+w_R)s) q(s) ds exactly."""

        lo = max(lower, self.config.n_minus)
        hi = min(upper, self.config.n_plus)
        if hi <= lo or self.config.omega_prod0 == 0.0:
            return 0.0
        alpha = 3.0 * self.config.w_reservoir
        delta = self.config.half_width
        x_lo = (lo - self.config.n_star) / delta
        x_hi = (hi - self.config.n_star) / delta
        b = alpha * delta
        anchor, integral_x = _scaled_exp_polynomial_integral(x_lo, x_hi, b)
        try:
            anchor_n = self.config.n_star + delta * anchor
            value = (
                self.config.omega_prod0
                * math.exp(alpha * anchor_n)
                * (15.0 / 16.0)
                * integral_x
            )
        except OverflowError as error:
            raise ValueError(
                "weighted source integral is not finite for this config"
            ) from error
        if not math.isfinite(value):
            raise ValueError("weighted source integral is not finite for this config")
        # Only suppress round-off below an abundance-scale tolerance; a material
        # negative value would violate the non-negative-source invariant.
        tolerance = 128.0 * math.ulp(max(1.0, self.config.omega_prod0))
        if value < -tolerance:
            raise ValueError("analytic weighted source integral became negative")
        return max(0.0, value)

    def reservoir_density(self, n: object) -> float:
        """Reservoir density from its analytic integrating-factor solution."""

        n_value = _finite_real(n, "n")
        if n_value < self.config.n_initial:
            raise ValueError("reservoir density is outside the declared n domain")
        if n_value > 0.0:
            raise ValueError("bridge is defined only through the present epoch n <= 0")
        lam = 3.0 * (1.0 + self.config.w_reservoir)
        paid_before_present = self._weighted_source_integral(n_value, 0.0)
        try:
            value = math.exp(-lam * n_value) * (
                self.config.reservoir_present_density + paid_before_present
            )
        except OverflowError as error:
            raise ValueError("reservoir density is not finite and non-negative") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("reservoir density is not finite and non-negative")
        return value

    def reservoir_derivative(self, n: object) -> float:
        """Return the reservoir ODE right-hand side, not a numerical derivative.

        The independent finite-difference check lives in the focused test suite;
        this method only evaluates ``-3(1+w_R) rho_R - q``.
        """

        n_value = _finite_real(n, "n")
        lam = 3.0 * (1.0 + self.config.w_reservoir)
        return -lam * self.reservoir_density(n_value) - self.source(n_value)

    def production_continuity_residual(self, n: object) -> float:
        n_value = _finite_real(n, "n")
        return self.production_derivative(n_value) + 3.0 * self.production_density(n_value) - self.source(n_value)

    def reservoir_continuity_residual(self, n: object) -> float:
        n_value = _finite_real(n, "n")
        lam = 3.0 * (1.0 + self.config.w_reservoir)
        return self.reservoir_derivative(n_value) + lam * self.reservoir_density(n_value) + self.source(n_value)

    def total_continuity_residual(self, n: object) -> float:
        """Conditional total residual; the compact source cancels exactly."""

        n_value = _finite_real(n, "n")
        return self.production_continuity_residual(n_value) + self.reservoir_continuity_residual(n_value)

    def total_continuity_relative_residual(self, n: object) -> float:
        """Scale-relative sampled round-off residual of the algebraic identity.

        This is not an independent derivation of the continuity equations. It
        measures floating-point cancellation after substituting their analytic
        right-hand sides; centered finite differences provide the independent
        numerical cross-check in the focused tests.
        """

        n_value = _finite_real(n, "n")
        lam = 3.0 * (1.0 + self.config.w_reservoir)
        rho_r = self.reservoir_density(n_value)
        rho_r_prime = self.reservoir_derivative(n_value)
        q_value = self.source(n_value)
        rho_p = self.production_density(n_value)
        rho_p_prime = self.production_derivative(n_value)
        residual = math.fsum(
            (rho_p_prime, 3.0 * rho_p, -q_value, rho_r_prime, lam * rho_r, q_value)
        )
        scale = max(
            1.0,
            abs(rho_p_prime),
            abs(3.0 * rho_p),
            abs(rho_r_prime),
            abs(lam * rho_r),
            abs(q_value),
        )
        return abs(residual) / scale

    def cold_density_error_bound(self) -> float:
        """Conditional bound for a nonrelativistic ``a^-2`` pressure envelope.

        The declared envelope is
        ``0 <= w_p(n) <= w_open*exp(-2*(n-n_plus))`` after the source. Without
        this axiom, ``w_open`` alone does not bound the accumulated dilution
        error and this formula must not be used.
        """

        return -math.expm1(-1.5 * self.config.w_open)

    def certificate(self) -> FiniteQuenchCertificate:
        """Return immutable evidence for the finite bridge's stated invariants."""

        nodes = (self.config.n_minus, self.config.n_star, self.config.n_plus, 0.0)
        residual = max(abs(self.total_continuity_residual(node)) for node in nodes)
        relative_residual = max(
            self.total_continuity_relative_residual(node) for node in nodes
        )
        early = self.reservoir_density(self.config.n_minus)
        present = self.reservoir_density(0.0)
        # For n <= 0, lambda >= 0 and q >= 0 imply rho_R' <= 0, so its
        # interval minimum is exactly its specified present density.
        roles = (
            ("n", "ln(a), dimensionless e-fold time"),
            ("omega_prod0", "rho_prod(0)/rho_crit,0, dimensionless"),
            ("q", "normalized density per dimensionless n"),
            ("w_reservoir", "dimensionless reservoir equation-of-state ratio"),
            ("w_open", "dimensionless post-opening coldness bound input"),
            ("n_initial", "dimensionless lower endpoint of the certified domain"),
            (
                "cold_envelope",
                "axiom: w_p <= w_open*exp(-2*(n-n_plus))",
            ),
        )
        return FiniteQuenchCertificate(
            support=(self.config.n_minus, self.config.n_plus),
            present_abundance_residual=self.production_density(0.0) - self.config.omega_prod0,
            early_reservoir_density=early,
            present_reservoir_density=present,
            min_reservoir_density=present,
            max_sampled_total_continuity_residual=residual,
            max_sampled_total_continuity_relative_residual=relative_residual,
            cold_density_error_bound=self.cold_density_error_bound(),
            cold_envelope=self.config.cold_envelope,
            dimensionless_roles=roles,
        )
