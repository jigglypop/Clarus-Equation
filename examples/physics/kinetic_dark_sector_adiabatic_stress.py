"""Fourth-order adiabatic stress subtraction for a constant-mass FLRW scalar.

The dimensionless contract is the same as ``kinetic_dark_sector_flrw_mode``:

    x = H0 * eta, q = k / H0, mu = m / H0, U = sqrt(H0) * u_phys.

For ``ds^2 = a^2(-d eta^2 + dx^2)`` and constant positive ``mu``, define

    w^2 = q^2 + a^2 mu^2,
    sigma = (6 xi - 1) a_xx / a.

The WKB frequency satisfies

    W^2 = w^2 + sigma - W_xx/(2 W) + 3 W_x^2/(4 W^2).

This module implements the Parker--Fulling/Bunch expansion
``W = w + W2 + W4`` as a formal adiabatic series.  Time derivatives are
tagged by an auxiliary epsilon, so the returned order-zero, order-two, and
order-four coefficients are projections rather than a finite-difference
fit.

Proof boundary
--------------
For the covariant scalar stress tensor,

    nabla_mu T^{mu nu} = (Box phi - m^2 phi - xi R phi) nabla^nu phi.

The displayed Riccati equation is exactly the rescaled mode equation under
the WKB ansatz.  Its residual after ``w + W2 + W4`` starts at adiabatic order
six.  Therefore the divergence of the stress built from that ansatz also
starts at order six; projecting the stress through order four gives the
mode-wise continuity identity through order five.  The implementation
returns both residuals as proof receipts and independently reconstructs W4
by two Riccati iterations.

This closes the mode-wise 0/2/4 subtraction identity for a constant positive
mass.  It deliberately does not infer finite local curvature counterterms, a
Hadamard state, or a time-dependent-mass energy-transfer law.  A separate
power-law-tail API below turns an externally certified ultraviolet bound into
an exact integrated remainder bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_MAX_TIME_DEGREE = 6
_MAX_ADIABATIC_ORDER = 6


class _FormalSeries:
    """Truncated series in time displacement and adiabatic epsilon."""

    def __init__(self, coefficients: dict[tuple[int, int], float] | None = None):
        self.coefficients: dict[tuple[int, int], float] = {}
        for (time_degree, order), value in (coefficients or {}).items():
            if (
                0 <= time_degree <= _MAX_TIME_DEGREE
                and 0 <= order <= _MAX_ADIABATIC_ORDER
                and value != 0.0
            ):
                self.coefficients[(time_degree, order)] = float(value)

    @classmethod
    def constant(cls, value: float) -> _FormalSeries:
        return cls({(0, 0): value})

    @staticmethod
    def _coerce(value: _FormalSeries | float) -> _FormalSeries:
        if isinstance(value, _FormalSeries):
            return value
        return _FormalSeries.constant(float(value))

    def __add__(self, other: _FormalSeries | float) -> _FormalSeries:
        other_series = self._coerce(other)
        result = dict(self.coefficients)
        for key, value in other_series.coefficients.items():
            result[key] = result.get(key, 0.0) + value
        return _FormalSeries(result)

    def __radd__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self + other

    def __neg__(self) -> _FormalSeries:
        return _FormalSeries({key: -value for key, value in self.coefficients.items()})

    def __sub__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self + (-self._coerce(other))

    def __rsub__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self._coerce(other) - self

    def __mul__(self, other: _FormalSeries | float) -> _FormalSeries:
        other_series = self._coerce(other)
        result: dict[tuple[int, int], float] = {}
        for (left_time, left_order), left_value in self.coefficients.items():
            for (right_time, right_order), right_value in other_series.coefficients.items():
                time_degree = left_time + right_time
                order = left_order + right_order
                if time_degree <= _MAX_TIME_DEGREE and order <= _MAX_ADIABATIC_ORDER:
                    key = (time_degree, order)
                    result[key] = result.get(key, 0.0) + left_value * right_value
        return _FormalSeries(result)

    def __rmul__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self * other

    def __truediv__(self, other: _FormalSeries | float) -> _FormalSeries:
        if isinstance(other, _FormalSeries):
            return self * other.inverse()
        scalar = float(other)
        if scalar == 0.0:
            raise ZeroDivisionError("formal-series scalar division by zero")
        return _FormalSeries(
            {key: value / scalar for key, value in self.coefficients.items()}
        )

    def __rtruediv__(self, other: _FormalSeries | float) -> _FormalSeries:
        return self._coerce(other) * self.inverse()

    def derivative(self, count: int = 1) -> _FormalSeries:
        if count < 0:
            raise ValueError("derivative count must be non-negative")
        result = self
        for _ in range(count):
            differentiated: dict[tuple[int, int], float] = {}
            for (time_degree, order), value in result.coefficients.items():
                if time_degree > 0:
                    differentiated[(time_degree - 1, order)] = time_degree * value
            result = _FormalSeries(differentiated)
        return result

    def inverse(self) -> _FormalSeries:
        constant = self.coefficient(0)
        if constant == 0.0:
            raise ZeroDivisionError("formal series has zero constant term")
        delta = self / constant - 1.0
        result = _FormalSeries.constant(1.0)
        term = _FormalSeries.constant(1.0)
        for _ in range(_MAX_TIME_DEGREE + _MAX_ADIABATIC_ORDER + 1):
            term = -term * delta
            if not term.coefficients:
                break
            result = result + term
        return result / constant

    def sqrt(self) -> _FormalSeries:
        constant = self.coefficient(0)
        if constant <= 0.0:
            raise ValueError("formal square root requires a positive constant term")
        delta = self / constant - 1.0
        result = _FormalSeries.constant(1.0)
        term = _FormalSeries.constant(1.0)
        binomial = 1.0
        for power in range(1, _MAX_TIME_DEGREE + _MAX_ADIABATIC_ORDER + 1):
            term = term * delta
            if not term.coefficients:
                break
            binomial *= (0.5 - (power - 1)) / power
            result = result + binomial * term
        return math.sqrt(constant) * result

    def coefficient(self, order: int, time_degree: int = 0) -> float:
        return self.coefficients.get((time_degree, order), 0.0)


@dataclass(frozen=True)
class ScaleFactorJet:
    """Scale factor and x-derivatives at one event, through sixth order."""

    a: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    d6: float

    def __post_init__(self) -> None:
        values = (self.a, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all scale-factor jet entries must be finite")
        if self.a <= 0.0:
            raise ValueError("scale factor must be positive")

    @property
    def derivatives(self) -> tuple[float, ...]:
        return (self.a, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)


@dataclass(frozen=True)
class MassSquaredJet:
    """mu(x)^2 and x-derivatives at one event, through sixth order."""

    value: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    d6: float

    def __post_init__(self) -> None:
        values = (
            self.value,
            self.d1,
            self.d2,
            self.d3,
            self.d4,
            self.d5,
            self.d6,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all mass-squared jet entries must be finite")
        if self.value <= 0.0:
            raise ValueError("mass squared must be positive")

    @property
    def derivatives(self) -> tuple[float, ...]:
        return (
            self.value,
            self.d1,
            self.d2,
            self.d3,
            self.d4,
            self.d5,
            self.d6,
        )


@dataclass(frozen=True)
class ModeStress:
    """Dimensionless stress per d^3q before the (2 pi)^-3 measure."""

    energy_density_over_h0_four: float
    pressure_over_h0_four: float


@dataclass(frozen=True)
class FourthOrderCounterterm:
    """Projected 0/2/4 counterterms and algebraic proof receipts."""

    w_orders: tuple[float, float, float]
    energy_density_orders: tuple[float, float, float]
    pressure_orders: tuple[float, float, float]
    max_riccati_residual_through_order_four: float
    max_ward_residual_through_order_five: float
    max_iterated_recurrence_disagreement: float
    status: str = (
        "FOURTH_ORDER_CONSTANT_MASS_COUNTERTERM_NO_FINITE_RENORMALIZATION_CONDITION"
    )

    @property
    def stress(self) -> ModeStress:
        return ModeStress(
            energy_density_over_h0_four=math.fsum(self.energy_density_orders),
            pressure_over_h0_four=math.fsum(self.pressure_orders),
        )


@dataclass(frozen=True)
class FourthOrderAdiabaticState:
    """Canonical initial data defined by the local W0+W2+W4 frequency."""

    u: complex
    du_dx: complex
    frequency: float
    frequency_derivative: float
    wronskian_residual: float
    status: str = "LOCAL_FOURTH_ORDER_ADIABATIC_INITIAL_STATE"


@dataclass(frozen=True)
class SixthOrderRemainder:
    """Leading formal term left after subtracting stress orders 0, 2, and 4."""

    energy_density_order_six: float
    pressure_order_six: float
    per_mode_large_q_power: int = -5
    radial_integrand_large_q_power: int = -3
    ultraviolet_integrable: bool = True


@dataclass(frozen=True)
class CertifiedPowerLawTail:
    """External certificate |s(q)| <= coefficient*q^-exponent above start_q."""

    coefficient: float
    exponent: float
    start_q: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("tail coefficient must be finite and non-negative")
        if not math.isfinite(self.exponent) or self.exponent <= 3.0:
            raise ValueError("tail exponent must be finite and greater than three")
        if not math.isfinite(self.start_q) or self.start_q <= 0.0:
            raise ValueError("tail start_q must be finite and positive")

    def isotropic_integral_bound_from(self, q: float) -> float:
        if not math.isfinite(q) or q < self.start_q:
            raise ValueError("tail bound is not certified at the requested q")
        return (
            self.coefficient
            * q ** (3.0 - self.exponent)
            / (2.0 * math.pi**2 * (self.exponent - 3.0))
        )


@dataclass(frozen=True)
class IntegratedStress:
    """Finite-grid central value plus rigorous certified UV-tail bounds."""

    energy_density_over_h0_four: float
    pressure_over_h0_four: float
    energy_tail_absolute_bound: float
    pressure_tail_absolute_bound: float
    status: str = "FINITE_GRID_PLUS_CERTIFIED_POWER_LAW_UV_TAIL"


@dataclass(frozen=True)
class TimeDependentMassCounterterm:
    """One local rho/p/<phi^2> counterterm triplet with energy transfer."""

    energy_density_orders: tuple[float, float, float]
    pressure_orders: tuple[float, float, float]
    field_squared_orders: tuple[float, float, float]
    transfer_orders: tuple[float, float, float]
    max_transfer_ward_residual_through_order_five: float
    status: str = "TIME_DEPENDENT_MASS_VARIATIONAL_COUNTERTERM_TRIPLET"


def _validate_parameters(q: float, mu: float, xi: float) -> None:
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q=k/H0 must be finite and positive")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("constant mu=m/H0 must be finite and positive")
    if not math.isfinite(xi):
        raise ValueError("xi must be finite")


def _scale_factor_series(jet: ScaleFactorJet) -> _FormalSeries:
    coefficients = {
        (degree, degree): derivative / math.factorial(degree)
        for degree, derivative in enumerate(jet.derivatives)
        if derivative != 0.0
    }
    return _FormalSeries(coefficients)


def _mass_squared_series(jet: MassSquaredJet) -> _FormalSeries:
    coefficients = {
        (degree, degree): derivative / math.factorial(degree)
        for degree, derivative in enumerate(jet.derivatives)
        if derivative != 0.0
    }
    return _FormalSeries(coefficients)


def _wkb_frequencies(
    a: _FormalSeries,
    q: float,
    mu: float,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries, _FormalSeries, _FormalSeries]:
    return _wkb_frequencies_from_mass_squared(
        a,
        q,
        _FormalSeries.constant(mu * mu),
        xi,
    )


def _wkb_frequencies_from_mass_squared(
    a: _FormalSeries,
    q: float,
    mass_squared_ratio: _FormalSeries,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries, _FormalSeries, _FormalSeries]:
    w = (q * q + mass_squared_ratio * a * a).sqrt()
    sigma = (6.0 * xi - 1.0) * a.derivative(2) / a
    inverse_w = w.inverse()
    w_prime = w.derivative()
    w_second = w.derivative(2)
    w2 = (
        0.5 * sigma * inverse_w
        - 0.25 * w_second * inverse_w * inverse_w
        + 0.375 * w_prime * w_prime * inverse_w * inverse_w * inverse_w
    )
    w2_prime = w2.derivative()
    w2_second = w2.derivative(2)
    # Expanding the Riccati equation at order four gives
    #
    #   2 w W4 + W2^2 = delta[-W''/(2W)+3W'^2/(4W^2)]|W2.
    #
    # Keeping this coefficient form avoids ambiguities in typeset nested
    # half-powers and provides a direct comparison with Riccati iteration.
    w4 = (
        -0.25 * w2_second * inverse_w * inverse_w
        + 0.25 * w_second * w2 * inverse_w * inverse_w * inverse_w
        + 0.75 * w_prime * w2_prime * inverse_w * inverse_w * inverse_w
        - 0.75
        * w_prime
        * w_prime
        * w2
        * inverse_w
        * inverse_w
        * inverse_w
        * inverse_w
        - 0.5 * w2 * w2 * inverse_w
    )
    return w, w2, w4, sigma


def _riccati_step(
    current: _FormalSeries,
    w: _FormalSeries,
    sigma: _FormalSeries,
) -> _FormalSeries:
    logarithmic_derivative = current.derivative() / current
    return (
        w * w
        + sigma
        - 0.5 * current.derivative(2) / current
        + 0.75 * logarithmic_derivative * logarithmic_derivative
    ).sqrt()


def _stress_series(
    a: _FormalSeries,
    wkb_frequency: _FormalSeries,
    q: float,
    mu: float,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries]:
    return _stress_series_from_mass_squared(
        a,
        wkb_frequency,
        q,
        _FormalSeries.constant(mu * mu),
        xi,
    )


def _stress_series_from_mass_squared(
    a: _FormalSeries,
    wkb_frequency: _FormalSeries,
    q: float,
    mass_squared_ratio: _FormalSeries,
    xi: float,
) -> tuple[_FormalSeries, _FormalSeries]:
    inverse_wkb = wkb_frequency.inverse()
    amplitude = 0.5 * inverse_wkb
    frequency_derivative = wkb_frequency.derivative()
    cross = -0.5 * frequency_derivative * inverse_wkb * inverse_wkb
    kinetic = (
        0.5 * wkb_frequency
        + 0.125
        * frequency_derivative
        * frequency_derivative
        * inverse_wkb
        * inverse_wkb
        * inverse_wkb
    )

    hubble = a.derivative() / a
    acceleration = a.derivative(2) / a
    mass_squared = mass_squared_ratio * a * a
    inverse_a_four = (a * a * a * a).inverse()

    energy_bracket = (
        kinetic
        + (q * q + mass_squared) * amplitude
        + (6.0 * xi - 1.0)
        * (hubble * cross - hubble * hubble * amplitude)
    )
    pressure_bracket = (
        kinetic
        - hubble * cross
        + (hubble * hubble - q * q / 3.0 - mass_squared) * amplitude
        + 2.0
        * xi
        * (
            -2.0 * kinetic
            + 3.0 * hubble * cross
            + (
                2.0 * q * q
                + 2.0 * mass_squared
                + (12.0 * xi - 2.0) * acceleration
                - 3.0 * hubble * hubble
            )
            * amplitude
        )
    )
    return 0.5 * inverse_a_four * energy_bracket, 0.5 * inverse_a_four * pressure_bracket


def fourth_order_counterterm(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> FourthOrderCounterterm:
    """Return the constant-mass fourth-order adiabatic stress counterterm."""

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, w2, w4, sigma = _wkb_frequencies(a, q, mu, xi)
    full_frequency = w + w2 + w4
    energy, pressure = _stress_series(a, full_frequency, q, mu, xi)

    logarithmic_derivative = full_frequency.derivative() / full_frequency
    riccati_residual = (
        full_frequency * full_frequency
        - w * w
        - sigma
        + 0.5 * full_frequency.derivative(2) / full_frequency
        - 0.75 * logarithmic_derivative * logarithmic_derivative
    )
    hubble = a.derivative() / a
    ward_residual = energy.derivative() + 3.0 * hubble * (energy + pressure)

    first_iteration = _riccati_step(w, w, sigma)
    second_iteration = _riccati_step(first_iteration, w, sigma)
    recurrence_disagreements = [
        abs(full_frequency.coefficient(order) - second_iteration.coefficient(order))
        for order in (0, 2, 4)
    ]

    return FourthOrderCounterterm(
        w_orders=tuple(full_frequency.coefficient(order) for order in (0, 2, 4)),
        energy_density_orders=tuple(energy.coefficient(order) for order in (0, 2, 4)),
        pressure_orders=tuple(pressure.coefficient(order) for order in (0, 2, 4)),
        max_riccati_residual_through_order_four=max(
            abs(riccati_residual.coefficient(order)) for order in range(5)
        ),
        max_ward_residual_through_order_five=max(
            abs(ward_residual.coefficient(order)) for order in range(6)
        ),
        max_iterated_recurrence_disagreement=max(recurrence_disagreements),
    )


def fourth_order_adiabatic_initial_state(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> FourthOrderAdiabaticState:
    """Construct canonical WKB data using the local fourth-order frequency."""

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, w2, w4, _ = _wkb_frequencies(a, q, mu, xi)
    full_frequency = w + w2 + w4
    frequency = math.fsum(full_frequency.coefficient(order) for order in (0, 2, 4))
    frequency_derivative = math.fsum(
        full_frequency.derivative().coefficient(order) for order in (1, 3, 5)
    )
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("fourth-order WKB frequency must be finite and positive")
    if not math.isfinite(frequency_derivative):
        raise ValueError("fourth-order WKB frequency derivative must be finite")
    u = complex(1.0 / math.sqrt(2.0 * frequency))
    du_dx = complex(-frequency_derivative / (2.0 * frequency), -frequency) * u
    wronskian = u * du_dx.conjugate() - u.conjugate() * du_dx
    return FourthOrderAdiabaticState(
        u=u,
        du_dx=du_dx,
        frequency=frequency,
        frequency_derivative=frequency_derivative,
        wronskian_residual=abs(wronskian - 1.0j),
    )


def sixth_order_remainder(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
) -> SixthOrderRemainder:
    """Return the leading formal stress left by fourth-order subtraction.

    For fixed background jets and positive mass, w~q.  W2~q^-1 and W4~q^-3,
    so the next stress coefficient is O(q^-5).  The isotropic measure changes
    this to q^2 O(q^-5)=O(q^-3), whose ultraviolet integral converges.
    """

    _validate_parameters(q, mu, xi)
    a = _scale_factor_series(jet)
    w, _, _, sigma = _wkb_frequencies(a, q, mu, xi)
    # Three Riccati iterations reconstruct W0+W2+W4+W6 through the retained
    # formal order.  W6 contributes to the sixth-order pressure coefficient
    # and therefore cannot be omitted from the leading remainder.
    first = _riccati_step(w, w, sigma)
    second = _riccati_step(first, w, sigma)
    third = _riccati_step(second, w, sigma)
    energy, pressure = _stress_series(a, third, q, mu, xi)
    return SixthOrderRemainder(
        energy_density_order_six=energy.coefficient(6),
        pressure_order_six=pressure.coefficient(6),
    )


def time_dependent_mass_counterterm(
    jet: ScaleFactorJet,
    mass_squared_jet: MassSquaredJet,
    *,
    q: float,
    xi: float,
) -> TimeDependentMassCounterterm:
    """Build the matched rho/p/<phi^2> triplet for mu(x)^2.

    The conformal-time transfer identity is

        rho_x + 3 (a_x/a) (rho+p) = (mu^2)_x <phi^2>/2.

    All four quantities are projected from the same WKB functional.  This is
    why the identity survives subtraction through adiabatic order five.
    """

    _validate_parameters(q, math.sqrt(mass_squared_jet.value), xi)
    a = _scale_factor_series(jet)
    mass_squared = _mass_squared_series(mass_squared_jet)
    w, w2, w4, _ = _wkb_frequencies_from_mass_squared(
        a, q, mass_squared, xi
    )
    full_frequency = w + w2 + w4
    energy, pressure = _stress_series_from_mass_squared(
        a, full_frequency, q, mass_squared, xi
    )
    field_squared = 0.5 * (a * a * full_frequency).inverse()
    transfer = 0.5 * mass_squared.derivative() * field_squared
    hubble = a.derivative() / a
    ward_residual = (
        energy.derivative()
        + 3.0 * hubble * (energy + pressure)
        - transfer
    )
    return TimeDependentMassCounterterm(
        energy_density_orders=tuple(energy.coefficient(order) for order in (0, 2, 4)),
        pressure_orders=tuple(
            pressure.coefficient(order) for order in (0, 2, 4)
        ),
        field_squared_orders=tuple(
            field_squared.coefficient(order) for order in (0, 2, 4)
        ),
        transfer_orders=tuple(
            transfer.coefficient(order) for order in (1, 3, 5)
        ),
        max_transfer_ward_residual_through_order_five=max(
            abs(ward_residual.coefficient(order)) for order in range(6)
        ),
    )


def integrate_isotropic_stress_with_certified_tail(
    q_values: tuple[float, ...],
    stresses: tuple[ModeStress, ...],
    *,
    energy_tail: CertifiedPowerLawTail,
    pressure_tail: CertifiedPowerLawTail,
) -> IntegratedStress:
    """Integrate q^2 s(q)/(2 pi^2) and attach exact UV-tail error bounds."""

    if len(q_values) != len(stresses) or len(q_values) < 2:
        raise ValueError("q_values and stresses must have the same length of at least two")
    if not all(math.isfinite(q) and q > 0.0 for q in q_values):
        raise ValueError("q grid must be finite and positive")
    if any(right <= left for left, right in zip(q_values, q_values[1:])):
        raise ValueError("q grid must be strictly increasing")
    for stress in stresses:
        if not all(
            math.isfinite(value)
            for value in (
                stress.energy_density_over_h0_four,
                stress.pressure_over_h0_four,
            )
        ):
            raise ValueError("stress samples must be finite")

    last_q = q_values[-1]
    if last_q < energy_tail.start_q or last_q < pressure_tail.start_q:
        raise ValueError("the q grid must reach both certified tail domains")
    if (
        abs(stresses[-1].energy_density_over_h0_four)
        > energy_tail.coefficient * last_q ** (-energy_tail.exponent)
    ):
        raise ValueError("last energy sample violates its certified tail bound")
    if (
        abs(stresses[-1].pressure_over_h0_four)
        > pressure_tail.coefficient * last_q ** (-pressure_tail.exponent)
    ):
        raise ValueError("last pressure sample violates its certified tail bound")

    energy_terms: list[float] = []
    pressure_terms: list[float] = []
    for left, right, left_stress, right_stress in zip(
        q_values,
        q_values[1:],
        stresses,
        stresses[1:],
    ):
        width = right - left
        energy_terms.append(
            0.5
            * width
            * (
                left * left * left_stress.energy_density_over_h0_four
                + right * right * right_stress.energy_density_over_h0_four
            )
        )
        pressure_terms.append(
            0.5
            * width
            * (
                left * left * left_stress.pressure_over_h0_four
                + right * right * right_stress.pressure_over_h0_four
            )
        )
    measure = 1.0 / (2.0 * math.pi**2)
    return IntegratedStress(
        energy_density_over_h0_four=measure * math.fsum(energy_terms),
        pressure_over_h0_four=measure * math.fsum(pressure_terms),
        energy_tail_absolute_bound=energy_tail.isotropic_integral_bound_from(last_q),
        pressure_tail_absolute_bound=pressure_tail.isotropic_integral_bound_from(last_q),
    )


def bare_mode_stress(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
    u: complex,
    du_dx: complex,
) -> ModeStress:
    """Evaluate the on-shell covariant stress of one canonical mode."""

    _validate_parameters(q, mu, xi)
    if not all(
        math.isfinite(value)
        for value in (u.real, u.imag, du_dx.real, du_dx.imag)
    ):
        raise ValueError("mode and derivative must be finite")

    a = jet.a
    hubble = jet.d1 / a
    acceleration = jet.d2 / a
    amplitude = abs(u) ** 2
    cross = 2.0 * (du_dx * u.conjugate()).real
    kinetic = abs(du_dx) ** 2
    mass_squared = a * a * mu * mu

    energy_bracket = (
        kinetic
        + (q * q + mass_squared) * amplitude
        + (6.0 * xi - 1.0)
        * (hubble * cross - hubble * hubble * amplitude)
    )
    pressure_bracket = (
        kinetic
        - hubble * cross
        + (hubble * hubble - q * q / 3.0 - mass_squared) * amplitude
        + 2.0
        * xi
        * (
            -2.0 * kinetic
            + 3.0 * hubble * cross
            + (
                2.0 * q * q
                + 2.0 * mass_squared
                + (12.0 * xi - 2.0) * acceleration
                - 3.0 * hubble * hubble
            )
            * amplitude
        )
    )
    scale = 0.5 / a**4
    return ModeStress(
        energy_density_over_h0_four=scale * energy_bracket,
        pressure_over_h0_four=scale * pressure_bracket,
    )


def renormalized_mode_stress(
    jet: ScaleFactorJet,
    *,
    q: float,
    mu: float,
    xi: float,
    u: complex,
    du_dx: complex,
) -> ModeStress:
    """Subtract the local 0/2/4 counterterm from one bare mode."""

    bare = bare_mode_stress(jet, q=q, mu=mu, xi=xi, u=u, du_dx=du_dx)
    subtraction = fourth_order_counterterm(jet, q=q, mu=mu, xi=xi).stress
    return ModeStress(
        energy_density_over_h0_four=(
            bare.energy_density_over_h0_four
            - subtraction.energy_density_over_h0_four
        ),
        pressure_over_h0_four=(
            bare.pressure_over_h0_four - subtraction.pressure_over_h0_four
        ),
    )
