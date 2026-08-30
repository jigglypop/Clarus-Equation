"""Analytic ultraviolet tail bound for the smooth-tanh created excess.

For ``p >= P > 0``, let

    omega_i,o = sqrt(p^2 + m_i,o^2),
    x = pi*tau*abs(omega_o-omega_i)/2.

Using ``abs(omega_o-omega_i) <= abs(m_o^2-m_i^2)/(2p)``,
``sinh(x) <= x exp(x)``, and the exponential lower bound on each
denominator sinh gives

    f_created(p) <= A(P) p^-2 exp(-2*pi*tau*p),

where ``f_created=(1+2*n_in)|beta_p|^2``.  The remaining number and
collisionless present-energy tails then have elementary exponential-integral
bounds.  This is an exact real-arithmetic inequality for the adopted
asymptotic Minkowski tanh spectrum.  It is not a renormalized FLRW stress
tensor or a certificate for the finite-window quadrature below ``P``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import sys

from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
)


_LOG_FLOAT_MAX = math.log(sys.float_info.max)
_MIN_SUBNORMAL = math.nextafter(0.0, 1.0)
_LOG_MIN_SUBNORMAL = math.log(_MIN_SUBNORMAL)


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _exp_upward(log_value: float) -> float:
    """Evaluate a positive analytic bound and nudge the final exp upward.

    This protects the final conversion and subnormal case, but is not interval
    arithmetic: preceding log/expm1 operations may round in either direction.
    """

    if log_value == -math.inf:
        return 0.0
    if not math.isfinite(log_value) or log_value >= _LOG_FLOAT_MAX:
        raise ValueError("tail bound is outside the finite numerical domain")
    if log_value <= _LOG_MIN_SUBNORMAL:
        return _MIN_SUBNORMAL
    value = math.exp(log_value)
    upper = math.nextafter(value, math.inf)
    if not math.isfinite(upper):
        raise ValueError("tail bound is outside the finite numerical domain")
    return upper


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


@dataclass(frozen=True)
class SmoothQuenchTailCertificate:
    """Closed-form tail receipt for one declared momentum threshold."""

    momentum_start: float
    exponential_decay_rate: float
    log_occupation_coefficient: float
    present_number_density_upper: float
    present_energy_density_upper: float
    present_pressure_upper: float
    omega_produced_upper: float
    scale_factor_at_production: float
    critical_density_today: float
    proof_assumptions: tuple[str, ...]
    numerical_status: str = (
        "FLOAT_EVALUATION_OF_ANALYTIC_BOUND_NOT_INTERVAL_CERTIFIED"
    )
    role: str = (
        "ANALYTIC_CREATED_EXCESS_UV_TAIL_BOUND_NOT_RENORMALIZED_FLRW_STRESS"
    )


def _log_occupation_coefficient(
    species: QuantumSeatSpecies,
    momentum_start: float,
) -> tuple[float, float]:
    rate = 2.0 * math.pi * species.duration
    mass_square_difference = abs(
        (species.mass_out - species.mass_in)
        * (species.mass_out + species.mass_in)
    )
    if mass_square_difference == 0.0:
        return rate, -math.inf
    if not math.isfinite(rate) or not math.isfinite(mass_square_difference):
        raise ValueError("quench tail parameters are outside the finite domain")
    c_value = math.pi * species.duration * mass_square_difference / 4.0
    rate_times_start = rate * momentum_start
    denominator_factor = -math.expm1(-rate_times_start)
    stimulation = 1.0 + 2.0 * species.initial_mode_occupation
    log_coefficient = (
        math.log(stimulation)
        + math.log(4.0)
        + 2.0 * math.log(c_value)
        + 2.0 * c_value / momentum_start
        - 2.0 * math.log(denominator_factor)
    )
    if not math.isfinite(log_coefficient):
        raise ValueError("occupation tail coefficient is not finite")
    return rate, log_coefficient


def smooth_quench_created_occupation_tail_upper(
    species: QuantumSeatSpecies,
    *,
    momentum: object,
    momentum_start: object,
) -> float:
    """Return the analytic created-occupation upper bound for ``p >= P``."""

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    p_value = _positive_finite(momentum, "momentum")
    start = _positive_finite(momentum_start, "momentum_start")
    if p_value < start:
        raise ValueError("momentum must be >= momentum_start")
    rate, log_coefficient = _log_occupation_coefficient(species, start)
    if log_coefficient == -math.inf:
        return 0.0
    return _exp_upward(
        log_coefficient - 2.0 * math.log(p_value) - rate * p_value
    )


def smooth_quench_present_tail_certificate(
    species: QuantumSeatSpecies,
    *,
    momentum_start: object,
    scale_factor_at_production: object,
    critical_density_today: object,
) -> SmoothQuenchTailCertificate:
    """Bound all omitted created-excess modes above ``momentum_start``."""

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    start = _positive_finite(momentum_start, "momentum_start")
    scale_factor = _positive_finite(
        scale_factor_at_production,
        "scale_factor_at_production",
    )
    if scale_factor > 1.0:
        raise ValueError("scale_factor_at_production must be <= 1")
    critical_density = _positive_finite(
        critical_density_today,
        "critical_density_today",
    )
    rate, log_coefficient = _log_occupation_coefficient(species, start)
    if log_coefficient == -math.inf:
        number_upper = 0.0
        energy_upper = 0.0
        pressure_upper = 0.0
        omega_upper = 0.0
    else:
        log_common = log_coefficient - rate * start
        log_prefactor = (
            math.log(species.degeneracy)
            - math.log(2.0 * math.pi * math.pi)
            + 3.0 * math.log(scale_factor)
        )
        log_number_upper = log_prefactor + log_common - math.log(rate)
        log_rest_term = math.log(species.mass_out) - math.log(rate)
        log_momentum_term = (
            math.log(scale_factor)
            - math.log(rate)
            + math.log(start + 1.0 / rate)
        )
        log_energy_bracket = _logaddexp(log_rest_term, log_momentum_term)
        log_energy_upper = log_prefactor + log_common + log_energy_bracket
        number_upper = _exp_upward(log_number_upper)
        energy_upper = _exp_upward(log_energy_upper)
        pressure_upper = math.nextafter(energy_upper / 3.0, math.inf)
        omega_upper = math.nextafter(
            energy_upper / critical_density,
            math.inf,
        )
        if not math.isfinite(omega_upper):
            raise ValueError("omega tail bound is outside the finite domain")

    return SmoothQuenchTailCertificate(
        momentum_start=start,
        exponential_decay_rate=rate,
        log_occupation_coefficient=log_coefficient,
        present_number_density_upper=number_upper,
        present_energy_density_upper=energy_upper,
        present_pressure_upper=pressure_upper,
        omega_produced_upper=omega_upper,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
        proof_assumptions=(
            "exact asymptotic Minkowski smooth-tanh Bogoliubov spectrum",
            "created excess f=(1+2*n_in)|beta|^2",
            "stable decoupled constant-mass propagation after production",
            "physical momentum redshifts as p0=a_star*p_star",
            "bound certifies p>=momentum_start only",
        ),
    )
