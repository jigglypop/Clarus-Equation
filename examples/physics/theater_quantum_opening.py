"""Exact conditional quantum-quench model for the theater-opening analogy.

Each seat species is represented by a multiplet of real scalar modes.  A smooth
tanh change of its mass produces a calculable Bogoliubov occupation.  The
species degeneracy and final rest mass implement the number of equivalent seats
and their material weight.

This is a four-dimensional effective model on an already declared opening
surface.  It does not derive spacetime itself, the initial vacuum, the quench
profile, or the energy scale from zero-dimensional data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _log_sinh_positive(value: float) -> float:
    """Return log(sinh(value)) stably for a strictly positive argument."""

    if value <= 0.0:
        raise ValueError("log-sinh argument must be positive")
    if value < 20.0:
        return math.log(math.sinh(value))
    return value - math.log(2.0) + math.log1p(-math.exp(-2.0 * value))


@dataclass(frozen=True)
class QuantumSeatSpecies:
    """One real-scalar multiplet and its smooth opening protocol."""

    label: str
    degeneracy: int
    mass_in: float
    mass_out: float
    duration: float
    initial_mode_occupation: float = 0.0
    role: str = "CONDITIONAL_SMOOTH_QUENCH_SPECIES"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("species label must be non-empty")
        if (
            isinstance(self.degeneracy, bool)
            or not isinstance(self.degeneracy, int)
            or self.degeneracy < 1
        ):
            raise ValueError("degeneracy must be a positive integer")
        for name, value in (
            ("mass_in", self.mass_in),
            ("mass_out", self.mass_out),
            ("duration", self.duration),
            ("initial_mode_occupation", self.initial_mode_occupation),
        ):
            _require_finite(name, value)
        if self.mass_in <= 0.0 or self.mass_out <= 0.0:
            raise ValueError("asymptotic masses must be positive")
        if self.duration <= 0.0:
            raise ValueError("smooth quench duration must be positive")
        if self.initial_mode_occupation < 0.0:
            raise ValueError("initial mode occupation must be non-negative")

    @classmethod
    def from_seat_weight(
        cls,
        *,
        label: str,
        degeneracy: int,
        mass_in: float,
        reference_energy: float,
        relative_rest_mass: float,
        duration: float,
        initial_mode_occupation: float = 0.0,
    ) -> QuantumSeatSpecies:
        """Match the seat weight to the final rest mass E_* epsilon_s."""

        _require_finite("reference_energy", reference_energy)
        _require_finite("relative_rest_mass", relative_rest_mass)
        if reference_energy <= 0.0 or relative_rest_mass <= 0.0:
            raise ValueError("seat energy scales must be positive")
        return cls(
            label=label,
            degeneracy=degeneracy,
            mass_in=mass_in,
            mass_out=reference_energy * relative_rest_mass,
            duration=duration,
            initial_mode_occupation=initial_mode_occupation,
        )


@dataclass(frozen=True)
class BogoliubovMode:
    momentum: float
    omega_in: float
    omega_out: float
    alpha_squared: float
    beta_squared: float
    created_occupation: float
    normalization_residual: float
    protocol: str


def _frequencies(
    species: QuantumSeatSpecies,
    momentum: float,
) -> tuple[float, float]:
    _require_finite("momentum", momentum)
    if momentum < 0.0:
        raise ValueError("momentum must be non-negative")
    return (
        math.hypot(momentum, species.mass_in),
        math.hypot(momentum, species.mass_out),
    )


def bosonic_out_occupation(
    *,
    beta_squared: float,
    initial_occupation: float,
) -> float:
    """Return n_out=n_in+(1+2*n_in)|beta|^2 for one bosonic mode."""

    for name, value in (
        ("beta_squared", beta_squared),
        ("initial_occupation", initial_occupation),
    ):
        _require_finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    return initial_occupation + (1.0 + 2.0 * initial_occupation) * beta_squared


def instantaneous_mode(
    species: QuantumSeatSpecies,
    momentum: float,
) -> BogoliubovMode:
    """Return the exact sudden-quench mode as a UV negative control."""

    omega_in, omega_out = _frequencies(species, momentum)
    root_ratio = math.sqrt(omega_out / omega_in)
    inverse_root_ratio = 1.0 / root_ratio
    alpha = 0.5 * (root_ratio + inverse_root_ratio)
    beta = 0.5 * (root_ratio - inverse_root_ratio)
    alpha_squared = alpha * alpha
    beta_squared = beta * beta
    out_occupation = bosonic_out_occupation(
        beta_squared=beta_squared,
        initial_occupation=species.initial_mode_occupation,
    )
    return BogoliubovMode(
        momentum=momentum,
        omega_in=omega_in,
        omega_out=omega_out,
        alpha_squared=alpha_squared,
        beta_squared=beta_squared,
        created_occupation=(
            out_occupation - species.initial_mode_occupation
        ),
        normalization_residual=alpha_squared - beta_squared - 1.0,
        protocol="INSTANTANEOUS_UV_NEGATIVE_CONTROL",
    )


def smooth_tanh_mode(
    species: QuantumSeatSpecies,
    momentum: float,
) -> BogoliubovMode:
    """Return the exact mode occupation for a smooth tanh mass-squared quench."""

    omega_in, omega_out = _frequencies(species, momentum)
    prefactor = math.pi * species.duration
    denominator_log = (
        _log_sinh_positive(prefactor * omega_in)
        + _log_sinh_positive(prefactor * omega_out)
    )
    omega_minus = 0.5 * abs(omega_out - omega_in)
    if omega_minus == 0.0:
        beta_squared = 0.0
    else:
        beta_squared = math.exp(
            2.0 * _log_sinh_positive(prefactor * omega_minus)
            - denominator_log
        )
    omega_plus = 0.5 * (omega_out + omega_in)
    alpha_squared = math.exp(
        2.0 * _log_sinh_positive(prefactor * omega_plus)
        - denominator_log
    )
    out_occupation = bosonic_out_occupation(
        beta_squared=beta_squared,
        initial_occupation=species.initial_mode_occupation,
    )
    return BogoliubovMode(
        momentum=momentum,
        omega_in=omega_in,
        omega_out=omega_out,
        alpha_squared=alpha_squared,
        beta_squared=beta_squared,
        created_occupation=(
            out_occupation - species.initial_mode_occupation
        ),
        normalization_residual=alpha_squared - beta_squared - 1.0,
        protocol="SMOOTH_TANH_EXACT",
    )


@dataclass(frozen=True)
class SuddenQuenchUVVerdict:
    beta_squared_power: int = -4
    radial_number_integrand_power: int = -2
    radial_energy_integrand_power: int = -1
    number_density_uv_convergent: bool = True
    energy_density_uv_convergent: bool = False
    status: str = "SUDDEN_ENERGY_LOG_DIVERGENT_NOT_COSMOLOGY_SOURCE"


@dataclass(frozen=True)
class QuenchDensityAudit:
    label: str
    number_density: float
    excess_energy_density: float
    dephased_pressure: float
    equation_of_state: float
    mean_energy_per_created_quantum: float
    rms_momentum: float
    rms_momentum_over_mass: float
    momentum_max: float
    intervals: int
    maximum_bogoliubov_residual: float
    protocol: str
    ultraviolet_status: str
    stress_role: str


def _default_smooth_momentum_max(species: QuantumSeatSpecies) -> float:
    return max(
        12.0 * species.mass_in,
        12.0 * species.mass_out,
        20.0 / species.duration,
    )


def integrate_quench_densities(
    species: QuantumSeatSpecies,
    *,
    protocol: str = "smooth",
    momentum_max: float | None = None,
    intervals: int = 2400,
) -> QuenchDensityAudit:
    """Integrate asymptotic out-particle number, energy and dephased pressure.

    For the instantaneous protocol an explicit UV cutoff is mandatory because
    its excitation energy is logarithmically divergent in three spatial
    dimensions.  The smooth result is the finite out-particle excess above the
    final Minkowski vacuum, not the full renormalized FLRW stress tensor.
    """

    if intervals < 200:
        raise ValueError("intervals must be at least 200")
    if intervals % 2:
        intervals += 1
    if protocol not in {"smooth", "instantaneous"}:
        raise ValueError("protocol must be smooth or instantaneous")
    if protocol == "instantaneous" and momentum_max is None:
        raise ValueError("instantaneous energy requires an explicit UV cutoff")
    upper = (
        _default_smooth_momentum_max(species)
        if momentum_max is None
        else momentum_max
    )
    _require_finite("momentum_max", upper)
    if upper <= 0.0:
        raise ValueError("momentum_max must be positive")

    mode_function = (
        smooth_tanh_mode if protocol == "smooth" else instantaneous_mode
    )
    step = upper / intervals
    number_sum = 0.0
    energy_sum = 0.0
    pressure_sum = 0.0
    momentum2_sum = 0.0
    max_residual = 0.0
    for index in range(intervals + 1):
        momentum = index * step
        mode = mode_function(species, momentum)
        weight = 1.0 if index in (0, intervals) else (4.0 if index % 2 else 2.0)
        occupation = mode.created_occupation
        radial_number = momentum * momentum * occupation
        number_sum += weight * radial_number
        energy_sum += weight * radial_number * mode.omega_out
        pressure_sum += (
            weight
            * momentum**4
            * occupation
            / (3.0 * mode.omega_out)
        )
        momentum2_sum += weight * momentum**4 * occupation
        max_residual = max(max_residual, abs(mode.normalization_residual))

    spherical_factor = (
        species.degeneracy / (2.0 * math.pi * math.pi) * step / 3.0
    )
    number_density = spherical_factor * number_sum
    energy_density = spherical_factor * energy_sum
    pressure = spherical_factor * pressure_sum
    momentum2_density = spherical_factor * momentum2_sum
    if number_density > 0.0:
        mean_energy = energy_density / number_density
        rms_momentum = math.sqrt(momentum2_density / number_density)
    else:
        mean_energy = 0.0
        rms_momentum = 0.0
    equation_of_state = pressure / energy_density if energy_density > 0.0 else 0.0
    return QuenchDensityAudit(
        label=species.label,
        number_density=number_density,
        excess_energy_density=energy_density,
        dephased_pressure=pressure,
        equation_of_state=equation_of_state,
        mean_energy_per_created_quantum=mean_energy,
        rms_momentum=rms_momentum,
        rms_momentum_over_mass=rms_momentum / species.mass_out,
        momentum_max=upper,
        intervals=intervals,
        maximum_bogoliubov_residual=max_residual,
        protocol=(
            "SMOOTH_TANH_EXACT"
            if protocol == "smooth"
            else "INSTANTANEOUS_CUTOFF_DEPENDENT"
        ),
        ultraviolet_status=(
            "FINITE_FOR_POSITIVE_DURATION"
            if protocol == "smooth"
            else "ENERGY_LOG_DIVERGENT_AS_CUTOFF_REMOVED"
        ),
        stress_role=(
            "ASYMPTOTIC_OUT_EXCESS_NOT_FULL_RENORMALIZED_FLRW_STRESS"
        ),
    )


@dataclass(frozen=True)
class OpeningSpeciesFraction:
    label: str
    degeneracy: int
    final_rest_mass: float
    number_density: float
    energy_density: float
    energy_fraction: float
    mean_energy_over_rest_mass: float


def multi_species_opening(
    species: tuple[QuantumSeatSpecies, ...],
    *,
    intervals: int = 2400,
) -> tuple[OpeningSpeciesFraction, ...]:
    """Return post-quench energy fractions for distinct seat materials."""

    if not species:
        raise ValueError("at least one quantum seat species is required")
    labels = tuple(item.label for item in species)
    if len(set(labels)) != len(labels):
        raise ValueError("quantum seat labels must be unique")
    audits = tuple(
        integrate_quench_densities(item, intervals=intervals)
        for item in species
    )
    total_energy = math.fsum(item.excess_energy_density for item in audits)
    if total_energy <= 0.0:
        raise ZeroDivisionError("the selected quench creates no excitation energy")
    return tuple(
        OpeningSpeciesFraction(
            label=item.label,
            degeneracy=item.degeneracy,
            final_rest_mass=item.mass_out,
            number_density=audit.number_density,
            energy_density=audit.excess_energy_density,
            energy_fraction=audit.excess_energy_density / total_energy,
            mean_energy_over_rest_mass=(
                audit.mean_energy_per_created_quantum / item.mass_out
            ),
        )
        for item, audit in zip(species, audits)
    )


def scalar_energy_transfer_rate(
    *,
    degeneracy: int,
    mass_squared_rate: float,
    renormalized_field_squared: float,
) -> float:
    """Return Q_s=g_s*dot(m_s^2)*<chi_s^2>_ren/2 from the action."""

    if (
        isinstance(degeneracy, bool)
        or not isinstance(degeneracy, int)
        or degeneracy < 1
    ):
        raise ValueError("degeneracy must be a positive integer")
    _require_finite("mass_squared_rate", mass_squared_rate)
    _require_finite("renormalized_field_squared", renormalized_field_squared)
    return (
        0.5
        * degeneracy
        * mass_squared_rate
        * renormalized_field_squared
    )


def total_ward_residual(
    *,
    scalar_transfer_rates: tuple[float, ...],
    clock_transfer_rate: float,
) -> float:
    """Return sum_s Q_s+Q_clock for the declared action split."""

    _require_finite("clock_transfer_rate", clock_transfer_rate)
    for rate in scalar_transfer_rates:
        _require_finite("scalar_transfer_rate", rate)
    return math.fsum((*scalar_transfer_rates, clock_transfer_rate))
