"""Conditional bridge from a smooth-quench spectrum to present abundance.

The theorem implemented here starts *after* a production hypersurface has been
declared.  All dimensionful inputs use one common natural-energy unit with
``hbar=c=k_B=1``.  Entropy bookkeeping fixes the production scale factor,
the exact smooth-tanh Bogoliubov spectrum fixes the created phase-space
occupation, and collisionless Liouville evolution redshifts physical momentum
as ``p_0=a_* p_*``.

For a stable, decoupled species and ``a_0=1``,

    n_0 = a_*^3 g/(2 pi^2) integral dp p^2 f_*(p),

    rho_0 = a_*^3 g/(2 pi^2)
            integral dp p^2 sqrt(m_f^2 + a_*^2 p^2) f_*(p),

    Omega_0 = rho_0 / (3 M_Pl^2 H_0^2).

This closes a conditional forward map once the quench, entropy history and
cosmological normalization are supplied.  It does not derive those inputs,
the finite-time source ``q(N)``, a renormalized FLRW stress tensor, or
perturbations.  The numerical receipt is a finite-window Simpson quadrature.
It carries a separate analytic upper bound on the omitted smooth-quench UV
tail, but not a discretization-error certificate for the finite window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
    integrate_quench_densities,
    smooth_tanh_mode,
)
from examples.physics.kinetic_dark_sector_quench_tail_bound import (
    smooth_quench_present_tail_certificate,
)


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


@dataclass(frozen=True)
class EntropyRedshiftContract:
    """Inputs that determine ``a_*`` from comoving-entropy bookkeeping.

    ``comoving_entropy_growth = S_0/S_*`` is one for adiabatic expansion and
    at least one when entropy is injected after production.
    """

    temperature_at_production: float
    temperature_today: float
    entropy_dof_at_production: float
    entropy_dof_today: float
    comoving_entropy_growth: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "temperature_at_production",
            "temperature_today",
            "entropy_dof_at_production",
            "entropy_dof_today",
            "comoving_entropy_growth",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name),
            )
        if self.comoving_entropy_growth < 1.0:
            raise ValueError("comoving_entropy_growth must be >= 1")
        if self.scale_factor_at_production > 1.0:
            raise ValueError(
                "entropy inputs place production after the normalized present"
            )

    @property
    def scale_factor_at_production(self) -> float:
        log_scale_factor = (
            (
                math.log(self.entropy_dof_today)
                - math.log(self.entropy_dof_at_production)
                - math.log(self.comoving_entropy_growth)
            )
            / 3.0
            + math.log(self.temperature_today)
            - math.log(self.temperature_at_production)
        )
        if log_scale_factor > 0.0:
            raise ValueError(
                "entropy inputs place production after the normalized present"
            )
        scale_factor = math.exp(log_scale_factor)
        if not math.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("production scale factor is not finite and positive")
        return scale_factor


@dataclass(frozen=True)
class NaturalUnitCosmology:
    """Present normalization in the same natural-energy unit as the species."""

    hubble_today: float
    reduced_planck_mass: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hubble_today",
            _positive_finite(self.hubble_today, "hubble_today"),
        )
        object.__setattr__(
            self,
            "reduced_planck_mass",
            _positive_finite(self.reduced_planck_mass, "reduced_planck_mass"),
        )
        _ = self.critical_density_today

    @property
    def critical_density_today(self) -> float:
        planck_hubble = self.reduced_planck_mass * self.hubble_today
        value = 3.0 * planck_hubble * planck_hubble
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("critical density is not finite and positive")
        return value


@dataclass(frozen=True)
class CollisionlessAbundanceCertificate:
    """Finite-window receipt for the conditional collisionless forward map."""

    label: str
    scale_factor_at_production: float
    critical_density_today: float
    production_number_density: float
    present_number_density: float
    present_number_density_from_dilution: float
    number_dilution_residual: float
    present_rest_density: float
    present_energy_density: float
    present_pressure: float
    present_equation_of_state: float
    omega_produced_today: float
    omitted_number_density_upper: float
    omitted_energy_density_upper: float
    omitted_omega_upper: float
    relative_kinetic_energy: float
    relative_kinetic_energy_upper_bound: float
    cold_bound_residual: float
    momentum_max: float
    intervals: int
    ultraviolet_status: str
    quadrature_status: str
    tail_numerical_status: str
    production_approximation: str
    role: str = (
        "CONDITIONAL_BETA_TO_ABUNDANCE_FORWARD_MAP_NOT_ABUNDANCE_PREDICTION"
    )


def smooth_quench_collisionless_abundance(
    species: QuantumSeatSpecies,
    *,
    entropy: EntropyRedshiftContract,
    cosmology: NaturalUnitCosmology,
    momentum_max: float | None = None,
    intervals: int = 2400,
) -> CollisionlessAbundanceCertificate:
    """Propagate the created smooth-quench excess to today's abundance.

    The occupation is the created excess
    ``(1+2*n_in)|beta_p|^2``, not a pre-existing population.  The species is
    assumed stable, decoupled and constant-mass after the quench.  The flat
    asymptotic tanh spectrum is adopted on the production surface; using it in
    FLRW additionally requires a local-quench approximation such as
    ``H_* duration << 1``, which this function does not certify.
    """

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    if not isinstance(entropy, EntropyRedshiftContract):
        raise ValueError("entropy must be an EntropyRedshiftContract")
    if not isinstance(cosmology, NaturalUnitCosmology):
        raise ValueError("cosmology must be a NaturalUnitCosmology")
    if isinstance(intervals, bool) or not isinstance(intervals, int):
        raise ValueError("intervals must be an integer")
    if intervals < 200:
        raise ValueError("intervals must be at least 200")
    if momentum_max is not None:
        momentum_max = _positive_finite(momentum_max, "momentum_max")

    try:
        production = integrate_quench_densities(
            species,
            protocol="smooth",
            momentum_max=momentum_max,
            intervals=intervals,
        )
    except OverflowError as error:
        raise ValueError(
            "production quadrature is outside the finite numerical domain"
        ) from error
    scale_factor = entropy.scale_factor_at_production
    step = production.momentum_max / production.intervals
    number_terms: list[float] = []
    kinetic_terms: list[float] = []
    pressure_terms: list[float] = []
    momentum2_terms: list[float] = []
    for index in range(production.intervals + 1):
        momentum = index * step
        mode = smooth_tanh_mode(species, momentum)
        simpson_weight = (
            1.0
            if index in (0, production.intervals)
            else (4.0 if index % 2 else 2.0)
        )
        occupation = mode.created_occupation
        if not math.isfinite(occupation) or occupation < 0.0:
            raise ValueError("created occupation is not finite and non-negative")
        momentum_squared = momentum * momentum
        if not math.isfinite(momentum_squared):
            raise ValueError("momentum window is outside the finite numerical domain")
        radial_number = momentum_squared * occupation
        momentum_today = scale_factor * momentum
        energy_today = math.hypot(species.mass_out, momentum_today)
        kinetic_energy_today = momentum_today * (
            momentum_today / (energy_today + species.mass_out)
        )
        pressure_energy = momentum_today * (momentum_today / energy_today) / 3.0
        number_terms.append(simpson_weight * radial_number)
        kinetic_terms.append(
            simpson_weight * radial_number * kinetic_energy_today
        )
        pressure_terms.append(
            simpson_weight * radial_number * pressure_energy
        )
        momentum2_terms.append(
            simpson_weight * radial_number * momentum_squared
        )

    spherical_factor_today = (
        species.degeneracy
        / (2.0 * math.pi * math.pi)
        * scale_factor**3
        * step
        / 3.0
    )
    number_from_quadrature = spherical_factor_today * math.fsum(number_terms)
    number_from_dilution = scale_factor**3 * production.number_density
    rest_density = species.mass_out * number_from_quadrature
    kinetic_density = spherical_factor_today * math.fsum(kinetic_terms)
    energy_density = rest_density + kinetic_density
    pressure = spherical_factor_today * math.fsum(pressure_terms)
    if rest_density > 0.0:
        relative_kinetic = kinetic_density / rest_density
        mean_momentum_squared = (
            math.fsum(momentum2_terms) / math.fsum(number_terms)
        )
        relative_kinetic_bound = (
            0.5
            * scale_factor**2
            * mean_momentum_squared
            / species.mass_out**2
        )
    else:
        relative_kinetic = 0.0
        relative_kinetic_bound = 0.0
    equation_of_state = pressure / energy_density if energy_density > 0.0 else 0.0
    critical_density = cosmology.critical_density_today
    omega_produced = energy_density / critical_density
    cold_bound_residual = relative_kinetic - relative_kinetic_bound
    tail = smooth_quench_present_tail_certificate(
        species,
        momentum_start=production.momentum_max,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
    )

    finite_outputs = (
        number_from_quadrature,
        number_from_dilution,
        energy_density,
        pressure,
        rest_density,
        relative_kinetic,
        relative_kinetic_bound,
        equation_of_state,
        critical_density,
        omega_produced,
        cold_bound_residual,
        tail.present_number_density_upper,
        tail.present_energy_density_upper,
        tail.omega_produced_upper,
    )
    if any(not math.isfinite(value) for value in finite_outputs):
        raise ValueError("abundance output is outside the finite numerical domain")
    invariant_scale = max(1.0, relative_kinetic_bound)
    if (
        rest_density > energy_density
        or pressure < 0.0
        or equation_of_state < 0.0
        or equation_of_state > 1.0 / 3.0 + 32.0 * math.ulp(1.0)
        or cold_bound_residual > 128.0 * math.ulp(invariant_scale)
    ):
        raise ValueError("collisionless abundance invariants failed")

    return CollisionlessAbundanceCertificate(
        label=species.label,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
        production_number_density=production.number_density,
        present_number_density=number_from_quadrature,
        present_number_density_from_dilution=number_from_dilution,
        number_dilution_residual=number_from_quadrature - number_from_dilution,
        present_rest_density=rest_density,
        present_energy_density=energy_density,
        present_pressure=pressure,
        present_equation_of_state=equation_of_state,
        omega_produced_today=omega_produced,
        omitted_number_density_upper=tail.present_number_density_upper,
        omitted_energy_density_upper=tail.present_energy_density_upper,
        omitted_omega_upper=tail.omega_produced_upper,
        relative_kinetic_energy=relative_kinetic,
        relative_kinetic_energy_upper_bound=relative_kinetic_bound,
        cold_bound_residual=cold_bound_residual,
        momentum_max=production.momentum_max,
        intervals=production.intervals,
        ultraviolet_status="ANALYTIC_EXPONENTIAL_OMITTED_TAIL_BOUND_ATTACHED",
        quadrature_status="FINITE_WINDOW_SIMPSON_NOT_UV_TAIL_CERTIFICATE",
        tail_numerical_status=tail.numerical_status,
        production_approximation=(
            "ASYMPTOTIC_MINKOWSKI_CREATED_EXCESS_HSTAR_DURATION_NOT_CERTIFIED"
        ),
    )
