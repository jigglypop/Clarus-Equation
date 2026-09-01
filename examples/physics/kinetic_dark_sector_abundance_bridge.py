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
from examples.physics.kinetic_dark_sector_perturbation_gate import (
    GaussianNormalModePerturbationAudit,
    audit_gaussian_normal_mode_perturbations,
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


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
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


@dataclass(frozen=True)
class FixedBasisBilinearQuenchCosmologyAudit:
    """Composition receipt for a two-field Gaussian quench and late WKB gate.

    The path is the explicitly supplied interpolation

    ``K(t) = K_in + s(t) (K_out-K_in)``,
    ``s(t) = (1+tanh(t/tau))/2``.

    Only commuting, nondegenerate endpoint matrices are admitted.  Under that
    restriction one fixed orthogonal basis diagonalizes the whole path, so the
    two exact scalar Bogoliubov spectra may be used without omitting a rotating-
    basis mixing term.  The produced state is a squeezed Gaussian state, not
    the finite coherent-product state used by the E45 cell witness.
    """

    initial_action_parameter_manifest: tuple[float, float, float, float]
    final_action_parameter_manifest: tuple[float, float, float, float]
    interpolation: str
    matrix_commutator_frobenius_ev4: float
    matrix_commutator_relative_residual: float
    fixed_basis_off_diagonal_relative_residual: float
    initial_normal_mode_gap_fraction: float
    final_normal_mode_gap_fraction: float
    fixed_basis_mode_mass_squared_at_initial_ev2: tuple[float, float]
    fixed_basis_mode_mass_squared_at_final_ev2: tuple[float, float]
    fixed_basis_mode_masses_at_initial_ev: tuple[float, float]
    fixed_basis_mode_masses_at_final_ev: tuple[float, float]
    quench_duration_ev_minus1: float
    hubble_at_production_ev: float
    local_quench_ratio: float
    local_quench_limit: float
    local_quench_pass: bool
    quench_species: tuple[QuantumSeatSpecies, QuantumSeatSpecies]
    abundance_certificates: tuple[
        CollisionlessAbundanceCertificate,
        CollisionlessAbundanceCertificate,
    ]
    present_created_number_densities_ev3: tuple[float, float]
    present_created_rest_density_constants_ev4: tuple[float, float]
    present_created_energy_densities_ev4: tuple[float, float]
    present_created_pressures_ev4: tuple[float, float]
    present_background_rms_momentum_ratio_upper: tuple[float, float]
    present_dust_replacement_relative_errors: tuple[float, float]
    present_created_energy_density_total_ev4: float
    present_dust_input_density_total_ev4: float
    omega_created_today: float
    omega_dust_input_today: float
    omega_vacuum_today: float
    omega_created_energy_plus_vacuum_today: float
    omega_dust_input_plus_vacuum_today: float
    background_cold_pass: bool
    perturbation_audit: GaussianNormalModePerturbationAudit
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    failed_gates: tuple[str, ...]
    status: str
    mode_order: str = "FINAL_MASS_ASCENDING_IN_FIXED_NORMAL_BASIS"
    fixed_normal_basis_path_derived: bool = True
    conditional_created_abundance_forward_map_derived: bool = True
    conditional_wkb_composition_derived: bool = True
    late_dephased_created_excess_only: bool = True
    bogoliubov_phase_and_anomalous_correlator_propagated: bool = False
    created_state_is_e45_product_gaussian: bool = False
    quench_profile_supplied: bool = True
    initial_in_state_supplied: bool = True
    entropy_history_supplied: bool = True
    quench_driver_energy_ledger_derived: bool = False
    covariant_qft_stress_renormalized: bool = False
    absolute_abundance_predicted: bool = False
    vacuum_energy_derived: bool = False
    integrated_out_environment_stress_added: bool = False
    full_growth_or_lensing_derived: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def _bilinear_mass_matrix_data(
    *,
    system_mass: float,
    environment_mass: float,
    coupling: float,
    endpoint: str,
) -> tuple[float, float, float, float, float]:
    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError(f"{endpoint} endpoint masses must be positive")
    system_mass_squared = system_mass * system_mass
    environment_mass_squared = environment_mass * environment_mass
    determinant = (
        system_mass_squared * environment_mass_squared - coupling * coupling
    )
    if determinant <= 0.0:
        raise ValueError(
            f"{endpoint} bilinear mass matrix must be positive definite"
        )
    frobenius = math.sqrt(
        system_mass_squared * system_mass_squared
        + environment_mass_squared * environment_mass_squared
        + 2.0 * coupling * coupling
    )
    return (
        system_mass_squared,
        environment_mass_squared,
        coupling,
        determinant,
        frobenius,
    )


def audit_fixed_basis_bilinear_quench_cosmology(
    *,
    initial_system_mass_ev: float,
    initial_environment_mass_ev: float,
    initial_bilinear_coupling_ev2: float,
    initial_vacuum_energy_density_ev4: float,
    final_system_mass_ev: float,
    final_environment_mass_ev: float,
    final_bilinear_coupling_ev2: float,
    final_vacuum_energy_density_ev4: float,
    quench_duration_ev_minus1: float,
    initial_mode_occupations: tuple[float, float],
    entropy: EntropyRedshiftContract,
    cosmology: NaturalUnitCosmology,
    hubble_at_production_ev: float,
    comoving_wavenumber_ev: float,
    momentum_max_ev: float | None = None,
    intervals: int = 2400,
    validity_limit: float = 0.1,
    commuting_tolerance: float = 2.0e-12,
) -> FixedBasisBilinearQuenchCosmologyAudit:
    """Map a fixed-basis Gaussian quench to abundance and a WKB discriminator.

    In each admitted normal mode the invisible in-vacuum superposition obeys
    ``a_out = alpha*a_in + beta*a_in^dagger``.  Its late dephased observable
    effect is the created excess ``(1+2*n_in)|beta_p|^2``.  This function
    integrates that excess over all momenta, redshifts it to today, replaces
    the cold part by its exactly conserved rest-density constant, and passes
    only that dust input to the retained two-field WKB perturbation gate.

    The calculation is exact for the supplied flat-space tanh profile and
    conditional for cosmology.  It deliberately does not infer the profile,
    its power source, the in-state, entropy history, or the independent vacuum
    constant from the endpoint action.
    """

    scalar_inputs = {
        "initial_system_mass_ev": initial_system_mass_ev,
        "initial_environment_mass_ev": initial_environment_mass_ev,
        "initial_bilinear_coupling_ev2": initial_bilinear_coupling_ev2,
        "initial_vacuum_energy_density_ev4": initial_vacuum_energy_density_ev4,
        "final_system_mass_ev": final_system_mass_ev,
        "final_environment_mass_ev": final_environment_mass_ev,
        "final_bilinear_coupling_ev2": final_bilinear_coupling_ev2,
        "final_vacuum_energy_density_ev4": final_vacuum_energy_density_ev4,
        "hubble_at_production_ev": hubble_at_production_ev,
        "comoving_wavenumber_ev": comoving_wavenumber_ev,
    }
    checked = {
        name: _finite_real(value, name) for name, value in scalar_inputs.items()
    }
    duration = _positive_finite(
        quench_duration_ev_minus1,
        "quench_duration_ev_minus1",
    )
    limit = _positive_finite(validity_limit, "validity_limit")
    if limit > 0.25:
        raise ValueError("validity_limit must lie in (0, 0.25]")
    tolerance = _positive_finite(
        commuting_tolerance,
        "commuting_tolerance",
    )
    if tolerance > 1.0e-6:
        raise ValueError("commuting_tolerance must be <= 1e-6")
    if checked["hubble_at_production_ev"] < 0.0:
        raise ValueError("hubble_at_production_ev must be nonnegative")
    if checked["comoving_wavenumber_ev"] <= 0.0:
        raise ValueError("comoving_wavenumber_ev must be positive")
    if not isinstance(entropy, EntropyRedshiftContract):
        raise ValueError("entropy must be an EntropyRedshiftContract")
    if not isinstance(cosmology, NaturalUnitCosmology):
        raise ValueError("cosmology must be a NaturalUnitCosmology")
    try:
        occupations_input = tuple(initial_mode_occupations)
    except TypeError as error:
        raise ValueError("initial_mode_occupations must contain two values") from error
    if len(occupations_input) != 2:
        raise ValueError("initial_mode_occupations must contain two values")
    occupations = tuple(
        _finite_real(value, f"initial_mode_occupations[{index}]")
        for index, value in enumerate(occupations_input)
    )
    if any(value < 0.0 for value in occupations):
        raise ValueError("initial mode occupations must be nonnegative")

    initial_data = _bilinear_mass_matrix_data(
        system_mass=checked["initial_system_mass_ev"],
        environment_mass=checked["initial_environment_mass_ev"],
        coupling=checked["initial_bilinear_coupling_ev2"],
        endpoint="initial",
    )
    final_data = _bilinear_mass_matrix_data(
        system_mass=checked["final_system_mass_ev"],
        environment_mass=checked["final_environment_mass_ev"],
        coupling=checked["final_bilinear_coupling_ev2"],
        endpoint="final",
    )
    initial_a, initial_d, initial_b, _, initial_norm = initial_data
    final_a, final_d, final_b, _, final_norm = final_data

    commutator_entry = (
        final_b * (initial_a - initial_d)
        - initial_b * (final_a - final_d)
    )
    commutator_norm = math.sqrt(2.0) * abs(commutator_entry)
    commutator_relative = commutator_norm / (initial_norm * final_norm)
    if commutator_relative > tolerance:
        raise ValueError(
            "endpoint mass matrices do not share a fixed normal basis"
        )

    initial_radius = math.hypot(0.5 * (initial_a - initial_d), initial_b)
    initial_gap_fraction = 2.0 * initial_radius / initial_norm
    if initial_gap_fraction <= tolerance:
        raise ValueError(
            "initial normal masses are degenerate; per-mode preparation is "
            "basis-dependent"
        )
    final_radius = math.hypot(0.5 * (final_a - final_d), final_b)
    final_gap_fraction = 2.0 * final_radius / final_norm
    if final_gap_fraction <= tolerance:
        raise ValueError(
            "final normal masses are degenerate; per-mode abundance is basis-dependent"
        )
    angle = 0.5 * math.atan2(2.0 * final_b, final_a - final_d)
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def fixed_basis_entries(
        diagonal_a: float,
        diagonal_d: float,
        off_diagonal: float,
    ) -> tuple[float, float, float]:
        mode_minus = (
            diagonal_a * sine * sine
            - 2.0 * off_diagonal * sine * cosine
            + diagonal_d * cosine * cosine
        )
        mode_plus = (
            diagonal_a * cosine * cosine
            + 2.0 * off_diagonal * sine * cosine
            + diagonal_d * sine * sine
        )
        rotated_off_diagonal = (
            off_diagonal * (cosine * cosine - sine * sine)
            + (diagonal_d - diagonal_a) * sine * cosine
        )
        return mode_minus, mode_plus, rotated_off_diagonal

    initial_minus, initial_plus, initial_rotated_off = fixed_basis_entries(
        initial_a,
        initial_d,
        initial_b,
    )
    final_minus, final_plus, final_rotated_off = fixed_basis_entries(
        final_a,
        final_d,
        final_b,
    )
    fixed_basis_residual = max(
        abs(initial_rotated_off) / initial_norm,
        abs(final_rotated_off) / final_norm,
    )
    if fixed_basis_residual > tolerance:
        raise ValueError("fixed-basis diagonalization residual exceeds tolerance")
    mode_mass_squared_in = (initial_minus, initial_plus)
    mode_mass_squared_out = (final_minus, final_plus)
    all_mode_mass_squared = (*mode_mass_squared_in, *mode_mass_squared_out)
    if any(value <= 0.0 for value in all_mode_mass_squared):
        raise ValueError("fixed-basis normal masses must remain positive")
    mode_masses_in = tuple(math.sqrt(value) for value in mode_mass_squared_in)
    mode_masses_out = tuple(math.sqrt(value) for value in mode_mass_squared_out)

    labels = ("fixed-basis-final-minus", "fixed-basis-final-plus")
    species = tuple(
        QuantumSeatSpecies(
            label=label,
            degeneracy=1,
            mass_in=mass_in,
            mass_out=mass_out,
            duration=duration,
            initial_mode_occupation=occupation,
        )
        for label, mass_in, mass_out, occupation in zip(
            labels,
            mode_masses_in,
            mode_masses_out,
            occupations,
            strict=True,
        )
    )
    abundance = tuple(
        smooth_quench_collisionless_abundance(
            mode_species,
            entropy=entropy,
            cosmology=cosmology,
            momentum_max=momentum_max_ev,
            intervals=intervals,
        )
        for mode_species in species
    )
    rest_densities = tuple(
        certificate.present_rest_density for certificate in abundance
    )
    if sum(rest_densities) <= 0.0:
        raise ValueError(
            "the supplied quench creates no positive dust input for the WKB gate"
        )

    energy_densities = tuple(
        certificate.present_energy_density for certificate in abundance
    )
    pressure_densities = tuple(
        certificate.present_pressure for certificate in abundance
    )
    number_densities = tuple(
        certificate.present_number_density for certificate in abundance
    )
    rms_momentum_ratio_upper = tuple(
        math.sqrt(2.0 * certificate.relative_kinetic_energy_upper_bound)
        for certificate in abundance
    )
    dust_replacement_errors = tuple(
        certificate.relative_kinetic_energy for certificate in abundance
    )
    background_cold_pass = max(rms_momentum_ratio_upper) <= limit
    local_quench_ratio = checked["hubble_at_production_ev"] * duration
    local_quench_pass = local_quench_ratio <= limit

    perturbation = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=checked["final_system_mass_ev"],
        environment_mass_ev=checked["final_environment_mass_ev"],
        bilinear_coupling_ev2=checked["final_bilinear_coupling_ev2"],
        vacuum_energy_density_ev4=checked["final_vacuum_energy_density_ev4"],
        scale_factor=1.0,
        hubble_ev=cosmology.hubble_today,
        comoving_wavenumber_ev=checked["comoving_wavenumber_ev"],
        comoving_mode_density_constants_ev4=rest_densities,
        reduced_planck_mass_ev=cosmology.reduced_planck_mass,
        validity_limit=limit,
    )
    failed_gates: list[str] = []
    if not local_quench_pass:
        failed_gates.append("local_quench_Hstar_tau")
    if not background_cold_pass:
        failed_gates.append("created_background_cold")
    failed_gates.extend(
        f"present_perturbation_{gate}" for gate in perturbation.failed_gates
    )

    critical_density = cosmology.critical_density_today
    total_energy_density = math.fsum(energy_densities)
    total_rest_density = math.fsum(rest_densities)
    omega_created = total_energy_density / critical_density
    omega_dust = total_rest_density / critical_density
    omega_vacuum = (
        checked["final_vacuum_energy_density_ev4"] / critical_density
    )
    dimension_manifest = {
        "normal_mode_mass_squared": 2.0,
        "quench_duration": -1.0,
        "hubble_at_production": 1.0,
        "matrix_commutator": 4.0,
        "created_number_density": 3.0,
        "created_energy_density": 4.0,
        "created_pressure": 4.0,
    }
    dimensionless_core_dimensions = {
        "pi_tau_omega": -1.0 + 1.0,
        "Hstar_tau": 1.0 - 1.0,
        "matrix_commutator_relative_residual": 4.0 - 4.0,
        "rms_momentum_over_mass": 1.0 - 1.0,
        "omega_created_today": 4.0 - 4.0,
    }
    dimensions_pass = all(
        dimension == 0.0 for dimension in dimensionless_core_dimensions.values()
    )
    if not dimensions_pass:
        failed_gates.append("dimensions")

    failures = tuple(failed_gates)
    return FixedBasisBilinearQuenchCosmologyAudit(
        initial_action_parameter_manifest=(
            checked["initial_system_mass_ev"],
            checked["initial_environment_mass_ev"],
            checked["initial_bilinear_coupling_ev2"],
            checked["initial_vacuum_energy_density_ev4"],
        ),
        final_action_parameter_manifest=(
            checked["final_system_mass_ev"],
            checked["final_environment_mass_ev"],
            checked["final_bilinear_coupling_ev2"],
            checked["final_vacuum_energy_density_ev4"],
        ),
        interpolation=(
            "K(t)=K_in+(1+tanh(t/tau))*(K_out-K_in)/2_IN_FIXED_BASIS"
        ),
        matrix_commutator_frobenius_ev4=commutator_norm,
        matrix_commutator_relative_residual=commutator_relative,
        fixed_basis_off_diagonal_relative_residual=fixed_basis_residual,
        initial_normal_mode_gap_fraction=initial_gap_fraction,
        final_normal_mode_gap_fraction=final_gap_fraction,
        fixed_basis_mode_mass_squared_at_initial_ev2=mode_mass_squared_in,
        fixed_basis_mode_mass_squared_at_final_ev2=mode_mass_squared_out,
        fixed_basis_mode_masses_at_initial_ev=mode_masses_in,
        fixed_basis_mode_masses_at_final_ev=mode_masses_out,
        quench_duration_ev_minus1=duration,
        hubble_at_production_ev=checked["hubble_at_production_ev"],
        local_quench_ratio=local_quench_ratio,
        local_quench_limit=limit,
        local_quench_pass=local_quench_pass,
        quench_species=species,
        abundance_certificates=abundance,
        present_created_number_densities_ev3=number_densities,
        present_created_rest_density_constants_ev4=rest_densities,
        present_created_energy_densities_ev4=energy_densities,
        present_created_pressures_ev4=pressure_densities,
        present_background_rms_momentum_ratio_upper=rms_momentum_ratio_upper,
        present_dust_replacement_relative_errors=dust_replacement_errors,
        present_created_energy_density_total_ev4=total_energy_density,
        present_dust_input_density_total_ev4=total_rest_density,
        omega_created_today=omega_created,
        omega_dust_input_today=omega_dust,
        omega_vacuum_today=omega_vacuum,
        omega_created_energy_plus_vacuum_today=omega_created + omega_vacuum,
        omega_dust_input_plus_vacuum_today=omega_dust + omega_vacuum,
        background_cold_pass=background_cold_pass,
        perturbation_audit=perturbation,
        mass_dimension_manifest=tuple(dimension_manifest.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        failed_gates=failures,
        status=(
            "PASS_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB"
            if not failures
            else "FAIL_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB_GATES"
        ),
    )
