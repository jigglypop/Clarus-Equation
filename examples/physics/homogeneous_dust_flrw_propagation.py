"""Conditional propagation of causal-record dust on a flat FLRW background.

The causal-record bridge already constructs positive comoving monokinetic dust
and a unit-scale-factor flat-FLRW Cauchy slice.  This module advances that one
slice through cosmic time under additional explicit assumptions: the dust is
homogeneous, collisionless, source free, pressureless, and is the only
background component.

For scale factor ``a`` normalized by ``a_*=1``, conservation of particle number
in a comoving cell gives

``n(a)=n_* a^-3`` and ``rho(a)=rho_* a^-3``.

The expanding flat Friedmann branch then has

``H(a)=H_* a^-3/2`` and
``a(t)=[1+(3/2)H_*(t-t_*)]^(2/3)``.

Thus ``dot(rho)+3 H rho=0`` and ``H^2=8 pi G rho/3`` hold exactly.  Scale
factors and their ratios are dimensionless; ``H_*(t-t_*)`` is the only power-
law time argument and is dimensionless.  This theorem propagates an already
declared dust readout.  It does not derive the readout choice, a renormalized
quantum stress tensor, observed dark-matter abundance, a vacuum ``w=-1``
sector, perturbation growth, or a CE-specific observational prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.causal_record_dust_bridge import (
    DustInitialData,
    FlatFLRWCauchyWitness,
    Tensor4,
)


DEFAULT_TOLERANCE = 1.0e-10


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_comoving_dust(
    dust: DustInitialData, *, tolerance: float
) -> None:
    mass = _finite_positive(dust.mass, "dust mass")
    number_density = _finite_positive(
        dust.rest_number_density, "dust rest number density"
    )
    density = _finite_positive(dust.rest_energy_density, "dust energy density")
    if any(
        not math.isfinite(float(value))
        for value in (
            dust.energy,
            dust.gamma,
            dust.surface_number_density,
            *dust.four_velocity,
            *dust.current,
            *(entry for row in dust.stress for entry in row),
        )
    ):
        raise ValueError("dust initial data must be finite")
    if abs(dust.gamma - 1.0) > tolerance or abs(dust.energy - mass) > tolerance:
        raise ValueError("homogeneous FLRW propagation requires rest-frame dust")
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(dust.four_velocity, (1.0, 0.0, 0.0, 0.0))
    ):
        raise ValueError("homogeneous FLRW propagation requires comoving dust")
    if abs(density - mass * number_density) > tolerance:
        raise ValueError("dust energy density must equal mass times number density")
    expected_current = (number_density, 0.0, 0.0, 0.0)
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(dust.current, expected_current)
    ):
        raise ValueError("dust current must equal n u")
    expected_stress = tuple(
        tuple(density if mu == 0 and nu == 0 else 0.0 for nu in range(4))
        for mu in range(4)
    )
    if any(
        abs(dust.stress[mu][nu] - expected_stress[mu][nu]) > tolerance
        for mu in range(4)
        for nu in range(4)
    ):
        raise ValueError("dust stress must be pressureless rho u tensor u")


@dataclass(frozen=True)
class HomogeneousDustFLRWPropagation:
    initial_scale_factor: float
    evaluation_scale_factor: float
    dimensionless_scale_factor_ratio: float
    reference_comoving_coordinate_volume: float
    physical_volume: float
    conserved_comoving_particle_number: float
    rest_number_density: float
    energy_density: float
    pressure: float
    equation_of_state_parameter: float
    hubble_rate: float
    elapsed_cosmic_time: float
    density_time_derivative: float
    hubble_time_derivative: float
    acceleration_over_scale_factor: float
    comoving_particle_number_residual: float
    comoving_rest_energy_residual: float
    continuity_equation_residual: float
    friedmann_equation_residual: float
    raychaudhuri_equation_residual: float
    scale_factor_solution_residual: float
    comoving_orthonormal_stress: Tensor4
    homogeneous_source_free_dust_propagation_closed: bool
    background_covariant_conservation_closed: bool
    record_to_renormalized_quantum_stress_derived: bool
    dust_readout_selected_by_ce_dynamics: bool
    observed_dark_matter_abundance_predicted: bool
    vacuum_dark_energy_sector_derived: bool
    perturbations_and_structure_growth_derived: bool
    ce_specific_independent_observational_prediction_derived: bool
    status: str
    claim_ceiling: str = (
        "CONDITIONAL_RECORD_DUST_FLRW_BACKGROUND_NOT_DARK_SECTOR_SELECTION_OR_ABUNDANCE"
    )


def propagate_homogeneous_flat_flrw_dust(
    dust: DustInitialData,
    initial_witness: FlatFLRWCauchyWitness,
    *,
    evaluation_scale_factor: float,
    reference_comoving_coordinate_volume: float = 1.0,
    source_energy_transfer_rate_density: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> HomogeneousDustFLRWPropagation:
    """Propagate unit-normalized comoving dust on the expanding flat branch."""

    scale_factor = _finite_positive(evaluation_scale_factor, "evaluation scale factor")
    comoving_volume = _finite_positive(
        reference_comoving_coordinate_volume,
        "reference comoving coordinate volume",
    )
    tolerance = _finite_positive(tolerance, "tolerance")
    source = float(source_energy_transfer_rate_density)
    if not math.isfinite(source):
        raise ValueError("source energy transfer rate density must be finite")
    if abs(source) > tolerance:
        raise ValueError("source-free dust theorem requires Q=0")
    _validate_comoving_dust(dust, tolerance=tolerance)

    newton_constant = _finite_positive(
        initial_witness.newton_constant, "Newton constant"
    )
    initial_density = dust.rest_energy_density
    expected_initial_hubble = math.sqrt(
        8.0 * math.pi * newton_constant * initial_density / 3.0
    )
    if abs(initial_witness.energy_density - initial_density) > tolerance:
        raise ValueError("FLRW witness energy density must match dust initial data")
    if abs(initial_witness.hubble_rate - expected_initial_hubble) > tolerance:
        raise ValueError("FLRW witness must lie on the expanding Friedmann branch")
    if abs(initial_witness.hamiltonian_residual) > tolerance or any(
        abs(value) > tolerance for value in initial_witness.momentum_residual
    ):
        raise ValueError("FLRW witness must satisfy the initial GR constraints")

    scale_ratio = scale_factor
    physical_volume = comoving_volume * scale_ratio**3
    initial_particle_number = dust.rest_number_density * comoving_volume
    number_density = dust.rest_number_density / scale_ratio**3
    density = initial_density / scale_ratio**3
    pressure = 0.0
    hubble = expected_initial_hubble / scale_ratio ** 1.5
    elapsed_time = 2.0 * (scale_ratio ** 1.5 - 1.0) / (
        3.0 * expected_initial_hubble
    )
    density_time_derivative = -3.0 * hubble * density
    hubble_time_derivative = -1.5 * hubble**2
    acceleration_over_scale = hubble_time_derivative + hubble**2
    propagated_particle_number = number_density * physical_volume
    propagated_rest_energy = density * physical_volume
    initial_rest_energy = initial_density * comoving_volume
    continuity_residual = density_time_derivative + 3.0 * hubble * density
    friedmann_residual = hubble**2 - 8.0 * math.pi * newton_constant * density / 3.0
    raychaudhuri_residual = (
        hubble_time_derivative + 4.0 * math.pi * newton_constant * density
    )
    reconstructed_scale_factor = (
        1.0 + 1.5 * expected_initial_hubble * elapsed_time
    ) ** (2.0 / 3.0)
    scale_solution_residual = reconstructed_scale_factor - scale_factor
    stress: Tensor4 = (
        (density, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    closed = all(
        residual <= tolerance
        for residual in (
            abs(propagated_particle_number - initial_particle_number),
            abs(propagated_rest_energy - initial_rest_energy),
            abs(continuity_residual),
            abs(friedmann_residual),
            abs(raychaudhuri_residual),
            abs(scale_solution_residual),
        )
    )

    return HomogeneousDustFLRWPropagation(
        initial_scale_factor=1.0,
        evaluation_scale_factor=scale_factor,
        dimensionless_scale_factor_ratio=scale_ratio,
        reference_comoving_coordinate_volume=comoving_volume,
        physical_volume=physical_volume,
        conserved_comoving_particle_number=propagated_particle_number,
        rest_number_density=number_density,
        energy_density=density,
        pressure=pressure,
        equation_of_state_parameter=0.0,
        hubble_rate=hubble,
        elapsed_cosmic_time=elapsed_time,
        density_time_derivative=density_time_derivative,
        hubble_time_derivative=hubble_time_derivative,
        acceleration_over_scale_factor=acceleration_over_scale,
        comoving_particle_number_residual=(
            propagated_particle_number - initial_particle_number
        ),
        comoving_rest_energy_residual=(
            propagated_rest_energy - initial_rest_energy
        ),
        continuity_equation_residual=continuity_residual,
        friedmann_equation_residual=friedmann_residual,
        raychaudhuri_equation_residual=raychaudhuri_residual,
        scale_factor_solution_residual=scale_solution_residual,
        comoving_orthonormal_stress=stress,
        homogeneous_source_free_dust_propagation_closed=closed,
        background_covariant_conservation_closed=abs(continuity_residual) <= tolerance,
        record_to_renormalized_quantum_stress_derived=False,
        dust_readout_selected_by_ce_dynamics=False,
        observed_dark_matter_abundance_predicted=False,
        vacuum_dark_energy_sector_derived=False,
        perturbations_and_structure_growth_derived=False,
        ce_specific_independent_observational_prediction_derived=False,
        status=(
            "CONDITIONAL_HOMOGENEOUS_RECORD_DUST_FLRW_PROPAGATION_CLOSED"
            if closed
            else "HOMOGENEOUS_DUST_FLRW_PROPAGATION_AUDIT_FAILED"
        ),
    )
