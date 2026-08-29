"""Conditional constant-vacuum propagation from a one-slice energy match.

``VacuumCauchyTransition`` provides an exact one-slice energy partition and a
declared orthonormal stress ``diag(rho,-rho,-rho,-rho)``.  A finite register
match alone does not establish that this stress remains a constant covariant
vacuum term on every later slice.  This module therefore requires the global
constant-vacuum action to be adopted explicitly.

Under that extra input, in a flat vacuum-only expanding FLRW background,

``rho(a)=rho_*``, ``p=-rho_*``, ``H^2=8 pi G rho_*/3``, and
``a(t)=exp(H(t-t_*))``.

The vacuum energy inside a fixed comoving cell grows as ``a^3``.  This is not
ordinary conserved dust energy: local conservation is instead the exact work
identity ``d(rho V)/dt + p dV/dt=0``.  The logarithm used to reconstruct time
has the dimensionless argument ``a/a_*=a``.  The theorem does not derive the
global vacuum action, select the vacuum readout, fix its density, or provide a
CE-specific observational prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.lattice_scalar_transition_bridge import (
    VacuumCauchyTransition,
)


DEFAULT_TOLERANCE = 1.0e-10


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ConstantVacuumFLRWPropagation:
    initial_scale_factor: float
    evaluation_scale_factor: float
    dimensionless_scale_factor_ratio: float
    initial_physical_volume: float
    physical_volume: float
    vacuum_energy_density: float
    pressure: float
    equation_of_state_parameter: float
    hubble_rate: float
    elapsed_cosmic_time: float
    vacuum_energy_in_comoving_cell: float
    vacuum_energy_change_from_initial_slice: float
    physical_volume_time_derivative: float
    vacuum_energy_time_derivative: float
    continuity_equation_residual: float
    friedmann_equation_residual: float
    raychaudhuri_equation_residual: float
    acceleration_equation_residual: float
    negative_pressure_work_residual: float
    scale_factor_solution_residual: float
    de_sitter_ricci_scalar: float
    global_constant_vacuum_action_adopted: bool
    conditional_constant_vacuum_flrw_propagation_closed: bool
    background_covariant_conservation_closed: bool
    one_slice_energy_match_derives_global_vacuum_action: bool
    finite_register_supplies_all_later_comoving_vacuum_energy: bool
    vacuum_readout_selected_by_ce_dynamics: bool
    observed_dark_energy_density_predicted: bool
    vacuum_renormalization_and_radiative_stability_derived: bool
    perturbations_and_ce_specific_observational_prediction_derived: bool
    status: str
    claim_ceiling: str = (
        "CONDITIONAL_CONSTANT_VACUUM_FLRW_NOT_GLOBAL_ACTION_OR_DENSITY_DERIVATION"
    )


def propagate_constant_vacuum_flat_flrw(
    transition: VacuumCauchyTransition,
    *,
    newton_constant: float,
    evaluation_scale_factor: float,
    global_constant_vacuum_action_adopted: bool,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ConstantVacuumFLRWPropagation:
    """Propagate the adopted constant vacuum on the expanding flat branch."""

    newton = _finite_positive(newton_constant, "Newton constant")
    scale_factor = _finite_positive(evaluation_scale_factor, "evaluation scale factor")
    tolerance = _finite_positive(tolerance, "tolerance")
    if not isinstance(global_constant_vacuum_action_adopted, bool):
        raise ValueError("global_constant_vacuum_action_adopted must be boolean")
    if not global_constant_vacuum_action_adopted:
        raise ValueError(
            "one-slice energy matching is insufficient without a global constant vacuum action"
        )
    density = _finite_positive(transition.vacuum_density, "vacuum density")
    initial_volume = _finite_positive(transition.spatial_volume, "spatial volume")
    if any(
        abs(value) > tolerance
        for value in (
            transition.no_double_counting_residual,
            transition.energy_preserving_rotation_commutator_residual,
            transition.unitary_total_energy_residual,
        )
    ):
        raise ValueError("one-slice vacuum transition must close its energy ledger")
    expected_stress = np.diag((density, -density, -density, -density))
    supplied_stress = np.asarray(transition.vacuum_stress, dtype=float)
    if (
        supplied_stress.shape != (4, 4)
        or not np.all(np.isfinite(supplied_stress))
        or np.linalg.norm(supplied_stress - expected_stress) > tolerance
    ):
        raise ValueError("vacuum stress must equal diag(rho,-rho,-rho,-rho)")
    if abs(transition.transferred_energy - density * initial_volume) > tolerance:
        raise ValueError("one-slice transferred energy must equal rho times volume")

    pressure = -density
    hubble = math.sqrt(8.0 * math.pi * newton * density / 3.0)
    elapsed_time = math.log(scale_factor) / hubble
    physical_volume = initial_volume * scale_factor**3
    vacuum_energy = density * physical_volume
    energy_change = vacuum_energy - transition.transferred_energy
    volume_time_derivative = 3.0 * hubble * physical_volume
    energy_time_derivative = density * volume_time_derivative
    density_time_derivative = 0.0
    hubble_time_derivative = 0.0
    acceleration_over_scale = hubble**2
    continuity_residual = density_time_derivative + 3.0 * hubble * (
        density + pressure
    )
    friedmann_residual = hubble**2 - 8.0 * math.pi * newton * density / 3.0
    raychaudhuri_residual = hubble_time_derivative + 4.0 * math.pi * newton * (
        density + pressure
    )
    acceleration_residual = acceleration_over_scale + 4.0 * math.pi * newton * (
        density + 3.0 * pressure
    ) / 3.0
    work_residual = energy_time_derivative + pressure * volume_time_derivative
    reconstructed_scale_factor = math.exp(hubble * elapsed_time)
    scale_residual = reconstructed_scale_factor - scale_factor
    ricci_scalar = 12.0 * hubble**2
    closed = all(
        abs(residual) <= tolerance
        for residual in (
            continuity_residual,
            friedmann_residual,
            raychaudhuri_residual,
            acceleration_residual,
            work_residual,
            scale_residual,
        )
    )

    return ConstantVacuumFLRWPropagation(
        initial_scale_factor=1.0,
        evaluation_scale_factor=scale_factor,
        dimensionless_scale_factor_ratio=scale_factor,
        initial_physical_volume=initial_volume,
        physical_volume=physical_volume,
        vacuum_energy_density=density,
        pressure=pressure,
        equation_of_state_parameter=-1.0,
        hubble_rate=hubble,
        elapsed_cosmic_time=elapsed_time,
        vacuum_energy_in_comoving_cell=vacuum_energy,
        vacuum_energy_change_from_initial_slice=energy_change,
        physical_volume_time_derivative=volume_time_derivative,
        vacuum_energy_time_derivative=energy_time_derivative,
        continuity_equation_residual=continuity_residual,
        friedmann_equation_residual=friedmann_residual,
        raychaudhuri_equation_residual=raychaudhuri_residual,
        acceleration_equation_residual=acceleration_residual,
        negative_pressure_work_residual=work_residual,
        scale_factor_solution_residual=scale_residual,
        de_sitter_ricci_scalar=ricci_scalar,
        global_constant_vacuum_action_adopted=True,
        conditional_constant_vacuum_flrw_propagation_closed=closed,
        background_covariant_conservation_closed=abs(continuity_residual) <= tolerance,
        one_slice_energy_match_derives_global_vacuum_action=False,
        finite_register_supplies_all_later_comoving_vacuum_energy=False,
        vacuum_readout_selected_by_ce_dynamics=False,
        observed_dark_energy_density_predicted=False,
        vacuum_renormalization_and_radiative_stability_derived=False,
        perturbations_and_ce_specific_observational_prediction_derived=False,
        status=(
            "CONDITIONAL_CONSTANT_VACUUM_DE_SITTER_PROPAGATION_CLOSED"
            if closed
            else "CONSTANT_VACUUM_FLRW_PROPAGATION_AUDIT_FAILED"
        ),
    )
