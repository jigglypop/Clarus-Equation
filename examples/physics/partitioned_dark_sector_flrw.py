"""A no-double-counting dust-plus-vacuum FLRW background bridge.

The existing record bridge can split one terminal energy ledger into a
mass-shell residual and a complementary remainder.  The residual can be read
as comoving monokinetic dust.  The remainder can be matched to a constant
vacuum channel only if both modules present the same source receipt: using the
full record energy once for dust and again for vacuum would double count it.

This module first certifies the shared receipt

``E_record = E_dust + E_vacuum``

and then propagates the two disjoint components on a flat expanding FLRW
background.  With ``x=a/a_*`` dimensionless,

``rho_m(x)=rho_m* x^-3`` and ``rho_Lambda(x)=rho_Lambda*``.

The result is the exact dust-plus-constant-vacuum background solution.  The
partition, the global vacuum action, and both absolute densities remain
inputs.  Consequently this is not a CE selection law or an independent
cosmological prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.causal_record_dust_bridge import (
    DustInitialData,
    KineticMatching,
    Tensor4,
)
from examples.physics.lattice_scalar_transition_bridge import (
    VacuumCauchyTransition,
    match_constant_vacuum_on_cauchy_slice,
)


DEFAULT_TOLERANCE = 1.0e-10


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _dimensionless_close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _relative_close(left: float, right: float, tolerance: float) -> bool:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return True
    return abs(left - right) <= tolerance * scale


def _scaled_zero(value: float, reference_scale: float, tolerance: float) -> bool:
    scale = abs(reference_scale)
    if scale == 0.0:
        return value == 0.0
    return abs(value) <= tolerance * scale


@dataclass(frozen=True)
class PartitionedDarkReceipt:
    source_receipt_id: str
    dust_allocation_id: str
    vacuum_allocation_id: str
    total_record_energy: float
    spatial_volume: float
    dust_energy: float
    vacuum_energy: float
    unassigned_energy: float
    dust_fraction: float
    vacuum_fraction: float
    record_partition_residual: float
    vacuum_channel_residual: float
    disjoint_allocation_ids: bool
    two_channel_partition_closed: bool
    status: str
    claim_ceiling: str = (
        "SHARED_RECEIPT_PARTITION_NOT_MICROSCOPIC_DARK_READOUT_SELECTION"
    )


def construct_record_complement_as_vacuum(
    matching: KineticMatching,
    dust: DustInitialData,
    *,
    source_receipt_id: str,
    dust_allocation_id: str,
    vacuum_allocation_id: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PartitionedDarkReceipt:
    """Construct dust and vacuum from disjoint parts of one record.

    The vacuum transition is built inside this function with the exact
    complement of ``matching`` as its battery.  An unrelated transition with
    the same numerical energy therefore cannot be presented as provenance.
    """

    tolerance = _finite_positive(tolerance, "tolerance")
    identifiers = (source_receipt_id, dust_allocation_id, vacuum_allocation_id)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("receipt and allocation identifiers must be non-empty")
    if dust_allocation_id == vacuum_allocation_id:
        raise ValueError("dust and vacuum allocation identifiers must be disjoint")

    volume = _finite_positive(matching.cell_volume, "record cell volume")
    total_energy = _finite_positive(
        matching.total_energy_density * volume, "total record energy"
    )
    dust_energy = _finite_positive(
        matching.residual_energy_density * volume, "dust allocation energy"
    )
    complement_energy = _finite_positive(
        matching.complement_energy_density * volume,
        "record complement energy",
    )
    if not _scaled_zero(
        matching.no_double_counting_residual,
        matching.total_energy_density,
        tolerance,
    ):
        raise ValueError("record residual and complement do not close their ledger")

    if not _relative_close(
        dust.rest_energy_density, matching.residual_energy_density, tolerance
    ):
        raise ValueError("dust density must be the record residual density")
    if not _dimensionless_close(dust.gamma, 1.0, tolerance) or any(
        not _dimensionless_close(actual, expected, tolerance)
        for actual, expected in zip(dust.four_velocity, (1.0, 0.0, 0.0, 0.0))
    ):
        raise ValueError("mixed homogeneous background requires comoving rest dust")
    if any(
        not (
            _relative_close(
                dust.stress[mu][nu], dust.rest_energy_density, tolerance
            )
            if mu == nu == 0
            else _scaled_zero(
                dust.stress[mu][nu], dust.rest_energy_density, tolerance
            )
        )
        for mu in range(4)
        for nu in range(4)
    ):
        raise ValueError("dust allocation must be pressureless")

    vacuum = match_constant_vacuum_on_cauchy_slice(
        battery_energy=complement_energy,
        spatial_volume=volume,
        residual_efficiency=1.0,
    )

    if not _relative_close(vacuum.spatial_volume, volume, tolerance):
        raise ValueError("dust and vacuum allocations must use the same Cauchy cell")
    if not _relative_close(
        vacuum.battery_energy_before, complement_energy, tolerance
    ):
        raise ValueError("vacuum battery must be the record complement, not a second ledger")
    if not _dimensionless_close(vacuum.residual_efficiency, 1.0, tolerance):
        raise ValueError("two-channel closure requires the full complement in vacuum")
    if not _relative_close(vacuum.transferred_energy, complement_energy, tolerance):
        raise ValueError("vacuum transfer must consume exactly the record complement")
    if not _scaled_zero(
        vacuum.complement_energy_after, complement_energy, tolerance
    ):
        raise ValueError("a nonzero leftover requires a third explicit channel")
    for residual in (
        vacuum.no_double_counting_residual,
        vacuum.energy_preserving_rotation_commutator_residual,
        vacuum.unitary_total_energy_residual,
    ):
        if not _scaled_zero(residual, complement_energy, tolerance):
            raise ValueError("vacuum transition does not close its energy ledger")
    if not _relative_close(
        vacuum.vacuum_density * volume, vacuum.transferred_energy, tolerance
    ):
        raise ValueError("vacuum density times volume must equal transferred energy")
    expected_vacuum_stress = tuple(
        tuple(
            vacuum.vacuum_density
            if mu == nu == 0
            else -vacuum.vacuum_density
            if mu == nu
            else 0.0
            for nu in range(4)
        )
        for mu in range(4)
    )
    if any(
        not (
            _relative_close(
                float(vacuum.vacuum_stress[mu, nu]),
                expected_vacuum_stress[mu][nu],
                tolerance,
            )
            if expected_vacuum_stress[mu][nu] != 0.0
            else _scaled_zero(
                float(vacuum.vacuum_stress[mu, nu]),
                vacuum.vacuum_density,
                tolerance,
            )
        )
        for mu in range(4)
        for nu in range(4)
    ):
        raise ValueError("vacuum stress must equal diag(rho,-rho,-rho,-rho)")

    vacuum_energy = vacuum.transferred_energy
    unassigned_energy = vacuum.complement_energy_after
    record_residual = total_energy - dust_energy - vacuum_energy - unassigned_energy
    vacuum_residual = complement_energy - vacuum_energy - unassigned_energy
    closed = (
        _scaled_zero(record_residual, total_energy, tolerance)
        and _scaled_zero(vacuum_residual, complement_energy, tolerance)
        and _scaled_zero(unassigned_energy, total_energy, tolerance)
    )
    return PartitionedDarkReceipt(
        source_receipt_id=source_receipt_id,
        dust_allocation_id=dust_allocation_id,
        vacuum_allocation_id=vacuum_allocation_id,
        total_record_energy=total_energy,
        spatial_volume=volume,
        dust_energy=dust_energy,
        vacuum_energy=vacuum_energy,
        unassigned_energy=unassigned_energy,
        dust_fraction=dust_energy / total_energy,
        vacuum_fraction=vacuum_energy / total_energy,
        record_partition_residual=record_residual,
        vacuum_channel_residual=vacuum_residual,
        disjoint_allocation_ids=True,
        two_channel_partition_closed=closed,
        status=(
            "SHARED_RECORD_DUST_VACUUM_PARTITION_CLOSED"
            if closed
            else "SHARED_RECORD_PARTITION_AUDIT_FAILED"
        ),
    )


@dataclass(frozen=True)
class PartitionedDustVacuumFLRW:
    scale_factor_ratio: float
    initial_total_density: float
    initial_dust_density: float
    vacuum_density: float
    dust_density: float
    total_density: float
    total_pressure: float
    effective_equation_of_state: float
    dust_fraction_initial: float
    vacuum_fraction_initial: float
    vacuum_fraction_at_evaluation: float
    initial_hubble_rate: float
    hubble_rate: float
    normalized_hubble_rate: float
    elapsed_cosmic_time: float
    dimensionless_elapsed_time: float
    reconstructed_scale_factor_ratio: float
    matter_vacuum_equality_scale_factor_ratio: float
    acceleration_transition_scale_factor_ratio: float
    density_time_derivative: float
    hubble_time_derivative: float
    acceleration_over_scale_factor: float
    continuity_equation_residual: float
    friedmann_equation_residual: float
    raychaudhuri_equation_residual: float
    acceleration_equation_residual: float
    scale_factor_solution_residual: float
    total_orthonormal_stress: Tensor4
    shared_receipt_partition_closed: bool
    conditional_mixed_background_closed: bool
    global_constant_vacuum_action_adopted: bool
    partition_fraction_selected_by_ce_dynamics: bool
    absolute_dark_density_predicted: bool
    vacuum_action_derived_from_one_slice: bool
    renormalized_quantum_stress_derived: bool
    perturbations_and_structure_growth_derived: bool
    ce_specific_independent_observational_prediction_derived: bool
    status: str
    claim_ceiling: str = (
        "CONDITIONAL_PARTITIONED_DUST_CONSTANT_VACUUM_BACKGROUND_NOT_SELECTION_ABUNDANCE_OR_PREDICTION"
    )


def propagate_partitioned_dust_vacuum_flat_flrw(
    receipt: PartitionedDarkReceipt,
    *,
    newton_constant: float,
    evaluation_scale_factor_ratio: float,
    global_constant_vacuum_action_adopted: bool,
    dust_vacuum_transfer_rate_density: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PartitionedDustVacuumFLRW:
    """Propagate a certified positive dust/vacuum partition exactly.

    Both allocations must be strictly positive.  The pure-dust and pure-vacuum
    endpoints are the separate one-component propagation theorems.
    """

    newton = _finite_positive(newton_constant, "Newton constant")
    x = _finite_positive(
        evaluation_scale_factor_ratio, "evaluation scale-factor ratio"
    )
    tolerance = _finite_positive(tolerance, "tolerance")
    if not isinstance(global_constant_vacuum_action_adopted, bool):
        raise ValueError("global_constant_vacuum_action_adopted must be boolean")
    if not global_constant_vacuum_action_adopted:
        raise ValueError("mixed propagation requires the global constant-vacuum action")
    transfer = float(dust_vacuum_transfer_rate_density)
    if not math.isfinite(transfer) or transfer != 0.0:
        raise ValueError("separately conserved dust and vacuum require Q=0")
    if not receipt.two_channel_partition_closed:
        raise ValueError("mixed propagation requires a closed shared receipt")
    if not receipt.disjoint_allocation_ids:
        raise ValueError("dust and vacuum allocations must have disjoint identifiers")
    if receipt.dust_allocation_id == receipt.vacuum_allocation_id:
        raise ValueError("dust and vacuum cannot consume the same allocation")

    volume = _finite_positive(receipt.spatial_volume, "spatial volume")
    dust_energy = _finite_positive(receipt.dust_energy, "dust energy")
    vacuum_energy = _finite_positive(receipt.vacuum_energy, "vacuum energy")
    total_energy = _finite_positive(receipt.total_record_energy, "record energy")
    if not _scaled_zero(
        receipt.record_partition_residual, total_energy, tolerance
    ) or not _scaled_zero(
        receipt.vacuum_channel_residual, vacuum_energy, tolerance
    ):
        raise ValueError("shared receipt residuals must vanish")
    if not _scaled_zero(receipt.unassigned_energy, total_energy, tolerance):
        raise ValueError("mixed two-component background cannot hide a third channel")
    if not _relative_close(total_energy, dust_energy + vacuum_energy, tolerance):
        raise ValueError("dust and vacuum energies must sum to the source receipt")

    rho_m_star = dust_energy / volume
    rho_lambda = vacuum_energy / volume
    rho_star = total_energy / volume
    f_m = rho_m_star / rho_star
    f_lambda = rho_lambda / rho_star
    rho_m = rho_m_star / x**3
    rho_total = rho_m + rho_lambda
    pressure = -rho_lambda
    w_effective = pressure / rho_total

    gravitational_factor = 8.0 * math.pi * newton / 3.0
    hubble_star = math.sqrt(gravitational_factor * rho_star)
    hubble_lambda = math.sqrt(gravitational_factor * rho_lambda)
    hubble = math.sqrt(gravitational_factor * rho_total)
    normalized_hubble = hubble / hubble_star
    initial_asinh_argument = math.sqrt(rho_lambda / rho_m_star)
    evaluation_asinh_argument = initial_asinh_argument * x ** 1.5
    elapsed_time = 2.0 * (
        math.asinh(evaluation_asinh_argument)
        - math.asinh(initial_asinh_argument)
    ) / (3.0 * hubble_lambda)
    reconstructed_x = (
        math.sqrt(rho_m_star / rho_lambda)
        * math.sinh(
            math.asinh(initial_asinh_argument) + 1.5 * hubble_lambda * elapsed_time
        )
    ) ** (2.0 / 3.0)

    density_dot = -3.0 * hubble * rho_m
    hubble_dot = -4.0 * math.pi * newton * rho_m
    acceleration = -(4.0 * math.pi * newton / 3.0) * (
        rho_m - 2.0 * rho_lambda
    )
    continuity_residual = density_dot + 3.0 * hubble * (
        rho_total + pressure
    )
    friedmann_residual = hubble**2 - gravitational_factor * rho_total
    raychaudhuri_residual = hubble_dot + 4.0 * math.pi * newton * (
        rho_total + pressure
    )
    acceleration_residual = acceleration + 4.0 * math.pi * newton * (
        rho_total + 3.0 * pressure
    ) / 3.0
    scale_residual = reconstructed_x - x
    equality_scale = (rho_m_star / rho_lambda) ** (1.0 / 3.0)
    acceleration_scale = (rho_m_star / (2.0 * rho_lambda)) ** (1.0 / 3.0)
    stress: Tensor4 = (
        (rho_total, 0.0, 0.0, 0.0),
        (0.0, pressure, 0.0, 0.0),
        (0.0, 0.0, pressure, 0.0),
        (0.0, 0.0, 0.0, pressure),
    )
    normalized_residuals = (
        abs(continuity_residual)
        / max(1.0, abs(density_dot), abs(3.0 * hubble * (rho_total + pressure))),
        abs(friedmann_residual)
        / max(1.0, hubble**2, abs(gravitational_factor * rho_total)),
        abs(raychaudhuri_residual)
        / max(
            1.0,
            abs(hubble_dot),
            abs(4.0 * math.pi * newton * (rho_total + pressure)),
        ),
        abs(acceleration_residual)
        / max(
            1.0,
            abs(acceleration),
            abs(
                4.0 * math.pi * newton * (rho_total + 3.0 * pressure) / 3.0
            ),
        ),
        abs(scale_residual) / max(1.0, abs(x), abs(reconstructed_x)),
    )
    closed = all(value <= tolerance for value in normalized_residuals)

    return PartitionedDustVacuumFLRW(
        scale_factor_ratio=x,
        initial_total_density=rho_star,
        initial_dust_density=rho_m_star,
        vacuum_density=rho_lambda,
        dust_density=rho_m,
        total_density=rho_total,
        total_pressure=pressure,
        effective_equation_of_state=w_effective,
        dust_fraction_initial=f_m,
        vacuum_fraction_initial=f_lambda,
        vacuum_fraction_at_evaluation=rho_lambda / rho_total,
        initial_hubble_rate=hubble_star,
        hubble_rate=hubble,
        normalized_hubble_rate=normalized_hubble,
        elapsed_cosmic_time=elapsed_time,
        dimensionless_elapsed_time=hubble_star * elapsed_time,
        reconstructed_scale_factor_ratio=reconstructed_x,
        matter_vacuum_equality_scale_factor_ratio=equality_scale,
        acceleration_transition_scale_factor_ratio=acceleration_scale,
        density_time_derivative=density_dot,
        hubble_time_derivative=hubble_dot,
        acceleration_over_scale_factor=acceleration,
        continuity_equation_residual=continuity_residual,
        friedmann_equation_residual=friedmann_residual,
        raychaudhuri_equation_residual=raychaudhuri_residual,
        acceleration_equation_residual=acceleration_residual,
        scale_factor_solution_residual=scale_residual,
        total_orthonormal_stress=stress,
        shared_receipt_partition_closed=True,
        conditional_mixed_background_closed=closed,
        global_constant_vacuum_action_adopted=True,
        partition_fraction_selected_by_ce_dynamics=False,
        absolute_dark_density_predicted=False,
        vacuum_action_derived_from_one_slice=False,
        renormalized_quantum_stress_derived=False,
        perturbations_and_structure_growth_derived=False,
        ce_specific_independent_observational_prediction_derived=False,
        status=(
            "CONDITIONAL_PARTITIONED_DUST_VACUUM_FLRW_BACKGROUND_CLOSED"
            if closed
            else "PARTITIONED_DUST_VACUUM_FLRW_AUDIT_FAILED"
        ),
    )
