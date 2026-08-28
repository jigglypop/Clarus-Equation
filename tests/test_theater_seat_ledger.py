from __future__ import annotations

import math

from examples.physics.theater_seat_ledger import (
    OpeningScale,
    PregeometricSeatSpectrum,
    SeatScalingAudit,
    SeatType,
    TransferChannel,
    open_spectrum,
    reservoir_closure_rate,
    source_ratio_from_number_current,
    total_transfer_residual,
    uniform_opening_energy_requirement,
)


def test_different_seat_energies_control_opened_density_not_counts_alone() -> None:
    spectrum = PregeometricSeatSpectrum(
        (
            SeatType("light", degeneracy=10.0, occupancy=0.5, relative_energy=1.0),
            SeatType("heavy", degeneracy=10.0, occupancy=0.5, relative_energy=4.0),
        )
    )

    assert spectrum.normalized_energy_fractions() == (
        ("light", 0.2),
        ("heavy", 0.8),
    )

    opened = open_spectrum(
        spectrum,
        OpeningScale(event_number_density=2.0, reference_event_energy=3.0),
    )
    assert tuple(item.energy_density for item in opened) == (30.0, 120.0)


def test_count_energy_reparameterization_preserves_the_ledger() -> None:
    seat = SeatType("folded", degeneracy=3.0, occupancy=0.4, relative_energy=5.0)
    rescaled = seat.count_energy_degenerate_copy(count_rescaling=7.0)

    assert math.isclose(
        seat.dimensionless_energy_weight,
        rescaled.dimensionless_energy_weight,
        rel_tol=1.0e-15,
    )


def test_separately_conserved_matter_radiation_and_vacuum_scalings() -> None:
    matter = SeatScalingAudit(0.0, 0.0)
    radiation = SeatScalingAudit(0.0, -1.0)
    vacuum = SeatScalingAudit(3.0, 0.0)

    assert (matter.density_exponent, matter.effective_w) == (-3.0, 0.0)
    assert (radiation.density_exponent, radiation.effective_w) == (-4.0, 1.0 / 3.0)
    assert (vacuum.density_exponent, vacuum.effective_w) == (0.0, -1.0)
    assert matter.intrinsic_w_given_current == matter.effective_w
    assert radiation.intrinsic_w_given_current == radiation.effective_w
    assert vacuum.intrinsic_w_given_current == vacuum.effective_w


def test_same_seat_scaling_allows_different_pressure_when_current_changes() -> None:
    separately_conserved = SeatScalingAudit(3.0, 0.0, 0.0)
    interacting_dust = SeatScalingAudit(3.0, 0.0, 3.0)

    assert separately_conserved.density_exponent == interacting_dust.density_exponent == 0.0
    assert separately_conserved.intrinsic_w_given_current == -1.0
    assert interacting_dust.intrinsic_w_given_current == 0.0


def test_covariant_number_current_identity_separates_pressure_and_source() -> None:
    conserved_vacuum_ratio = source_ratio_from_number_current(
        number_creation_over_hubble_number=3.0,
        energy_drift_over_hubble_energy=0.0,
        intrinsic_w=-1.0,
    )
    interacting_dust_ratio = source_ratio_from_number_current(
        number_creation_over_hubble_number=3.0,
        energy_drift_over_hubble_energy=0.0,
        intrinsic_w=0.0,
    )

    assert conserved_vacuum_ratio == 0.0
    assert interacting_dust_ratio == 3.0


def test_reservoir_rate_closes_total_transfer_and_detects_omission() -> None:
    destinations = (1.25, -0.25)
    reservoir = reservoir_closure_rate(destinations)
    closed = (
        TransferChannel("visible", destinations[0]),
        TransferChannel("kinetic", destinations[1]),
        TransferChannel("fold", reservoir),
    )

    assert total_transfer_residual(closed) == 0.0
    assert total_transfer_residual(closed[:-1]) == 1.0


def test_uniform_acoustic_rescaling_keeps_fraction_denominators_distinct() -> None:
    ratio = 136.95509676045015 / 147.09
    audit = uniform_opening_energy_requirement(ratio)

    assert math.isclose(audit.hubble_ratio, 1.0 / ratio, rel_tol=1.0e-15)
    assert math.isclose(
        audit.extra_density_over_baseline,
        0.1534795458,
        rel_tol=1.0e-9,
    )
    assert math.isclose(
        audit.extra_fraction_of_total,
        0.1330578825,
        rel_tol=1.0e-9,
    )
    assert math.isclose(
        audit.extra_fraction_of_total,
        1.0 - ratio * ratio,
        rel_tol=2.0e-15,
    )
