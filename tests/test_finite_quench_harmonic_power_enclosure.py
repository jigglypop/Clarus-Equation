"""Focused tests for the conditional all-k harmonic power sum."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.finite_quench_harmonic_power_enclosure import (
    ConvergenceHarmonicExteriorTailBoundCertificate,
    ConvergenceHarmonicLogKCellEnvelope,
    enclose_convergence_harmonic_auto_power,
)


def _cell(
    log_cell: tuple[object, object],
    primordial: tuple[object, object],
    real: tuple[object, object],
    imaginary: tuple[object, object] = (0, 0),
) -> ConvergenceHarmonicLogKCellEnvelope:
    return ConvergenceHarmonicLogKCellEnvelope.freeze(
        log_k_over_pivot_interval=log_cell,
        primordial_curvature_power_interval=primordial,
        convergence_transfer_real_interval=real,
        convergence_transfer_imaginary_interval=imaginary,
    )


def _tail_certificate(
    *,
    ell: int,
    band: tuple[object, object],
    low: object,
    high: object,
    proof_reference: str = "synthetic exact exterior-tail proof",
) -> ConvergenceHarmonicExteriorTailBoundCertificate:
    return ConvergenceHarmonicExteriorTailBoundCertificate.freeze(
        ell=ell,
        band_log_k_over_pivot_interval=band,
        low_k_reduced_tail_upper_bound=low,
        high_k_reduced_tail_upper_bound=high,
        proof_reference=proof_reference,
        exterior_integral_bounds_certified=True,
    )


def test_exact_two_cell_band_power_encloses_complex_transfer() -> None:
    receipt = enclose_convergence_harmonic_auto_power(
        ell=12,
        cells=(
            _cell((-2, -1), (2, 3), (-1, 2), (0, 1)),
            _cell((-1, 1), (1, 2), (1, 2), (-3, -2)),
        ),
        cellwide_envelopes_certified=True,
    )

    assert receipt.convergence_transfer_modulus_squared_cell_intervals == (
        (Fraction(0), Fraction(5)),
        (Fraction(5), Fraction(13)),
    )
    assert receipt.reduced_integrand_cell_intervals == (
        (Fraction(0), Fraction(15)),
        (Fraction(5), Fraction(26)),
    )
    assert receipt.reduced_band_angular_power_interval == (
        Fraction(10),
        Fraction(67),
    )
    assert receipt.reduced_full_angular_power_interval is None
    assert receipt.band_limited_angular_auto_power_enclosed
    assert not receipt.full_angular_auto_power_enclosed


def test_two_certified_tails_promote_band_to_full_power_enclosure() -> None:
    receipt = enclose_convergence_harmonic_auto_power(
        ell=2,
        cells=(
            _cell((-2, -1), (2, 3), (-1, 2), (0, 1)),
            _cell((-1, 1), (1, 2), (1, 2), (-3, -2)),
        ),
        cellwide_envelopes_certified=True,
        exterior_tail_certificate=_tail_certificate(
            ell=2,
            band=(-2, 1),
            low=Fraction(1, 2),
            high=Fraction(3, 2),
        ),
    )

    assert receipt.reduced_full_angular_power_interval == (
        Fraction(10),
        Fraction(69),
    )
    assert receipt.band_log_k_over_pivot_interval == (-2, 1)
    assert receipt.certified_low_k_tail_domain_upper_endpoint == -2
    assert receipt.certified_high_k_tail_domain_lower_endpoint == 1
    assert receipt.both_exterior_tail_integral_upper_bounds_certified
    assert receipt.certified_tail_domains_locked_to_partition_exterior
    assert receipt.exterior_tail_bound_certificate is not None
    assert (
        receipt.exterior_tail_bound_certificate.proof_reference
        == "synthetic exact exterior-tail proof"
    )
    assert receipt.full_angular_auto_power_enclosed


def test_zero_transfer_gives_exact_zero_auto_power() -> None:
    receipt = enclose_convergence_harmonic_auto_power(
        ell=3,
        cells=(_cell((-1, 1), (1, 4), (0, 0), (0, 0)),),
        cellwide_envelopes_certified=True,
        exterior_tail_certificate=_tail_certificate(
            ell=3,
            band=(-1, 1),
            low=0,
            high=0,
            proof_reference="exact zero exterior transfer proof",
        ),
    )

    assert receipt.reduced_band_angular_power_interval == (0, 0)
    assert receipt.reduced_full_angular_power_interval == (0, 0)
    assert receipt.exact_rational_nonnegative_cell_sum_proven


def test_receipt_locks_dimensionless_four_pi_convention_and_nonclaims() -> None:
    receipt = enclose_convergence_harmonic_auto_power(
        ell=7,
        cells=(_cell((0, 1), (1, 1), (-2, -1), (-4, -3)),),
        cellwide_envelopes_certified=True,
    )

    assert receipt.convergence_transfer_modulus_squared_cell_intervals == (
        (Fraction(10), Fraction(20)),
    )
    assert receipt.dimensionless_log_k_coordinate_used
    assert receipt.dimensionless_primordial_curvature_power_used
    assert receipt.dimensionless_convergence_transfer_used
    assert receipt.curvature_fourier_covariance_convention_adopted
    assert receipt.four_pi_d_log_k_angular_power_identity_adopted
    assert receipt.four_pi_factor_kept_symbolic
    assert receipt.input_is_supplied_convergence_transfer_contract
    assert receipt.binwide_not_nodewise_envelopes_required
    assert not receipt.lensing_potential_to_convergence_factor_derived
    assert not receipt.weyl_transfer_from_ce_dynamics_derived
    assert not receipt.all_k_einstein_boltzmann_transfer_derived
    assert not receipt.primordial_curvature_spectrum_derived
    assert not receipt.source_redshift_distribution_derived
    assert not receipt.post_born_or_relativistic_corrections_enclosed
    assert not receipt.covariance_or_likelihood_enclosed


def test_node_samples_cannot_be_mislabeled_as_binwide_envelopes() -> None:
    with pytest.raises(ValueError, match="grid nodes are insufficient"):
        enclose_convergence_harmonic_auto_power(
            ell=2,
            cells=(_cell((0, 1), (1, 1), (1, 1)),),
            cellwide_envelopes_certified=False,
        )


def test_log_k_cells_must_be_positive_and_contiguous() -> None:
    with pytest.raises(ValueError, match="positive width"):
        _cell((0, 0), (1, 1), (1, 1))
    with pytest.raises(ValueError, match="contiguous partition"):
        enclose_convergence_harmonic_auto_power(
            ell=2,
            cells=(
                _cell((0, 1), (1, 1), (1, 1)),
                _cell((2, 3), (1, 1), (1, 1)),
            ),
            cellwide_envelopes_certified=True,
        )


def test_invalid_power_harmonic_and_tail_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _cell((0, 1), (-1, 1), (1, 1))
    with pytest.raises(ValueError, match="ell >= 2"):
        enclose_convergence_harmonic_auto_power(
            ell=1,
            cells=(_cell((0, 1), (1, 1), (1, 1)),),
            cellwide_envelopes_certified=True,
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        _tail_certificate(
            ell=2,
            band=(0, 1),
            low=-1,
            high=0,
        )
    with pytest.raises(ValueError, match="must be certified"):
        ConvergenceHarmonicExteriorTailBoundCertificate.freeze(
            ell=2,
            band_log_k_over_pivot_interval=(0, 1),
            low_k_reduced_tail_upper_bound=0,
            high_k_reduced_tail_upper_bound=0,
            proof_reference="not yet certified",
            exterior_integral_bounds_certified=False,
        )
    with pytest.raises(ValueError, match="nonempty proof reference"):
        _tail_certificate(ell=2, band=(0, 1), low=0, high=0, proof_reference="")


def test_missing_falsified_or_mismatched_tail_certificate_fails_closed() -> None:
    receipt = enclose_convergence_harmonic_auto_power(
        ell=2,
        cells=(_cell((0, 1), (1, 1), (1, 1)),),
        cellwide_envelopes_certified=True,
    )

    assert receipt.certified_low_k_reduced_tail_upper_bound is None
    assert receipt.certified_high_k_reduced_tail_upper_bound is None
    assert receipt.reduced_full_angular_power_interval is None
    assert not receipt.certified_tail_domains_locked_to_partition_exterior
    assert not receipt.full_angular_auto_power_enclosed

    certificate = _tail_certificate(ell=2, band=(0, 1), low=0, high=0)
    falsified = replace(
        certificate,
        high_k_exterior_reduced_integral_upper_bound_certified=False,
    )
    with pytest.raises(ValueError, match="proof prerequisites"):
        enclose_convergence_harmonic_auto_power(
            ell=2,
            cells=(_cell((0, 1), (1, 1), (1, 1)),),
            cellwide_envelopes_certified=True,
            exterior_tail_certificate=falsified,
        )
    mismatched = _tail_certificate(ell=2, band=(-1, 1), low=0, high=0)
    with pytest.raises(ValueError, match="provenance mismatch"):
        enclose_convergence_harmonic_auto_power(
            ell=2,
            cells=(_cell((0, 1), (1, 1), (1, 1)),),
            cellwide_envelopes_certified=True,
            exterior_tail_certificate=mismatched,
        )
