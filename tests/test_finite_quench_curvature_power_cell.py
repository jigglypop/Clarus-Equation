"""Focused tests for theorem-37 compact-kappa harmonic power cells."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    construct_compact_kappa_bin_weyl_transfer_enclosure,
)
from examples.physics.finite_quench_curvature_power_cell import (
    CertifiedPrimordialCurvaturePowerKappaBinEnvelope,
    construct_compact_kappa_bin_curvature_power_cell,
    enclose_positive_rational_log_ratio,
)
from examples.physics.finite_quench_harmonic_power_enclosure import (
    _square_interval,
)
from examples.physics.finite_quench_primordial_curvature_normalization import (
    normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature,
)
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_source_plane_harmonic_transfer import (
    project_compact_kappa_bin_to_source_plane_harmonic_transfer,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _bridge() -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


@pytest.fixture(scope="module")
def curvature_transfer():
    evolution = FiniteQuenchRegularMetricEvolution(
        _bridge(),
        n_initial=-5.0,
        kappa_initial=0.05,
    )
    compact = construct_compact_kappa_bin_weyl_transfer_enclosure(
        FiniteQuenchTraceEndpointEnclosure(evolution),
        initial_kappa_lower=Fraction(49, 1000),
        initial_kappa_upper=Fraction(51, 1000),
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )
    harmonic = project_compact_kappa_bin_to_source_plane_harmonic_transfer(
        compact,
        ell=2,
    )
    return normalize_compact_kappa_bin_harmonic_transfer_to_comoving_curvature(
        compact,
        harmonic,
    )


@pytest.fixture(scope="module")
def primordial_certificate(curvature_transfer):
    harmonic = curvature_transfer.source_plane_harmonic_transfer_receipt
    return CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
        initial_kappa_interval=curvature_transfer.initial_kappa_interval,
        dimensionless_fixed_wavenumber_interval=(
            harmonic.dimensionless_fixed_wavenumber_interval
        ),
        primordial_curvature_power_interval=(
            Fraction(2, 1_000_000_000),
            Fraction(11, 5_000_000_000),
        ),
        proof_reference="synthetic bin-wide primordial-power envelope",
        binwide_power_envelope_certified=True,
    )


@pytest.fixture(scope="module")
def power_cell(curvature_transfer, primordial_certificate):
    return construct_compact_kappa_bin_curvature_power_cell(
        curvature_transfer,
        primordial_certificate,
    )


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (Fraction(51), Fraction(49)),
        (Fraction(2), Fraction(1)),
        (Fraction(9), Fraction(2)),
        (Fraction(10**20), Fraction(3)),
    ],
)
def test_exact_rational_log_ratio_contains_reference_for_small_and_large_ratios(
    numerator: Fraction,
    denominator: Fraction,
) -> None:
    receipt = enclose_positive_rational_log_ratio(
        numerator=numerator,
        denominator=denominator,
    )
    lower, upper = receipt.logarithm_interval
    with localcontext() as context:
        context.prec = 120
        reference = (
            _decimal_fraction(numerator) / _decimal_fraction(denominator)
        ).ln()
        assert _decimal_fraction(lower) <= reference
        assert reference <= _decimal_fraction(upper)
    assert 0 < lower <= upper
    assert receipt.positive_tail_upper_bound >= 0
    assert receipt.reduced_ratio_in_one_to_two_proven
    assert receipt.exact_rational_positive_series_remainder_proven


def test_default_kappa_bin_log_width_has_exact_one_over_fifty_argument(
    power_cell,
) -> None:
    log_receipt = power_cell.logarithmic_width_enclosure
    lower, upper = log_receipt.logarithm_interval

    assert log_receipt.ratio == Fraction(51, 49)
    assert log_receipt.power_of_two_range_reduction_exponent == 0
    assert log_receipt.reduced_ratio == Fraction(51, 49)
    assert log_receipt.reduced_ratio_atanh_argument == Fraction(1, 50)
    with localcontext() as context:
        context.prec = 120
        reference = (Decimal(51) / Decimal(49)).ln()
        assert _decimal_fraction(lower) <= reference
        assert reference <= _decimal_fraction(upper)
    assert upper - lower < Fraction(1, 10**50)


def test_signed_curvature_transfer_is_squared_before_power_aggregation(
    power_cell,
) -> None:
    transfer = power_cell.convergence_transfer_per_comoving_curvature_interval
    squared = power_cell.convergence_transfer_modulus_squared_interval

    assert transfer[1] < 0
    assert squared == (transfer[1] ** 2, transfer[0] ** 2)
    assert squared[0] > 0
    assert _square_interval((Fraction(-2), Fraction(3))) == (
        Fraction(0),
        Fraction(9),
    )


def test_default_one_bin_reduced_power_is_nonnegative_and_contains_reference(
    power_cell,
) -> None:
    lower, upper = power_cell.reduced_angular_power_cell_interval
    transfer_lower, transfer_upper = (
        power_cell.convergence_transfer_per_comoving_curvature_interval
    )

    assert 0 < lower <= upper
    with localcontext() as context:
        context.prec = 120
        log_width = (Decimal(51) / Decimal(49)).ln()
        reference_lower = (
            log_width
            * _decimal_fraction(Fraction(2, 1_000_000_000))
            * _decimal_fraction(transfer_upper) ** 2
        )
        reference_upper = (
            log_width
            * _decimal_fraction(Fraction(11, 5_000_000_000))
            * _decimal_fraction(transfer_lower) ** 2
        )
        assert _decimal_fraction(lower) <= reference_lower
        assert reference_upper <= _decimal_fraction(upper)
    assert power_cell.reduced_power_integrand_interval[0] > 0


def test_power_cell_locks_measure_provenance_and_nonclaims(power_cell) -> None:
    assert power_cell.fixed_initial_slice_shared_wavenumber_scale_proven
    assert power_cell.d_log_k_equals_d_log_q_equals_d_log_kappa_proven
    assert power_cell.exact_rational_logarithmic_bin_width_enclosed
    assert power_cell.binwide_primordial_power_and_transfer_envelopes_used
    assert power_cell.signed_transfer_squared_before_power_aggregation
    assert power_cell.nonnegative_reduced_power_integrand_enclosed
    assert power_cell.four_pi_d_log_k_angular_power_identity_reused
    assert power_cell.four_pi_factor_kept_symbolic
    assert power_cell.single_compact_bin_reduced_angular_power_contribution_enclosed
    assert not power_cell.physical_wavenumber_mpc_inverse_calibrated
    assert not power_cell.primordial_pivot_wavenumber_calibrated
    assert not power_cell.primordial_spectrum_derived_from_ce
    assert not power_cell.all_k_compact_bin_coverage_enclosed
    assert not power_cell.exterior_tail_integrals_enclosed
    assert not power_cell.source_population_distribution_supplied
    assert not power_cell.post_born_or_relativistic_corrections_enclosed
    assert not power_cell.full_angular_power_spectrum_enclosed
    assert not power_cell.covariance_or_likelihood_enclosed


def test_primordial_power_certificate_rejects_nodes_negative_power_and_bad_q(
    curvature_transfer,
) -> None:
    harmonic = curvature_transfer.source_plane_harmonic_transfer_receipt
    common = dict(
        initial_kappa_interval=curvature_transfer.initial_kappa_interval,
        dimensionless_fixed_wavenumber_interval=(
            harmonic.dimensionless_fixed_wavenumber_interval
        ),
        proof_reference="synthetic proof",
    )
    with pytest.raises(ValueError, match="nodes are insufficient"):
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
            **common,
            primordial_curvature_power_interval=(1, 2),
            binwide_power_envelope_certified=False,
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
            **common,
            primordial_curvature_power_interval=(-1, 2),
            binwide_power_envelope_certified=True,
        )
    with pytest.raises(ValueError, match="must be positive"):
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
            initial_kappa_interval=curvature_transfer.initial_kappa_interval,
            dimensionless_fixed_wavenumber_interval=(0, 1),
            primordial_curvature_power_interval=(1, 2),
            proof_reference="synthetic proof",
            binwide_power_envelope_certified=True,
        )
    with pytest.raises(ValueError, match="positive and nonempty"):
        CertifiedPrimordialCurvaturePowerKappaBinEnvelope.freeze(
            initial_kappa_interval=curvature_transfer.initial_kappa_interval,
            dimensionless_fixed_wavenumber_interval=(1, 1),
            primordial_curvature_power_interval=(1, 2),
            proof_reference="synthetic proof",
            binwide_power_envelope_certified=True,
        )


def test_mismatched_bins_and_falsified_proofs_fail_closed(
    curvature_transfer,
    primordial_certificate,
) -> None:
    wrong_bin = replace(
        primordial_certificate,
        initial_kappa_interval=(Fraction(1, 20), Fraction(51, 1000)),
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        construct_compact_kappa_bin_curvature_power_cell(
            curvature_transfer,
            wrong_bin,
        )
    falsified = replace(
        primordial_certificate,
        binwide_not_nodewise_primordial_power_envelope_certified=False,
    )
    with pytest.raises(ValueError, match="proof prerequisites"):
        construct_compact_kappa_bin_curvature_power_cell(
            curvature_transfer,
            falsified,
        )
    unproven_transfer = replace(
        curvature_transfer,
        compact_bin_harmonic_transfer_per_comoving_curvature_enclosed=False,
    )
    with pytest.raises(ValueError, match="proof prerequisites"):
        construct_compact_kappa_bin_curvature_power_cell(
            unproven_transfer,
            primordial_certificate,
        )


def test_tampered_fixed_slice_q_scaling_fails_closed(
    curvature_transfer,
    primordial_certificate,
) -> None:
    harmonic = curvature_transfer.source_plane_harmonic_transfer_receipt
    q_lower, q_upper = harmonic.dimensionless_fixed_wavenumber_interval
    changed_q = (q_lower, q_upper + Fraction(1, 10_000))
    changed_harmonic = replace(
        harmonic,
        dimensionless_fixed_wavenumber_interval=changed_q,
    )
    changed_transfer = replace(
        curvature_transfer,
        source_plane_harmonic_transfer_receipt=changed_harmonic,
    )
    changed_certificate = replace(
        primordial_certificate,
        dimensionless_fixed_wavenumber_interval=changed_q,
    )

    with pytest.raises(ValueError, match="q-to-kappa scaling"):
        construct_compact_kappa_bin_curvature_power_cell(
            changed_transfer,
            changed_certificate,
        )


def test_invalid_log_ratio_and_order_fail_closed() -> None:
    with pytest.raises(ValueError, match="0 < denominator < numerator"):
        enclose_positive_rational_log_ratio(numerator=1, denominator=1)
    with pytest.raises(ValueError, match="0 < denominator < numerator"):
        enclose_positive_rational_log_ratio(numerator=1, denominator=0)
    with pytest.raises(ValueError, match="must be an integer"):
        enclose_positive_rational_log_ratio(
            numerator=2,
            denominator=1,
            highest_partial_sum_order=True,
        )
    with pytest.raises(ValueError, match="lie in"):
        enclose_positive_rational_log_ratio(
            numerator=2,
            denominator=1,
            highest_partial_sum_order=129,
        )
