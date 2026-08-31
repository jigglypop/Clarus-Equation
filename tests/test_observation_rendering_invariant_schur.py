from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.observation_rendering_invariant_schur import (
    DEFAULT_D_EFF,
    DEFAULT_DELTA,
    DEFAULT_Q_LOW,
    audit_constructed_observation_rendering,
    audit_probability_flow,
    conditional_hubble_readout,
    controlled_depth_rendering_sequence,
    continuous_event_flow_beta,
    metric_representative_family,
    next_controlled_depth_spatial_scale,
    normalized_matrix_schur_spectrum,
    normalized_scalar_schur_loss,
    relative_pole_readout,
    required_event_rate_for_target_flow,
    rendering_composition_candidates,
)


def test_scalar_normalized_loss_survives_independent_p_and_q_rescalings() -> None:
    boundary = 2.3
    mixing = -0.7
    internal = 1.9
    original = normalized_scalar_schur_loss(boundary, mixing, internal)
    p_scale = 4.0
    q_scale = -2.5
    transformed = normalized_scalar_schur_loss(
        boundary / p_scale**2,
        mixing / (p_scale * q_scale),
        internal / q_scale**2,
    )

    assert transformed.normalized_loss == pytest.approx(original.normalized_loss)
    assert transformed.retained_factor == pytest.approx(original.retained_factor)
    assert transformed.raw_effective_kernel == pytest.approx(
        original.raw_effective_kernel / p_scale**2
    )
    assert original.p_and_q_rescaling_invariant
    assert not original.raw_effective_kernel_is_normalization_invariant


def test_scalar_normalized_loss_exposes_unstable_branch_without_hiding_it() -> None:
    stable = normalized_scalar_schur_loss(1.0, 0.4, 1.0)
    unstable = normalized_scalar_schur_loss(1.0, 1.1, 1.0)

    assert stable.strictly_positive
    assert stable.retained_factor == pytest.approx(0.84)
    assert not unstable.strictly_positive
    assert unstable.retained_factor < 0.0
    with pytest.raises(ValueError, match="positive"):
        normalized_scalar_schur_loss(0.0, 0.4, 1.0)


def test_matrix_generalized_spectrum_is_invariant_under_coordinates() -> None:
    boundary = np.asarray(((3.0, 0.4), (0.4, 2.1)))
    mixing = np.asarray(((0.4, -0.2), (0.1, 0.3)))
    internal = np.asarray(((2.4, 0.2), (0.2, 1.7)))
    original = normalized_matrix_schur_spectrum(boundary, mixing, internal)

    p_change = np.asarray(((1.7, 0.3), (-0.2, 0.8)))
    q_change = np.asarray(((0.9, -0.4), (0.2, 1.5)))
    p_inverse = np.linalg.inv(p_change)
    q_inverse = np.linalg.inv(q_change)
    transformed = normalized_matrix_schur_spectrum(
        p_inverse.T @ boundary @ p_inverse,
        p_inverse.T @ mixing @ q_inverse,
        q_inverse.T @ internal @ q_inverse,
    )

    assert transformed.normalized_loss_eigenvalues == pytest.approx(
        original.normalized_loss_eigenvalues
    )
    assert transformed.retained_generalized_eigenvalues == pytest.approx(
        original.retained_generalized_eigenvalues
    )
    assert transformed.determinant_ratio == pytest.approx(original.determinant_ratio)
    assert original.determinant_ratio == pytest.approx(
        original.determinant_ratio_from_spectrum
    )
    assert original.coordinate_redefinition_invariant


def test_matrix_audit_requires_positive_physical_blocks() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        normalized_matrix_schur_spectrum(
            ((1.0, 0.0), (0.0, -1.0)),
            ((0.1,), (0.2,)),
            ((1.0,),),
        )


def test_parallel_schur_is_additive_but_other_compositions_need_axioms() -> None:
    candidates = rendering_composition_candidates(DEFAULT_DELTA, 3.0, DEFAULT_Q_LOW)
    x = DEFAULT_DELTA / 3.0
    y = DEFAULT_Q_LOW**2

    assert candidates.geometric_loss == pytest.approx(0.05925280780332461)
    assert candidates.probability_loss == pytest.approx(0.0023665033321246818)
    assert candidates.simultaneous_retained_factor == pytest.approx(1.0 - x - y)
    assert candidates.simultaneous_scale_factor == pytest.approx(0.9687005155694667)
    assert candidates.sequential_retained_factor == pytest.approx(
        (1.0 - x) * (1.0 - y)
    )
    assert candidates.sequential_scale_factor == pytest.approx(0.9687728891910915)
    assert candidates.exponential_retained_factor == pytest.approx(
        math.exp(-(x + y))
    )
    assert candidates.exponential_scale_factor == pytest.approx(0.9696601249172384)
    assert (
        candidates.sequential_retained_factor
        - candidates.simultaneous_retained_factor
    ) == pytest.approx(candidates.sequential_cross_term)
    assert candidates.simultaneous_is_exact_for_block_diagonal_q
    assert candidates.sequential_requires_rewhitened_cascade_axiom
    assert candidates.exponential_requires_log_semigroup_axiom
    assert not candidates.composition_selected
    assert not candidates.prediction


def test_unit_q_branch_makes_simultaneous_factor_nonpositive() -> None:
    candidates = rendering_composition_candidates(DEFAULT_DELTA, 3.0, 1.0)

    assert candidates.simultaneous_retained_factor < 0.0
    assert candidates.simultaneous_scale_factor is None


def test_zero_d_flow_keeps_time_and_energy_maps_open() -> None:
    low = audit_probability_flow()
    unit = audit_probability_flow(DEFAULT_D_EFF, 1.0)

    assert abs(low.fixed_point_residual) < 1.0e-15
    assert low.local_derivative == pytest.approx(0.15458752312007412)
    assert low.locally_attracting
    assert unit.fixed_point_residual == pytest.approx(0.0)
    assert not unit.locally_attracting
    assert unit.unit_branch_repelling
    assert continuous_event_flow_beta(
        DEFAULT_Q_LOW, DEFAULT_D_EFF, 1.0
    ) == pytest.approx(0.0, abs=1.0e-15)
    assert not low.event_depth_to_scale_factor_derived
    assert not low.probability_to_kernel_derived
    assert not low.probability_is_energy_density


def test_controlled_depth_low_branch_is_a_certified_unique_sequence() -> None:
    sequence = controlled_depth_rendering_sequence(8)

    assert sequence.q_values == pytest.approx(
        (
            0.0,
            0.04167897710728743,
            0.04758143081297484,
            0.048482317708456984,
            0.04862131179349705,
            0.048642792065905284,
            0.048646112493243234,
            0.04864662578618275,
            0.048646705134778526,
        )
    )
    assert sequence.spatial_scale_factors == pytest.approx(
        (
            0.9699212298927555,
            0.9690253118799145,
            0.9687534256137965,
            0.9687087576079261,
            0.9687017911804205,
            0.9687007127986995,
            0.9687005460594983,
            0.9687005202828656,
            0.9687005162980999,
        )
    )
    assert sequence.contraction_bound == pytest.approx(0.360024795514655)
    assert sequence.minimum_spatial_radicand == pytest.approx(
        0.8417191386540371
    )
    assert sequence.interval_invariant
    assert sequence.contraction_certified
    assert sequence.monotone_low_branch
    assert sequence.unique_within_supplied_discrete_map
    assert sequence.lambda_step_relation_closed
    assert sequence.dimensionless_core_arguments == (
        ("D * (1 - q_n)", "D and q_n are dimensionless"),
        ("1 - x - q_n**2", "x=delta/d and q_n are dimensionless"),
    )
    assert not sequence.physical_protocol_depth_derived
    assert not sequence.probability_to_spatial_residue_derived
    assert not sequence.event_depth_to_scale_factor_derived
    assert not sequence.absolute_dark_energy_density_derived
    assert not sequence.prediction


def test_lambda_only_step_relation_reproduces_every_controlled_depth() -> None:
    sequence = controlled_depth_rendering_sequence(8)

    reconstructed = tuple(
        next_controlled_depth_spatial_scale(previous)
        for previous in sequence.spatial_scale_factors[:-1]
    )

    assert reconstructed == pytest.approx(sequence.spatial_scale_factors[1:])


def test_continuous_embedding_can_inverse_fit_any_compatible_target_flow() -> None:
    q = 0.02
    slow_beta = 0.001
    fast_beta = 0.004
    slow_rate = required_event_rate_for_target_flow(q, DEFAULT_D_EFF, slow_beta)
    fast_rate = required_event_rate_for_target_flow(q, DEFAULT_D_EFF, fast_beta)

    assert continuous_event_flow_beta(q, DEFAULT_D_EFF, slow_rate) == pytest.approx(
        slow_beta
    )
    assert continuous_event_flow_beta(q, DEFAULT_D_EFF, fast_rate) == pytest.approx(
        fast_beta
    )
    assert fast_rate == pytest.approx(4.0 * slow_rate)
    with pytest.raises(ValueError, match="not identifiable at a fixed point"):
        required_event_rate_for_target_flow(
            DEFAULT_Q_LOW, DEFAULT_D_EFF, target_beta=0.0
        )


def test_relative_pole_ratio_separates_three_channel_assignments() -> None:
    retained = rendering_composition_candidates(
        DEFAULT_DELTA, 3.0, DEFAULT_Q_LOW
    ).simultaneous_retained_factor
    spatial = relative_pole_readout(1.0, retained)
    temporal = relative_pole_readout(retained, 1.0)
    universal = relative_pole_readout(retained, retained)

    assert spatial.relative_clock_ruler_factor == pytest.approx(math.sqrt(retained))
    assert temporal.relative_clock_ruler_factor == pytest.approx(
        1.0 / math.sqrt(retained)
    )
    assert universal.relative_clock_ruler_factor == pytest.approx(1.0)
    assert not spatial.pure_conformal_cancellation
    assert universal.pure_conformal_cancellation

    rescaled = relative_pole_readout(7.0, 7.0 * retained, 5.0, 5.0)
    assert rescaled.relative_clock_ruler_factor == pytest.approx(
        spatial.relative_clock_ruler_factor
    )
    assert spatial.common_field_rescaling_invariant
    assert not spatial.separate_lapse_and_ruler_identified


def test_conformal_family_does_not_fix_lapse_and_ruler_separately() -> None:
    relative_factor = 0.97
    first = metric_representative_family(relative_factor, 1.0)
    second = metric_representative_family(relative_factor, 3.0)

    assert first != pytest.approx(second)
    assert first[1] / first[0] == pytest.approx(relative_factor)
    assert second[1] / second[0] == pytest.approx(relative_factor)


def test_conditional_hubble_formula_is_not_promoted_to_prediction() -> None:
    reference = conditional_hubble_readout(1.0, 1.0, 0.0)
    running = conditional_hubble_readout(0.96, 0.94, -0.02)

    assert reference.h_rendered_over_h_reference == pytest.approx(1.0)
    assert running.h_rendered_over_h_reference == pytest.approx(
        0.99 / math.sqrt(0.96)
    )
    assert running.same_field_normalization_axiom_required
    assert not running.protocol_clock_ruler_calibration_derived
    assert not running.unique_from_relative_pole_ratio
    assert not running.prediction


def test_default_audit_closes_invariants_but_keeps_physics_open() -> None:
    audit = audit_constructed_observation_rendering()
    candidates = audit.composition_candidates

    assert audit.status == "INVARIANT_FORMULATION_CLOSED_PHYSICAL_BRIDGES_OPEN"
    assert audit.raw_legacy_kernel_numerically_equals_simultaneous_factor_only
    assert audit.legacy_raw_witness.raw_effective_kernel == pytest.approx(
        candidates.simultaneous_retained_factor
    )
    assert audit.legacy_raw_witness.retained_factor != pytest.approx(
        candidates.simultaneous_retained_factor
    )
    assert (
        audit.simultaneous_schur_witness.retained_generalized_eigenvalues
        == pytest.approx((candidates.simultaneous_retained_factor,))
    )
    assert audit.universal_channel_readout.relative_clock_ruler_factor == pytest.approx(
        1.0
    )
    assert audit.spatial_only_channel_readout.relative_clock_ruler_factor == pytest.approx(
        candidates.simultaneous_scale_factor
    )
    assert audit.temporal_only_channel_readout.relative_clock_ruler_factor == pytest.approx(
        1.0 / candidates.simultaneous_scale_factor
    )
    assert (
        audit.conditional_spatial_hubble_readout.h_rendered_over_h_reference
        == pytest.approx(1.0, abs=1.0e-15)
    )
    assert audit.controlled_depth_sequence.q_values[-1] == pytest.approx(
        0.048646705134778526
    )
    assert not audit.controlled_depth_sequence.prediction
    assert not audit.q_to_kernel_map_derived
    assert not audit.temporal_spatial_channel_assignment_derived
    assert not audit.event_depth_to_redshift_map_derived
    assert not audit.retarded_ctp_background_variation_derived
    assert not audit.absolute_dark_energy_density_derived
    assert not audit.prediction
