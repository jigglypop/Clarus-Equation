import math

import numpy as np
import pytest

from reality_stone.clarus.dual_scc_basal_ganglia import (
    DualSCCBasalGanglia,
    DualSCCConfig,
    DualSCCConvergenceError,
)


def test_coloured_layers_are_two_sccs_but_reciprocal_union_is_one_macro_scc() -> None:
    system = DualSCCBasalGanglia()
    audit = system.topology_audit()
    assert audit.slow_is_strongly_connected
    assert audit.fast_is_strongly_connected
    assert audit.slow_component_count == 1
    assert audit.fast_component_count == 1
    assert audit.union_component_count == 1
    assert audit.union_is_single_macro_scc


def test_small_gain_certificate_and_residual_bound_close_the_fixed_point() -> None:
    system = DualSCCBasalGanglia()
    certificate = system.certificate
    assert certificate.certified
    assert certificate.spectral_radius < 1.0
    assert certificate.determinant_margin > 0.0
    assert certificate.weighted_contraction < 1.0

    result_a = system.settle((0.4, -0.3), (0.2, -0.5, 0.1))
    result_b = system.settle(
        (0.4, -0.3),
        (0.2, -0.5, 0.1),
        initial_slow=(1.0, -1.0),
        initial_fast=(-1.0, 1.0, 0.5),
    )
    assert np.allclose(result_a.slow_state, result_b.slow_state, atol=2e-10)
    assert np.allclose(result_a.fast_state, result_b.fast_state, atol=2e-10)
    assert result_a.error_bound <= system.config.tolerance
    assert result_a.residual <= result_a.error_bound
    assert all(bound >= 0.0 for bound in result_a.error_bound_by_layer)
    assert max(abs(value) for value in (*result_a.slow_state, *result_a.fast_state)) <= 1.0


def test_two_separately_contracting_layers_can_fail_when_cross_gain_is_too_large() -> None:
    system = DualSCCBasalGanglia(
        DualSCCConfig(
            slow_recurrence=0.80,
            fast_recurrence=0.80,
            slow_from_fast=0.30,
            fast_from_slow=0.30,
        )
    )
    assert system.config.slow_recurrence < 1.0
    assert system.config.fast_recurrence < 1.0
    assert system.certificate.spectral_radius > 1.0
    assert not system.certificate.certified
    with pytest.raises(ValueError, match="small-gain certificate failed"):
        system.settle((0.0, 0.0), (0.0, 0.0, 0.0))


def test_finite_budget_fails_closed_instead_of_returning_an_uncertified_state() -> None:
    system = DualSCCBasalGanglia(
        DualSCCConfig(max_iterations=1, tolerance=1e-16)
    )
    with pytest.raises(DualSCCConvergenceError, match="finite budget"):
        system.settle((0.9, -0.8), (0.7, -0.6, 0.5))


def test_broken_coloured_cycle_is_rejected_before_decision() -> None:
    broken = np.zeros((2, 2), dtype=np.float64)
    system = DualSCCBasalGanglia(slow_matrix=broken)
    assert not system.topology_audit().slow_is_strongly_connected
    with pytest.raises(ValueError, match="strongly connected"):
        system.settle((0.0, 0.0), (0.0, 0.0, 0.0))


def test_hold_channel_is_normalized_and_does_not_reorder_conditional_actions() -> None:
    system = DualSCCBasalGanglia()
    state = (0.8, -0.2, 0.1)
    low = system.policy(state, hold_bias_delta=-1.0)
    high = system.policy(state, hold_bias_delta=1.0)
    assert high.hold_probability > low.hold_probability
    assert np.allclose(
        high.conditional_action_probabilities,
        low.conditional_action_probabilities,
        rtol=0.0,
        atol=1e-15,
    )
    assert low.normalization_error <= 1e-15
    assert high.normalization_error <= 1e-15
    assert math.isclose(
        sum(high.action_probabilities) + high.hold_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_core_rejects_nonfinite_or_dimensionally_unresolved_inputs() -> None:
    system = DualSCCBasalGanglia()
    with pytest.raises(ValueError, match="finite"):
        system.settle((math.inf, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        system.settle((0.0,), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dimensionless"):
        system.settle((0.0, 0.0), (0.0, 0.0, 0.0), hold_bias_delta=math.nan)


def test_slow_to_fast_cross_path_is_causal_for_action_readout() -> None:
    coupled = DualSCCBasalGanglia()
    cut = DualSCCBasalGanglia(
        DualSCCConfig(slow_from_fast=0.0, fast_from_slow=0.0)
    )
    positive = coupled.settle((-0.8, 0.8), (0.0, 0.0, -0.5)).policy
    negative = coupled.settle((0.8, -0.8), (0.0, 0.0, -0.5)).policy
    cut_positive = cut.settle((-0.8, 0.8), (0.0, 0.0, -0.5)).policy
    cut_negative = cut.settle((0.8, -0.8), (0.0, 0.0, -0.5)).policy
    assert positive.conditional_action_probabilities[1] > 0.5
    assert negative.conditional_action_probabilities[1] < 0.5
    assert np.allclose(
        cut_positive.conditional_action_probabilities,
        cut_negative.conditional_action_probabilities,
        rtol=0.0,
        atol=1e-15,
    )
