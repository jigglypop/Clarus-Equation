import numpy as np

from reality_stone.clarus.dual_scc_basal_ganglia import DualSCCBasalGanglia
from reality_stone.clarus.dual_scc_probe_benchmark import (
    DualSCCProbeBenchConfig,
    _fixed_point_policy,
    _logit,
    evaluate_dual_scc_probe_benchmark,
)


def test_reduced_readout_comes_from_recurrent_policy_not_analytic_probability() -> None:
    core = DualSCCBasalGanglia()
    context_probability = 0.82
    evidence_probability = 0.67
    initial_slow = (0.15, -0.10)
    initial_fast = (0.0, 0.0, 0.0)
    observed = _fixed_point_policy(
        core,
        context_probability,
        evidence_probability,
        0.12,
        initial_slow=initial_slow,
        initial_fast=initial_fast,
        logit_clip=7.0,
        drive_scale=3.0,
        slow_memory_gain=0.72,
    )
    context_logit = _logit(context_probability)
    evidence_logit = _logit(evidence_probability)
    entropy = -(
        evidence_probability * np.log(evidence_probability)
        + (1.0 - evidence_probability) * np.log(1.0 - evidence_probability)
    ) / np.log(2.0)
    recurrent = core.settle(
        (
            -context_logit / 3.0 + 0.72 * initial_slow[0],
            context_logit / 3.0 + 0.72 * initial_slow[1],
        ),
        (-evidence_logit / 3.0, evidence_logit / 3.0, 2.0 * entropy - 1.0),
        initial_slow=initial_slow,
        initial_fast=initial_fast,
    )
    expected = core.policy(recurrent.fast_state, hold_bias_delta=0.12)
    assert observed[0] == expected.conditional_action_probabilities[1]
    assert observed[1] == expected.hold_probability
    assert not np.isclose(observed[0], evidence_probability, rtol=0.0, atol=1e-6)


def test_reduced_benchmark_cannot_issue_a_promotion_go() -> None:
    result = evaluate_dual_scc_probe_benchmark(
        seeds=(7001, 7002),
        config=DualSCCProbeBenchConfig(episodes_per_seed=8, quadrature_points=5),
    )
    assert result["verdict"] == "HOLD"
    assert not result["promotion_eligibility"]["eligible"]
    assert not result["gates"]["causal_integrity_instrumented"]
