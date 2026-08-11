import math

import pytest

from reality_stone.clarus.hazard_mixture_benchmark import (
    evaluate_hazard_ensemble,
    small_hazard_ensemble_config,
)
from reality_stone.clarus.hazard_mixture_dag import HazardMixtureDecisionDag


def test_hazard_mixture_joint_filter_and_pending_order() -> None:
    model = HazardMixtureDecisionDag()
    assert math.isclose(
        sum(sum(row) for row in model.joint_posterior),
        1.0,
    )
    output = model.forward_step((0.4, -0.3, 0.8), (0.2, -0.1, 0.3, 0.0))
    assert output.joint_sum_error <= 1e-14
    assert output.action_mixture_residual <= 1e-14
    with pytest.raises(RuntimeError, match="pending mixture"):
        model.forward_step((0.4, -0.3, 0.8), (0.2, -0.1, 0.3, 0.0))
    result = model.commit_outcome(-1.0)
    assert result.joint_sum_error <= 1e-14
    assert result.outcome_bayes_residual <= 1e-14
    assert math.isclose(sum(result.hazard_weights), 1.0)


def test_hazard_ensemble_small_benchmark_integrity() -> None:
    result = evaluate_hazard_ensemble(small_hazard_ensemble_config())
    assert result["schema"] == "clarus.recurrent-bg-dag-hazard-ensemble.validation.v1"
    assert result["gates"]["joint_simplex"]
    assert result["gates"]["filter_identities"]
    assert result["track_terminal"]
    assert result["future_reads"] == 0
