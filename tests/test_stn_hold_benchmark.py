import numpy as np

from reality_stone.clarus.stn_hold_benchmark import (
    StnHoldBenchConfig,
    evaluate_stn_hold,
    solve_stopping_policy,
)


def test_stopping_policy_is_finite_and_conflict_sensitive() -> None:
    config = StnHoldBenchConfig(llr_points=501, quadrature_points=9, episodes_per_seed=100)
    policy = solve_stopping_policy(config)
    assert all(np.all(np.isfinite(value)) for value in policy.values)
    assert policy.should_hold(np.asarray((0.0,)), 0)[0]
    assert not policy.should_hold(np.asarray((config.llr_limit,)), 0)[0]
    assert not policy.should_hold(np.asarray((0.0,)), config.horizon - 1)[0]


def test_small_stn_hold_benchmark_reports_preregistered_stop_without_retuning() -> None:
    result = evaluate_stn_hold(
        StnHoldBenchConfig(
            llr_points=801,
            quadrature_points=11,
            episodes_per_seed=1200,
            seeds=6,
        )
    )
    assert result["schema"] == "clarus.stn-value-of-information-hold.validation.v1"
    assert result["gates"]["common_offset_exact_no_effect"]
    assert result["gates"]["utility_over_immediate"]
    assert result["gates"]["utility_over_always_wait"]
    assert not result["gates"]["conflict_selective_hold"]
    assert result["verdict"] == "STOP"
