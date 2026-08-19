import torch

import reality_stone.clarus.runtime_metric_intervention as g1_module
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig
from reality_stone.clarus.runtime_metric_intervention import (
    MetricInterventionConfig,
    _base_fixture,
)
from reality_stone.clarus.runtime_metric_sufficiency import (
    G2Config,
    G2_ENVIRONMENTS,
    _g2_bootstrap_lcb,
    _g2_calibrate,
    _g2_codebook,
    _g2_environment_snapshot,
    _g2_fixture,
    _g2_frozen_protocol,
    _g2_noise_schedule,
    _g2_noise_start,
    g2_metric_feature_utility,
)


def _legacy_selection(salience: torch.Tensor, threshold: float, budget: int) -> torch.Tensor:
    mask = torch.zeros_like(salience, dtype=torch.bool)
    eligible = salience >= threshold
    count = min(int(eligible.sum().item()), budget)
    if count:
        scored = salience.masked_fill(~eligible, float("-inf"))
        _, indices = torch.topk(scored, k=count)
        mask[indices] = True
    return mask


def test_force_all_active_is_default_off_legacy_exact_and_snapshot_safe() -> None:
    weight = torch.zeros(6, 6)
    salience = torch.tensor([-1.0, 0.23, 0.80, 0.10, 0.50, -0.20])
    legacy = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(dim=6, active_ratio=0.5, axon_delay=False),
        backend="torch",
        device="cpu",
    )
    assert not legacy.config.force_all_active_selection
    actual = legacy._select_active(salience, 3)
    expected = _legacy_selection(salience, legacy.config.active_threshold, 3)
    assert torch.equal(actual, expected)

    forced = BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=6,
            active_ratio=1.0,
            axon_delay=False,
            force_all_active_selection=True,
        ),
        backend="torch",
        device="cpu",
    )
    assert forced._select_active(torch.full((6,), -100.0), 0).all()
    restored = BrainRuntime.from_snapshot(forced.snapshot(), backend="torch", device="cpu")
    assert restored.config.force_all_active_selection
    assert restored._select_active(torch.full((6,), -100.0), 0).all()


def test_g2_fixture_is_dedicated_but_reproduces_g1_partition(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("G2 must not call G1 frozen-protocol validation")

    monkeypatch.setattr(g1_module, "_frozen_protocol", forbidden)
    config = G2Config(seed=97501)  # retired smoke seed, never an official G2 unit
    g2_snapshot, g2_injection, g2_groups = _g2_fixture(config.seed, config)
    g1_snapshot, g1_injection, g1_groups = _base_fixture(
        config.seed,
        MetricInterventionConfig(seed=config.seed, active_threshold=0.0),
    )
    assert _g2_frozen_protocol(config)
    assert g2_snapshot.config.force_all_active_selection
    assert not g1_snapshot.config.force_all_active_selection
    assert g2_snapshot.config.active_ratio == 1.0
    assert torch.equal(g2_snapshot.weight, g1_snapshot.weight)
    assert torch.equal(g2_injection, g1_injection)
    assert all(torch.equal(g2_groups[name], g1_groups[name]) for name in ("S", "T", "N"))


def test_g2_codebook_and_noise_intervals_are_frozen_and_disjoint() -> None:
    config = G2Config(seed=97501)
    first = _g2_codebook(config)
    second = _g2_codebook(config)
    assert first["audit"] == second["audit"]
    assert first["audit"]["direction_sha256"] == (
        "b7e218f6910309486ecb216f5428ee89c2b5860b8892c57943e36cc8b8e7ce0f"
    )
    assert first["audit"]["max_axis_alignment"] <= 0.95
    assert first["audit"]["max_pair_alignment"] < 1.0 - 1e-10
    assert first["audit"]["fit_test_pair_disjoint"]

    current = _g2_noise_schedule(97601)
    next_seed = _g2_noise_schedule(97602)
    assert current["interval_count"] == 108
    assert current["pairwise_disjoint"] and next_seed["pairwise_disjoint"]
    assert _g2_noise_start(97601, 0, "calibration", 0) == 8 * (97601 * 512)
    intervals = sorted(
        (row["start"], row["stop_inclusive"])
        for schedule in (current, next_seed) for row in schedule["rows"]
    )
    assert all(first_interval[1] < second_interval[0] for first_interval, second_interval in zip(intervals, intervals[1:]))


def test_g2_calibration_is_all_active_fixed_weight_and_covariant() -> None:
    config = G2Config(seed=97501)
    base, injection, _ = _g2_fixture(config.seed, config)
    for environment_index, (gain, noise) in enumerate(G2_ENVIRONMENTS):
        snapshot, audit = _g2_environment_snapshot(base, gain, noise)
        calibration = _g2_calibrate(
            snapshot, injection, config.seed, environment_index, config,
        )
        assert audit["weight_unchanged"]
        assert calibration["integrity"]
        assert calibration["B_h"].shape == (6, 3, 3)
        assert calibration["transform_covariance_residual"] <= 1e-6
        assert calibration["transform_metric_residual"] <= 1e-6
        assert min(calibration["eigenvalues_C"]) > 0
        assert min(calibration["eigenvalues_g"]) > 0
        assert min(calibration["eigenvalues_Q_raw"]) > 0
        assert all(
            row["active_counts"] == [48] * 7 and row["weight_unchanged"]
            for row in calibration["positive"] + calibration["negative"]
        )


def test_g2_full_retired_smoke_seed_closes_all_integrity_and_budget_gates() -> None:
    result = g2_metric_feature_utility(97501)
    assert result["frozen_protocol"]
    assert result["dedicated_g2_fixture"]
    assert result["integrity"]
    assert result["transform_feature_residual_max"] <= 1e-6
    assert result["cterms_formula_residual_max"] <= 1e-10
    assert result["no_repackaging"]["fit_raw_feature_residual"] <= 1e-10
    assert result["no_repackaging"]["fit_standardized_feature_residual"] <= 1e-10
    assert result["no_repackaging"]["prediction_residual"] <= 1e-8
    assert result["no_repackaging"]["nonaliased_feature_arrays"]
    assert result["coefficient_ledger"]["D"] == 6
    assert result["coefficient_ledger"]["D+g"] == 7
    assert result["coefficient_ledger"]["D+Cterms"] == 12
    assert result["coefficient_ledger"]["D+Craw"] == 12
    assert result["coefficient_ledger"]["D2"] == 21
    assert set(result["deltas_vs_metric"]) == {
        "D", "D+C", "D+E", "D+perm", "D+Bpath", "D+Qraw", "D+Cterms",
        "D+Craw", "D2", "persistence", "global_mean", "raw_Bpath",
    }
    assert _g2_bootstrap_lcb([0.1, 0.2, 0.3]) == _g2_bootstrap_lcb([0.1, 0.2, 0.3])
