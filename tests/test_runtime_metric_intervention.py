import math

import torch

from reality_stone.clarus.runtime_metric_intervention import (
    MetricInterventionConfig,
    _arm_snapshot,
    _base_fixture,
    _bootstrap_lcb,
    _edge_delta,
    _fixed_transform,
    _frozen_protocol,
    _metric_from_calibration,
    _pulse_rollout,
    g1_edge_intervention,
)


def test_g1_fixture_is_seeded_orthonormal_and_mechanism_open() -> None:
    config = MetricInterventionConfig(seed=97401)
    first, injection, groups = _base_fixture(config.seed, config)
    again, again_injection, again_groups = _base_fixture(config.seed, config)
    other, _, _ = _base_fixture(config.seed + 1, config)

    torch.testing.assert_close(injection.T @ injection, torch.eye(3), atol=0.0, rtol=0.0)
    assert torch.equal(injection, again_injection)
    assert all(torch.equal(groups[name], again_groups[name]) for name in ("S", "T", "N"))
    assert torch.unique(torch.cat(tuple(groups.values()))).numel() == config.dim
    assert torch.equal(first.weight, again.weight)
    assert not torch.equal(first.weight, other.weight)
    assert torch.count_nonzero(torch.diag(first.weight)) == 0
    assert first.config.active_threshold == config.active_threshold == 0.04
    assert _frozen_protocol(config)


def test_g1_edge_install_is_exact_and_receiver_matched() -> None:
    config = MetricInterventionConfig(seed=97401)
    base, injection, groups = _base_fixture(config.seed, config)
    treatment_delta = _edge_delta(config, groups, "T")
    scramble_delta = _edge_delta(config, groups, "N")
    _, treatment = _arm_snapshot(base, injection, groups, config, "treatment")
    _, scrambled = _arm_snapshot(base, injection, groups, config, "scrambled")

    assert torch.count_nonzero(treatment_delta) == 256
    assert torch.count_nonzero(scramble_delta) == 256
    assert not torch.equal(treatment_delta, scramble_delta)
    assert treatment["declared_receiver"] == "T"
    assert scrambled["declared_receiver"] == "N"
    for audit in (treatment, scrambled):
        assert audit["delta_nonzero_count"] == 256
        assert audit["applied_nonzero_count"] == 256
        assert audit["delta_positive_count"] == 256
        assert audit["applied_positive_count"] == 256
        assert math.isclose(audit["delta_frobenius_norm"], 1.28, abs_tol=1e-6)
        assert math.isclose(audit["installed_delta_norm"], 1.28, abs_tol=1e-6)
        assert audit["applied_reconstruction_residual"] <= 1e-7
        assert audit["declared_block_inside_max_error"] <= 1e-7
        assert audit["declared_block_outside_max_abs"] <= 1e-7
        assert audit["only_declared_block"]
        assert audit["dense_sparse_parity"]


def test_g1_pulses_restore_state_and_metric_transform_covaries() -> None:
    config = MetricInterventionConfig(seed=97401)
    base, injection, groups = _base_fixture(config.seed, config)
    treatment, _ = _arm_snapshot(base, injection, groups, config, "treatment")
    first = _pulse_rollout(treatment, injection, 0, config.calibration_amplitude, config)
    second = _pulse_rollout(treatment, injection, 0, config.calibration_amplitude, config)
    metric = _metric_from_calibration(treatment, injection, config)

    torch.testing.assert_close(first["trajectory"], second["trajectory"], atol=0.0, rtol=0.0)
    assert first["driven_coordinates_active"]
    assert first["active_count_after_pulse"] >= config.dim // 3
    assert first["rows_before"] == first["rows_after_pulse"] == first["rows_after_rollout"] == 0
    assert first["weight_unchanged"]
    assert first["automatic_stdp_updates"] == 0
    assert first["first_passage"] == 1
    assert metric["cross_response"] > 0.0
    assert min(metric["eigenvalues_C"]) > 0.0
    assert min(metric["eigenvalues_g"]) > 0.0
    assert metric["target_variance_identity_residual"] <= 1e-8
    assert metric["covariance_transform_residual"] <= 1e-6
    assert metric["metric_transform_residual"] <= 1e-6
    assert metric["transform_finite"] and metric["transform_invertible"]
    assert bytes.fromhex(metric["transform_bytes_hex"]) == (
        _fixed_transform().numpy().tobytes()
    )


def test_g1_full_audit_and_nonfrozen_config_cannot_go() -> None:
    result = g1_edge_intervention(97401)
    assert result["frozen_protocol"]
    assert result["snapshot_restore_parity"]
    assert result["edge_match"]
    assert result["integrity"]
    assert result["arms"]["treatment"]["calibration"]["cross_response"] > (
        result["arms"]["scrambled"]["calibration"]["cross_response"]
    )
    assert all(row["passes"] for row in result["first_passage_by_sign"])

    diagnostic = g1_edge_intervention(
        97401, MetricInterventionConfig(seed=97401, active_threshold=0.03),
    )
    assert not diagnostic["frozen_protocol"]
    assert not diagnostic["integrity"]
    assert diagnostic["status"] == "STOP"


def test_g1_bootstrap_is_seed_level_and_deterministic() -> None:
    values = [0.10, 0.12, 0.14, 0.16]
    first = _bootstrap_lcb(values)
    second = _bootstrap_lcb(values)
    assert first == second
    assert 0.10 <= first <= 0.16
