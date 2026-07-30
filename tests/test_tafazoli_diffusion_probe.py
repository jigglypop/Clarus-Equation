from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import reality_stone.clarus.tafazoli_diffusion_probe as diffusion
from reality_stone.clarus.tafazoli_call_graph_probe import TrajectoryDesign
from reality_stone.clarus.tafazoli_diffusion_probe import (
    OU_FULL,
    OU_ISO,
    QUADRATIC_DRIFT_FULL_Q,
    STATE_SCALE,
    TIME_SCALE,
    DiffusionClaimLocks,
    DiffusionProbeConfig,
    DriftModel,
    OOFResiduals,
    apply_drift_model,
    crossfit_drift_residuals,
    fit_drift_model,
    fit_noise_model,
    score_multivariate_gaussian,
    run_tafazoli_diffusion_probe_from_arrays,
    validate_diffusion_claim_locks,
    validate_tafazoli_diffusion_report,
)
from reality_stone.clarus.tafazoli_session_operator_probe import SessionSpec


def _oof(
    residuals: np.ndarray,
    current: np.ndarray,
    anchors: np.ndarray,
) -> OOFResiduals:
    count = residuals.shape[0]
    return OOFResiduals(
        residuals=residuals,
        current=current,
        anchor_indices=anchors,
        trial_indices=np.arange(count),
        fold_count=2,
        every_trial_held_out_once=True,
    )


def _score(
    family: str,
    noise,
    residuals: np.ndarray,
    current: np.ndarray,
    anchors: np.ndarray,
    *,
    train_count: int,
    drift_parameters: int = 0,
):
    return score_multivariate_gaussian(
        family,
        residuals,
        noise,
        oof_train_vector_count=train_count,
        drift_parameter_count=drift_parameters,
        anchor_indices=anchors,
        current=current,
    )


def test_full_covariance_is_spd_and_scores_near_singular_data_finitely() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    residuals = np.column_stack((base, base, 2.0 * base))
    current = rng.normal(size=residuals.shape)
    anchors = np.zeros(residuals.shape[0], dtype=np.int64)
    config = DiffusionProbeConfig()

    model = fit_noise_model(OU_FULL, _oof(residuals, current, anchors), config=config)
    score = _score(
        OU_FULL,
        model,
        residuals[:80],
        current[:80],
        anchors[:80],
        train_count=residuals.shape[0],
    )

    assert np.min(np.linalg.eigvalsh(model.base_covariance)) > 0.0
    np.linalg.cholesky(model.base_covariance)
    assert np.isfinite(score.total_codelength_bits)
    assert np.isfinite(score.bits_per_test_scalar)


@pytest.mark.parametrize("correlation, expected", [(0.92, OU_FULL), (0.0, OU_ISO)])
def test_correlated_and_isotropic_bic_controls(
    correlation: float,
    expected: str,
) -> None:
    train_rng = np.random.default_rng(20)
    test_rng = np.random.default_rng(21)
    covariance = np.asarray(((1.0, correlation), (correlation, 1.0)))
    train = train_rng.multivariate_normal(np.zeros(2), covariance, size=1600)
    test = test_rng.multivariate_normal(np.zeros(2), covariance, size=1600)
    current_train = train_rng.normal(size=train.shape)
    current_test = test_rng.normal(size=test.shape)
    train_anchors = np.zeros(train.shape[0], dtype=np.int64)
    test_anchors = np.zeros(test.shape[0], dtype=np.int64)
    oof = _oof(train, current_train, train_anchors)
    config = DiffusionProbeConfig()
    scores = {}
    for family in (OU_ISO, OU_FULL):
        noise = fit_noise_model(family, oof, config=config)
        scores[family] = _score(
            family,
            noise,
            test,
            current_test,
            test_anchors,
            train_count=train.shape[0],
        ).total_codelength_bits
    assert min(scores, key=scores.get) == expected


def test_state_scale_and_time_scale_win_only_on_their_own_signal() -> None:
    rng = np.random.default_rng(33)
    count = 1200
    anchors = np.tile(np.arange(2), count // 2)
    current = np.column_stack(
        (rng.choice((-3.0, 3.0), count), rng.normal(scale=0.2, size=count))
    )
    state_scale = np.where(current[:, 0] < 0.0, 0.2, 2.0)
    time_scale = np.where(anchors == 0, 0.2, 2.0)
    config = DiffusionProbeConfig(kmeans_restarts=2, kmeans_max_iterations=30)

    for scales, winner in ((state_scale, STATE_SCALE), (time_scale, TIME_SCALE)):
        train = rng.normal(size=(count, 2)) * scales[:, None]
        test = rng.normal(size=(count, 2)) * scales[:, None]
        oof = _oof(train, current, anchors)
        scores = {}
        for family in (OU_FULL, TIME_SCALE, STATE_SCALE):
            noise = fit_noise_model(
                family,
                oof,
                config=config,
                full_training_current=current.reshape(30, 40, 2),
            )
            scores[family] = _score(
                family,
                noise,
                test,
                current,
                anchors,
                train_count=count,
            ).total_codelength_bits
        assert min(scores, key=scores.get) == winner


def test_quadratic_mean_beats_spurious_state_dependent_noise() -> None:
    rng = np.random.default_rng(71)
    train_x = rng.uniform(-2.0, 2.0, size=(1, 1000, 1))
    test_x = rng.uniform(-2.0, 2.0, size=(1, 1000, 1))

    def design(x: np.ndarray, noise: np.ndarray) -> TrajectoryDesign:
        y = 0.4 * x + 0.9 * np.square(x) + noise
        return TrajectoryDesign(x, x, y, np.arange(x.shape[1]))

    train = design(train_x, rng.normal(scale=0.25, size=train_x.shape))
    test = design(test_x, rng.normal(scale=0.25, size=test_x.shape))
    linear = fit_drift_model(train, order=1, quadratic=False)
    quadratic = fit_drift_model(train, order=1, quadratic=True)
    train_linear = (train.successor - apply_drift_model(linear, train)).reshape(-1, 1)
    train_quad = (train.successor - apply_drift_model(quadratic, train)).reshape(-1, 1)
    test_linear = (test.successor - apply_drift_model(linear, test)).reshape(-1, 1)
    test_quad = (test.successor - apply_drift_model(quadratic, test)).reshape(-1, 1)
    current_train = train.current.reshape(-1, 1)
    current_test = test.current.reshape(-1, 1)
    anchors = np.arange(1000)
    config = DiffusionProbeConfig(kmeans_restarts=2, kmeans_max_iterations=30)
    state = fit_noise_model(
        STATE_SCALE,
        _oof(train_linear, current_train, anchors),
        config=config,
        full_training_current=train.current,
    )
    quad = fit_noise_model(
        QUADRATIC_DRIFT_FULL_Q,
        _oof(train_quad, current_train, anchors),
        config=config,
    )
    state_score = _score(
        STATE_SCALE,
        state,
        test_linear,
        current_test,
        anchors,
        train_count=1000,
        drift_parameters=linear.parameter_count,
    )
    quad_score = _score(
        QUADRATIC_DRIFT_FULL_Q,
        quad,
        test_quad,
        current_test,
        anchors,
        train_count=1000,
        drift_parameters=quadratic.parameter_count,
    )
    assert quad_score.total_codelength_bits < state_score.total_codelength_bits


def test_ar2_requires_second_order_history() -> None:
    rng = np.random.default_rng(82)
    trials, anchors = 12, 250
    current = rng.normal(size=(trials, anchors, 1))
    second_lag = rng.normal(size=(trials, anchors, 1))
    successor = 0.05 * current + 0.9 * second_lag + rng.normal(
        scale=0.05, size=current.shape
    )
    order_one = TrajectoryDesign(
        history=current,
        current=current,
        successor=successor,
        anchor_indices=np.arange(anchors),
    )
    order_two = replace(
        order_one,
        history=np.concatenate((current, second_lag), axis=-1),
    )
    first = fit_drift_model(order_one, order=1, quadratic=False)
    second = fit_drift_model(order_two, order=2, quadratic=False)
    first_sse = np.sum(np.square(successor - apply_drift_model(first, order_one)))
    second_sse = np.sum(np.square(successor - apply_drift_model(second, order_two)))
    assert second_sse < first_sse * 0.02


def test_ou_semigroup_composes_but_switching_operator_does_not() -> None:
    operator = np.asarray(((0.8, 0.1), (-0.2, 0.7)))
    drift = DriftModel(
        family="VAR_ORDER_1",
        order=1,
        latent_rank=2,
        coefficients=np.vstack((np.zeros(2), operator)),
        parameter_count=6,
    )
    current = np.asarray(((1.0, -0.5), (0.2, 1.4)))
    composed = diffusion._semigroup_prediction(drift, current, horizon_steps=3)
    np.testing.assert_allclose(composed, current @ np.linalg.matrix_power(operator, 3))

    alternate = np.asarray(((0.2, 0.8), (0.6, -0.1)))
    switching_truth = current @ operator @ alternate @ operator
    assert np.linalg.norm(composed - switching_truth) > 0.1


def test_reverse_classification_is_explicitly_descriptive_not_a_gate() -> None:
    assert not diffusion.DirectionClassification(
        forward_full_bits_per_scalar=2.0,
        reverse_full_bits_per_scalar=1.0,
        forward_full_total_bits=200.0,
        reverse_full_total_bits=100.0,
        test_scalar_count=100,
        lower_code_direction="REVERSE_LOWER_CODE",
        used_in_primary_diffusion_gate=False,
    ).used_in_primary_diffusion_gate
    config = diffusion.DiffusionProbeConfig()
    assert config.minimum_codelength_advantage_bits_per_scalar == 0.01
    assert config.semigroup_max_excess_bits_per_scalar == 0.02


def test_window_lag_stride_and_preregistered_threshold_are_locked() -> None:
    for change in (
        {"observation_window_bins": 9},
        {"lag_bins": 9},
        {"primary_stride_bins": 9},
        {"minimum_codelength_advantage_bits_per_scalar": 0.005},
        {"minimum_codelength_advantage_bits_per_scalar": 0.02},
    ):
        with pytest.raises(ValueError):
            DiffusionProbeConfig(**change)


def test_covariance_crossfit_holds_each_training_trial_out_once() -> None:
    rng = np.random.default_rng(404)
    latent = rng.normal(size=(6, 71, 1))
    config = DiffusionProbeConfig(covariance_oof_fold_count=2)
    first = crossfit_drift_residuals(
        latent, order=1, quadratic=False, config=config, reverse=False
    )
    second = crossfit_drift_residuals(
        latent, order=1, quadratic=False, config=config, reverse=False
    )
    assert "outer_test" not in inspect.signature(crossfit_drift_residuals).parameters
    assert first.every_trial_held_out_once
    assert set(first.trial_indices) == set(range(latent.shape[0]))
    np.testing.assert_array_equal(first.residuals, second.residuals)


def test_claim_locks_and_config_are_deterministic_and_strict_json() -> None:
    config = DiffusionProbeConfig()
    assert json.dumps(config.__dict__, sort_keys=True, allow_nan=False)
    with pytest.raises(ValueError, match="claim lock"):
        validate_diffusion_claim_locks(
            replace(DiffusionClaimLocks(), d1_d3_rows_treated_as_paired_trials=True)
        )


def test_small_report_is_deterministic_serializable_and_leakage_locked() -> None:
    rng = np.random.default_rng(903)
    dim1 = rng.lognormal(mean=1.0, sigma=0.15, size=(6, 2, 71))
    dim3 = rng.lognormal(mean=1.0, sigma=0.15, size=(6, 2, 71))
    config = DiffusionProbeConfig(
        outer_fold_count=2,
        covariance_oof_fold_count=2,
        kmeans_restarts=2,
        kmeans_max_iterations=20,
        run_reverse_classification=False,
        run_markov_order_sensitivity=False,
        run_semigroup_sensitivity=False,
    )
    specs = (
        SessionSpec(1, "animal_a", 0, 1),
        SessionSpec(2, "animal_b", 1, 2),
    )

    first = run_tafazoli_diffusion_probe_from_arrays(
        dim1, dim3, config=config, session_specs=specs
    )
    second = run_tafazoli_diffusion_probe_from_arrays(
        dim1, dim3, config=config, session_specs=specs
    )
    validate_tafazoli_diffusion_report(first)
    assert json.dumps(first.to_dict(), sort_keys=True, allow_nan=False) == json.dumps(
        second.to_dict(), sort_keys=True, allow_nan=False
    )
    assert len(first.session_specs) == 2
    assert len(first.results) == 2 * 2 * 2
    assert all(len(item.fold_results) == 2 for item in first.results)
    assert {
        (item.session_index_one_based, item.dimension, item.event_mean_removed)
        for item in first.results
    } == {
        (session, dimension, event)
        for session in (1, 2)
        for dimension in (1, 3)
        for event in (False, True)
    }
    assert {item.dimension for item in first.results} == {1, 3}
    assert first.source_file_md5 is None
    assert first.blind_fields_used == ()
    assert first.saved_test_role == "not_used"
    assert first.train_only_preprocessing
    assert not any(first.claim_locks.__dict__.values())
    assert all(
        fold.covariance_fit_from_outer_train_trial_oof
        and not fold.outer_test_used_for_covariance_or_gate
        and not fold.d1_d3_rows_treated_as_paired_trials
        for item in first.results
        for fold in item.fold_results
    )


def test_protocol_and_official_result_keep_diffusion_claims_separate() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository_root / "benchmarks" / "tafazoli_diffusion_probe_v1.json"
        ).read_text(encoding="utf-8")
    )
    result = json.loads(
        (
            repository_root
            / "benchmarks"
            / "tafazoli_diffusion_probe_v1_result.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["schema_version"] == "clarus-tafazoli-diffusion-probe/v1"
    assert protocol["snapshot_constraints"]["allowed_dimensions_one_based"] == [1, 3]
    assert protocol["screening_thresholds"] == {
        "minimum_codelength_advantage_bits_per_scalar": 0.01,
        "minimum_session_unit_win_fraction": 0.5,
        "semigroup_max_excess_bits_per_scalar_vs_direct_refit": 0.02,
        "required_groups": [
            "raw_all",
            "raw_Chico",
            "raw_Silas",
            "event_mean_removed_all",
            "event_mean_removed_Chico",
            "event_mean_removed_Silas",
        ],
    }
    assert result["official_checksum_verified"]
    assert result["completed_grid"]["session_dimension_mode_units"] == 108
    assert result["observed_markov_order"]["order_1_units"] == 108
    assert (
        result["claim_verdicts"][
            "model_relative_local_affine_isotropic_gaussian_proxy_winner"
        ]
        == "YES"
    )
    for key in (
        "gaussian_innovation_law_identified",
        "state_dependent_noise_proxy_survived_controls",
        "biological_diffusion_identified",
        "generative_reverse_process_identified",
        "score_function_identified",
        "causal_diffusion_mechanism_identified",
        "spatial_graph_diffusion_identified",
    ):
        assert result["claim_verdicts"][key] == "NO"
    assert (
        result["claim_verdicts"]["biological_diffusion_exists_or_is_absent"]
        == "TEST_UNAVAILABLE"
    )
