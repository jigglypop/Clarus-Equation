from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from reality_stone.clarus.resonant_spatiotemporal_mask import (
    ResonantMaskStage,
    resonant_mask_manifest_sha256,
    resonant_spatiotemporal_mask_audit,
    validate_resonant_spatiotemporal_mask_audit,
)


DESIGN = np.array(
    [
        [1.0, 0.8, 1.2, 0.9],
        [1.1, 0.0, 0.0, 1.3],
        [0.7, 0.0, 0.0, 0.0],
    ]
)
TRAINING = np.zeros(DESIGN.shape, dtype=bool)
TRAINING.flat[[0, 1, 2, 3, 4, 7]] = True
HELDOUT = ~TRAINING
PREARRIVAL = np.zeros(DESIGN.shape, dtype=bool)
PREARRIVAL.flat[[6, 11]] = True
OFF_SUPPORT = np.zeros(DESIGN.shape, dtype=bool)
OFF_SUPPORT.flat[[5, 9, 10]] = True
TARGET = np.zeros(DESIGN.shape, dtype=bool)
TARGET.flat[8] = True
PREPROCESSING_HASH = hashlib.sha256(b"synthetic paired subtraction v2").hexdigest()
CALIBRATION_HASH = hashlib.sha256(b"synthetic frozen mask calibration v2").hexdigest()

CONFIG = {
    "observations_are_independent_blocks": True,
    "gaussian_mean_model_declared": True,
    "expected_response_sign": 1,
    "familywise_alpha": 0.05,
    "equivalence_bound": 0.05,
    "minimum_target_response": 0.5,
    "maximum_training_reduced_chi_square": 4.0,
    "maximum_covariance_condition_number": 1.0e8,
    "covariance_rank_relative_tolerance": 1.0e-10,
    "minimum_paired_covariance_eigenvalue": 1.0e-8,
    "minimum_residual_mean_variance": 1.0e-10,
    "minimum_trials": 64,
}


def _block_ids(trials: int) -> tuple[str, ...]:
    return tuple(f"block-{index:04d}" for index in range(trials))


def _responses(*, seed: int = 731, trials: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    common_baseline = rng.normal(3.0, 0.1, size=(trials, *DESIGN.shape))
    paired_noise = rng.normal(0.0, 0.02, size=(trials, *DESIGN.shape))
    sham = common_baseline
    matched = common_baseline + 2.0 * DESIGN + paired_noise
    return matched, sham


def _manifest_hash(
    *,
    design: np.ndarray = DESIGN,
    training: np.ndarray = TRAINING,
    heldout: np.ndarray = HELDOUT,
    prearrival: np.ndarray = PREARRIVAL,
    off_support: np.ndarray = OFF_SUPPORT,
    target: np.ndarray = TARGET,
    matched_ids: tuple[str, ...] | None = None,
    sham_ids: tuple[str, ...] | None = None,
    config: dict[str, object] | None = None,
) -> str:
    effective_config = {**CONFIG, **(config or {})}
    ids = matched_ids or _block_ids(256)
    return resonant_mask_manifest_sha256(
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        matched_block_ids=ids,
        sham_block_ids=sham_ids or ids,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        **effective_config,
    )


def _audit(
    *,
    matched: np.ndarray | None = None,
    sham: np.ndarray | None = None,
    design: np.ndarray = DESIGN,
    training: np.ndarray = TRAINING,
    heldout: np.ndarray = HELDOUT,
    prearrival: np.ndarray = PREARRIVAL,
    off_support: np.ndarray = OFF_SUPPORT,
    target: np.ndarray = TARGET,
    matched_ids: tuple[str, ...] | None = None,
    sham_ids: tuple[str, ...] | None = None,
    declared_hash: str | None = None,
    frozen: bool = True,
    config: dict[str, object] | None = None,
):
    if matched is None or sham is None:
        matched, sham = _responses()
    ids = matched_ids or _block_ids(matched.shape[0])
    other_ids = sham_ids or ids
    effective_config = {**CONFIG, **(config or {})}
    manifest = declared_hash or _manifest_hash(
        design=design,
        training=training,
        heldout=heldout,
        prearrival=prearrival,
        off_support=off_support,
        target=target,
        matched_ids=ids,
        sham_ids=other_ids,
        config=effective_config,
    )
    return resonant_spatiotemporal_mask_audit(
        matched_response=matched,
        sham_response=sham,
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        matched_block_ids=ids,
        sham_block_ids=other_ids,
        preprocessing_artifact_sha256=PREPROCESSING_HASH,
        design_calibration_artifact_sha256=CALIBRATION_HASH,
        declared_manifest_sha256=manifest,
        manifest_frozen_before_data=frozen,
        masks_fixed_before_holdout=True,
        **effective_config,
    )


def test_frozen_joint_design_predicts_crossed_holdout_cells() -> None:
    report = _audit()

    assert report.manifest_hash_matches
    assert report.paired_block_ids_aligned
    assert report.paired_block_ids_unique
    assert report.covariance_nonvacuous
    assert report.training_model_degrees_of_freedom == 5
    assert report.simultaneous_comparison_count == 18
    assert report.simultaneous_confidence_multiplier == pytest.approx(
        3.020753839710795,
        abs=1.0e-12,
    )
    assert report.joint_mask_gls_pass
    assert report.heldout_prediction_pass
    assert report.prearrival_equivalence_pass
    assert report.off_support_equivalence_pass
    assert report.target_response_pass
    assert report.heldout_localization_pass
    assert report.conditional_spatiotemporal_response_mask
    assert report.maximum_supported_stage is (
        ResonantMaskStage.CONDITIONAL_SPATIOTEMPORAL_RESPONSE_MASK
    )
    assert report.fitted_global_amplitude is not None
    assert abs(report.fitted_global_amplitude - 2.0) < 0.01
    assert report.factor_rescaling_counterexample_exact
    assert not report.individual_factor_normalizations_identifiable
    assert not any(vars(report.claim_locks).values())


def test_heldout_mutation_does_not_change_training_fit_and_fails_prediction() -> None:
    matched, sham = _responses()
    baseline = _audit(matched=matched, sham=sham)
    mutated = matched.copy()
    mutated[:, TARGET] += 0.2
    report = _audit(matched=mutated, sham=sham)

    assert report.fitted_global_amplitude == baseline.fitted_global_amplitude
    assert report.joint_mask_gls_pass
    assert not report.heldout_prediction_pass
    assert not report.conditional_spatiotemporal_response_mask


def test_rank_two_training_interaction_fails_one_amplitude_gls() -> None:
    matched, sham = _responses()
    interacted = matched.copy()
    interacted[:, 0, 0] += 0.2
    report = _audit(matched=interacted, sham=sham)

    assert not report.joint_mask_gls_pass
    assert not report.heldout_prediction_pass
    assert "training GLS gate" in " ".join(report.blockers)


def test_prearrival_response_is_an_early_window_failure_not_a_causality_claim() -> None:
    matched, sham = _responses()
    leaked = matched.copy()
    leaked[:, PREARRIVAL] += 0.2
    report = _audit(matched=leaked, sham=sham)

    assert not report.prearrival_equivalence_pass
    assert not report.heldout_prediction_pass
    assert not report.claim_locks.relativistic_causality_derived
    assert "early-time control" in " ".join(report.blockers)


def test_broadcast_off_support_response_fails_localization() -> None:
    matched, sham = _responses()
    broadcast = matched.copy()
    broadcast[:, OFF_SUPPORT] += 0.2
    report = _audit(matched=broadcast, sham=sham)

    assert not report.off_support_equivalence_pass
    assert not report.heldout_localization_pass
    assert not report.conditional_spatiotemporal_response_mask


def test_posthoc_design_change_breaks_frozen_manifest_hash() -> None:
    changed_design = DESIGN.copy()
    changed_design[0, 0] *= 1.1
    report = _audit(
        design=changed_design,
        declared_hash=_manifest_hash(design=DESIGN),
    )

    assert not report.manifest_hash_matches
    assert not report.joint_mask_gls_pass
    assert report.maximum_supported_stage is ResonantMaskStage.PAIRED_RESPONSE_CONTROL
    assert "hash does not match" in report.first_blocker


def test_exact_zero_covariance_returns_a_structured_failure() -> None:
    trials = 128
    sham = np.zeros((trials, *DESIGN.shape))
    matched = sham + 2.0 * DESIGN
    report = _audit(
        matched=matched,
        sham=sham,
        matched_ids=_block_ids(trials),
    )

    assert not report.covariance_nonvacuous
    assert report.fitted_global_amplitude is None
    assert not report.joint_mask_gls_pass
    assert not report.conditional_spatiotemporal_response_mask


def test_zero_product_and_non_boolean_masks_are_rejected() -> None:
    matched, sham = _responses()
    with pytest.raises(ValueError, match="zero-product design"):
        _audit(matched=matched, sham=sham, design=np.zeros_like(DESIGN))

    with pytest.raises(ValueError, match="training_mask must be a boolean array"):
        _audit(matched=matched, sham=sham, training=TRAINING.astype(int))


def test_validator_recomputes_every_field_and_rejects_adversarial_tampering() -> None:
    report = _audit()
    tampered = replace(
        report,
        manifest_sha256="0" * 64,
        computed_manifest_sha256="f" * 64,
        manifest_hash_matches=True,
        covariance_nonvacuous=False,
        training_reduced_chi_square=float("nan"),
        first_blocker="PASS",
        blockers=(),
    )

    with pytest.raises(ValueError, match="canonical recomputation"):
        validate_resonant_spatiotemporal_mask_audit(tampered)

    relaxed_raw = replace(report.raw_inputs, minimum_trials=128)
    with pytest.raises(ValueError, match="canonical recomputation"):
        validate_resonant_spatiotemporal_mask_audit(
            replace(report, raw_inputs=relaxed_raw)
        )


def test_unfrozen_manifest_stage_and_claim_tampering_fail_closed() -> None:
    unfrozen = _audit(frozen=False)
    assert not unfrozen.joint_mask_gls_pass
    assert unfrozen.maximum_supported_stage is ResonantMaskStage.PAIRED_RESPONSE_CONTROL

    report = _audit()
    with pytest.raises(ValueError, match="claim locks"):
        validate_resonant_spatiotemporal_mask_audit(
            replace(
                report,
                claim_locks=replace(report.claim_locks, ce_coupling_derived=True),
            )
        )
    with pytest.raises(ValueError, match="canonical recomputation"):
        validate_resonant_spatiotemporal_mask_audit(
            replace(report, maximum_supported_stage=ResonantMaskStage.PAIRED_RESPONSE_CONTROL)
        )


def test_one_training_cell_is_saturated_and_cannot_pass_gls() -> None:
    design = np.array([[1.0, 0.0], [0.0, 1.0]])
    training = np.array([[True, False], [False, False]])
    heldout = ~training
    prearrival = np.array([[False, True], [False, False]])
    off_support = np.array([[False, False], [True, False]])
    target = np.array([[False, False], [False, True]])
    trials = 128
    rng = np.random.default_rng(44)
    sham = rng.normal(size=(trials, *design.shape))
    matched = sham + design + rng.normal(0.0, 0.02, size=sham.shape)

    report = _audit(
        matched=matched,
        sham=sham,
        design=design,
        training=training,
        heldout=heldout,
        prearrival=prearrival,
        off_support=off_support,
        target=target,
        matched_ids=_block_ids(trials),
    )

    assert report.training_model_degrees_of_freedom == 0
    assert not report.training_design_non_saturated
    assert not report.joint_mask_gls_pass
    assert report.maximum_supported_stage is ResonantMaskStage.FROZEN_MANIFEST_CONTROL


def test_rank_deficient_protected_cells_cannot_create_zero_width_certainty() -> None:
    design = np.array(
        [
            [1.0, 0.8, 1.2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    training = np.zeros(design.shape, dtype=bool)
    training.flat[[0, 1, 2]] = True
    heldout = ~training
    prearrival = np.zeros(design.shape, dtype=bool)
    prearrival.flat[[3, 4]] = True
    off_support = np.zeros(design.shape, dtype=bool)
    off_support.flat[[5, 6, 7]] = True
    target = np.zeros(design.shape, dtype=bool)
    target.flat[8] = True
    trials = 256
    rng = np.random.default_rng(48)
    sham = np.zeros((trials, *design.shape))
    paired = 2.0 * design[np.newaxis, ...] + np.zeros_like(sham)
    paired[:, training] += rng.normal(0.0, 0.02, size=(trials, 3))
    matched = sham + paired

    report = _audit(
        matched=matched,
        sham=sham,
        design=design,
        training=training,
        heldout=heldout,
        prearrival=prearrival,
        off_support=off_support,
        target=target,
    )

    assert report.paired_covariance_rank == 3
    assert not report.heldout_covariance_nonvacuous
    assert not report.heldout_prediction_pass
    assert report.maximum_heldout_residual_upper_bound is None
    assert not report.conditional_spatiotemporal_response_mask


def test_minimum_trials_and_inference_settings_are_manifest_bound() -> None:
    baseline = _manifest_hash()
    changed_minimum = _manifest_hash(config={"minimum_trials": 128})
    changed_alpha = _manifest_hash(config={"familywise_alpha": 0.01})

    assert baseline != changed_minimum
    assert baseline != changed_alpha


def test_duplicated_or_permuted_block_ids_fail_independence_control() -> None:
    matched, sham = _responses(trials=128)
    unique = _block_ids(128)
    duplicated = tuple(f"block-{index // 4:04d}" for index in range(128))
    duplicate_report = _audit(
        matched=matched,
        sham=sham,
        matched_ids=duplicated,
        sham_ids=duplicated,
    )
    permuted = tuple(reversed(unique))
    permuted_report = _audit(
        matched=matched,
        sham=sham,
        matched_ids=unique,
        sham_ids=permuted,
    )

    assert not duplicate_report.paired_block_ids_unique
    assert not duplicate_report.minimum_independent_blocks_met
    assert not duplicate_report.joint_mask_gls_pass
    assert not permuted_report.paired_block_ids_aligned
    assert not permuted_report.joint_mask_gls_pass


def test_high_variance_exact_mean_fails_simultaneous_training_interval() -> None:
    trials = 256
    rng = np.random.default_rng(902)
    noise = rng.normal(0.0, 1.0, size=(trials, *DESIGN.shape))
    noise -= np.mean(noise, axis=0, keepdims=True)
    sham = np.zeros_like(noise)
    matched = 2.0 * DESIGN + noise
    report = _audit(matched=matched, sham=sham)

    assert report.maximum_training_absolute_residual is not None
    assert report.maximum_training_absolute_residual < 1.0e-12
    assert report.maximum_training_residual_upper_bound is not None
    assert report.maximum_training_residual_upper_bound > CONFIG["equivalence_bound"]
    assert not report.joint_mask_gls_pass
