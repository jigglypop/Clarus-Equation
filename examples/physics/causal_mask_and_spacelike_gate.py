"""Synthetic spatiotemporal-mask and spacelike-marginal controls."""

from __future__ import annotations

import hashlib

import numpy as np

from reality_stone.clarus.resonant_spatiotemporal_mask import (
    resonant_mask_manifest_sha256,
    resonant_spatiotemporal_mask_audit,
)
from reality_stone.clarus.spacelike_marginal_gate import (
    BinnedSelectorCounts,
    spacelike_marginal_gate,
)


def spacelike_control():
    local = BinnedSelectorCounts(
        detector_id="local-A",
        bin_labels=("low", "high"),
        selector_0_counts=(950_000, 50_000),
        selector_1_counts=(50_000, 950_000),
    )
    remote = tuple(
        BinnedSelectorCounts(
            detector_id=f"spacelike-B{index}",
            bin_labels=("low", "high"),
            selector_0_counts=(500_000, 500_000),
            selector_1_counts=(500_000, 500_000),
        )
        for index in (1, 2)
    )
    return spacelike_marginal_gate(
        local_a=local,
        spacelike_b=remote,
        delta_min=0.8,
        delta_ns=0.02,
        selector_randomized=True,
        bins_predeclared_before_unblinding=True,
        familywise_alpha=0.05,
        minimum_count_per_selector=100,
    )


def mask_control():
    design = np.array(
        [
            [1.0, 0.8, 1.2, 0.9],
            [1.1, 0.0, 0.0, 1.3],
            [0.7, 0.0, 0.0, 0.0],
        ]
    )
    training = np.zeros(design.shape, dtype=bool)
    training.flat[[0, 1, 2, 3, 4, 7]] = True
    heldout = ~training
    prearrival = np.zeros(design.shape, dtype=bool)
    prearrival.flat[[6, 11]] = True
    off_support = np.zeros(design.shape, dtype=bool)
    off_support.flat[[5, 9, 10]] = True
    target = np.zeros(design.shape, dtype=bool)
    target.flat[8] = True
    trials = 256
    block_ids = tuple(f"synthetic-block-{index:04d}" for index in range(trials))
    preprocessing_hash = hashlib.sha256(
        b"synthetic paired subtraction v2"
    ).hexdigest()
    calibration_hash = hashlib.sha256(
        b"synthetic frozen mask calibration v2"
    ).hexdigest()
    config = {
        "observations_are_independent_blocks": True,
        "gaussian_mean_model_declared": True,
        "expected_response_sign": 1,
        "familywise_alpha": 0.05,
        "equivalence_bound": 0.05,
        "minimum_target_response": 0.5,
        "maximum_covariance_condition_number": 1.0e8,
        "covariance_rank_relative_tolerance": 1.0e-10,
        "minimum_paired_covariance_eigenvalue": 1.0e-8,
        "minimum_residual_mean_variance": 1.0e-10,
        "minimum_trials": 64,
    }
    manifest_sha256 = resonant_mask_manifest_sha256(
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        matched_block_ids=block_ids,
        sham_block_ids=block_ids,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        **config,
    )

    rng = np.random.default_rng(731)
    baseline = rng.normal(3.0, 0.1, size=(trials, *design.shape))
    paired_noise = rng.normal(0.0, 0.02, size=(trials, *design.shape))
    return resonant_spatiotemporal_mask_audit(
        matched_response=baseline + 2.0 * design + paired_noise,
        sham_response=baseline,
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        matched_block_ids=block_ids,
        sham_block_ids=block_ids,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        declared_manifest_sha256=manifest_sha256,
        manifest_frozen_before_data=True,
        masks_fixed_before_holdout=True,
        **config,
    )


def main() -> None:
    spacelike = spacelike_control()
    mask = mask_control()
    print("randomized spacelike-marginal / frozen response-mask control")
    print(f"  spacelike stage             {spacelike.maximum_supported_stage.value}")
    print(f"  local TV LCB                {spacelike.local_a.tv_lower_confidence_bound:.6f}")
    print(
        "  maximum spacelike TV UCB   "
        f"{spacelike.maximum_spacelike_upper_confidence_bound:.6f}"
    )
    print(f"  spatiotemporal stage        {mask.maximum_supported_stage.value}")
    print(f"  fitted global amplitude     {mask.fitted_global_amplitude:.9f}")
    print(
        "  maximum heldout upper      "
        f"{mask.maximum_heldout_residual_upper_bound:.9f}"
    )
    print(
        "  declared-block mask        "
        f"{mask.conditional_declared_block_spatiotemporal_response_mask}"
    )
    print("  CE/new matter/stress        False (claim locked)")


if __name__ == "__main__":
    main()
