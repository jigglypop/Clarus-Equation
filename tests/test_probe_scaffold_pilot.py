from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.probe_scaffold_pilot import (
    CommonKernelProbeReadout,
    PhaseNoiseSweepPoint,
    ProbeScaffoldClaimLocks,
    energy_ledger_audit,
    phase_lock_order_parameter,
    phase_noise_sweep_audit,
    post_pump_common_kernel_audit,
    probe_scaffold_pilot_report,
    probe_selectivity_audit,
    validate_probe_scaffold_pilot_report,
)


JITTER = (-0.02, 0.01, 0.0, 0.02, -0.01, 0.015, -0.015, 0.0)
ZEROS = (0.0,) * len(JITTER)
BASELINE = (10.0,) * len(JITTER)


def _bias_corrected_symmetric_resultant(angle: float) -> float:
    raw = math.cos(angle)
    count = len(JITTER)
    return math.sqrt(max(0.0, (count * raw * raw - 1.0) / (count - 1.0)))


def _phase_point(
    noise: float,
    angle: float,
    *,
    held_out: bool = False,
    response_override: float | None = None,
    reference_response: float = 0.0,
) -> PhaseNoiseSweepPoint:
    phases = tuple(value for _ in range(len(JITTER) // 2) for value in (-angle, angle))
    response = (
        0.2 + 2.0 * _bias_corrected_symmetric_resultant(angle)
        if response_override is None
        else response_override
    )
    a_on_matched = tuple(base + response + jitter for base, jitter in zip(BASELINE, JITTER))
    a_on_sham = BASELINE
    a_off_matched = BASELINE
    a_off_sham = BASELINE
    b_on_matched = tuple(base + reference_response for base in BASELINE)
    return PhaseNoiseSweepPoint(
        noise_strength=noise,
        phase_offsets_rad=phases,
        probe_a_pump_on_matched=a_on_matched,
        probe_a_pump_on_sham=a_on_sham,
        probe_a_pump_off_matched=a_off_matched,
        probe_a_pump_off_sham=a_off_sham,
        reference_pump_on_matched=b_on_matched,
        reference_pump_on_sham=BASELINE,
        reference_pump_off_matched=BASELINE,
        reference_pump_off_sham=BASELINE,
        held_out=held_out,
    )


def _passing_sweep():
    return phase_noise_sweep_audit(
        [
            _phase_point(0.0, 0.10),
            _phase_point(1.0, 0.50),
            _phase_point(1.5, 0.70, held_out=True),
            _phase_point(2.0, 0.90),
            _phase_point(3.0, 1.25),
        ],
        minimum_probe_a_effect=0.5,
        minimum_selective_effect=0.5,
        minimum_absolute_correlation=0.95,
        heldout_prediction_equivalence_bound=0.15,
    )


def _kernel_probe(
    probe_id: str,
    gain: float,
    *,
    held_out: bool = False,
    kernel: float = 1.2,
    pre_effect: float = 0.0,
    jitter: tuple[float, ...] = JITTER,
) -> CommonKernelProbeReadout:
    sham = (4.0,) * len(JITTER)
    post = tuple(4.0 + gain * (kernel + offset) for offset in jitter)
    pre = tuple(4.0 + gain * pre_effect for _ in JITTER)
    return CommonKernelProbeReadout(
        probe_id=probe_id,
        calibrated_response_gain=gain,
        post_pump_response=post,
        post_pump_sham=sham,
        pre_pump_response=pre,
        pre_pump_sham=sham,
        held_out=held_out,
    )


def _passing_common_kernel(**overrides):
    arguments = dict(
        probes=[
            _kernel_probe("A", 1.0),
            _kernel_probe("B", 2.0),
            _kernel_probe("C", 0.5, held_out=True),
        ],
        residual_drive_monitor_post=ZEROS,
        residual_drive_monitor_sham=ZEROS,
        nuisance_monitor_post=ZEROS,
        nuisance_monitor_sham=ZEROS,
        pump_start_time_s=0.0,
        pump_off_time_s=10.0,
        post_readout_start_time_s=12.0,
        post_readout_end_time_s=13.0,
        minimum_pump_off_dwell_s=1.0,
        minimum_common_kernel=0.5,
        kernel_factorization_equivalence_bound=0.15,
        pre_response_equivalence_bound=0.1,
        monitor_equivalence_bound=0.1,
        residual_drive_to_kernel_gain_upper_bound=1.0,
        apparatus_memory_kernel_upper_bound=0.05,
        minimum_unexplained_kernel_margin=0.5,
        heldout_probe_designation_declared=True,
        calibration_fixed_before_pump=True,
        blind_analysis_declared=True,
        separate_heldout_readout_chain_declared=True,
    )
    arguments.update(overrides)
    return post_pump_common_kernel_audit(**arguments)


def _diagonal_covariance(variance: float) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(variance if row == column else 0.0 for column in range(10))
        for row in range(10)
    )


def _repeated(value: float) -> tuple[float, ...]:
    return (value,) * len(JITTER)


def _passing_energy(**overrides):
    arguments = dict(
        pump_work_j=_repeated(10.0),
        controller_work_j=_repeated(1.0),
        probe_work_j=_repeated(0.2),
        transfer_work_j=_repeated(0.3),
        preexisting_reservoir_release_j=_repeated(0.5),
        candidate_decoupled_energy_j=_repeated(4.0),
        radiated_energy_j=_repeated(3.0),
        thermal_mechanical_energy_j=_repeated(2.0),
        reservoir_storage_j=_repeated(2.0),
        recovered_work_j=_repeated(1.0),
        energy_covariance_j2=_diagonal_covariance(1.0e-4),
        pump_and_controller_decoupled_at_endpoint_declared=True,
    )
    arguments.update(overrides)
    return energy_ledger_audit(**arguments)


def test_phase_lock_order_parameter_has_finite_sample_bias_control() -> None:
    locked = phase_lock_order_parameter([0.0] * 8)
    uniform = phase_lock_order_parameter(
        [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0] * 2
    )

    assert locked.raw_resultant_length == 1.0
    assert locked.bias_corrected_resultant_length == 1.0
    assert uniform.bias_corrected_resultant_length == 0.0
    assert locked.effective_sample_size_method == "KISH_WEIGHT_ONLY"
    assert not locked.time_autocorrelation_corrected


def test_phase_lock_rejects_vacuous_weights_and_nonfinite_phase() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        phase_lock_order_parameter([0.0] * 4, weights=[0.0] * 4)
    with pytest.raises(ValueError, match="finite"):
        phase_lock_order_parameter([0.0, 0.0, 0.0, math.nan])
    with pytest.raises(ValueError, match="real scalar"):
        phase_lock_order_parameter([0.0, 0.0, 0.0, True])


def test_factorial_selectivity_removes_controller_only_effect() -> None:
    controller_only = tuple(base + 2.0 for base in BASELINE)
    result = probe_selectivity_audit(
        controller_only,
        BASELINE,
        controller_only,
        BASELINE,
        BASELINE,
        BASELINE,
        BASELINE,
        BASELINE,
    )

    assert result.probe_a_effect.mean_effect == 0.0
    assert not result.private_dressing_conditionally_supported

    with pytest.raises(ValueError, match="minimum_probe_a_effect must be positive"):
        probe_selectivity_audit(
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            minimum_probe_a_effect=0.0,
            minimum_selective_effect=0.0,
        )
    with pytest.raises(ValueError, match="reference_equivalence_bound"):
        probe_selectivity_audit(
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            reference_equivalence_bound=1_000.0,
        )
    with pytest.raises(ValueError, match="confidence_multiplier"):
        probe_selectivity_audit(
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            BASELINE,
            confidence_multiplier=1.0e-12,
        )


def test_global_heating_is_not_probe_selectivity() -> None:
    point = _phase_point(0.0, 0.1, reference_response=2.0)
    result = probe_selectivity_audit(
        point.probe_a_pump_on_matched,
        point.probe_a_pump_on_sham,
        point.probe_a_pump_off_matched,
        point.probe_a_pump_off_sham,
        point.reference_pump_on_matched,
        point.reference_pump_on_sham,
        point.reference_pump_off_matched,
        point.reference_pump_off_sham,
    )

    assert result.probe_a_response_detected
    assert not result.reference_equivalent_to_zero
    assert not result.private_dressing_conditionally_supported
    assert not result.public_environment_change_implied


def test_phase_noise_sweep_predicts_designated_heldout_level() -> None:
    sweep = _passing_sweep()

    assert sweep.noise_phase_correlation is not None
    assert sweep.noise_phase_correlation < -0.95
    assert sweep.phase_selectivity_correlation is not None
    assert sweep.phase_selectivity_correlation > 0.999
    assert sweep.heldout_response_matches_designated_prediction
    assert sweep.phase_lock_dependence_conditionally_supported
    assert sweep.heldout_prediction_residual.confidence_multiplier >= 4.3026
    heldout = next(point for point in sweep.points if point.held_out)
    training_errors = [
        point.selectivity.difference_in_differences.standard_error
        for point in sweep.points
        if not point.held_out
    ]
    assert sweep.heldout_prediction_residual.standard_error >= (
        heldout.selectivity.difference_in_differences.standard_error
        + min(training_errors)
    )
    assert not sweep.noise_to_phase_lock_dynamics_derived
    assert not sweep.causation_by_phase_lock_derived


def test_phase_sweep_rejects_missing_holdout_constant_response_or_extrapolation() -> None:
    no_heldout = [
        _phase_point(index, angle)
        for index, angle in enumerate((0.1, 0.5, 0.7, 0.9, 1.2))
    ]
    with pytest.raises(ValueError, match="exactly one noise point"):
        phase_noise_sweep_audit(no_heldout)

    designated = [
        _phase_point(0.0, 0.1),
        _phase_point(1.0, 0.5),
        _phase_point(1.5, 0.7, held_out=True),
        _phase_point(2.0, 0.9),
        _phase_point(3.0, 1.2),
    ]
    with pytest.raises(ValueError, match="must not exceed half"):
        phase_noise_sweep_audit(
            designated,
            heldout_prediction_equivalence_bound=2_000.0,
        )

    extrapolated = [
        _phase_point(0.0, 0.5),
        _phase_point(1.0, 0.7),
        _phase_point(2.0, 0.9),
        _phase_point(3.0, 1.1),
        _phase_point(4.0, 0.1, held_out=True),
    ]
    with pytest.raises(ValueError, match="within the training range"):
        phase_noise_sweep_audit(extrapolated)

    constant = [
        _phase_point(0.0, 0.1, response_override=1.0),
        _phase_point(1.0, 0.5, response_override=1.0),
        _phase_point(1.5, 0.7, response_override=1.0, held_out=True),
        _phase_point(2.0, 0.9, response_override=1.0),
        _phase_point(3.0, 1.25, response_override=1.0),
    ]
    sweep = phase_noise_sweep_audit(constant)
    assert not sweep.selective_response_tracks_phase_lock
    assert not sweep.phase_lock_dependence_conditionally_supported


def test_duplicate_noise_and_mismatched_trial_counts_are_rejected() -> None:
    points = [
        _phase_point(0.0, 0.1),
        _phase_point(1.0, 0.5),
        _phase_point(1.0, 0.7, held_out=True),
        _phase_point(2.0, 0.9),
        _phase_point(3.0, 1.2),
    ]
    with pytest.raises(ValueError, match="unique"):
        phase_noise_sweep_audit(points)

    point = _phase_point(0.0, 0.1)
    with pytest.raises(ValueError, match="equal length"):
        probe_selectivity_audit(
            point.probe_a_pump_on_matched[:-1],
            point.probe_a_pump_on_sham,
            point.probe_a_pump_off_matched,
            point.probe_a_pump_off_sham,
            point.reference_pump_on_matched,
            point.reference_pump_on_sham,
            point.reference_pump_off_matched,
            point.reference_pump_off_sham,
        )


def test_three_calibrated_probes_factorize_a_common_post_pump_kernel() -> None:
    audit = _passing_common_kernel()

    raw_means = [item.raw_post_effect.mean_effect for item in audit.probe_audits]
    assert raw_means == pytest.approx([1.2, 2.4, 0.6])
    assert audit.training_probes_factorize_common_kernel
    assert audit.heldout_probe_matches_common_kernel
    assert audit.monitors_and_apparatus_memory_cannot_explain_kernel
    assert audit.probe_correlation_model == "UNMEASURED_WORST_CASE_CORRELATION"
    assert not audit.probe_covariance_measured
    training_errors = [
        item.normalized_post_kernel.standard_error
        for item in audit.probe_audits
        if not item.held_out
    ]
    assert audit.fitted_training_kernel.standard_error >= max(training_errors)
    assert audit.post_pump_persistence_conditionally_supported
    heldout = next(item for item in audit.probe_audits if item.held_out)
    assert audit.heldout_kernel_residual.standard_error == pytest.approx(
        heldout.normalized_post_kernel.standard_error
        + audit.fitted_training_kernel.standard_error
    )
    assert audit.heldout_separate_chain_response_conditionally_supported
    assert not audit.physical_material_phase_derived


def test_equal_raw_response_with_wrong_calibrated_gain_fails_heldout_prediction() -> None:
    wrong_heldout = _kernel_probe("C", 0.5, held_out=True, kernel=2.4)
    audit = _passing_common_kernel(
        probes=[_kernel_probe("A", 1.0), _kernel_probe("B", 2.0), wrong_heldout]
    )

    assert not audit.heldout_probe_matches_common_kernel
    assert not audit.heldout_separate_chain_response_conditionally_supported


def test_signed_calibrated_gain_predicts_opposite_raw_response() -> None:
    audit = _passing_common_kernel(
        probes=[
            _kernel_probe("A", 1.0),
            _kernel_probe("B", -2.0),
            _kernel_probe("C", 0.5, held_out=True),
        ]
    )

    raw_means = [item.raw_post_effect.mean_effect for item in audit.probe_audits]
    assert raw_means == pytest.approx([1.2, -2.4, 0.6])
    assert audit.heldout_separate_chain_response_conditionally_supported


def test_common_kernel_requires_three_probes_and_one_heldout() -> None:
    with pytest.raises(ValueError, match="at least three"):
        _passing_common_kernel(probes=[_kernel_probe("A", 1.0), _kernel_probe("B", 2.0)])
    with pytest.raises(ValueError, match="exactly one"):
        _passing_common_kernel(
            probes=[_kernel_probe("A", 1.0), _kernel_probe("B", 2.0), _kernel_probe("C", 0.5)]
        )
    with pytest.raises(ValueError, match="minimum_common_kernel must be positive"):
        _passing_common_kernel(minimum_common_kernel=0.0)
    with pytest.raises(ValueError, match="minimum_pump_off_dwell_s must be positive"):
        _passing_common_kernel(minimum_pump_off_dwell_s=0.0)
    with pytest.raises(ValueError, match="minimum_unexplained_kernel_margin must be positive"):
        _passing_common_kernel(minimum_unexplained_kernel_margin=0.0)
    with pytest.raises(ValueError, match="must not exceed half"):
        _passing_common_kernel(kernel_factorization_equivalence_bound=1_000.0)
    with pytest.raises(ValueError, match="must not exceed half"):
        _passing_common_kernel(pre_response_equivalence_bound=1_000.0)
    with pytest.raises(ValueError, match="monitor_equivalence_bound"):
        _passing_common_kernel(monitor_equivalence_bound=1_000.0)
    with pytest.raises(ValueError, match="must be positive"):
        _passing_common_kernel(residual_drive_to_kernel_gain_upper_bound=0.0)

    anticorrelated = _passing_common_kernel(
        probes=[
            _kernel_probe("A", 1.0, kernel=1.19, jitter=JITTER),
            _kernel_probe(
                "B",
                2.0,
                kernel=1.21,
                jitter=tuple(-value for value in JITTER),
            ),
            _kernel_probe("C", 0.5, held_out=True, kernel=1.2),
        ],
        kernel_factorization_equivalence_bound=0.045,
    )
    assert not anticorrelated.training_probes_factorize_common_kernel
    assert not anticorrelated.heldout_separate_chain_response_conditionally_supported


def test_pre_response_monitors_and_apparatus_memory_are_vetoes() -> None:
    pre = _passing_common_kernel(
        probes=[
            _kernel_probe("A", 1.0, pre_effect=0.5),
            _kernel_probe("B", 2.0, pre_effect=0.5),
            _kernel_probe("C", 0.5, held_out=True, pre_effect=0.5),
        ]
    )
    assert not pre.all_pre_responses_equivalent_to_zero
    assert not pre.post_pump_persistence_conditionally_supported

    drive = _passing_common_kernel(
        residual_drive_monitor_post=(0.4,) * len(JITTER),
        residual_drive_to_kernel_gain_upper_bound=3.0,
    )
    assert not drive.residual_drive_equivalent_to_zero
    assert not drive.monitors_and_apparatus_memory_cannot_explain_kernel
    assert not drive.post_pump_persistence_conditionally_supported

    memory = _passing_common_kernel(apparatus_memory_kernel_upper_bound=1.1)
    assert not memory.monitors_and_apparatus_memory_cannot_explain_kernel
    assert not memory.post_pump_persistence_conditionally_supported

    amplified_nuisance = _passing_common_kernel(
        nuisance_monitor_post=(0.05,) * len(JITTER),
        nuisance_to_kernel_gain_upper_bound=100.0,
    )
    assert amplified_nuisance.nuisance_monitor_equivalent_to_zero
    assert not amplified_nuisance.monitors_and_apparatus_memory_cannot_explain_kernel
    assert not amplified_nuisance.post_pump_persistence_conditionally_supported


def test_metadata_and_time_ordering_cannot_be_silently_skipped() -> None:
    unblinded = _passing_common_kernel(blind_analysis_declared=False)
    assert unblinded.post_pump_persistence_conditionally_supported
    assert not unblinded.heldout_separate_chain_response_conditionally_supported

    posthoc_calibration = _passing_common_kernel(calibration_fixed_before_pump=False)
    assert not posthoc_calibration.post_pump_persistence_conditionally_supported
    assert not posthoc_calibration.heldout_separate_chain_response_conditionally_supported

    badly_timed = _passing_common_kernel(post_readout_start_time_s=9.0)
    assert not badly_timed.time_ordering_valid
    assert not badly_timed.post_pump_persistence_conditionally_supported


def test_fixed_signed_energy_ledger_closes_with_informative_covariance() -> None:
    audit = _passing_energy()

    assert audit.mean_balance_residual_j == pytest.approx(0.0)
    assert audit.covariance_symmetric_positive_semidefinite
    assert audit.balance_statistically_consistent_with_zero
    assert audit.balance_residual_small
    assert audit.uncertainty_nonvacuous
    assert audit.energy_ledger_closed_conditionally
    assert not audit.microscopic_energy_transfer_mechanism_derived


def test_energy_misbalance_giant_uncertainty_and_coupled_endpoint_fail() -> None:
    missing_output = _passing_energy(recovered_work_j=_repeated(0.0))
    assert not missing_output.balance_residual_small
    assert not missing_output.energy_ledger_closed_conditionally

    giant_error = _passing_energy(energy_covariance_j2=_diagonal_covariance(1.0))
    assert giant_error.balance_statistically_consistent_with_zero
    assert not giant_error.uncertainty_nonvacuous
    assert not giant_error.energy_ledger_closed_conditionally

    giant_absolute_tolerance = _passing_energy(absolute_closure_tolerance_j=100.0)
    assert not giant_absolute_tolerance.uncertainty_nonvacuous
    assert not giant_absolute_tolerance.energy_ledger_closed_conditionally

    absolute_tolerance_cannot_override_relative_residual = _passing_energy(
        recovered_work_j=_repeated(0.7),
        absolute_closure_tolerance_j=0.4,
    )
    assert absolute_tolerance_cannot_override_relative_residual.uncertainty_nonvacuous
    assert not absolute_tolerance_cannot_override_relative_residual.balance_residual_small
    assert not absolute_tolerance_cannot_override_relative_residual.energy_ledger_closed_conditionally

    coupled = _passing_energy(pump_and_controller_decoupled_at_endpoint_declared=False)
    assert not coupled.energy_ledger_closed_conditionally

    with pytest.raises(ValueError, match="maximum_relative_balance_residual"):
        _passing_energy(maximum_relative_balance_residual=1.0)
    with pytest.raises(ValueError, match="maximum_relative_uncertainty"):
        _passing_energy(maximum_relative_uncertainty=100.0)


def test_energy_covariance_and_inputs_are_strictly_validated() -> None:
    bad_shape = ((0.0,) * 9,) * 9
    with pytest.raises(ValueError, match="shape"):
        _passing_energy(energy_covariance_j2=bad_shape)

    negative_variance = list(list(row) for row in _diagonal_covariance(1.0e-4))
    negative_variance[0][0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _passing_energy(energy_covariance_j2=negative_variance)

    tiny_negative_variance = [list(row) for row in _diagonal_covariance(0.0)]
    tiny_negative_variance[0][0] = -1.0e-13
    with pytest.raises(ValueError, match="positive semidefinite"):
        _passing_energy(energy_covariance_j2=tiny_negative_variance)

    dynamic_range_negative = [list(row) for row in _diagonal_covariance(0.0)]
    dynamic_range_negative[0][0] = -1.0e5
    dynamic_range_negative[1][1] = 1.0e20
    with pytest.raises(ValueError, match="positive semidefinite"):
        _passing_energy(energy_covariance_j2=dynamic_range_negative)

    with pytest.raises(ValueError, match="must be a bool"):
        _passing_energy(pump_and_controller_decoupled_at_endpoint_declared=1)


def test_negative_energy_channel_does_not_pass_the_fixed_ledger() -> None:
    audit = _passing_energy(recovered_work_j=_repeated(-1.0))
    assert not audit.all_channels_nonnegative
    assert not audit.energy_ledger_closed_conditionally


def test_report_keeps_private_and_public_branches_independent_and_claim_locked() -> None:
    report = probe_scaffold_pilot_report(
        phase_noise_sweep=_passing_sweep(),
        post_pump_common_kernel=_passing_common_kernel(),
        energy_ledger=_passing_energy(),
    )

    assert report.maximum_private_branch_stage == "CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING"
    assert report.maximum_public_branch_stage == "CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE"
    assert report.conditional_public_response_kernel_candidate
    assert not report.conditional_public_scaffold_candidate
    assert not report.physical_public_scaffold_derived
    assert not any(vars(report.claim_locks).values())


def test_public_branch_does_not_require_a_private_phase_lock_story() -> None:
    null_points = [
        _phase_point(0.0, 0.1, response_override=0.0),
        _phase_point(1.0, 0.5, response_override=0.0),
        _phase_point(1.5, 0.7, response_override=0.0, held_out=True),
        _phase_point(2.0, 0.9, response_override=0.0),
        _phase_point(3.0, 1.25, response_override=0.0),
    ]
    report = probe_scaffold_pilot_report(
        phase_noise_sweep=phase_noise_sweep_audit(null_points),
        post_pump_common_kernel=_passing_common_kernel(),
        energy_ledger=_passing_energy(),
    )

    assert report.maximum_private_branch_stage == "PRIVATE_NULL_CONTROL_ONLY"
    assert report.maximum_public_branch_stage == "CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE"


def test_report_validator_rejects_physical_claim_or_stage_tampering() -> None:
    report = probe_scaffold_pilot_report(
        phase_noise_sweep=_passing_sweep(),
        post_pump_common_kernel=_passing_common_kernel(),
        energy_ledger=_passing_energy(),
    )
    with pytest.raises(ValueError, match="claim locks"):
        validate_probe_scaffold_pilot_report(
            replace(report, physical_public_scaffold_derived=True)
        )
    with pytest.raises(ValueError, match="claim locks"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                claim_locks=replace(ProbeScaffoldClaimLocks(), new_material_derived=True),
            )
        )
    with pytest.raises(ValueError, match="private branch stage"):
        validate_probe_scaffold_pilot_report(
            replace(report, maximum_private_branch_stage="PHYSICAL_MATTER")
        )
    with pytest.raises(ValueError, match="phase causation"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                phase_noise_sweep=replace(
                    report.phase_noise_sweep,
                    causation_by_phase_lock_derived=True,
                ),
            )
        )
    first_point = report.phase_noise_sweep.points[0]
    tampered_point = replace(
        first_point,
        selectivity=replace(
            first_point.selectivity,
            public_environment_change_implied=True,
        ),
    )
    with pytest.raises(ValueError, match="public environment"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                phase_noise_sweep=replace(
                    report.phase_noise_sweep,
                    points=(tampered_point, *report.phase_noise_sweep.points[1:]),
                ),
            )
        )
    raw_phase_summary_tamper = replace(
        first_point,
        phase_lock=phase_lock_order_parameter([0.0] * len(JITTER)),
    )
    with pytest.raises(ValueError, match="raw phases"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                phase_noise_sweep=replace(
                    report.phase_noise_sweep,
                    points=(
                        raw_phase_summary_tamper,
                        *report.phase_noise_sweep.points[1:],
                    ),
                ),
            )
        )
    config_tamper = replace(
        first_point,
        selectivity=replace(
            first_point.selectivity,
            minimum_probe_a_effect=0.6,
        ),
    )
    with pytest.raises(ValueError, match="point configuration"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                phase_noise_sweep=replace(
                    report.phase_noise_sweep,
                    points=(config_tamper, *report.phase_noise_sweep.points[1:]),
                ),
            )
        )

    first_probe = report.post_pump_common_kernel.probe_audits[0]
    shifted_raw = replace(
        first_probe.raw_post_effect,
        mean_effect=first_probe.raw_post_effect.mean_effect + 0.1,
        lower_bound=first_probe.raw_post_effect.lower_bound + 0.1,
        upper_bound=first_probe.raw_post_effect.upper_bound + 0.1,
    )
    shifted_normalized = replace(
        first_probe.normalized_post_kernel,
        mean_effect=first_probe.normalized_post_kernel.mean_effect + 0.1,
        lower_bound=first_probe.normalized_post_kernel.lower_bound + 0.1,
        upper_bound=first_probe.normalized_post_kernel.upper_bound + 0.1,
    )
    raw_probe_summary_tamper = replace(
        first_probe,
        raw_post_effect=shifted_raw,
        normalized_post_kernel=shifted_normalized,
    )
    with pytest.raises(ValueError, match="raw readouts"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                post_pump_common_kernel=replace(
                    report.post_pump_common_kernel,
                    probe_audits=(
                        raw_probe_summary_tamper,
                        *report.post_pump_common_kernel.probe_audits[1:],
                    ),
                ),
            )
        )

    amplified_nuisance = _passing_common_kernel(
        nuisance_monitor_post=(0.05,) * len(JITTER),
        nuisance_to_kernel_gain_upper_bound=100.0,
    )
    monitor_summary_tamper = replace(
        amplified_nuisance,
        nuisance_monitor=report.post_pump_common_kernel.nuisance_monitor,
        nuisance_kernel_explanation_upper_bound=0.0,
        monitors_and_apparatus_memory_cannot_explain_kernel=True,
        post_pump_persistence_conditionally_supported=True,
        heldout_separate_chain_response_conditionally_supported=True,
    )
    with pytest.raises(ValueError, match="raw monitor data"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=monitor_summary_tamper,
            energy_ledger=_passing_energy(),
        )
    with pytest.raises(ValueError, match="physical material phase"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                post_pump_common_kernel=replace(
                    report.post_pump_common_kernel,
                    physical_material_phase_derived=True,
                ),
            )
        )
    with pytest.raises(ValueError, match="must be a bool"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                post_pump_common_kernel=replace(
                    report.post_pump_common_kernel,
                    heldout_probe_designation_declared="false",
                ),
            )
        )
    with pytest.raises(ValueError, match="must be a bool"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                energy_ledger=replace(
                    report.energy_ledger,
                    pump_and_controller_decoupled_at_endpoint_declared="false",
                ),
            )
        )
    with pytest.raises(ValueError, match="microscopic transfer mechanism"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                energy_ledger=replace(
                    report.energy_ledger,
                    microscopic_energy_transfer_mechanism_derived=True,
                ),
            )
        )
    with pytest.raises(ValueError, match="maximum_relative_uncertainty"):
        validate_probe_scaffold_pilot_report(
            replace(
                report,
                energy_ledger=replace(
                    report.energy_ledger,
                    maximum_relative_uncertainty=100.0,
                ),
            )
        )

    failed_kernel = _passing_common_kernel(
        probes=[
            _kernel_probe("A", 1.0),
            _kernel_probe("B", 2.0),
            _kernel_probe("C", 0.5, held_out=True, kernel=2.4),
        ]
    )
    tampered_kernel = replace(
        failed_kernel,
        heldout_probe_matches_common_kernel=True,
        heldout_separate_chain_response_conditionally_supported=True,
    )
    with pytest.raises(ValueError, match="common-kernel pass flags"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=tampered_kernel,
            energy_ledger=_passing_energy(),
        )


def test_report_builder_recomputes_energy_pass_instead_of_trusting_boolean() -> None:
    failed_energy = _passing_energy(recovered_work_j=_repeated(0.0))
    tampered_energy = replace(failed_energy, energy_ledger_closed_conditionally=True)

    with pytest.raises(ValueError, match="energy ledger pass flags"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=_passing_common_kernel(),
            energy_ledger=tampered_energy,
        )

    giant_covariance = _passing_energy(
        energy_covariance_j2=_diagonal_covariance(1.0)
    )
    zero_error_interval = replace(
        giant_covariance.balance_residual_interval,
        standard_error=0.0,
        lower_bound=giant_covariance.mean_balance_residual_j,
        upper_bound=giant_covariance.mean_balance_residual_j,
    )
    covariance_summary_tamper = replace(
        giant_covariance,
        balance_residual_interval=zero_error_interval,
        total_balance_sigma_j=0.0,
        relative_balance_uncertainty=0.0,
        uncertainty_nonvacuous=True,
        energy_ledger_closed_conditionally=True,
    )
    with pytest.raises(ValueError, match="below covariance sigma"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=_passing_common_kernel(),
            energy_ledger=covariance_summary_tamper,
        )

    scattered_energy = _passing_energy(
        pump_work_j=(20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0)
    )
    covariance_floor = scattered_energy.declared_covariance_balance_sigma_j
    scatter_tampered_interval = replace(
        scattered_energy.balance_residual_interval,
        standard_error=covariance_floor,
        lower_bound=(
            scattered_energy.mean_balance_residual_j
            - covariance_floor
            * scattered_energy.balance_residual_interval.confidence_multiplier
        ),
        upper_bound=(
            scattered_energy.mean_balance_residual_j
            + covariance_floor
            * scattered_energy.balance_residual_interval.confidence_multiplier
        ),
    )
    scatter_summary_tamper = replace(
        scattered_energy,
        balance_residual_interval=scatter_tampered_interval,
        total_balance_sigma_j=covariance_floor,
        relative_balance_uncertainty=(
            covariance_floor / scattered_energy.mean_candidate_decoupled_energy_j
        ),
        uncertainty_nonvacuous=True,
        energy_ledger_closed_conditionally=True,
    )
    with pytest.raises(ValueError, match="raw trials"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=_passing_common_kernel(),
            energy_ledger=scatter_summary_tamper,
        )

    negative_trial = _passing_energy(
        pump_work_j=_repeated(11.0),
        candidate_decoupled_energy_j=_repeated(5.0),
        recovered_work_j=(-0.01, 2.01, -0.01, 2.01, -0.01, 2.01, -0.01, 2.01)
    )
    assert negative_trial.uncertainty_nonvacuous
    tampered_minima = replace(
        negative_trial,
        minimum_channel_values_j=(*negative_trial.minimum_channel_values_j[:-1], 0.0),
        all_channels_nonnegative=True,
        energy_ledger_closed_conditionally=True,
    )
    with pytest.raises(ValueError, match="minima"):
        probe_scaffold_pilot_report(
            phase_noise_sweep=_passing_sweep(),
            post_pump_common_kernel=_passing_common_kernel(),
            energy_ledger=tampered_minima,
        )
