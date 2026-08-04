"""Run the synthetic control for private dressing versus a public response kernel."""

from __future__ import annotations

import math

from reality_stone.clarus.probe_scaffold_pilot import (
    CommonKernelProbeReadout,
    PhaseNoiseSweepPoint,
    energy_ledger_audit,
    phase_noise_sweep_audit,
    post_pump_common_kernel_audit,
    probe_scaffold_pilot_report,
)


JITTER = (-0.02, 0.01, 0.0, 0.02, -0.01, 0.015, -0.015, 0.0)
ZEROS = (0.0,) * len(JITTER)
BASELINE = (10.0,) * len(JITTER)


def _repeated(value: float) -> tuple[float, ...]:
    return (value,) * len(JITTER)


def _bias_corrected_resultant(angle: float) -> float:
    raw = math.cos(angle)
    count = len(JITTER)
    return math.sqrt(max(0.0, (count * raw * raw - 1.0) / (count - 1.0)))


def _phase_point(noise: float, angle: float, *, held_out: bool = False) -> PhaseNoiseSweepPoint:
    phases = tuple(value for _ in range(len(JITTER) // 2) for value in (-angle, angle))
    response = 0.2 + 2.0 * _bias_corrected_resultant(angle)
    return PhaseNoiseSweepPoint(
        noise_strength=noise,
        phase_offsets_rad=phases,
        probe_a_pump_on_matched=tuple(
            base + response + jitter for base, jitter in zip(BASELINE, JITTER, strict=True)
        ),
        probe_a_pump_on_sham=BASELINE,
        probe_a_pump_off_matched=BASELINE,
        probe_a_pump_off_sham=BASELINE,
        reference_pump_on_matched=BASELINE,
        reference_pump_on_sham=BASELINE,
        reference_pump_off_matched=BASELINE,
        reference_pump_off_sham=BASELINE,
        held_out=held_out,
    )


def _kernel_probe(
    probe_id: str,
    gain: float,
    *,
    held_out: bool = False,
) -> CommonKernelProbeReadout:
    sham = _repeated(4.0)
    return CommonKernelProbeReadout(
        probe_id=probe_id,
        calibrated_response_gain=gain,
        post_pump_response=tuple(4.0 + gain * (1.2 + jitter) for jitter in JITTER),
        post_pump_sham=sham,
        pre_pump_response=sham,
        pre_pump_sham=sham,
        held_out=held_out,
    )


def _diagonal_covariance(variance: float) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(variance if row == column else 0.0 for column in range(10))
        for row in range(10)
    )


def main() -> None:
    sweep = phase_noise_sweep_audit(
        [
            _phase_point(0.0, 0.10),
            _phase_point(1.0, 0.50),
            _phase_point(1.5, 0.70, held_out=True),
            _phase_point(2.0, 0.90),
            _phase_point(3.0, 1.25),
        ],
        minimum_absolute_correlation=0.95,
        heldout_prediction_equivalence_bound=0.15,
    )
    kernel = post_pump_common_kernel_audit(
        [_kernel_probe("A", 1.0), _kernel_probe("B", 2.0), _kernel_probe("C", 0.5, held_out=True)],
        residual_drive_monitor_post=ZEROS,
        residual_drive_monitor_sham=ZEROS,
        nuisance_monitor_post=ZEROS,
        nuisance_monitor_sham=ZEROS,
        pump_start_time_s=0.0,
        pump_off_time_s=10.0,
        post_readout_start_time_s=12.0,
        post_readout_end_time_s=13.0,
        minimum_pump_off_dwell_s=1.0,
        kernel_factorization_equivalence_bound=0.15,
        apparatus_memory_kernel_upper_bound=0.05,
        minimum_unexplained_kernel_margin=0.5,
        heldout_probe_designation_declared=True,
        calibration_fixed_before_pump=True,
        blind_analysis_declared=True,
        separate_heldout_readout_chain_declared=True,
    )
    energy = energy_ledger_audit(
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
    report = probe_scaffold_pilot_report(
        phase_noise_sweep=sweep,
        post_pump_common_kernel=kernel,
        energy_ledger=energy,
    )

    print("CLARUS PROBE/SCAFFOLD SYNTHETIC CONTROL")
    print(" private branch")
    print(f"  noise-phase correlation      {sweep.noise_phase_correlation:+.9f}")
    print(f"  phase-response correlation   {sweep.phase_selectivity_correlation:+.9f}")
    print(f"  heldout residual             {sweep.heldout_prediction_residual.mean_effect:+.3e}")
    print(f"  stage                        {report.maximum_private_branch_stage}")
    print(" public branch")
    print(f"  fitted common kernel         {kernel.fitted_training_kernel.mean_effect:.9f}")
    print(f"  heldout kernel residual      {kernel.heldout_kernel_residual.mean_effect:+.3e}")
    print(
        "  residual-drive explanation  "
        f"{kernel.residual_drive_kernel_explanation_upper_bound:.3e}"
    )
    print(f"  energy balance residual J    {energy.mean_balance_residual_j:+.3e}")
    print(f"  energy balance sigma J       {energy.total_balance_sigma_j:.3e}")
    print(f"  stage                        {report.maximum_public_branch_stage}")
    print(" claim locks")
    print(f"  public response candidate    {report.conditional_public_response_kernel_candidate}")
    print(f"  public scaffold candidate    {report.conditional_public_scaffold_candidate}")
    print(f"  physical scaffold            {report.physical_public_scaffold_derived}")
    print(f"  new matter                   {report.claim_locks.new_material_derived}")


if __name__ == "__main__":
    main()
