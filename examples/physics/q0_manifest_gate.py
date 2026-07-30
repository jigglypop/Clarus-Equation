"""Run the fixed minimal Abelian-Higgs Q0 structural control benchmark."""

from __future__ import annotations

from pathlib import Path

from reality_stone.clarus.q0_manifest_gate import (
    load_q0_control_benchmark,
    q0_manifest_gate_report,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    benchmark_path = (
        repository_root / "benchmarks" / "q0_minimal_abelian_higgs_v1.json"
    )
    benchmark = load_q0_control_benchmark(benchmark_path)
    report = q0_manifest_gate_report(
        benchmark.manifest,
        benchmark.control_inputs,
    )
    field_space = report.field_space_audit
    background = report.background_audit
    gauge = report.gauge_audit

    print("Q0 MINIMAL ABELIAN-HIGGS + Z2 SINGLET R_XI STRUCTURAL GATE")
    print(f"  benchmark                   {benchmark_path}")
    print(f"  scope                       {report.scope}")
    print(f"  status                      {report.structural_status}")
    print(
        "  control Q0.0 manifest       "
        f"{report.control_q0_0_pass}"
    )
    print(
        "  control Q0.1 field space    "
        f"{report.control_q0_1_pass}"
    )
    print(
        "  control Q0.2 background     "
        f"{report.control_q0_2_pass}"
    )
    print(
        "  control Q0.3 gauge/ghost    "
        f"{report.control_q0_3_pass}"
    )
    print(
        "  control through Q0.3        "
        f"{report.control_through_q0_3_pass}"
    )
    print(
        "  ordinary Hessian extra      "
        f"{field_space.non_tensor_extra_term:.12g}"
    )
    print(
        "  covariant Hessian residual  "
        f"{field_space.covariance_residual:.12g}"
    )
    print(
        "  Higgs tadpole               "
        f"{background.higgs_tadpole:.12g}"
    )
    print(
        "  Z2 singlet tadpole          "
        f"{background.singlet_tadpole:.12g}"
    )
    print(
        "  singlet effective mass^2    "
        f"{background.singlet_effective_mass_squared:.12g}"
    )
    print(
        "  independent lambda_HP       "
        f"{background.lambda_hp:.12g}"
    )
    print(
        "  A.dchi mixing residual      "
        f"{gauge.mixing_cancellation_residual:.12g}"
    )
    print(
        "  FP ghost mass residual      "
        f"{gauge.fp_operator_residual:.12g}"
    )
    print(f"  Abelian control slice pass  {report.abelian_control_slice_pass}")
    print(f"  full Q0.0 complete          {report.full_q0_0_complete}")
    print(f"  full Q0.1 complete          {report.full_q0_1_complete}")
    print(f"  full Q0.2 complete          {report.full_q0_2_complete}")
    print(f"  full Q0.3 complete          {report.full_q0_3_complete}")
    print(f"  full Q0 pass                {report.full_q0_pass}")
    print(f"  full CE+SM complete         {report.full_ce_sm_complete}")
    print(f"  excluded sectors            {report.excluded_sectors}")
    print(f"  stress tensor derived       {report.stress_tensor_derived}")
    print(f"  spectral density derived    {report.spectral_density_derived}")
    print(f"  conclusion                  {report.conclusion}")


if __name__ == "__main__":
    main()
