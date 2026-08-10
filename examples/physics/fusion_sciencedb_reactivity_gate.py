from __future__ import annotations

from reality_stone.clarus.fusion_sciencedb_reactivity_loop import (
    current_sciencedb_dt_reactivity_audit,
)


def main() -> None:
    audit = current_sciencedb_dt_reactivity_audit()

    print("FUSION SCIENCEDB UNPOLARIZED REACTIVITY GATE")
    print(f" source dataset                  {audit.source_dataset_doi}")
    print(f" source version                  {audit.source_dataset_version}")
    print(f" D-T table SHA-256               {audit.dt_cross_section_runtime_sha256}")
    print(f" D-T table rows                  {audit.dt_table_row_count}")
    print(f" temperature                     {audit.temperature_kev:.6f} keV")
    print(f" deuteron lab -> cm factor       {audit.deuteron_lab_to_cm_energy_factor:.12f}")
    print(
        f" Bosch-Hale closed <sigma v>     {audit.bosch_hale_closed_reactivity_cm3_s:.12e} cm^3/s"
    )
    print(
        " Bosch-Hale same-kernel value    "
        f"{audit.bosch_hale_same_kernel_reactivity_cm3_s:.12e} cm^3/s"
    )
    for envelope in audit.interpolation_envelopes:
        print(f" {envelope.method}")
        print(f"   central                       {envelope.central_reactivity_cm3_s:.12e}")
        print(
            f"   all points -ERR              {envelope.all_points_minus_err_reactivity_cm3_s:.12e}"
        )
        print(
            f"   all points +ERR              {envelope.all_points_plus_err_reactivity_cm3_s:.12e}"
        )
        print(f"   central / Bosch-Hale closed  {envelope.central_to_bosch_hale_closed_ratio:.12f}")

    print(f" direct-sigma interpolation span {audit.sigma_interpolation_relative_spread:.9%}")
    print(f" S-factor interpolation span     {audit.s_factor_interpolation_relative_spread:.9%}")
    print(f" all-method central span          {audit.all_method_central_relative_spread:.9%}")
    print(
        " grid-refinement max residual    "
        f"{audit.grid_refinement_max_relative_residual:.9%}"
    )
    print(f" numeric covariance matrix       {audit.numeric_covariance_matrix_available}")
    print(f" initial-state spin operator      {audit.initial_state_spin_operator_available}")
    print(
        " unpolarized <1% certification   "
        f"{audit.unpolarized_sub_one_percent_certification_gate_pass}"
    )
    print(
        " physical state-resolved >=1%    "
        f"{audit.physical_state_resolved_one_percent_branch_gate_pass}"
    )
    print(f" maximum supported stage         {audit.maximum_supported_stage}")

    if not audit.payload_audit.payload_integrity_gate_pass:
        raise RuntimeError("ScienceDB payload integrity regressed")
    if not audit.dt_table_parsed_from_integrity_verified_raw_bytes:
        raise RuntimeError("D-T table was not parsed from verified bytes")
    if audit.interpolation_spread_below_one_percent:
        raise RuntimeError("pinned interpolation-spread control changed; review required")
    if not audit.grid_refinement_gate_pass:
        raise RuntimeError("Maxwellian integration grid refinement regressed")
    if audit.numeric_covariance_matrix_available:
        raise RuntimeError("unexpected covariance promotion requires independent review")
    if audit.initial_state_spin_operator_available:
        raise RuntimeError("unexpected spin-operator promotion requires independent review")
    if audit.physical_state_resolved_one_percent_branch_gate_pass:
        raise RuntimeError("unexpected physical branch promotion requires independent review")


if __name__ == "__main__":
    main()
