from __future__ import annotations

from reality_stone.clarus.fusion_polarized_evidence_loop import (
    current_fusion_polarized_evidence_report,
)


def main() -> None:
    report = current_fusion_polarized_evidence_report()
    reaction = report.reaction
    source = report.source

    print("FUSION POLARIZED D-T PRIMARY-EVIDENCE LOOP")
    print(" published energy-dependent control")
    print(f"  source                          {reaction.source_doi}")
    print(f"  figure image sha256             {reaction.expected_source_image_sha256}")
    print(f"  runtime image hash verified     {reaction.source_image_sha256_verified}")
    print(f"  runtime image size verified     {reaction.source_image_dimensions_verified}")
    print(
        "  full-alignment Maxwellian ratio "
        f"{reaction.full_alignment_maxwellian_reactivity_ratio:.12f}"
    )
    print(
        "  digitization-lower ratio        "
        f"{reaction.digitization_lower_maxwellian_reactivity_ratio:.12f}"
    )
    print(
        "  10 keV kernel central 90% keV   "
        f"{reaction.maxwellian_kernel_central_90_low_energy_kev:.6f} .. "
        f"{reaction.maxwellian_kernel_central_90_high_energy_kev:.6f}"
    )
    print(
        "  author numeric grid             "
        f"{reaction.author_machine_readable_energy_grid_available}"
    )
    print(
        "  model covariance                "
        f"{reaction.nuclear_model_systematic_covariance_available}"
    )
    print(
        "  EXFOR single-angle Azz table    "
        f"{reaction.dries_exfor_machine_readable_single_angle_azz_available}"
    )
    print(
        "  Han covariance metadata         "
        f"{reaction.han_sciencedb_metadata_declares_unpolarized_rmatrix_covariance}"
    )
    print(
        f"  Han numeric files verified      {reaction.han_sciencedb_numeric_files_locally_verified}"
    )
    print(
        "  Han sigma interpolation spread  "
        f"{reaction.han_sciencedb_unpolarized_reactivity_audit.sigma_interpolation_relative_spread:.6%}"
    )
    print(
        "  Han numeric covariance matrix   "
        f"{reaction.han_sciencedb_numeric_covariance_matrix_available}"
    )
    print(
        "  Han double-polarized operator   "
        f"{reaction.han_initial_double_polarized_state_operator_available}"
    )
    print(f"  direct Czz measurement          {reaction.spin_correlation_czz_directly_measured}")
    print(f"  target-state operator           {reaction.target_state_resolved_operator_available}")
    print(
        "  operator bytes verified         "
        f"{reaction.target_state_operator_artifact.runtime_artifact_gate_pass}"
    )
    print(" source, retention, and pump")
    print(f"  required D or T rate /s         {source.required_per_species_fuel_rate_s:.9e}")
    print(f"  reference plant scale invariant {source.reference_plant_scale_invariant_pass}")
    print(
        "  Coulter polarized-D rate /s     "
        f"{source.coulter_measured_continuous_deuterium_rate_s:.9e}"
    )
    print(f"  required / measured-D gap       {source.required_to_coulter_rate_ratio:.9e}")
    print(
        "  ANKE D rate / pz / pzz          "
        f"{source.anke_measured_deuterium_rate_s:.3e} / "
        f"{source.anke_measured_deuteron_vector_polarization_abs:.2f} / "
        f"{source.anke_measured_deuteron_tensor_polarization:.2f}"
    )
    print(
        "  ANKE 300 W is complete ledger   "
        f"{not source.anke_rf_power_is_partial_component_not_complete_wall_plug}"
    )
    print(
        "  reactor-rate polarized T        "
        f"{source.reactor_rate_polarized_tritium_source_demonstrated}"
    )
    print(
        "  D/T source artifacts verified   "
        f"{source.validated_reactor_rate_deuterium_source_artifact.runtime_artifact_gate_pass}/"
        f"{source.validated_reactor_rate_tritium_source_artifact.runtime_artifact_gate_pass}"
    )
    print(
        "  Utsuro HD proxy / actual T      "
        f"{source.utsuro_hd_proxy_proof_of_concept_performed}/"
        f"{source.utsuro_actual_polarized_tritium_source_demonstrated}"
    )
    print(
        f"  burn-weighted D-T product       {source.burn_weighted_dt_polarization_product_measured}"
    )
    print(
        "  actual solid-DT preburn NMR     "
        f"{source.souers_actual_solid_dt_triton_relaxation_measured and source.collins_actual_solid_dt_deuteron_nmr_relaxation_measured}"
    )
    print("  actual D-T burn retention       False")
    print(
        "  complete wall-plug eV/pair      "
        f"{source.complete_wall_plug_energy_per_injected_pair_measured}"
    )
    print(
        "  uncertainty/flow/provenance     "
        f"{source.complete_wall_plug_uncertainty_flow_and_provenance_available}"
    )
    print(" final")
    print(f"  figure control                  {report.energy_dependent_figure_control_reproduced}")
    print(f"  physical reaction evidence      {report.physical_reaction_evidence_gate_pass}")
    print(f"  physical source evidence        {report.physical_source_evidence_gate_pass}")
    print(f"  physical spin-fusion evidence   {report.physical_spin_fusion_evidence_gate_pass}")
    print(f"  maximum supported stage         {report.maximum_supported_stage}")

    if not report.energy_dependent_figure_control_reproduced:
        raise RuntimeError("published figure reproduction control regressed")
    if report.physical_reaction_evidence_gate_pass:
        raise RuntimeError("unexpected physical reaction-evidence promotion requires review")
    if report.physical_source_evidence_gate_pass:
        raise RuntimeError("unexpected physical source-evidence promotion requires review")
    if report.physical_spin_fusion_evidence_gate_pass:
        raise RuntimeError("unexpected physical evidence promotion requires review")


if __name__ == "__main__":
    main()
