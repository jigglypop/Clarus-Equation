from __future__ import annotations

from reality_stone.clarus.fusion_scalar_current_loop import (
    current_fusion_scalar_current_report,
)


def main() -> None:
    report = current_fusion_scalar_current_report()
    nucleon = report.nucleon_scalar_charge
    shape = report.one_body_nuclear_shape
    barrier = report.barrier_window
    radius = report.intrinsic_scalar_radius
    proxy = report.sigma_term_proxy
    two_body = report.two_body_scalar_current
    gate = report.certification

    print("CE FUSION SCALAR-CURRENT LOOP")
    print(" one-nucleon normalization")
    print(
        f"  candidate g_p / g_n             {nucleon.candidate_proton_coupling:.9e} / {nucleon.candidate_neutron_coupling:.9e}"
    )
    print(
        f"  modern uds / candidate ratio    {nucleon.modern_to_candidate_isoscalar_numerator_ratio:.9e}"
    )
    print(
        f"  fixed-scale D/T product ratio   {nucleon.fixed_scale_dt_product_ratio_diagnostic:.9e}"
    )
    print(
        "  modern p=n isoscalar proxy      "
        f"{nucleon.modern_proton_equals_neutron_isoscalar_proxy_assumed}"
    )
    print(" one-body shape")
    print(
        f"  sampled max Helm/Gauss residual {shape.maximum_sampled_spacelike_relative_residual:.9e}"
    )
    print(
        f"  q=i*m exterior diagnostic       {shape.imaginary_helm_to_gaussian_relative_residual:.9e}"
    )
    print(f"  central benchmark pass          {gate.helm_gaussian_central_benchmark_pass}")
    print(" barrier coverage")
    print(f"  mediator range fm               {barrier.mediator_compton_length_fm:.9e}")
    print(
        f"  qmax resolution fm              {barrier.smallest_spatial_scale_resolved_at_qmax_fm:.9e}"
    )
    print(f"  q needed at 3.24 fm MeV         {barrier.momentum_needed_for_inner_radius_mev:.9e}")
    print(f"  q grid resolves inner edge      {barrier.q_grid_resolves_inner_radius}")
    print(" scalar radius / sigma proxy")
    q40 = radius.spacelike_points[-1]
    print(
        f"  q40 amplitude range             {q40.correction_at_radius_min:.9e} / {q40.correction_at_radius_max:.9e}"
    )
    print(
        f"  q40 central radius-endpoint     {q40.exact_coupling_correction_at_radius_min:.9e} / {q40.exact_coupling_correction_at_radius_max:.9e}"
    )
    print(
        f"  sigma proxy coupling            {proxy.required_common_coupling_correction:.9e} +/- {proxy.required_common_coupling_correction_std:.9e}"
    )
    print(
        f"  He3 used as T proxy             {proxy.assumptions.helium3_used_as_triton_isospin_proxy}"
    )
    print(
        f"  actual T supplied               {proxy.assumptions.actual_triton_sigma_term_supplied}"
    )
    print(" two-body closure")
    print(
        f"  exact uds D amplitude range     {two_body.exact_modern_uds_deuteron_amplitude_correction_min:.9e} / {two_body.exact_modern_uds_deuteron_amplitude_correction_max:.9e}"
    )
    print(f"  fitted contact supplied         {gate.calibrated_two_body_contact_supplied}")
    print(f"  D/T covariance supplied         {gate.momentum_dependent_dt_covariance_supplied}")
    print(f"  real-space barrier supplied     {gate.full_real_space_barrier_response_supplied}")
    print(" direct leaf gates")
    print(f"  p/n covariance                  {nucleon.proton_neutron_sigma_covariance_supplied}")
    print(f"  modern sigma covariance         {nucleon.modern_sigma_term_covariance_supplied}")
    print(f"  normalization likelihood        {nucleon.normalization_likelihood_supplied}")
    print(f"  ab-initio density covariance    {shape.ab_initio_density_covariance_supplied}")
    print(f"  scalar-radius covariance        {radius.scalar_radius_covariance_supplied}")
    print(
        f"  full scalar form factor         {radius.low_q_expansion_promoted_to_full_form_factor}"
    )
    print(
        "  regulator-consistent current    "
        f"{two_body.regulator_consistent_current_and_potential_supplied}"
    )
    print(
        "  two-body D/T likelihood         "
        f"{two_body.momentum_dependent_dt_joint_likelihood_supplied}"
    )
    print(f"  two-body covariance             {two_body.two_body_covariance_supplied}")
    print(
        "  derived nuclear leaf gates      "
        f"{gate.nucleon_normalization_leaf_gate_pass}/"
        f"{gate.one_body_shape_leaf_gate_pass}/"
        f"{gate.scalar_radius_leaf_gate_pass}/"
        f"{gate.triton_sigma_response_leaf_gate_pass}/"
        f"{gate.two_body_leaf_gate_pass}"
    )
    print(" final")
    print(
        f"  comparison band                 +/- {gate.comparison_band_absolute_coupling_correction:.6f}"
    )
    print(f"  scalar-current certification    {gate.scalar_current_certification_pass}")
    print(f"  upstream UV/action gate         {gate.upstream_uv_action_gate_pass}")
    print(f"  upstream constraint gates       {gate.upstream_existing_constraints_gate_pass}")
    print(f"  physical branch accepted        {report.physical_ce_fusion_branch_accepted}")
    print(f"  status                          {gate.status}")


if __name__ == "__main__":
    main()
