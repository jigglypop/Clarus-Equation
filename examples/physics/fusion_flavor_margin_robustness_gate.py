from __future__ import annotations

from reality_stone.clarus.fusion_flavor_margin_robustness_loop import (
    current_fusion_flavor_margin_robustness_report,
)


def main() -> None:
    report = current_fusion_flavor_margin_robustness_report()
    lead = report.lead_shape_proxy
    latest = report.latest_kaon_data
    print("CE FUSION FLAVOR-ALIGNED MARGIN ROBUSTNESS")
    print(" D/T one-body folding")
    for audit in report.folding_scenarios:
        print(
            f"  {audit.scenario:29s} "
            f"P/P0={audit.required_product_to_point_ratio:.9f} "
            f"g/g0={audit.required_coupling_to_point_ratio:.9f}"
        )
    print(" D/T one-body morphology linear-response envelope")
    for audit in report.morphology_scenarios:
        print(
            f"  {audit.radius_scenario:29s} {audit.density_morphology:14s} "
            f"P/P0={audit.linearized_required_product_to_point_ratio:.9f} "
            f"g/g0={audit.linearized_required_coupling_to_point_ratio:.9f}"
        )
    print(" neutron--Pb finite-shape proxy")
    print(
        "  q range MeV                    "
        f"{lead.minimum_momentum_transfer_mev:.6f} .. "
        f"{lead.maximum_momentum_transfer_mev:.6f}"
    )
    print(
        "  local response envelope        "
        f"{lead.combined_shape_response_minimum:.9f} .. "
        f"{lead.combined_shape_response_maximum:.9f}"
    )
    print(
        "  q4-weighted response proxy     "
        f"{lead.q4_weighted_shape_response_minimum:.9f} .. "
        f"{lead.q4_weighted_shape_response_maximum:.9f}"
    )
    print(
        "  angular p-wave projection      "
        f"{lead.angular_p_wave_projection_response_minimum:.9f} .. "
        f"{lead.angular_p_wave_projection_response_maximum:.9f}"
    )
    print(
        "  low-E sigma2 projection        "
        f"{lead.low_energy_sigma2_finite_window_response_minimum:.6f} .. "
        f"{lead.low_energy_sigma2_finite_window_response_maximum:.6f}"
    )
    print(
        "  sigma2 grid refinement shift   "
        f"{lead.low_energy_sigma2_grid_refinement_max_relative_shift:.3e} "
        f"pass={lead.low_energy_sigma2_numerical_convergence_pass}"
    )
    print(f"  point critical response        {report.point_pb_response_critical:.9f}")
    print(
        f"  favorable-proxy critical       {report.most_favorable_proxy_pb_response_critical:.9f}"
    )
    print(" rare-kaon tightening axis")
    print(f"  point critical NLO factor      {report.point_kaon_nlo_tightening_critical:.9f}")
    print(
        "  favorable-proxy critical       "
        f"{report.most_favorable_proxy_kaon_nlo_tightening_critical:.9f}"
    )
    print(
        "  lower digitized line critical  "
        f"{report.robust_lower_line_kaon_nlo_tightening_critical:.9f}"
    )
    print(
        "  acknowledged factor passes     "
        f"{report.acknowledged_nlo_factor_passes_any_proxy_scenario}"
    )
    print(" latest NA62 invisible data")
    print(
        "  BR improvement range           "
        f"{latest.branching_ratio_improvement_factor_minimum:.3f} .. "
        f"{latest.branching_ratio_improvement_factor_maximum:.3f}"
    )
    print(
        "  coupling-bound multiplier      "
        f"{latest.coupling_bound_multiplier_minimum:.9f} .. "
        f"{latest.coupling_bound_multiplier_maximum:.9f}"
    )
    print(
        "  point critical NLO range       "
        f"{latest.point_nlo_tightening_critical_minimum:.9f} .. "
        f"{latest.point_nlo_tightening_critical_maximum:.9f}"
    )
    print(
        "  Figure 2 BR at 29.65 MeV       "
        f"new={latest.figure2_interpolated_2016_2022_observed_br_limit:.4e} "
        f"old={latest.figure2_interpolated_2016_2018_observed_br_limit:.4e}"
    )
    print(
        "  Figure 2 central I / sqrt      "
        f"{latest.figure2_interpolated_br_improvement_factor:.6f} / "
        f"{latest.figure2_interpolated_coupling_bound_multiplier:.6f}"
    )
    print(
        "  Figure 2 I readout envelope    "
        f"{latest.figure2_br_improvement_factor_minimum:.6f} .. "
        f"{latest.figure2_br_improvement_factor_maximum:.6f}"
    )
    print(
        "  Figure 2 point NLO envelope    "
        f"{latest.figure2_point_nlo_tightening_critical_minimum:.6f} .. "
        f"{latest.figure2_point_nlo_tightening_critical_maximum:.6f}"
    )
    print(
        "  Figure 2 curve entered         "
        f"{latest.figure2_candidate_mass_curve_interpolation_entered}"
    )
    print(f"  exact 29.65 MeV limit entered  {latest.exact_candidate_mass_observed_limit_entered}")
    print(f" margin gate pass                {report.margin_robustness_gate_pass}")
    print(f" physical branch accepted        {report.physical_ce_fusion_branch_accepted}")
    print(f" conclusion                      {report.conclusion}")


if __name__ == "__main__":
    main()
