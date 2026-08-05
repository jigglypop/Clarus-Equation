from __future__ import annotations

from reality_stone.clarus.fusion_spin_operator_loop import (
    current_fusion_spin_operator_report,
)


def main() -> None:
    report = current_fusion_spin_operator_report()
    spin = report.spin_average
    pseudoscalar = report.pseudoscalar
    axial = report.axial_vector
    vector = report.vector
    spin_two = report.spin_two
    node = report.derivative_node

    print("CE FUSION SPIN / OPERATOR LOOP")
    print(f" scalar required D-T product       {report.required_dt_charge_product:.9e}")
    print(" spin average")
    print(f"  raw unpolarized Tr(O)/6          {spin.raw_unpolarized_operator_trace:.9e}")
    print(f"  quartet-projected Tr(P4 O)/6    {spin.quartet_projected_unpolarized_trace:.9e}")
    print(" pseudoscalar")
    print(
        "  required |g_PD g_PT|            "
        f"{pseudoscalar.required_abs_effective_nuclear_coupling_product:.9e}"
    )
    print(
        "  equal |g_P| / alpha_P           "
        f"{pseudoscalar.equal_abs_effective_nuclear_coupling:.9e} / "
        f"{pseudoscalar.equal_coupling_fine_structure:.9e}"
    )
    print(" axial vector")
    print(
        "  required g_AD g_AT / equal g_A  "
        f"{axial.required_effective_nuclear_coupling_product:.9e} / "
        f"{axial.equal_effective_nuclear_coupling:.9e}"
    )
    print(
        f"  naive nuclear / universal-q K   {axial.naive_nuclear_coupling_to_quark_bound_ratio:.9e}"
    )
    print(" vector")
    print(
        "  minimax gp / gn                 "
        f"{vector.minimax_proton_coupling:.9e} / "
        f"{vector.minimax_neutron_coupling:.9e}"
    )
    print(
        "  Pb-blind gp / gn                "
        f"{vector.lead_blind_proton_coupling:.9e} / "
        f"{vector.lead_blind_neutron_coupling:.9e}"
    )
    print(f"  Pb-blind D-T product             {vector.lead_blind_dt_charge_product:.9e}")
    print(f"  Pb-blind u-d                     {vector.lead_blind_isovector_quark_coupling:.9e}")
    print(" spin 2")
    print(f"  required c/Lambda GeV^-1        {spin_two.required_equal_c_over_lambda_per_gev:.9e}")
    print(
        "  required / visible, invisible   "
        f"{spin_two.required_to_visible_bound_ratio:.9e} / "
        f"{spin_two.required_to_invisible_bound_ratio:.9e}"
    )
    print(" derivative node")
    print(f"  Yukawa range fm                  {node.yukawa_range_fm:.9e}")
    print(f"  Gamow incoming momentum MeV      {node.incoming_gamow_momentum_mev:.9e}")
    print(f"  node cancels Yukawa residue      {node.on_shell_node_cancels_yukawa_pole_residue}")
    print(
        f" exact NCSMC/R-matrix supplied     {report.exact_ncsmc_or_rmatrix_calculation_supplied}"
    )
    print(
        " mass-specific pi/K/BaBar supplied "
        f"{report.mass_specific_pion_kaon_babar_likelihoods_supplied}"
    )
    print(
        f" physical one-percent branch       {report.physical_one_percent_fusion_branch_accepted}"
    )
    print(f" maximum supported stage           {report.maximum_supported_stage}")

    if report.any_physical_operator_gate_pass:
        raise RuntimeError("fail-closed invariant violated: an operator gate opened")
    if report.physical_one_percent_fusion_branch_accepted:
        raise RuntimeError("fail-closed invariant violated: physical branch accepted")


if __name__ == "__main__":
    main()
