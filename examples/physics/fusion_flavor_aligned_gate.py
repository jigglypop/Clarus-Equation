from __future__ import annotations

from reality_stone.clarus.fusion_flavor_aligned_loop import (
    current_fusion_flavor_aligned_report,
)


def main() -> None:
    report = current_fusion_flavor_aligned_report()
    operator = report.operator
    neutron = report.neutron_constraint
    rare = report.rare_decay_constraint
    invisible = report.invisible_completion
    print("CE FUSION FLAVOR-ALIGNED DIRECT CANDIDATE")
    print(" operator / UV")
    print(f"  required universal g_N          {operator.universal_required_nucleon_coupling:.9e}")
    print(f"  aligned scale GeV               {operator.aligned_scale_gev:.9e}")
    print(f"  proton / neutron coupling       {operator.proton_coupling:.9e} / {operator.neutron_coupling:.9e}")
    print(f"  kappa v / M                     {operator.required_plot_coordinate_kappa_v_over_m:.9e}")
    print(f"  largest VLQ Yukawa              {operator.strange_vlq_yukawa:.9e}")
    print(" neutron constraint")
    print(f"  extrapolated equal bound        {neutron.extrapolated_equal_coupling_bound:.9e}")
    print(f"  flavor-matched Pb coupling      {neutron.flavor_matched_lead_effective_coupling:.9e}")
    print(f"  central fractional margin       {neutron.central_fractional_margin:.9e}")
    print(f"  q2 / m2 diagnostic              {neutron.representative_q2_over_m2:.9e}")
    print(" rare decay / invisible")
    print(f"  digitized central bound ratio   {rare.central_bound_to_candidate_ratio:.9e}")
    print(f"  conservative NLO allows         {rare.conservative_nlo_envelope_allows_candidate}")
    print(f"  invisible decay length m        {invisible.decay_length_m:.9e}")
    print(f" physical branch accepted         {report.physical_ce_fusion_branch_accepted}")
    print(f" classification                   {report.candidate_classification}")


if __name__ == "__main__":
    main()
