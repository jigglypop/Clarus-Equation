from __future__ import annotations

from reality_stone.clarus.fusion_operator_alternatives_loop import (
    current_fusion_operator_alternatives_report,
)


def main() -> None:
    report = current_fusion_operator_alternatives_report()
    trace = report.trace_gluon
    isospin = report.isospin
    disformal = report.disformal
    print("CE FUSION OPERATOR ALTERNATIVES")
    print(" trace / gluon")
    print(f"  required f/K GeV                {trace.required_scale_over_trace_coefficient_gev:.9e}")
    print(f"  required / rare-decay bound     {trace.required_to_bound_ratio:.9e}")
    print(" isospin endpoints")
    print(f"  protophobic Pb / bound          {isospin.protophobic_to_neutron_bound_ratio:.9e}")
    print(f"  neutron-phobic kaon violations  {isospin.neutron_phobic_kaon_combination_one_violation:.9e} / {isospin.neutron_phobic_kaon_combination_two_violation:.9e}")
    print(f"  Pb cancellation D-T coefficient {isospin.lead_cancellation_dt_product_coefficient:.9e}")
    print(" disformal massless upper")
    print(f"  M for 1 percent MeV             {disformal.required_scale_for_one_percent_mev:.9e}")
    print(f"  gain at M=200 MeV               {disformal.gain_at_hydrogen_bound:.9e}")
    print(f"  gain at M=810 MeV               {disformal.gain_at_stellar_bound:.9e}")
    print(f"  gain at M=1.2 TeV               {disformal.gain_at_atlas_bound:.9e}")
    print(f" any alternative cleared          {report.any_alternative_constraint_cleared}")
    print(f" maximum supported stage          {report.maximum_supported_stage}")


if __name__ == "__main__":
    main()
