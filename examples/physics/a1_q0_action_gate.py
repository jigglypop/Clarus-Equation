"""Run the conditional A1-to-Q0 local algebra gates."""

from __future__ import annotations

from reality_stone.clarus.a1_q0_action_bridge import (
    a1_q0_action_report,
    audit_higgs_invisible_width,
)


def main() -> None:
    report = a1_q0_action_report(
        action_gradient_x=3.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
        lambda_hp=0.13,
        higgs_vev=246.0,
    )
    hessian = report.hessian_coordinate_audit
    portal = report.portal_vacuum_audit
    legacy_width = audit_higgs_invisible_width(
        lambda_hp=0.0316,
        higgs_vev=246.22,
        higgs_mass=125.25,
        scalar_mass=43.77,
        sm_higgs_width=0.00407,
        branching_fraction_upper_limit=0.11,
    )

    print("CE A1 -> Q0 ACTION BRIDGE CONDITIONAL GATE")
    print(f"  scope                       {report.scope}")
    print(f"  status                      {report.conditional_status}")
    print(f"  tensor pullback Hessian     {hessian.tensor_pullback_hessian_y:.12g}")
    print(f"  ordinary Hessian            {hessian.ordinary_hessian_y:.12g}")
    print(f"  non-tensor extra term       {hessian.non_tensor_extra_term:.12g}")
    print(f"  covariant Hessian           {hessian.covariant_hessian_y:.12g}")
    print(f"  h-phi cross Hessian         {portal.h_phi_cross_hessian:.12g}")
    print(f"  phi mass shift              {portal.phi_mass_shift:.12g}")
    print(f"  h-phi-phi cubic             {portal.h_phi_phi_cubic:.12g}")
    print(f"  h-h-phi-phi quartic         {portal.h_h_phi_phi_quartic:.12g}")
    print(
        "  legacy portal width (GeV)   "
        f"{legacy_width.partial_width:.12g}"
    )
    print(
        "  legacy portal invisible BR  "
        f"{legacy_width.branching_fraction:.12g}"
    )
    print(
        "  legacy benchmark allowed    "
        f"{legacy_width.benchmark_allowed}"
    )
    print(f"  covariant action complete   {report.covariant_action_complete}")
    print(f"  stress tensor derived       {report.stress_tensor_derived}")
    print(f"  spectral density derived    {report.spectral_density_derived}")
    print(f"  conclusion                  {report.conclusion}")


if __name__ == "__main__":
    main()
