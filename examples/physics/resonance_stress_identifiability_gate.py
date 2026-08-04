from __future__ import annotations

from reality_stone.clarus.resonance_stress_identifiability import (
    ce_resonance_bridge_audit,
    pole_family_countermodel_audit,
)


def main() -> None:
    q_value = 1.0 / 6.65e-15
    audit = pole_family_countermodel_audit(resonance_q=q_value)
    bridge = ce_resonance_bridge_audit()

    print("CE RESONANCE-STRESS IDENTIFIABILITY LOOP")
    print(" Q", audit.resonance_q)
    print(" same correlation gain", audit.all_countermodels_have_same_correlation_length)
    for model in audit.countermodels:
        print(
            " p",
            model.requested_stress_exponent,
            "residue gain",
            model.residue_gain,
            "proxy gain",
            model.dimensional_stress_proxy_gain,
        )
    print(" stress scaling unique", audit.stress_scaling_unique_from_correlation_length)
    print(" physical null stress", audit.physical_null_stress_derived)
    print(" CE maximum stage", bridge.maximum_supported_stage)


if __name__ == "__main__":
    main()
