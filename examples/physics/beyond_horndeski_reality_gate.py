from __future__ import annotations

from reality_stone.clarus.beyond_horndeski_reality import (
    beyond_horndeski_reality_audit,
    ce_higher_derivative_extension_audit,
)


def main() -> None:
    audit = beyond_horndeski_reality_audit()
    extension = ce_higher_derivative_extension_audit()
    print("BEYOND-HORNDESKI REALITY GATE")
    for candidate in audit.candidates:
        print(candidate.name, "=>", candidate.verdict)
        print(" complete same-model pass", candidate.complete_same_model_pass)
    print(" GW speed relative bound", audit.gw_speed_relative_bound)
    print(" cross-model evidence splicing", audit.cross_model_evidence_splicing_allowed)
    print(" current reality pass", audit.current_reality_pass)
    print(" lone CE higher-derivative Hessian", extension.highest_derivative_hessian)
    print(" valid CE DHOST extension", extension.valid_minimal_extension)


if __name__ == "__main__":
    main()
