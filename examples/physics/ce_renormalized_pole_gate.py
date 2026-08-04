"""Print the current CE pole stage and optional portal one-loop diagnostic."""

from __future__ import annotations

import json

from reality_stone.clarus.portal_one_loop_control import (
    audit_portal_one_loop_scalar_control,
)
from reality_stone.clarus.renormalized_pole_certificate import (
    current_ce_renormalized_pole_certificate,
)


def main() -> None:
    """Run the fail-closed current-evidence and scalar-loop controls."""

    pole = current_ce_renormalized_pole_certificate()
    loop = audit_portal_one_loop_scalar_control()
    print(
        json.dumps(
            {
                "current_ce_pole_certificate": pole.to_dict(),
                "optional_portal_one_loop_control": loop.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("\nSummary")
    print(f"current CE stage: {pole.maximum_supported_stage.value}")
    print(f"first blocker: {pole.first_blocker}")
    print(
        "two-real-scalar finite loop / target mass^2: "
        f"{loop.finite_sum_to_target_mass_squared_ratio:.6f}"
    )
    print(f"bubble residue derivative: {loop.bubble_self_energy_derivative:.9e}")
    print(f"first h-phi cut energy: {loop.first_mixed_cut_energy_gev:.9f} GeV")
    print(f"physical CE pole derived: {pole.physical_lsz_particle_derived}")


if __name__ == "__main__":
    main()
